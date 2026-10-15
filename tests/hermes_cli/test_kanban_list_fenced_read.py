"""Invariant: ``kanban list`` answers a fenced reader (follow-up to t_346363f3, card t_b5eaf819).

``_cmd_list`` ran ``kb.recompute_ready`` — a ``write_txn`` — before listing, so the one read verb
a cabinet agent is told to run was the one read verb that failed from its shell: a
dispatcher-spawned worker carries ``HERMES_DELEGATED_CHILD_CONTEXT`` (stamped by the launcher,
which also strips ``HERMES_KANBAN_TASK``), that marker fences the board root, and the refused
promotion took the whole listing down with ``exit=1`` and "delegate_task child contexts cannot
mutate Kanban tasks or boards" — the wrong failure for a read, described as a write the caller
never asked for.

``kb.recompute_ready_for_read`` makes that promotion best-effort: skipped (``None``) when writes
at this board are fenced for this process, because promotions are the dispatcher tick's job and
it re-runs them on every pass. Every reader that MAY write still gets the mini-dispatch, and the
fence itself is untouched — the same fenced shell is still refused ``create``, ``claim``, ``edit``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc

# The child processes below import ``hermes_cli`` through whatever is installed; pin the repo
# root so they always import the tree under test (same reason as
# tests/tools/test_delegate_kanban_isolation.py).
_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def board(tmp_path, monkeypatch):
    """A board with one promotion-ready pair and one pair that must stay put.

    ``promote me`` is ``todo`` with a ``done`` parent -> ``recompute_ready`` promotes it.
    ``hold me`` is ``todo`` with a ``ready`` parent -> it must stay ``todo`` either way.
    """
    root = tmp_path / ".hermes"
    root.mkdir()
    db = root / "kanban.db"
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(root))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.discard(str(db.resolve()))
    kb.init_db()
    with kbc.connect() as conn:
        done = kb.create_task(conn, title="done parent", assignee="cab", created_by="probe")
        hold = kb.create_task(conn, title="open parent", assignee="cab", created_by="probe")
        promote = kb.create_task(conn, title="promote me", assignee="cab", created_by="probe",
                                 parents=[done])
        held = kb.create_task(conn, title="hold me", assignee="cab", created_by="probe",
                              parents=[hold])
        # create_task runs its own mini-dispatch; force the pair back so each test measures the
        # promotion the *read* performs, not the one the setup performed.
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='done' WHERE id=?", (done,))
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (hold,))
            conn.execute("UPDATE tasks SET status='todo' WHERE id IN (?,?)", (promote, held))
    return SimpleNamespace(root=root, db=db, promote=promote, held=held)


def _child_env(board, *, fenced: bool) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_REPO_ROOT)
    env["HERMES_HOME"] = str(board.root)
    env["HERMES_KANBAN_HOME"] = str(board.root)
    env["HERMES_KANBAN_DB"] = str(board.db)
    env.pop("HERMES_KANBAN_TASK", None)
    if fenced:
        env["HERMES_DELEGATED_CHILD_CONTEXT"] = str(board.root)
    else:
        env.pop("HERMES_DELEGATED_CHILD_CONTEXT", None)
    return env


def _run_cli(board, argv, *, fenced: bool) -> subprocess.CompletedProcess:
    """Run one ``hermes kanban <argv>`` in a real child process, through the CLI dispatcher."""
    code = (
        "from hermes_cli import kanban; import argparse; "
        "p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='cmd'); "
        "kanban.build_parser(sub); "
        f"args=p.parse_args(['kanban'] + {list(argv)!r}); "
        "raise SystemExit(kanban.kanban_command(args))"
    )
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env=_child_env(board, fenced=fenced), cwd=str(board.root), timeout=60)


def _status(board, task_id: str) -> str:
    with kbc.connect() as conn:
        task = kb.get_task(conn, task_id)
    assert task is not None, task_id
    return task.status


def _fence(board, monkeypatch) -> None:
    monkeypatch.setenv("HERMES_DELEGATED_CHILD_CONTEXT", str(board.root))


# ---------------------------------------------------------------------------
# The defect: a fenced reader must get its list
# ---------------------------------------------------------------------------


def test_fenced_reader_gets_its_list_where_the_read_used_to_die(board):
    """The card's command: exit 0, the rows, and no promotion performed from this shell."""
    result = _run_cli(board, ["list", "--assignee", "cab"], fenced=True)

    assert result.returncode == 0, result.stderr
    assert "cannot mutate" not in result.stderr
    assert "promote me" in result.stdout and "hold me" in result.stdout
    # Honest about what it skipped: stderr only, so stdout/--json stays machine-readable.
    assert "read-only here" in result.stderr
    assert "promote me" not in result.stderr
    assert _status(board, board.promote) == "todo"


def test_fenced_reader_can_still_use_json(board):
    result = _run_cli(board, ["list", "--assignee", "cab", "--json"], fenced=True)

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert {row["title"] for row in payload} == {"done parent", "open parent", "promote me", "hold me"}


def test_fenced_reader_is_still_refused_every_write(board):
    """The fence is untouched: the same shell that may now read may still not write."""
    with kbc.connect() as conn:
        before = len(kb.list_tasks(conn))

    result = _run_cli(board, ["create", "fence probe", "--assignee", "cab"], fenced=True)

    assert result.returncode == 1
    assert "cannot mutate Kanban tasks" in result.stderr
    with kbc.connect() as conn:
        assert len(kb.list_tasks(conn)) == before
    assert _status(board, board.promote) == "todo"


def test_unfenced_list_still_runs_the_mini_dispatch(board):
    """The convenience the read used to provide is not lost for readers that may write."""
    result = _run_cli(board, ["list", "--assignee", "cab"], fenced=False)

    assert result.returncode == 0, result.stderr
    assert "read-only here" not in result.stderr
    assert _status(board, board.promote) == "ready"
    assert _status(board, board.held) == "todo"


def test_in_process_delegate_child_gets_its_list(board):
    """An in-process delegate child (ContextVar fence, no spawned marker) reads too."""
    from agent.delegation_context import delegated_child_context

    with delegated_child_context():
        out = kc.run_slash("list --assignee cab")

    assert "promote me" in out
    assert "cannot mutate" not in out
    assert "read-only here" in out
    assert _status(board, board.promote) == "todo"


def test_kanban_list_tool_reports_skipped_promotions(board, monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    _fence(board, monkeypatch)

    from tools import kanban_tools as kt

    payload = json.loads(kt._handle_list({"assignee": "cab", "limit": 10}))

    assert payload["promotions_skipped"] is True
    assert payload["promoted"] == 0
    assert "promote me" in {t["title"] for t in payload["tasks"]}
    assert _status(board, board.promote) == "todo"


# ---------------------------------------------------------------------------
# The helper's own contract
# ---------------------------------------------------------------------------


def test_recompute_ready_for_read_promotes_when_the_reader_may_write(board):
    with kbc.connect() as conn:
        assert kb.recompute_ready_for_read(conn) == 1
    assert _status(board, board.promote) == "ready"


def test_recompute_ready_for_read_returns_none_when_fenced(board, monkeypatch):
    _fence(board, monkeypatch)
    with kbc.connect() as conn:
        assert kb.recompute_ready_for_read(conn) is None
    assert _status(board, board.promote) == "todo"
