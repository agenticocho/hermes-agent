import json
import os
from pathlib import Path


def test_router_uses_protected_api_key_file(tmp_path, monkeypatch):
    from hermes_cli.local_runtime import supervisor as supervisor_module

    runtime_root = tmp_path / "runtimes" / "llamacpp"
    install_dir = runtime_root / "b10107" / "cuda"
    models_dir = tmp_path / "models"
    captured = {}

    class FakeProcess:
        pid = 424242

        @staticmethod
        def poll():
            return None

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(
        supervisor_module,
        "runtimes_root",
        lambda: runtime_root,
    )
    monkeypatch.setattr(
        supervisor_module,
        "server_binary",
        lambda _install_dir: Path("/fake/llama-server"),
    )
    monkeypatch.setattr(
        supervisor_module.subprocess,
        "Popen",
        fake_popen,
    )

    instance = supervisor_module.LlamaServerSupervisor(
        install_dir,
        models_dir,
        port=19099,
        log_path=tmp_path / "llama-server.log",
    )

    try:
        instance._spawn()
    finally:
        if instance._log_handle is not None:
            instance._log_handle.close()
            instance._log_handle = None

    command = captured["cmd"]
    key_path = supervisor_module.api_key_path()

    assert "--api-key-file" in command
    assert command[command.index("--api-key-file") + 1] == str(key_path)
    assert "--api-key" not in command
    assert instance.api_key not in command

    assert key_path.read_text(encoding="utf-8").strip() == instance.api_key
    if os.name != "nt":
        assert key_path.stat().st_mode & 0o777 == 0o600

    state = json.loads(
        supervisor_module.state_path().read_text(encoding="utf-8")
    )
    assert state["api_key"] == instance.api_key
    assert state["pid"] == FakeProcess.pid
    if os.name != "nt":
        assert supervisor_module.state_path().stat().st_mode & 0o777 == 0o600
