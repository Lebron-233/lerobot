"""Run the registered D3 child once, with the original 295/300-second limits."""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-head", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--preparation", required=True, type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    output, prep = args.output.resolve(), args.preparation.resolve()
    registration = json.loads((prep / "registration_readback.json").read_text())
    if registration["body"] != (prep / "registration.md").read_text():
        raise ValueError("Registration readback differs")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if head != args.execution_head:
        raise ValueError("Execution HEAD changed")
    expected = repo / "outputs" / f"smolvla_graph_async_contract_{head[:8]}"
    if output != expected or output.exists():
        raise ValueError("Registered exclusive output path is invalid or already exists")
    python = "/home/rp/Workspace/SmolVLA_RTC/libero-reference-venv/bin/python"
    if Path(sys.executable) != Path(python):
        raise ValueError("Supervisor must use the unchanged model interpreter")
    command = [
        python,
        "-u",
        "examples/advanced/predictive_async/validate_smolvla_graph_worker.py",
        "--execution-head",
        head,
        "--output",
        str(output),
    ]
    if head not in registration["body"] or str(output) not in registration["body"]:
        raise ValueError("Registration does not name this HEAD and output")
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    log = prep / "model.log"
    started_at = datetime.now(UTC).isoformat()
    start = time.monotonic()
    soft_timeout = hard_timeout = False
    # Exclusive log also prevents rerun after a failure before output creation.
    with log.open("x") as stream:
        child = subprocess.Popen(
            command, cwd=repo, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
        )
        print(f"D3-r1 single attempt: head={head} pid={child.pid} hard_limit=300s", flush=True)
        try:
            exit_code = child.wait(timeout=295)
        except subprocess.TimeoutExpired:
            soft_timeout = True
            os.killpg(child.pid, signal.SIGTERM)
            try:
                exit_code = child.wait(timeout=max(0.01, 300 - (time.monotonic() - start)))
            except subprocess.TimeoutExpired:
                hard_timeout = True
                os.killpg(child.pid, signal.SIGKILL)
                exit_code = child.wait()
    receipt = {
        "execution_head": head,
        "supervisor_python": sys.executable,
        "command": command,
        "cwd": str(repo),
        "environment_overrides": {"PYTHONPATH": None, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "wall_seconds": time.monotonic() - start,
        "soft_shutdown_seconds": 295,
        "max_wall_seconds": 300,
        "soft_timeout": soft_timeout,
        "hard_timeout": hard_timeout,
        "child_pid": child.pid,
        "child_exit_code": exit_code,
        "child_exit_confirmed": True,
        "normal_exit": exit_code == 0 and not soft_timeout,
        "attempts": 1,
        "retries": 0,
        "output": str(output),
        "preparation": str(prep),
        "registration_url": registration["html_url"],
        "registration_readback_body_exact": True,
    }
    output.mkdir(parents=True, exist_ok=True)
    with (output / "execution.json").open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    for name in (
        "model.log",
        "registration.md",
        "registration_readback.json",
        "model_environment_before.json",
        "cpu_environment_before.json",
        "model_import_tests.log",
        "cpu_targeted_tests.log",
        "model_entry_help.log",
    ):
        shutil.copyfile(prep / name, output / name)
    shutil.copyfile(__file__, output / "supervise.py")
    print(json.dumps(receipt), flush=True)
    return 0 if receipt["normal_exit"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
