"""Run edited ``train.py`` code inside a ``unionai-sandbox`` interactive session."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

OOM_MARKERS = (
    "out of memory",
    "oom",
    "cannot allocate memory",
    "can't allocate memory",
    "unable to allocate",
    "memoryerror",
    "killed",
    "signal 9",
    "std::bad_alloc",
    "defaultcpuallocator",
    "bad_alloc",
)


def is_oom(stderr: str, returncode: int | None, *, stdout: str = "") -> bool:
    """Detect OOM from sandbox stderr / exit code (137 = SIGKILL/OOM-kill)."""
    if returncode in (137, -9):
        return True
    text = f"{stderr}\n{stdout}".lower()
    return any(marker in text for marker in OOM_MARKERS)


def parse_metrics(stdout: str) -> dict[str, Any] | None:
    """Parse the ``AUTORESEARCH_METRICS=`` line emitted by the driver script."""
    for line in stdout.splitlines():
        if line.startswith("AUTORESEARCH_METRICS="):
            return json.loads(line.split("=", 1)[1])
    return None


def write_driver_script(title: str, time_budget_sec: int, eval_tokens: int) -> str:
    """Return a small driver that imports the agent-edited ``train.py`` and prints metrics."""
    return textwrap.dedent(
        f'''
        import json
        import os
        import sys

        workdir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(workdir)
        os.environ["AUTORESEARCH_CACHE"] = workdir
        sys.path.insert(0, workdir)
        os.environ.setdefault("AUTORESEARCH_EVAL_TOKENS", "{eval_tokens}")

        from autoresearch_types import ExperimentConfig
        import train

        overrides = {{}}
        overrides_path = os.path.join(workdir, "config_overrides.json")
        if os.path.exists(overrides_path):
            with open(overrides_path) as f:
                overrides = json.load(f)

        config = ExperimentConfig(title={title!r}, time_budget_sec={time_budget_sec})
        if overrides:
            import dataclasses
            config = dataclasses.replace(config, **overrides)
        result = train.run_training(config)
        payload = {{
            "title": result.title,
            "val_bpb": round(result.val_bpb, 6),
            "model_name": result.model_name,
            "n_params": result.n_params,
            "steps": result.steps,
            "device": result.device,
            "notes": result.notes,
        }}
        print("AUTORESEARCH_METRICS=" + json.dumps(payload))
        '''
    ).strip()


def stage_sandbox_files(
    work_dir: str,
    train_py: str,
    *,
    title: str,
    time_budget_sec: int,
    eval_tokens: int | None = None,
    config_overrides: dict[str, Any] | None = None,
) -> Path:
    """Copy support modules + edited train code into the sandbox work directory."""
    import autoresearch_types
    import prepare

    if eval_tokens is None:
        eval_tokens = 32 * prepare.MAX_SEQ_LEN
    root = Path(work_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "train.py").write_text(train_py)
    if config_overrides:
        (root / "config_overrides.json").write_text(json.dumps(config_overrides))
    (root / "prepare.py").write_text(Path(prepare.__file__).read_text())
    (root / "autoresearch_types.py").write_text(Path(autoresearch_types.__file__).read_text())
    driver = write_driver_script(title, time_budget_sec, eval_tokens)
    driver_path = root / "driver.py"
    driver_path.write_text(driver)
    return driver_path


async def run_train_in_sandbox(
    work_dir: str,
    train_py: str,
    *,
    title: str,
    time_budget_sec: int,
    config_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute ``train.py`` via ``async with sb.on_device.session(backend='userns')``."""
    from union import sandbox as sb

    driver_path = stage_sandbox_files(
        work_dir,
        train_py,
        title=title,
        time_budget_sec=time_budget_sec,
        config_overrides=config_overrides,
    )
    timeout_s = max(time_budget_sec + 180, 300)

    try:
        async with sb.on_device.session(backend="userns", host_work_dir=work_dir) as sbx:
            proc = await sbx.run(
                f"python {driver_path}",
                stdout=True,
                stderr=True,
                network_mode="blocked",
                timeout_s=timeout_s,
            )
            stdout, stderr = await proc.communicate_text()
    except Exception as exc:
        err_text = str(exc)
        oom = is_oom(err_text, None)
        return {
            "success": False,
            "oom": oom,
            "title": title,
            "exit_code": None,
            "stdout_tail": "",
            "stderr": err_text,
            "error": (
                "Training run was OOM-killed; the platform will retry with more memory."
                if oom
                else f"Sandbox execution failed: {err_text}"
            ),
        }

    metrics = parse_metrics(stdout or "")
    oom = is_oom(stderr or "", proc.returncode, stdout=stdout or "")

    if metrics is not None and proc.returncode == 0:
        return {
            "success": True,
            "oom": False,
            **metrics,
            "exit_code": proc.returncode,
            "stderr_tail": (stderr or "")[-800:],
        }

    return {
        "success": False,
        "oom": oom,
        "title": title,
        "exit_code": proc.returncode,
        "stdout_tail": (stdout or "")[-1500:],
        "stderr": stderr or "",
        "error": (
            "Training run was OOM-killed; the platform will retry with more memory."
            if oom
            else f"Training failed (exit {proc.returncode}). See stderr for details."
        ),
    }
