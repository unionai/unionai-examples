"""The `environments` layer of Modal's SWE-Gym example, on Union sandboxes.

Same interface as Modal's `helpers/environments.py` (`Environments`), async:

    async with envs.sandbox(inst) as sb      a sandbox on a fresh copy of the instance's testbed, network blocked
    await envs.prepare_agent(sb)             strip git history so the fix can't be read from it
    rc, out = await envs.command(sb, cmd, timeout)
    patch = await envs.patch(sb)             `git diff` of what the agent changed
    report = await envs.grade(inst, patch, path)   hidden tests in another fresh copy; report["reward"]
    tb = await envs.image(inst)              the instance's testbed (Modal: its image)

What differs is underneath. Modal starts each sandbox from the instance's Docker image. Here
`image(inst)` builds the testbed once, in a sandbox with network, and commits its disk as a
`SandboxVolume`: sealed, never written again. Every attempt (and every grading run) is a new
sandbox on a copy-on-write fork of that volume, so four attempts at one instance start at once,
in a couple of seconds, and none sees what another wrote.

It is the same code in a Flyte task, in an app, and on a laptop: from outside the cluster the
sandbox SDK reaches each sandbox through Union's data proxy, as the user who is signed in.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import shlex
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import flyte
import swe_gym_tasks as tasks

from flyteplugins.union import sandbox
from flyteplugins.union.sandbox import Sandbox, SandboxEnvironment, SandboxVolume

logger = logging.getLogger(__name__)

#: How often a sandbox is asked for again when the node it went to is full.
SESSION_TRIES = 6


class InfrastructureError(RuntimeError):
    """The sandbox, not the agent, failed: the attempt is retried, not scored."""


@dataclass
class Testbed:
    """An instance's sealed testbed (Modal: its image)."""

    instance_id: str
    volume: SandboxVolume
    #: PASS_TO_PASS tests that pass with the gold patch on this testbed (None: not validated).
    pass_to_pass: Optional[List[str]] = None


def _short(inst: Dict[str, Any]) -> str:
    """`getmoto__moto-7023` -> `moto-7023`: what its sandboxes are named for."""
    return inst["instance_id"].split("__")[-1].lower()


class UnionEnvironments:
    """Modal's `Environments`, on Union sandboxes.

    ``attempt_env`` runs the agent and the hidden tests: no network, so nothing the agent runs
    can reach out, and no secrets. ``build_env`` builds testbeds: it needs the network for apt,
    PyPI and GitHub, and is used once per instance.

    ``cache_testbeds`` publishes each testbed as an artifact and starts from a published one
    when there is one: built once ever, not once per process (as a Modal image is).

    ``labels`` go on every sandbox it starts. Outside a task each sandbox is a run of its own,
    so one label shared by all of them (``{"swe-gym-run": "0612-a3f9"}``) groups a job's
    sandboxes together.

    ``queue`` sends every sandbox to that queue's cluster instead of the project's default one.
    """

    def __init__(
        self,
        *,
        attempt_env: Optional[SandboxEnvironment] = None,
        build_env: Optional[SandboxEnvironment] = None,
        validate: bool = True,
        cache_testbeds: bool = False,
        labels: Optional[Dict[str, str]] = None,
        queue: str = "",
    ) -> None:
        self.validate = validate
        self.cache_testbeds = cache_testbeds
        self.labels = labels
        self.queue = queue
        self.attempt_env = attempt_env or SandboxEnvironment(
            name="swe-attempt",
            resources=flyte.Resources(cpu=2, memory="4Gi"),
            max_duration=7200,
            network=sandbox.Network.blocked(),
            labels=labels,
            queue=queue,
        )
        self.build_env = build_env or SandboxEnvironment(
            name="swe-testbed-build",
            resources=flyte.Resources(cpu=2, memory="4Gi"),
            max_duration=3600,
            network=sandbox.Network.internet(),
            labels=labels,
            queue=queue,
        )
        self.testbeds: Dict[str, Testbed] = {}
        self._building: Dict[str, asyncio.Lock] = {}

    # -- Modal's interface --------------------------------------------------------------------

    async def image(self, inst: Dict[str, Any]) -> Testbed:
        """The instance's testbed, built the first time it is asked for (once, however many
        attempts ask at the same time)."""
        key = inst["instance_id"]
        async with self._building.setdefault(key, asyncio.Lock()):
            if key not in self.testbeds:
                self.testbeds[key] = await self._testbed(inst)
        return self.testbeds[key]

    def adopt(self, testbed: Testbed) -> None:
        """Uses a testbed built elsewhere (another task's output)."""
        self.testbeds[testbed.instance_id] = testbed

    @contextlib.asynccontextmanager
    async def sandbox(self, inst: Dict[str, Any], purpose: str = "attempt") -> AsyncIterator[Sandbox]:
        """A sandbox on a fresh copy of the instance's testbed, at its base commit."""
        testbed = await self.image(inst)
        name = f"{purpose}-{_short(inst)}-{uuid.uuid4().hex[:6]}"
        async with self._session(self.attempt_env, name, volume=testbed.volume) as sb:
            rc, output = await self.command(
                sb, f"cd /testbed && git reset --hard {shlex.quote(inst['base_commit'])} && git clean -fd"
            )
            if rc:
                raise InfrastructureError("Cannot reset repository: " + output[-2000:])
            yield sb

    async def prepare_agent(self, sb: Sandbox) -> None:
        rc, out = await self.command(
            sb,
            """set -e
rm -f /eval.sh /patch.diff /test.patch /root/patch.diff
cd /testbed
git checkout --detach HEAD
git remote | xargs -r -n1 git remote remove
git for-each-ref --format='%(refname)' | while read r; do git update-ref -d "$r"; done
git reflog expire --expire=now --all
git gc --prune=now
""",
            180,
        )
        if rc:
            raise InfrastructureError("Agent preparation failed: " + out[-2000:])

    async def command(self, sb: Sandbox, code: str, timeout: float = 120) -> Tuple[int, str]:
        """Runs `code` under `bash -lc`; returns (exit code, stdout + stderr). 124 on timeout."""
        r = await sb.run(["/bin/bash", "-lc", code], timeout=timeout, check=False)
        return (124 if r.timed_out else r.exit_code), r.stdout + r.stderr

    async def patch(self, sb: Sandbox) -> str:
        rc, out = await self.command(
            sb,
            "cd /testbed && git add -A && git diff --cached --binary HEAD -- . "
            "':(exclude)*.pyc' ':(exclude)__pycache__' ':(exclude).pytest_cache'",
        )
        if rc or len(out) > 2_000_000:
            raise InfrastructureError(f"Cannot capture patch: exit={rc}, bytes={len(out)}; {out[-2000:]}")
        return out

    async def grade(self, inst: Dict[str, Any], patch: str, directory: str | Path) -> Dict[str, Any]:
        """Applies `patch` in a sandbox of its own and runs the hidden tests: `report["reward"]`."""
        directory = Path(directory) / inst["instance_id"].lower()
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "prediction.patch").write_text(patch)
        testbed = await self.image(inst)
        async with self.sandbox(inst, purpose="grade") as sb:
            if patch:
                await sb.fs.write_text("/prediction.patch", patch)
                rc, out = await self.command(sb, "cd /testbed && git apply --whitespace=nowarn /prediction.patch")
                if rc:
                    (directory / "apply.log").write_text(out)
                    return {"reward": 0.0, "invalid_patch": True}
            await sb.fs.write_text("/tmp/hidden_test.patch", inst["test_patch"])
            await sb.fs.write_text("/hidden_eval.sh", tasks.eval_script(inst))
            rc, out = await self.command(sb, "bash /hidden_eval.sh", 1200)
            (directory / "test_output.txt").write_text(out)
            report = tasks.grade_log(inst, out, testbed.pass_to_pass)
            if not report["parsed"]:
                raise InfrastructureError(f"No trustworthy test results: {directory / 'test_output.txt'}; exit={rc}")
            report["reward"] = float(report["resolved"])
            (directory / "report.json").write_text(json.dumps(report, indent=2))
            return report

    # -- Testbeds -----------------------------------------------------------------------------

    async def _testbed(self, inst: Dict[str, Any]) -> Testbed:
        """A published testbed that passes its own gold patch, or one built now."""
        if self.cache_testbeds:
            testbed = await self._published(inst)
            if testbed is not None:
                try:
                    return await self._validated(testbed, inst)
                except InfrastructureError as e:
                    # Someone's earlier build, interrupted: built afresh, and published in its place.
                    logger.warning("published testbed for %s is not usable (%s); building it", testbed.instance_id, e)
        return await self._validated(await self._build(inst), inst)

    async def _published(self, inst: Dict[str, Any]) -> Optional[Testbed]:
        """The testbed an earlier run published for this instance and recipe, if any."""
        try:
            volume = await SandboxVolume.from_artifact(tasks.testbed_artifact(inst))
        except Exception as e:  # none yet (or unreadable)
            logger.info("no published testbed for %s (%s)", inst["instance_id"], e)
            return None
        return Testbed(inst["instance_id"], volume)

    async def _build(self, inst: Dict[str, Any]) -> Testbed:
        """Builds the testbed in a sandbox with network, and commits its disk."""
        env = self.build_env
        if self.cache_testbeds:
            # A session with `publish=` has its disk published when it closes: here under a
            # name of the instance and its recipe, for the next run (and anyone's) to start from.
            env = SandboxEnvironment(
                name=f"swe-build-{_short(inst)}".replace("_", "-"),
                resources=env.resources,
                max_duration=env.max_duration,
                network=env.network,
                publish=tasks.testbed_artifact(inst),
                labels=self.labels,
                queue=self.queue,
            )
        async with self._session(env, f"build-{_short(inst)}") as sb:
            try:
                r = await sb.run(["/bin/bash", "-c", tasks.build_script(inst)], timeout=3000, check=False)
                if r.exit_code:
                    raise InfrastructureError(
                        f"testbed build for {inst['instance_id']} failed:\n{(r.stdout + r.stderr)[-4000:]}"
                    )
                volume = await sb.commit()  # sealed: sandboxes start on copies of it, never on it
            except BaseException:
                # Closing would publish a half-built disk as the testbed; cancelling publishes nothing.
                with contextlib.suppress(Exception):
                    await asyncio.shield(sb.cancel("testbed build did not finish"))
                raise
        return Testbed(inst["instance_id"], volume)

    async def _validated(self, testbed: Testbed, inst: Dict[str, Any]) -> Testbed:
        """Runs the hidden tests with the gold patch, on a copy of the testbed (which never sees
        the patch). FAIL_TO_PASS must pass; PASS_TO_PASS tests that fail here fail for the
        testbed, not the agent, and are left out of grading."""
        if not self.validate:
            return testbed
        async with self._session(self.attempt_env, f"validate-{_short(inst)}", volume=testbed.volume) as sb:
            await sb.fs.write_text("/tmp/gold.patch", inst["patch"])
            await sb.fs.write_text("/tmp/hidden_test.patch", inst["test_patch"])
            await sb.fs.write_text("/tmp/eval.sh", tasks.eval_script(inst))
            r = await sb.run(
                ["/bin/bash", "-c", "cd /testbed && git apply /tmp/gold.patch && bash /tmp/eval.sh"],
                timeout=1200,
                check=False,
            )
        report = tasks.grade_log(inst, r.stdout + r.stderr)
        if report["FAIL_TO_PASS"]["failure"]:
            raise InfrastructureError(
                f"{inst['instance_id']}: the gold patch fails {report['FAIL_TO_PASS']['failure']} on this testbed:\n"
                f"{(r.stdout + r.stderr)[-3000:]}"
            )
        if report["PASS_TO_PASS"]["failure"]:
            logger.warning(
                "%s: %d PASS_TO_PASS tests fail on this testbed; not graded",
                inst["instance_id"],
                len(report["PASS_TO_PASS"]["failure"]),
            )
        testbed.pass_to_pass = report["PASS_TO_PASS"]["success"]
        return testbed

    # -- Sandboxes ----------------------------------------------------------------------------

    @contextlib.asynccontextmanager
    async def _session(self, env: SandboxEnvironment, name: str, **disk: Any) -> AsyncIterator[Sandbox]:
        """`env.session(name)`, asked for again when the node it went to is full."""
        async with contextlib.AsyncExitStack() as stack:
            for attempt in range(SESSION_TRIES):
                try:
                    sb = await stack.enter_async_context(env.session(name, **disk))
                    break
                except sandbox.SandboxSessionError as e:
                    if "refused" not in str(e) or attempt == SESSION_TRIES - 1:
                        raise InfrastructureError(f"sandbox {name} did not start: {e}") from e
                    await asyncio.sleep(2)
            yield sb
