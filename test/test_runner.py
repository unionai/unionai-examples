import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import tomllib
import yaml
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Default number of test scripts to execute concurrently. Test submission is a
# blocking subprocess call (``uv run`` / ``flyte run``) with no async client, so the
# runner spawns the runs across a thread pool and collects results as they complete.
# The cap keeps in-flight cloud/local runs bounded so we don't overwhelm the backend.
DEFAULT_CONCURRENCY = 8

def get_verbosity_flag(verbose_arg: str) -> str:
    """Convert verbosity argument to flyte flag.

    Args:
        verbose_arg: Can be '', '0', '1', '2', '3', 'v', 'vv', 'vvv'

    Returns:
        Appropriate flyte verbosity flag or empty string
    """
    if not verbose_arg or verbose_arg == "0":
        return ""
    elif verbose_arg in ["1", "v"]:
        return "-v"
    elif verbose_arg in ["2", "vv"]:
        return "-vv"
    elif verbose_arg in ["3", "vvv"]:
        return "-vvv"
    else:
        # Default to -v for any other non-empty value
        return "-v"

def parse_flyte_config(config_path: Path) -> Dict[str, str]:
    """Parse Flyte config file to extract host, domain, and project."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Extract endpoint and remove dns:// prefix
        endpoint = config.get('admin', {}).get('endpoint', '')
        host = endpoint.replace('dns:///', '')

        # Extract domain and project from task section
        task_config = config.get('task', {})
        domain = task_config.get('domain', '')
        project = task_config.get('project', '')

        return {
            'host': host,
            'domain': domain,
            'project': project
        }
    except Exception as e:
        print(f"⚠️  Error parsing Flyte config: {e}")
        return {}

def extract_execution_url(stdout: Optional[str], config_info: Dict[str, str]) -> Optional[str]:
    """Extract Flyte execution URL from test stdout."""
    if not stdout or not all(config_info.values()):
        return None

    # Construct the base URL pattern
    host = config_info['host']
    domain = config_info['domain']
    project = config_info['project']

    # Pattern to match execution URLs
    url_pattern = rf"https://{re.escape(host)}/v2/runs/project/{re.escape(project)}/domain/{re.escape(domain)}/[a-zA-Z0-9\-_]+"

    # Search for the pattern in stdout
    match = re.search(url_pattern, stdout)
    if match:
        return match.group(0)

    return None

@dataclass
class TestResult:
    script_path: str
    status: str  # "passed", "failed", "timeout", "skipped"
    duration: float
    timestamp: str  # ISO format timestamp when test was run
    exit_code: Optional[int] = None
    error_message: Optional[str] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    log_file: Optional[str] = None
    execution_url: Optional[str] = None

    @classmethod
    def create_with_timestamp(cls, script_path: str, status: str, duration: float, **kwargs):
        """Create a TestResult with current timestamp."""
        return cls(
            script_path=script_path,
            status=status,
            duration=duration,
            timestamp=datetime.now(timezone.utc).isoformat(),
            **kwargs
        )

@dataclass
class TestConfig:
    timeout: int = 300  # 5 minutes default timeout
    excluded_patterns: List[str] = None
    required_env_vars: Dict[str, str] = None
    concurrency: int = DEFAULT_CONCURRENCY  # max test scripts to run in parallel

    def __post_init__(self):
        if self.excluded_patterns is None:
            self.excluded_patterns = []
        if self.required_env_vars is None:
            self.required_env_vars = {}
        if not self.concurrency or self.concurrency < 1:
            self.concurrency = DEFAULT_CONCURRENCY

def load_persistent_results(reports_dir: Path) -> Dict[str, TestResult]:
    """Load existing test results from persistent storage."""
    persistent_file = reports_dir / "persistent_results.json"

    if not persistent_file.exists():
        print("📋 No existing persistent results found - starting fresh")
        return {}

    try:
        with open(persistent_file, "r") as f:
            data = json.load(f)

        results = {}
        for item in data:
            # Convert dict back to TestResult object
            result = TestResult(**item)
            results[result.script_path] = result

        print(f"📋 Loaded {len(results)} existing test results from persistent storage")
        return results

    except Exception as e:
        print(f"⚠️  Error loading persistent results: {e}")
        print("📋 Starting with empty results")
        return {}

def download_previous_results_from_github_pages(reports_dir: Path, github_pages_url: Optional[str] = None):
    """Download previous test results from GitHub Pages if available."""
    if not github_pages_url:
        # Try to auto-detect from environment or use default
        repo_owner = os.getenv("GITHUB_REPOSITORY_OWNER", "unionai")
        repo_name = os.getenv("GITHUB_REPOSITORY", "unionai-examples").split("/")[-1]
        github_pages_url = f"https://{repo_owner}.github.io/{repo_name}"

    persistent_file = reports_dir / "persistent_results.json"
    download_url = f"{github_pages_url}/persistent_results.json"

    print(f"🔄 Attempting to download previous results from: {download_url}")

    try:
        with urllib.request.urlopen(download_url) as response:
            data = response.read()

        with open(persistent_file, "wb") as f:
            f.write(data)

        print(f"✅ Downloaded previous results from GitHub Pages")
        return True

    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"📋 No previous results found on GitHub Pages (404) - starting fresh")
        else:
            print(f"⚠️  HTTP error downloading results: {e.code} {e.reason}")
        return False
    except urllib.error.URLError as e:
        print(f"⚠️  Network error downloading results: {e.reason}")
        return False
    except Exception as e:
        print(f"⚠️  Error downloading previous results: {e}")
        return False

def save_persistent_results(results: Dict[str, TestResult], reports_dir: Path):
    """Save test results to persistent storage."""
    persistent_file = reports_dir / "persistent_results.json"

    try:
        # Convert TestResult objects to dicts for JSON serialization
        data = [asdict(result) for result in results.values()]

        with open(persistent_file, "w") as f:
            json.dump(data, f, indent=2)

        print(f"💾 Saved {len(results)} test results to persistent storage")

    except Exception as e:
        print(f"⚠️  Error saving persistent results: {e}")

def merge_test_results(existing_results: Dict[str, TestResult],
                      new_results: List[TestResult]) -> Dict[str, TestResult]:
    """Merge new test results with existing ones, updating only files that were tested."""
    merged = existing_results.copy()

    # Update with new results
    for result in new_results:
        merged[result.script_path] = result

    print(f"🔄 Merged results: {len(new_results)} newly tested, {len(merged)} total in history")
    return merged

def find_runnable_scripts(root_dir: Path, config: TestConfig) -> List[Path]:
    """Find all python scripts with flyte.init calls."""
    runnable_scripts = []

    for file_path in root_dir.rglob("*.py"):
        # Skip excluded patterns
        if any(pattern in str(file_path) for pattern in config.excluded_patterns):
            print(f"⏭️  Skipping {file_path} (excluded pattern)")
            continue

        try:
            with open(file_path, "r") as f:
                content = f.read()
                # Only include files with both main guard and flyte.init
                if 'if __name__ == "__main__":' in content and 'flyte.init' in content:
                    runnable_scripts.append(file_path)
        except (UnicodeDecodeError, PermissionError) as e:
            print(f"⚠️  Could not read {file_path}: {e}")
            continue

    return runnable_scripts

def parse_inline_metadata(script_path: Path) -> Dict[str, Any]:
    """Parse inline metadata from script PEP 723 block using reference implementation."""
    REGEX = r'(?m)^# /// (?P<type>[a-zA-Z0-9-]+)$\s(?P<content>(^#(| .*)$\s)+)^# ///$'

    try:
        with open(script_path, "r") as f:
            script_content = f.read()

        name = 'script'
        matches = list(
            filter(lambda m: m.group('type') == name, re.finditer(REGEX, script_content))
        )
        if len(matches) > 1:
            print(f"⚠️  Multiple {name} blocks found in {script_path}")
            return {}
        elif len(matches) == 1:
            content = ''.join(
                line[2:] if line.startswith('# ') else line[1:]
                for line in matches[0].group('content').splitlines(keepends=True)
            )
            metadata = tomllib.loads(content)
            print(f"   📋 Found PEP 723 metadata with {len(metadata.get('dependencies', []))} dependencies")
            return metadata
        else:
            print(f"   📋 No PEP 723 metadata block found in {script_path}")
            return {}

    except Exception as e:
        print(f"⚠️  Error parsing metadata from {script_path}: {e}")
        return {}

def create_isolated_venv_and_install_deps(script_path: Path, metadata: Dict[str, Any], root_dir: Path) -> tuple[bool, Optional[Path]]:
    """Create an isolated virtual environment and install dependencies from script metadata."""
    dependencies = metadata.get("dependencies", [])
    if not dependencies:
        return True, None

    try:
        # Create venv directory in the repo under test/venvs/
        venvs_dir = root_dir / "test" / "venvs"
        venvs_dir.mkdir(parents=True, exist_ok=True)

        # Derive a UNIQUE venv name from the script's path relative to the repo root.
        # Using just the filename stem would collide when two scripts share a name
        # (e.g. several main.py) — a real hazard now that local tests run concurrently.
        try:
            rel = script_path.relative_to(root_dir)
        except ValueError:
            rel = Path(script_path.name)
        safe_name = re.sub(r"[^A-Za-z0-9]+", "_", str(rel.with_suffix(""))).strip("_")
        venv_path = venvs_dir / f"{safe_name}_venv"

        # Clean up any existing venv for this script
        if venv_path.exists():
            shutil.rmtree(venv_path)

        # Create virtual environment
        create_venv_cmd = ["uv", "venv", str(venv_path)]
        result = subprocess.run(
            create_venv_cmd,
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode != 0:
            print(f"   ❌ Failed to create virtual environment")
            if result.stderr:
                print(f"   📦 Error: {result.stderr}")
            return False, None

        # Install dependencies using the isolated venv
        install_cmd = ["uv", "pip", "install", "--requirement", str(script_path)]

        # Set up environment to use the isolated venv
        env = os.environ.copy()
        env["VIRTUAL_ENV"] = str(venv_path)
        env["PATH"] = f"{venv_path}/bin:{env.get('PATH', '')}"

        result = subprocess.run(
            install_cmd,
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout for dependency installation
            env=env
        )

        if result.returncode != 0:
            print(f"   ❌ Failed to install dependencies")
            if result.stderr:
                print(f"   📦 Error: {result.stderr}")
            return False, None

        print(f"   ✅ Dependencies installed successfully")

        # Check and display Flyte version
        try:
            version_cmd = ["uv", "run", "python", "-c", "import flyte; print(flyte.__version__)"]
            version_result = subprocess.run(
                version_cmd,
                capture_output=True,
                text=True,
                timeout=10,
                env=env
            )
            if version_result.returncode == 0:
                flyte_version = version_result.stdout.strip()
                print(f"   🚀 Using Flyte version: {flyte_version}")
            else:
                print(f"   ⚠️  Could not determine Flyte version")
        except Exception:
            print(f"   ⚠️  Could not determine Flyte version")

        return True, venv_path

    except subprocess.TimeoutExpired:
        print(f"   ⏰ Virtual environment creation or dependency installation timed out")
        return False, None
    except Exception as e:
        print(f"   ❌ Error creating environment or installing dependencies: {e}")
        return False, None

def resolve_flyte_config_path(test_dir: Path) -> Optional[str]:
    """Resolve the absolute path to the Flyte config template.

    Returns the path as a string, or None if the template is missing. Unlike the
    previous ``setup_flyte_config_env`` this does NOT mutate ``os.environ`` — the
    config path is passed explicitly to each subprocess via its ``env`` mapping, which
    keeps the function side-effect free and therefore safe to use from worker threads.
    """
    config_path = test_dir / "config.flyte.yaml"
    if not config_path.exists():
        print(f"⚠️  Config template not found: {config_path}")
        return None
    return str(config_path.absolute())


# Serializes multi-line completion blocks so concurrent runs don't interleave output.
_PRINT_LOCK = threading.Lock()

_STATUS_EMOJI = {
    "passed": "✅",
    "failed": "❌",
    "timeout": "⏰",
    "skipped": "⏭️",
}


def run_single_test(
    script: Path,
    config: TestConfig,
    root_dir: Path,
    flyte_config_path: Optional[str],
    config_info: Dict[str, str],
) -> TestResult:
    """Run a single example script against the cloud Flyte backend and return its result.

    Pure worker: it runs the script in a subprocess and returns a ``TestResult``. It
    does not stream output or write log files — the caller does that once the result is
    in hand, so concurrent runs don't interleave their console output. The Flyte config
    path and parsed ``config_info`` are computed once by the caller and passed in.
    """
    relative_path = str(script.relative_to(root_dir))
    start_time = time.time()

    # Build the subprocess environment. FLYTECTL_CONFIG is passed explicitly rather than
    # mutating the shared os.environ, so this is safe to call from multiple threads.
    env = {**os.environ, **config.required_env_vars}
    if flyte_config_path:
        env["FLYTECTL_CONFIG"] = flyte_config_path

    try:
        result = subprocess.run(
            ["uv", "run", str(script)],
            capture_output=True,
            text=True,
            cwd=script.parent,
            timeout=config.timeout,
            env=env,
        )

        stdout_captured = result.stdout
        stderr_captured = result.stderr
        returncode = result.returncode

        # Some Flyte failures surface in the output while the process still exits 0.
        output_text = (stdout_captured or "") + (stderr_captured or "")
        flyte_failed = any(pattern in output_text for pattern in [
            "PHASE_FAILED",
            "exited unsuccessfully",
            "Run failed",
            "execution failed",
        ])
        if flyte_failed and returncode == 0:
            returncode = 1

        duration = time.time() - start_time
        execution_url = extract_execution_url(stdout_captured, config_info)

        if returncode != 0:
            return TestResult.create_with_timestamp(
                script_path=relative_path,
                status="failed",
                duration=duration,
                exit_code=returncode,
                error_message=f"Exit code {returncode}",
                stdout=stdout_captured,
                stderr=stderr_captured,
                execution_url=execution_url,
            )

        return TestResult.create_with_timestamp(
            script_path=relative_path,
            status="passed",
            duration=duration,
            exit_code=returncode,
            stdout=stdout_captured,
            stderr=stderr_captured,
            execution_url=execution_url,
        )

    except subprocess.TimeoutExpired as e:
        duration = time.time() - start_time
        return TestResult.create_with_timestamp(
            script_path=relative_path,
            status="timeout",
            duration=duration,
            error_message=f"Timed out after {config.timeout}s",
            stdout=e.stdout,
            stderr=e.stderr,
            execution_url=None,
        )

    except Exception as e:
        duration = time.time() - start_time
        return TestResult.create_with_timestamp(
            script_path=relative_path,
            status="failed",
            duration=duration,
            error_message=f"Unexpected error: {str(e)}",
            execution_url=None,
        )


def _write_test_log(result: TestResult, log_dir: Path, local: bool) -> None:
    """Write the per-test log file and record its relative path on the result."""
    safe_log_name = result.script_path.replace("/", "__")
    suffix = "_local" if local else ""
    log_filename = f"{safe_log_name}{suffix}.log"
    result.log_file = f"logs/{log_filename}"

    log_file = log_dir / log_filename
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"Script: {result.script_path}\n")
        f.write(f"Status: {result.status}\n")
        f.write(f"Duration: {result.duration:.2f}s\n")
        if local:
            f.write("Mode: Local execution\n")
        if result.exit_code is not None:
            f.write(f"Exit Code: {result.exit_code}\n")
        if result.error_message:
            f.write(f"Error: {result.error_message}\n")
        f.write("\n--- STDOUT ---\n")
        f.write(result.stdout or "")
        f.write("\n--- STDERR ---\n")
        f.write(result.stderr or "")


def _print_completion(result: TestResult, local: bool, progress: str) -> None:
    """Print an atomic per-test completion block (safe under concurrency)."""
    emoji = _STATUS_EMOJI.get(result.status, "❓")
    mode = "local" if local else "cloud"
    with _PRINT_LOCK:
        print(f"\n{'=' * 60}")
        print(f"{emoji} {result.script_path} ({result.duration:.1f}s) [{mode}] {progress}")
        if result.error_message:
            print(f"   └─ {result.error_message}")
        # Surface captured output for non-passing tests so failures stay diagnosable in
        # the console log even though we no longer stream output live under concurrency.
        if result.status != "passed":
            if result.stdout:
                print("   ┌─ stdout ─────────────────────────────────────────")
                for line in result.stdout.rstrip().splitlines():
                    print(f"   │ {line}")
            if result.stderr:
                print("   ┌─ stderr ─────────────────────────────────────────")
                for line in result.stderr.rstrip().splitlines():
                    print(f"   │ {line}")
        print(f"{'─' * 60}")


def _run_concurrently(
    scripts: List[Path],
    worker: Callable[[Path], TestResult],
    root_dir: Path,
    log_dir: Path,
    max_workers: int,
    local: bool,
) -> List[TestResult]:
    """Spawn every test run at once and collect results as they complete.

    Uses a thread pool because test submission is a blocking subprocess call
    (``uv run`` / ``flyte run``) with no async client — threads let all runs wait on
    their cloud/local executions concurrently while ``as_completed`` reports each result
    the moment it lands. ``max_workers`` caps the number of in-flight runs so the backend
    isn't overwhelmed. Per-test timeout, result attribution, and reporting are preserved
    exactly; only the sequential wait is removed.
    """
    log_dir.mkdir(exist_ok=True, parents=True)
    results: List[TestResult] = []

    total = len(scripts)
    workers = max(1, min(max_workers, total))
    print(f"🚀 Spawning {total} test run(s), up to {workers} in flight...")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_script = {executor.submit(worker, script): script for script in scripts}
        completed = 0
        for future in as_completed(future_to_script):
            script = future_to_script[future]
            try:
                result = future.result()
            except Exception as e:
                # Defensive: workers catch their own errors, but never lose a result.
                try:
                    relative_path = str(script.relative_to(root_dir))
                except ValueError:
                    relative_path = str(script)
                result = TestResult.create_with_timestamp(
                    script_path=relative_path,
                    status="failed",
                    duration=0.0,
                    error_message=f"Runner error: {e}",
                )
            completed += 1
            _write_test_log(result, log_dir, local)
            _print_completion(result, local, f"({completed}/{total})")
            results.append(result)

    # Deterministic ordering for the reports, regardless of completion order.
    results.sort(key=lambda r: r.script_path)
    return results


def run_tests(scripts: List[Path], config: TestConfig, root_dir: Path, log_dir: Path) -> List[TestResult]:
    """Run all example scripts against the cloud Flyte backend, concurrently."""
    flyte_config_path = resolve_flyte_config_path(root_dir / "test")
    config_info = parse_flyte_config(Path(flyte_config_path)) if flyte_config_path else {}

    def worker(script: Path) -> TestResult:
        return run_single_test(script, config, root_dir, flyte_config_path, config_info)

    return _run_concurrently(scripts, worker, root_dir, log_dir, config.concurrency, local=False)

def run_single_test_local(script: Path, config: TestConfig, root_dir: Path, verbose: str = "") -> TestResult:
    """Run a single example script locally via 'flyte run --local' and return its result.

    Pure worker: it provisions an isolated venv, runs the script, and returns a
    ``TestResult``. Console and log output are handled by the caller once the result is
    available, so concurrent runs don't interleave. Each script gets a uniquely-named
    venv (see create_isolated_venv_and_install_deps) so parallel local runs don't clash.
    """
    relative_path = str(script.relative_to(root_dir))
    venv_path = None
    verbose_flag = get_verbosity_flag(verbose)

    try:
        # Parse inline metadata to get main function and params
        metadata = parse_inline_metadata(script)
        main_func = metadata.get("main", "main")
        params_str = metadata.get("params", "")

        # Create isolated environment and install dependencies
        deps_success, venv_path = create_isolated_venv_and_install_deps(script, metadata, root_dir)
        if not deps_success:
            return TestResult.create_with_timestamp(
                script_path=relative_path,
                status="failed",
                duration=0.0,
                error_message="Failed to create isolated environment or install dependencies"
            )

        # Build flyte run command
        cmd = ["flyte"]
        if verbose_flag:
            cmd.append(verbose_flag)
        cmd.extend(["--output-format", "json", "run", "--local", str(script), main_func])

        # Parse and add parameters if provided
        if params_str:
            try:
                # Use shlex to handle quotes and spaces robustly
                for arg in shlex.split(params_str):
                    if '=' in arg:
                        key, value = arg.split('=', 1)
                        cmd.append(f"--{key}={value}")
                    else:
                        print(f"   ⚠️  Skipping malformed parameter in {relative_path}: {arg}")
            except Exception as e:
                print(f"   ⚠️  Error parsing parameters '{params_str}' in {relative_path}: {e}")

        # Set up environment to use the isolated venv (if created)
        env = os.environ.copy()
        if venv_path:
            env["VIRTUAL_ENV"] = str(venv_path)
            env["PATH"] = f"{venv_path}/bin:{env.get('PATH', '')}"

        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=config.timeout,
                env=env,
            )
            duration = time.time() - start_time

            if result.returncode != 0:
                return TestResult.create_with_timestamp(
                    script_path=relative_path,
                    status="failed",
                    duration=duration,
                    exit_code=result.returncode,
                    error_message=f"Exit code {result.returncode}",
                    stdout=result.stdout,
                    stderr=result.stderr,
                    execution_url=None,  # Local tests don't have execution URLs
                )

            return TestResult.create_with_timestamp(
                script_path=relative_path,
                status="passed",
                duration=duration,
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                execution_url=None,  # Local tests don't have execution URLs
            )

        except subprocess.TimeoutExpired as e:
            duration = time.time() - start_time
            return TestResult.create_with_timestamp(
                script_path=relative_path,
                status="timeout",
                duration=duration,
                error_message=f"Timed out after {config.timeout}s",
                stdout=e.stdout,
                stderr=e.stderr,
                execution_url=None
            )

        except Exception as e:
            duration = time.time() - start_time
            return TestResult.create_with_timestamp(
                script_path=relative_path,
                status="failed",
                duration=duration,
                error_message=f"Unexpected error: {str(e)}",
                execution_url=None
            )

    finally:
        # Clean up isolated virtual environment
        if venv_path and venv_path.exists():
            try:
                shutil.rmtree(venv_path)
            except Exception as e:
                print(f"   ⚠️  Warning: Failed to clean up venv for {relative_path}: {e}")

def run_tests_local(scripts: List[Path], config: TestConfig, root_dir: Path, log_dir: Path, verbose: str = "") -> List[TestResult]:
    """Run all example scripts locally via 'flyte run --local', concurrently."""
    def worker(script: Path) -> TestResult:
        return run_single_test_local(script, config, root_dir, verbose)

    return _run_concurrently(scripts, worker, root_dir, log_dir, config.concurrency, local=True)

def generate_report(results: List[TestResult], root_dir: Path):
    """Generate a comprehensive test report with persistent results."""
    # Create reports directory
    reports_dir = root_dir / "test" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Download previous results from GitHub Pages if in CI environment
    if os.getenv("GITHUB_ACTIONS") == "true":
        print("🤖 GitHub Actions detected - downloading previous results...")
        downloaded = download_previous_results_from_github_pages(reports_dir)

        # If GitHub Pages download failed, try to use repository fallback
        if not downloaded:
            repo_persistent_file = root_dir / "test" / "persistent_results.json"
            if repo_persistent_file.exists():
                print("📂 GitHub Pages unavailable, using repository fallback...")
                import shutil
                shutil.copy2(repo_persistent_file, reports_dir / "persistent_results.json")
                print(f"✅ Loaded {repo_persistent_file} from repository")
            else:
                print("📋 No repository fallback found - starting fresh")

    # Load existing persistent results
    persistent_results = load_persistent_results(reports_dir)

    # Merge new results with existing ones
    merged_results = merge_test_results(persistent_results, results)

    # Save updated persistent results
    save_persistent_results(merged_results, reports_dir)

    # Convert merged results back to list for report generation
    all_results = list(merged_results.values())

    # Sort by script path for consistent ordering
    all_results.sort(key=lambda x: x.script_path)

    # Write current run JSON report (only new results)
    json_report = reports_dir / "test_report.json"
    with open(json_report, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    # Write complete historical JSON report
    historical_json_report = reports_dir / "historical_results.json"
    with open(historical_json_report, "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)

    # Write HTML report with all historical results
    html_report = reports_dir / "index.html"
    with open(html_report, "w") as f:
        f.write(generate_html_report(all_results))

    print(f"\n📊 Reports generated:")
    print(f"   Current run JSON: {json_report}")
    print(f"   Historical JSON: {historical_json_report}")
    print(f"   HTML (all results): {html_report}")
    print(f"   📈 Showing {len(results)} new results, {len(all_results)} total historical results")

def generate_html_report(results: List[TestResult]) -> str:
    """Generate an HTML test report with collapsible details."""
    passed = [r for r in results if r.status == "passed"]
    failed = [r for r in results if r.status == "failed"]
    timeout = [r for r in results if r.status == "timeout"]
    skipped = [r for r in results if r.status == "skipped"]

    def format_timestamp(timestamp_str: str) -> str:
        """Format ISO timestamp for display with both UTC and local time."""
        try:
            # Parse the ISO timestamp string
            # Handle both 'Z' suffix and '+00:00' suffix for UTC
            if timestamp_str.endswith('Z'):
                dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            elif '+' in timestamp_str or timestamp_str.endswith('+00:00'):
                dt = datetime.fromisoformat(timestamp_str)
            else:
                # If no timezone info, assume UTC
                dt = datetime.fromisoformat(timestamp_str).replace(tzinfo=timezone.utc)

            # Ensure we have a timezone-aware datetime in UTC
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)

            # Convert to UTC for consistent display
            utc_dt = dt.astimezone(timezone.utc)
            utc_formatted = utc_dt.strftime("%Y-%m-%d %H:%M:%S UTC")

            # Convert to local timezone
            local_dt = utc_dt.astimezone()
            local_formatted = local_dt.strftime("%Y-%m-%d %H:%M:%S %Z")

            # Return with tooltip showing both times
            iso_timestamp = dt.isoformat()
            return f'<span data-timestamp="{iso_timestamp}" title="UTC: {utc_formatted}&#10;Local: {local_formatted}" style="cursor: help;">{local_formatted}</span>'
        except Exception as e:
            # Fallback - return original timestamp with debug info in tooltip
            return f'<span title="Parse error: {str(e)}" style="cursor: help;">{timestamp_str}</span>'

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Flyte Examples Test Report</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 20px;
                background-color: #f8f9fa;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .summary {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 20px;
            }}
            .summary h2 {{ margin-top: 0; }}
            .stats {{ display: flex; gap: 20px; flex-wrap: wrap; }}
            .stat {{
                background: rgba(255,255,255,0.1);
                padding: 10px 15px;
                border-radius: 5px;
                backdrop-filter: blur(10px);
            }}
            .passed {{ color: #28a745; }}
            .failed {{ color: #dc3545; }}
            .timeout {{ color: #fd7e14; }}
            .skipped {{ color: #6c757d; }}

            table {{
                border-collapse: collapse;
                width: 100%;
                margin-top: 20px;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }}
            th, td {{
                border: 1px solid #e9ecef;
                padding: 12px;
                text-align: left;
            }}
            th {{
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                font-weight: 600;
                cursor: pointer;
                user-select: none;
                position: relative;
                transition: background-color 0.2s ease;
            }}
            th:hover {{
                background: linear-gradient(135deg, #e9ecef 0%, #dee2e6 100%);
            }}
            th.sortable:after {{
                content: ' ↕️';
                font-size: 0.8em;
                opacity: 0.5;
            }}
            th.sort-asc:after {{
                content: ' ▲';
                color: #007bff;
                opacity: 1;
            }}
            th.sort-desc:after {{
                content: ' ▼';
                color: #007bff;
                opacity: 1;
            }}
            tr:nth-child(even) {{ background-color: #f8f9fa; }}
            tr:hover {{ background-color: #e3f2fd; }}

            .error-msg {{
                font-family: 'Courier New', monospace;
                font-size: 0.9em;
                color: #dc3545;
                max-width: 300px;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
            }}

            .log-link {{
                display: inline-block;
                padding: 6px 12px;
                background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
                color: white;
                text-decoration: none;
                border-radius: 4px;
                font-size: 0.85em;
                font-weight: 500;
                transition: all 0.2s ease;
            }}
            .log-link:hover {{
                background: linear-gradient(135deg, #0056b3 0%, #004085 100%);
                transform: translateY(-1px);
                box-shadow: 0 2px 4px rgba(0,123,255,0.3);
                color: white;
                text-decoration: none;
            }}

            .script-link {{
                color: #007bff;
                text-decoration: none;
                font-weight: 600;
                transition: all 0.2s ease;
            }}
            .script-link:hover {{
                color: #0056b3;
                text-decoration: underline;
            }}

            .execution-link {{
                display: inline-block;
                padding: 6px 12px;
                background: linear-gradient(135deg, #28a745 0%, #1e7e34 100%);
                color: white;
                text-decoration: none;
                border-radius: 4px;
                font-size: 0.85em;
                font-weight: 500;
                transition: all 0.2s ease;
            }}
            .execution-link:hover {{
                background: linear-gradient(135deg, #1e7e34 0%, #155724 100%);
                transform: translateY(-1px);
                box-shadow: 0 2px 4px rgba(40,167,69,0.3);
                color: white;
                text-decoration: none;
            }}

            .no-execution {{
                color: #6c757d;
                font-style: italic;
                font-size: 0.9em;
            }}

            /* Collapsible styling */
            .collapsible {{
                background-color: #f1f3f4;
                color: #444;
                cursor: pointer;
                padding: 8px 12px;
                width: 100%;
                border: none;
                text-align: left;
                outline: none;
                font-size: 14px;
                border-radius: 4px;
                margin-top: 5px;
                transition: background-color 0.3s;
            }}
            .collapsible:hover {{ background-color: #e0e0e0; }}
            .collapsible:after {{
                content: '\\002B'; /* Plus sign */
                color: #777;
                font-weight: bold;
                float: right;
                margin-left: 5px;
            }}
            .collapsible.active:after {{
                content: "\\2212"; /* Minus sign */
            }}

            .content {{
                padding: 0 15px;
                max-height: 0;
                overflow: hidden;
                transition: max-height 0.3s ease-out;
                background-color: #fafafa;
                border-left: 3px solid #007bff;
                margin-top: 5px;
                border-radius: 0 4px 4px 0;
            }}
            .content.active {{
                max-height: 1000px;
                padding: 15px;
            }}

            .detail-section {{
                margin-bottom: 15px;
            }}
            .detail-label {{
                font-weight: bold;
                color: #495057;
                margin-bottom: 5px;
            }}
            .detail-value {{
                font-family: 'Courier New', monospace;
                background: #f8f9fa;
                padding: 8px;
                border-radius: 4px;
                border-left: 3px solid #007bff;
                white-space: pre-wrap;
                font-size: 0.9em;
                max-height: 200px;
                overflow-y: auto;
            }}

            .status-badge {{
                padding: 4px 8px;
                border-radius: 12px;
                font-size: 0.85em;
                font-weight: 500;
            }}
            .status-passed {{ background: #d4edda; color: #155724; }}
            .status-failed {{ background: #f8d7da; color: #721c24; }}
            .status-timeout {{ background: #fff3cd; color: #856404; }}
            .status-skipped {{ background: #d1ecf1; color: #0c5460; }}

            /* Navigation bar styles */
            .nav-bar {{
                background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
                color: white;
                padding: 15px 20px;
                border-radius: 8px;
                margin: 20px;
                margin-bottom: 10px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                flex-wrap: wrap;
                gap: 10px;
            }}
            .nav-links {{
                display: flex;
                gap: 15px;
                align-items: center;
                flex-wrap: wrap;
            }}
            .nav-link {{
                color: white;
                text-decoration: none;
                padding: 8px 16px;
                background: rgba(255,255,255,0.2);
                border-radius: 4px;
                font-weight: 500;
                transition: all 0.2s ease;
            }}
            .nav-link:hover {{
                background: rgba(255,255,255,0.3);
                color: white;
                text-decoration: none;
                transform: translateY(-1px);
            }}
            .nav-info {{
                font-size: 0.9em;
                opacity: 0.9;
            }}
        </style>
        <script>
            function toggleCollapsible(element) {{
                element.classList.toggle("active");
                var content = element.nextElementSibling;
                content.classList.toggle("active");
            }}

            // Table sorting functionality
            let sortDirection = {{}};

            function sortTable(columnIndex, table) {{
                const tbody = table.querySelector('tbody');
                const rows = Array.from(tbody.querySelectorAll('tr'));
                const isDescending = sortDirection[columnIndex] || false;
                sortDirection[columnIndex] = !isDescending;

                // Clear all sort indicators
                table.querySelectorAll('th').forEach(th => {{
                    th.classList.remove('sort-asc', 'sort-desc');
                }});

                // Add sort indicator to current column
                const currentHeader = table.querySelectorAll('th')[columnIndex];
                currentHeader.classList.add(isDescending ? 'sort-desc' : 'sort-asc');

                rows.sort((a, b) => {{
                    // Try to get clean sort values from data-sort attributes first
                    let aValue = a.cells[columnIndex].getAttribute('data-sort');
                    let bValue = b.cells[columnIndex].getAttribute('data-sort');

                    // Fallback to text content if no data-sort attribute
                    if (!aValue) aValue = a.cells[columnIndex].textContent.trim();
                    if (!bValue) bValue = b.cells[columnIndex].textContent.trim();

                    // Handle different data types based on column
                    let result = 0;

                    if (columnIndex === 2) {{ // Duration column - numeric comparison
                        const aNum = parseFloat(aValue) || 0;
                        const bNum = parseFloat(bValue) || 0;
                        result = aNum - bNum;
                    }} else if (columnIndex === 3) {{ // Timestamp column - date comparison
                        const aDate = new Date(aValue);
                        const bDate = new Date(bValue);
                        result = aDate.getTime() - bDate.getTime();
                    }} else if (columnIndex === 1) {{ // Status column - logical order (passed first, then issues)
                        const statusOrder = {{ 'passed': 1, 'failed': 2, 'timeout': 3, 'skipped': 4 }};
                        const aOrder = statusOrder[aValue] || 99;
                        const bOrder = statusOrder[bValue] || 99;
                        result = aOrder - bOrder;
                    }} else {{ // String comparison for script names and other columns
                        result = aValue.toLowerCase().localeCompare(bValue.toLowerCase());
                    }}

                    return isDescending ? -result : result;
                }});

                // Re-append sorted rows
                rows.forEach(row => tbody.appendChild(row));
            }}

            // Initialize sortable headers when page loads
            document.addEventListener('DOMContentLoaded', function() {{
                const table = document.querySelector('table');
                if (table) {{
                    const headers = table.querySelectorAll('th');
                    headers.forEach((header, index) => {{
                        // Make specific columns sortable
                        if (index === 0 || index === 1 || index === 2 || index === 3) {{ // Script, Status, Duration, Last Tested
                            header.classList.add('sortable');
                            header.addEventListener('click', () => sortTable(index, table));
                        }}
                    }});
                }}

                // Convert all timestamps to browser's local timezone
                convertTimestampsToBrowserTimezone();
            }});

            // Convert server-side timestamps to browser's local timezone
            function convertTimestampsToBrowserTimezone() {{
                const timestampElements = document.querySelectorAll('[data-timestamp]');

                timestampElements.forEach(element => {{
                    const isoTimestamp = element.getAttribute('data-timestamp');

                    try {{
                        const date = new Date(isoTimestamp);

                        // Get browser's local timezone
                        const browserLocal = date.toLocaleString(undefined, {{
                            year: 'numeric',
                            month: '2-digit',
                            day: '2-digit',
                            hour: '2-digit',
                            minute: '2-digit',
                            second: '2-digit',
                            timeZoneName: 'short'
                        }});

                        // Get UTC time
                        const utcTime = date.toLocaleString('en-US', {{
                            year: 'numeric',
                            month: '2-digit',
                            day: '2-digit',
                            hour: '2-digit',
                            minute: '2-digit',
                            second: '2-digit',
                            timeZone: 'UTC',
                            timeZoneName: 'short'
                        }});

                        // Update the display text to browser's local time
                        element.textContent = browserLocal;

                        // Update tooltip to show both UTC and browser local time
                        element.title = `UTC: ${{utcTime}}\\nLocal: ${{browserLocal}}`;

                    }} catch (error) {{
                        console.warn('Failed to convert timestamp:', isoTimestamp, error);
                    }}
                }});
            }}
        </script>
    </head>
    <body>
        <!-- Navigation Bar -->
        <div class="nav-bar">
            <div class="nav-links">
                <a href="test_report.json" class="nav-link">📄 Current Run</a>
                <a href="historical_results.json" class="nav-link">📊 All Results</a>
                <a href="persistent_results.json" class="nav-link">💾 Raw Data</a>
            </div>
            <div class="nav-info">
                Interactive Test Report with Historical Data
            </div>
        </div>

        <div class="container">
            <h1>🚀 Flyte Examples Test Report</h1>

            <div class="summary">
                <h2>📊 Test Summary</h2>
                <div class="stats">
                    <div class="stat"><strong>✅ Passed:</strong> {len(passed)}</div>
                    <div class="stat"><strong>❌ Failed:</strong> {len(failed)}</div>
                    <div class="stat"><strong>⏰ Timeout:</strong> {len(timeout)}</div>
                    <div class="stat"><strong>⏭️ Skipped:</strong> {len(skipped)}</div>
                    <div class="stat"><strong>📊 Total:</strong> {len(results)}</div>
                </div>
            </div>

            <h2>📋 Test Results</h2>
            <p style="color: #6c757d; font-size: 0.9em; margin-bottom: 15px;">
                💡 Click on <strong>Script</strong>, <strong>Status</strong>, <strong>Duration</strong>, or <strong>Last Tested</strong> column headers to sort the table.
            </p>
            <table>
                <thead>
                    <tr>
                        <th class="sortable" title="Click to sort by script name">Script</th>
                        <th class="sortable" title="Click to sort by status">Status</th>
                        <th class="sortable" title="Click to sort by duration">Duration</th>
                        <th class="sortable" title="Click to sort by timestamp (hover for UTC/Local time)">Last Tested</th>
                        <th>Summary</th>
                        <th>Log</th>
                        <th>Execution</th>
                        <th>Details</th>
                    </tr>
                </thead>
                <tbody>
    """

    for i, result in enumerate(results):
        status_class = result.status
        status_emoji = {"passed": "✅", "failed": "❌", "timeout": "⏰", "skipped": "⏭️"}
        emoji = status_emoji.get(result.status, "❓")

        # Escape HTML characters in output
        def escape_html(text):
            if not text:
                return ""
            return (text.replace('&', '&amp;')
                       .replace('<', '&lt;')
                       .replace('>', '&gt;')
                       .replace('"', '&quot;')
                       .replace("'", '&#x27;'))

        # Create GitHub link for the script
        github_base_url = "https://github.com/unionai/unionai-examples/blob/main"
        script_link = f"{github_base_url}/{result.script_path}"

        # Create execution link if available
        execution_cell = ""
        if result.execution_url:
            execution_cell = f"<a href='{result.execution_url}' target='_blank' class='execution-link'>🚀 View Execution</a>"
        else:
            execution_cell = "<span class='no-execution'>No execution</span>"

        html += f"""
            <tr class="{status_class}">
                <td data-sort="{result.script_path.lower()}"><strong><a href="{script_link}" target="_blank" class="script-link">{result.script_path}</a></strong></td>
                <td data-sort="{result.status}"><span class="status-badge status-{status_class}">{emoji} {result.status.title()}</span></td>
                <td data-sort="{result.duration}"><strong>{result.duration:.2f}s</strong></td>
                <td data-sort="{result.timestamp}"><small>{format_timestamp(result.timestamp)}</small></td>
                <td class="error-msg">{escape_html(result.error_message) if result.error_message else "Success" if result.status == "passed" else "-"}</td>
                <td>{"<a href='" + result.log_file + "' target='_blank' class='log-link'>📄 View Log</a>" if result.log_file else "-"}</td>
                <td>{execution_cell}</td>
                <td>
                    <button class="collapsible" onclick="toggleCollapsible(this)">
                        View Details
                    </button>
                    <div class="content">
                        <div class="detail-section">
                            <div class="detail-label">📁 Script Path:</div>
                            <div class="detail-value">{escape_html(result.script_path)}</div>
                        </div>

                        <div class="detail-section">
                            <div class="detail-label">📊 Status:</div>
                            <div class="detail-value">{emoji} {result.status.upper()}</div>
                        </div>

                        <div class="detail-section">
                            <div class="detail-label">⏱️ Duration:</div>
                            <div class="detail-value">{result.duration:.3f} seconds</div>
                        </div>

                        <div class="detail-section">
                            <div class="detail-label">📅 Last Tested:</div>
                            <div class="detail-value">{format_timestamp(result.timestamp)}</div>
                        </div>

                        {"" if result.exit_code is None else f'''
                        <div class="detail-section">
                            <div class="detail-label">🔢 Exit Code:</div>
                            <div class="detail-value">{result.exit_code}</div>
                        </div>
                        '''}

                        {"" if not result.error_message else f'''
                        <div class="detail-section">
                            <div class="detail-label">❌ Error Message:</div>
                            <div class="detail-value">{escape_html(result.error_message)}</div>
                        </div>
                        '''}

                        {"" if not result.stdout else f'''
                        <div class="detail-section">
                            <div class="detail-label">📤 Standard Output:</div>
                            <div class="detail-value">{escape_html(result.stdout)}</div>
                        </div>
                        '''}

                        {"" if not result.stderr else f'''
                        <div class="detail-section">
                            <div class="detail-label">🚨 Standard Error:</div>
                            <div class="detail-value">{escape_html(result.stderr)}</div>
                        </div>
                        '''}
                    </div>
                </td>
            </tr>
        """

    html += """
                </tbody>
            </table>
        </div>
    </body>
    </html>
    """
    return html

def load_config(config_path: Optional[Path] = None) -> TestConfig:
    """Load test configuration from file or use defaults."""
    # Auto-detect GitHub Actions environment
    is_github_actions = os.getenv("GITHUB_ACTIONS") == "true"

    # Use GitHub Actions config if available and in CI
    if is_github_actions and not config_path:
        ci_config_path = Path(__file__).parent / "config.github.json"
        if ci_config_path.exists():
            config_path = ci_config_path
            print(f"🤖 Detected GitHub Actions environment, using {ci_config_path}")

    if config_path and config_path.exists():
        try:
            with open(config_path, "r") as f:
                config_data = json.load(f)
            return TestConfig(**config_data)
        except Exception as e:
            print(f"⚠️  Error loading config from {config_path}: {e}")
            print("Using default configuration")

    # Default configuration
    return TestConfig(
        timeout=300,  # 5 minutes
        excluded_patterns=[
            "__pycache__",
            ".git",
            "venv",
            ".venv",
            "node_modules",
            ".pytest_cache",
            ".github",
            "templates"
        ],
        required_env_vars={"PYTHONPATH": "."} if not is_github_actions else {"PYTHONPATH": ".", "GITHUB_ACTIONS": "true"}
    )

def main():
    """Main function to run the test framework."""
    import argparse

    parser = argparse.ArgumentParser(description="Test runner for unionai-examples")
    parser.add_argument("subdirectory", nargs="?",
                       default="v2",
                       help="Subdirectory to scan (e.g., 'v2', 'tutorials', 'integrations')")
    parser.add_argument("--root", type=Path,
                       help="Root directory of unionai-examples repo (auto-detected if not provided)")
    parser.add_argument("--config", type=Path, help="Configuration file path")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds")
    parser.add_argument("--exclude", action="append", help="Patterns to exclude")
    parser.add_argument("--logs", type=Path, help="Directory to store logs and reports")
    parser.add_argument("--filter", help="Only run scripts matching this pattern")
    parser.add_argument("--file", help="Run only a specific file (relative to repo root)")
    parser.add_argument("--preview", action="store_true", help="Show what would be run without executing")
    parser.add_argument("--local", action="store_true", help="Run locally using 'flyte run --local' instead of cloud execution")
    parser.add_argument("--verbose", type=str, default="", help="Set verbosity level: v (-v), vv (-vv), vvv (-vvv), or 1/2/3")
    parser.add_argument("--concurrency", type=int, default=None,
                       help="Max number of tests to run concurrently (default: config value or 8; use 1 for sequential)")

    args = parser.parse_args()

    search_subdir = args.subdirectory

    # Auto-detect root directory if not provided
    if args.root:
        repo_root = args.root
    else:
        # Try to find unionai-examples repo from current directory
        current_path = Path.cwd()
        repo_root = None

        # Check if we're already in unionai-examples
        if current_path.name == "unionai-examples" or "unionai-examples" in str(current_path):
            # Walk up to find the root
            for parent in [current_path] + list(current_path.parents):
                if parent.name == "unionai-examples":
                    repo_root = parent
                    break

        if not repo_root:
            print("❌ Could not auto-detect unionai-examples repository root.")
            print("Please specify --root or run from within the unionai-examples directory.")
            sys.exit(1)

    # Set target directory to scan
    target_dir = repo_root / search_subdir

    # Set logs directory relative to repo root if not specified
    if args.logs:
        log_dir = args.logs
    else:
        reports_dir = repo_root / "test" / "reports"
        log_dir = reports_dir / "logs"


    # Load configuration
    config = load_config(args.config)

    # Override with command line arguments
    if args.timeout != 300:
        config.timeout = args.timeout
    if args.exclude:
        config.excluded_patterns.extend(args.exclude)
    if args.concurrency is not None:
        config.concurrency = max(1, args.concurrency)

    if not target_dir.exists():
        print(f"❌ Target directory {target_dir} does not exist")
        print(f"Available subdirectories in {repo_root}:")
        for item in repo_root.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                print(f"   - {item.name}")
        sys.exit(1)

    # Find all runnable scripts
    scripts_to_run = find_runnable_scripts(target_dir, config)

    # Apply file filter if specified (takes precedence over pattern filter)
    if args.file:
        # Match scripts that contain the file path
        scripts_to_run = [s for s in scripts_to_run if args.file in str(s)]
        if not scripts_to_run:
            print(f"❌ File not found: {args.file}")
            return 1
    elif args.filter:
        scripts_to_run = [s for s in scripts_to_run if args.filter in str(s)]
        print(f"📋 Filtered to {len(scripts_to_run)} scripts matching '{args.filter}'")

    if not scripts_to_run:
        print("No runnable scripts found!")
        return

    # Handle preview mode
    if args.preview:
        print(f"\n👀 Preview mode: showing {len(scripts_to_run)} scripts that would be tested")
        print("=" * 80)
        for i, script_path in enumerate(scripts_to_run, 1):
            rel_path = script_path.relative_to(repo_root)
            print(f"{i:3d}. {rel_path}")
        print("=" * 80)
        print(f"\n📋 Preview Summary:")
        print(f"   - Total scripts: {len(scripts_to_run)}")
        print(f"   - Target directory: {target_dir.relative_to(repo_root)}")
        if args.file:
            print(f"   - File filter: {args.file}")
        elif args.filter:
            print(f"   - Pattern filter: {args.filter}")
        print(f"   - Timeout: {config.timeout}s")
        print(f"   - Concurrency: {config.concurrency}")
        print(f"   - Mode: {'Local execution' if args.local else 'Cloud execution'}")
        print(f"\n✨ Preview complete - no tests were executed")
        return

    # All scripts now have flyte.init (due to filtering), so no need for breakdown

    # Run tests
    print(f"\n🧪 Running tests with {config.timeout}s timeout (concurrency: {config.concurrency})...")
    if args.local:
        print("🏠 Running tests locally using 'flyte run --local'")
        results = run_tests_local(scripts_to_run, config, repo_root, log_dir, args.verbose)
    else:
        print("☁️ Running tests on cloud Flyte backend")
        results = run_tests(scripts_to_run, config, repo_root, log_dir)

    # Generate reports
    generate_report(results, repo_root)

    # Print summary
    passed = [r for r in results if r.status == "passed"]
    failed = [r for r in results if r.status == "failed"]
    timeout = [r for r in results if r.status == "timeout"]
    skipped = [r for r in results if r.status == "skipped"]

    print(f"\n📋 Test Summary:")
    print(f"✅ Passed: {len(passed)}")
    print(f"❌ Failed: {len(failed)}")
    print(f"⏰ Timeout: {len(timeout)}")
    print(f"⏭️ Skipped: {len(skipped)}")
    print(f"📊 Total: {len(results)}")

    if failed or timeout:
        print(f"\n💥 Issues found:")
        for result in failed + timeout:
            print(f"   {result.script_path}: {result.error_message}")
        sys.exit(1)

    print(f"\n🎉 All tests completed successfully!")

if __name__ == "__main__":
    main()
