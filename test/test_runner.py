import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any

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

@dataclass
class TestResult:
    script_path: str
    status: str  # "passed", "failed", "timeout", "skipped"
    duration: float
    exit_code: Optional[int] = None
    error_message: Optional[str] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None

@dataclass
class TestConfig:
    timeout: int = 300  # 5 minutes default timeout
    excluded_patterns: List[str] = None
    required_env_vars: Dict[str, str] = None

    def __post_init__(self):
        if self.excluded_patterns is None:
            self.excluded_patterns = []
        if self.required_env_vars is None:
            self.required_env_vars = {}

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

        # Use script name for venv directory (safer than full path)
        script_name = script_path.stem
        venv_path = venvs_dir / f"{script_name}_venv"

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
        return True, venv_path

    except subprocess.TimeoutExpired:
        print(f"   ⏰ Virtual environment creation or dependency installation timed out")
        return False, None
    except Exception as e:
        print(f"   ❌ Error creating environment or installing dependencies: {e}")
        return False, None

def setup_flyte_config_env(test_dir: Path) -> Optional[str]:
    """Set up FLYTECTL_CONFIG environment variable to point to config template."""
    config_path = test_dir / "config.flyte.yaml"
    if not config_path.exists():
        print(f"⚠️  Config template not found: {config_path}")
        return None

    absolute_config_path = str(config_path.absolute())
    print(f"   ⚙️  Using Flyte config: {absolute_config_path}")
    return absolute_config_path



def run_single_test(script: Path, config: TestConfig, root_dir: Path) -> TestResult:
    """Run a single test script and return results."""
    script_name = script.stem
    relative_path = str(script.relative_to(root_dir))

    print(f"🏃 Running {relative_path}...")

    # Add cloud execution warning for Flyte scripts
    print(f"   ☁️  Note: This script will execute remotely on Flyte backend in the cloud")

    # Setup Flyte config environment variable
    test_dir = root_dir / "test"
    flyte_config_path = setup_flyte_config_env(test_dir)

    start_time = time.time()

    try:
        # Set up environment with Flyte config
        env = {**os.environ, **config.required_env_vars}
        if flyte_config_path:
            env["FLYTECTL_CONFIG"] = flyte_config_path

        # For Flyte scripts, capture output but also stream it
        print(f"   📺 Streaming logs from cloud execution...")
        print(f"   📦 Using uv run to handle inline script dependencies...")
        result = subprocess.run(
            ["uv", "run", str(script)],
            capture_output=True,
            text=True,
            cwd=script.parent,
            timeout=config.timeout,
            env=env
        )

        # Display the output in real-time for Flyte scripts
        if result.stdout:
            print(result.stdout, end='')
        if result.stderr:
            print(result.stderr, end='', file=sys.stderr)

        stdout_captured = result.stdout
        stderr_captured = result.stderr

        # Check for Flyte failure patterns in output
        output_text = (result.stdout or "") + (result.stderr or "")
        flyte_failed = any(pattern in output_text for pattern in [
            "PHASE_FAILED",
            "exited unsuccessfully",
            "Run failed",
            "execution failed"
        ])

        # Override exit code if we detect Flyte failure
        if flyte_failed and result.returncode == 0:
            print(f"   ⚠️  Detected Flyte failure in output despite exit code 0")
            # Simulate failed exit code
            class FakeResult:
                def __init__(self, returncode, stdout, stderr):
                    self.returncode = returncode
                    self.stdout = stdout
                    self.stderr = stderr
            result = FakeResult(1, stdout_captured, stderr_captured)

        # Check for success/failure
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, ["uv", "run", str(script)], stdout_captured, stderr_captured)

        duration = time.time() - start_time
        return TestResult(
            script_path=relative_path,
            status="passed",
            duration=duration,
            exit_code=result.returncode,
            stdout=stdout_captured,
            stderr=stderr_captured
        )

    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        return TestResult(
            script_path=relative_path,
            status="timeout",
            duration=duration,
            error_message=f"Timed out after {config.timeout}s"
        )

    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        # Now we always capture output, so use the captured values
        return TestResult(
            script_path=relative_path,
            status="failed",
            duration=duration,
            exit_code=e.returncode,
            error_message=f"Exit code {e.returncode}",
            stdout=e.stdout,
            stderr=e.stderr
        )

    except Exception as e:
        duration = time.time() - start_time
        return TestResult(
            script_path=relative_path,
            status="failed",
            duration=duration,
            error_message=f"Unexpected error: {str(e)}"
        )

    finally:
        # No cleanup needed when using environment variables
        pass

def run_tests(scripts: List[Path], config: TestConfig, root_dir: Path, log_dir: Path) -> List[TestResult]:
    """Run all test scripts and return results."""
    log_dir.mkdir(exist_ok=True, parents=True)
    results = []

    for script in scripts:
        result = run_single_test(script, config, root_dir)
        results.append(result)

        # Write individual log file
        log_file = log_dir / f"{script.stem}.log"
        with open(log_file, "w") as f:
            f.write(f"Script: {result.script_path}\n")
            f.write(f"Status: {result.status}\n")
            f.write(f"Duration: {result.duration:.2f}s\n")
            if result.exit_code is not None:
                f.write(f"Exit Code: {result.exit_code}\n")
            if result.error_message:
                f.write(f"Error: {result.error_message}\n")
            f.write("\n--- STDOUT ---\n")
            f.write(result.stdout or "")
            f.write("\n--- STDERR ---\n")
            f.write(result.stderr or "")

        # Print status
        status_emoji = {
            "passed": "✅",
            "failed": "❌",
            "timeout": "⏰",
            "skipped": "⏭️"
        }
        emoji = status_emoji.get(result.status, "❓")
        print(f"{emoji} {result.script_path} ({result.duration:.1f}s)")
        if result.error_message:
            print(f"   └─ {result.error_message}")

    return results

def run_single_test_local(script: Path, config: TestConfig, root_dir: Path, verbose: str = "") -> TestResult:
    """Run a single test script locally using 'flyte run --local'."""
    script_name = script.stem
    relative_path = str(script.relative_to(root_dir))
    venv_path = None
    verbose_flag = get_verbosity_flag(verbose)

    try:
        # Parse inline metadata to get main function and params
        metadata = parse_inline_metadata(script)
        main_func = metadata.get("main", "main")
        params_str = metadata.get("params", "")

        print(f"🏃 Running {relative_path} locally...")
        if verbose_flag:
            print(f"   🎯 Main function: {main_func}")
            if params_str:
                print(f"   ⚙️ Parameters: {params_str}")

        # Create isolated environment and install dependencies
        deps_success, venv_path = create_isolated_venv_and_install_deps(script, metadata, root_dir)
        if not deps_success:
            return TestResult(
                script_path=relative_path,
                status="failed",
                duration=0.0,
                error_message="Failed to create isolated environment or install dependencies"
            )

        start_time = time.time()

        try:
            # Build flyte run command
            cmd = ["flyte"]
            if verbose_flag:
                cmd.append(verbose_flag)
            cmd.extend(["--output-format", "json", "run", "--local", str(script), main_func])

            # Parse and add parameters if provided
            if params_str:
                try:
                    # Use shlex to handle quotes and spaces robustly
                    param_args = shlex.split(params_str)
                    for arg in param_args:
                        if '=' in arg:
                            key, value = arg.split('=', 1)
                            cmd.append(f"--{key}={value}")
                        else:
                            print(f"   ⚠️  Skipping malformed parameter: {arg}")
                except Exception as e:
                    print(f"   ⚠️  Error parsing parameters '{params_str}': {e}")

            # Only show command in verbose mode
            if verbose_flag:
                print(f"   💻 Command: {' '.join(cmd)}")

            # Set up environment to use the isolated venv (if created)
            env = os.environ.copy()
            if venv_path:
                env["VIRTUAL_ENV"] = str(venv_path)
                env["PATH"] = f"{venv_path}/bin:{env.get('PATH', '')}"

            # Run the local command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=config.timeout,
                env=env
            )

            # Display flyte output with clear formatting
            if result.stdout or result.stderr:
                print("   ┌─ Flyte Output ─────────────────────────────────────")
                if result.stdout:
                    # Indent flyte stdout
                    for line in result.stdout.splitlines():
                        print(f"   │ {line}")
                if result.stderr:
                    # Indent flyte stderr
                    for line in result.stderr.splitlines():
                        print(f"   │ {line}", file=sys.stderr)
                print("   └────────────────────────────────────────────────────")

            # Check for success/failure
            if result.returncode != 0:
                raise subprocess.CalledProcessError(
                    result.returncode, cmd, result.stdout, result.stderr
                )

            duration = time.time() - start_time
            return TestResult(
                script_path=relative_path,
                status="passed",
                duration=duration,
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr
            )

        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return TestResult(
                script_path=relative_path,
                status="timeout",
                duration=duration,
                error_message=f"Timed out after {config.timeout}s"
            )

        except subprocess.CalledProcessError as e:
            duration = time.time() - start_time
            return TestResult(
                script_path=relative_path,
                status="failed",
                duration=duration,
                exit_code=e.returncode,
                error_message=f"Exit code {e.returncode}",
                stdout=e.stdout,
                stderr=e.stderr
            )

        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                script_path=relative_path,
                status="failed",
                duration=duration,
                error_message=f"Unexpected error: {str(e)}"
            )

    finally:
        # Clean up isolated virtual environment
        if venv_path and venv_path.exists():
            try:
                shutil.rmtree(venv_path)
            except Exception as e:
                print(f"   ⚠️  Warning: Failed to clean up venv: {e}")

def run_tests_local(scripts: List[Path], config: TestConfig, root_dir: Path, log_dir: Path, verbose: str = "") -> List[TestResult]:
    """Run all test scripts locally and return results."""
    log_dir.mkdir(exist_ok=True, parents=True)
    results = []

    for script in scripts:
        result = run_single_test_local(script, config, root_dir, verbose)
        results.append(result)

        # Write individual log file
        safe_log_name = result.script_path.replace("/", "__")
        log_file = log_dir / f"{safe_log_name}_local.log"
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"Script: {result.script_path}\n")
            f.write(f"Status: {result.status}\n")
            f.write(f"Duration: {result.duration:.2f}s\n")
            f.write(f"Mode: Local execution\n")
            if result.exit_code is not None:
                f.write(f"Exit Code: {result.exit_code}\n")
            if result.error_message:
                f.write(f"Error: {result.error_message}\n")
            f.write("\n--- STDOUT ---\n")
            f.write(result.stdout or "")
            f.write("\n--- STDERR ---\n")
            f.write(result.stderr or "")

        # Print status
        status_emoji = {
            "passed": "✅",
            "failed": "❌",
            "timeout": "⏰",
            "skipped": "⏭️"
        }
        emoji = status_emoji.get(result.status, "❓")
        print(f"{emoji} {result.script_path} ({result.duration:.1f}s)")
        if result.error_message:
            print(f"   └─ {result.error_message}")

    return results

def generate_report(results: List[TestResult], root_dir: Path):
    """Generate a comprehensive test report."""
    # Create reports directory
    reports_dir = root_dir / "test" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Write JSON report
    json_report = reports_dir / "test_report.json"
    with open(json_report, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    # Write HTML report
    html_report = reports_dir / "test_report.html"
    with open(html_report, "w") as f:
        f.write(generate_html_report(results))

    print(f"\n📊 Reports generated:")
    print(f"   JSON: {json_report}")
    print(f"   HTML: {html_report}")

def generate_html_report(results: List[TestResult]) -> str:
    """Generate an HTML test report with collapsible details."""
    passed = [r for r in results if r.status == "passed"]
    failed = [r for r in results if r.status == "failed"]
    timeout = [r for r in results if r.status == "timeout"]
    skipped = [r for r in results if r.status == "skipped"]

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
        </style>
        <script>
            function toggleCollapsible(element) {{
                element.classList.toggle("active");
                var content = element.nextElementSibling;
                content.classList.toggle("active");
            }}
        </script>
    </head>
    <body>
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
            <table>
                <tr>
                    <th>Script</th>
                    <th>Status</th>
                    <th>Duration</th>
                    <th>Summary</th>
                    <th>Details</th>
                </tr>
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

        html += f"""
            <tr class="{status_class}">
                <td><strong>{result.script_path}</strong></td>
                <td><span class="status-badge status-{status_class}">{emoji} {result.status.title()}</span></td>
                <td><strong>{result.duration:.2f}s</strong></td>
                <td class="error-msg">{escape_html(result.error_message) if result.error_message else "Success" if result.status == "passed" else "-"}</td>
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
            ".github"
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
        log_dir = repo_root / "test" / "logs"


    # Load configuration
    config = load_config(args.config)

    # Override with command line arguments
    if args.timeout != 300:
        config.timeout = args.timeout
    if args.exclude:
        config.excluded_patterns.extend(args.exclude)

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

    # All scripts now have flyte.init (due to filtering), so no need for breakdown

    # Run tests
    print(f"\n🧪 Running tests with {config.timeout}s timeout...")
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
