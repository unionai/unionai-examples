from pathlib import Path

from union.remote import UnionRemote

remote = UnionRemote()

execution = remote.fetch_execution(name="a9srv9c7r8j7bw4h6n45")
fp = Path("output")
for k in execution.outputs:
    (fp / k).mkdir(parents=True, exist_ok=True)
remote.download(execution.outputs, "output", recursive=True)
