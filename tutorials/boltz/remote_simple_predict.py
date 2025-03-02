# Imports and init remote
import os
from union import UnionRemote
from flytekit.configuration import Config
from simple_predict import wf

os.environ["UNION_CONFIG"] = "/Users/pryceturner/.union/config_serving.yaml"
remote = UnionRemote(
    config=Config.auto(config_file="/Users/pryceturner/.union/config_serving.yaml")
)

remote.fast_register_workflow(entity=wf)
execution = remote.execute(entity=wf, inputs={"input": "prot.yaml"}, wait=True)
output = execution.outputs
print(output)
