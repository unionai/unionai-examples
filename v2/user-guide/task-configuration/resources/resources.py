# {{docs-fragment params}}
resources = Resources(
    cpu: Union[int, float, str, Tuple[Union[int, float, str], Union[int, float, str]], None] = None,
    memory: Union[str, Tuple[str, str], None] = None,
    gpu: Union[str, int, Device, None] = None,  # Accelerators string, count, or Device object
    disk: Union[str, None] = None,
    shm: Union[str, Literal["auto"], None] = None
)
# {{/docs-fragment params}}

# {{docs-fragment task-env}}
import flyte

# Define a TaskEnvironment for ML training tasks
env = flyte.TaskEnvironment(
    name="ml-training",
    resources=flyte.Resources(
        cpu=("2", "8"),        # Request 2 cores, allow up to 8 cores for scaling
        memory=("8Gi", "32Gi"), # Request 8 GiB, allow up to 32 GiB for large datasets
        gpu="A100:2",          # 2 NVIDIA A100 GPUs for training
        disk="50Gi",           # 50 GiB ephemeral storage for checkpoints
        shm="8Gi"              # 8 GiB shared memory for efficient data loading
    )
)

# Use the environment for tasks
@env.task
async def train_model(dataset_path: str) -> str:
    # This task will run with flexible resource allocation
    return "model_trained_successfully"
# {{/docs-fragment task-env}}

# {{docs-fragment override}}
# Override resources for specific tasks
@env.task(
    resources=flyte.Resources(
        cpu="16",
        memory="64Gi",
        gpu="H100:2",
        disk="50Gi",
        shm="8Gi"
    )
)
async def heavy_training_task() -> str:
    return "heavy_model_trained"
# {{/docs-fragment override}}

# {{docs-fragment cpu}}
# String formats (Kubernetes-style)
flyte.Resources(cpu="500m")        # 500 milliCPU (0.5 cores)
flyte.Resources(cpu="2")           # 2 CPU cores
flyte.Resources(cpu="1.5")         # 1.5 CPU cores

# Numeric formats
flyte.Resources(cpu=1)             # 1 CPU core
flyte.Resources(cpu=0.5)           # 0.5 CPU cores

# Request and limit ranges
flyte.Resources(cpu=("1", "2"))    # Request 1 core, limit to 2 cores
flyte.Resources(cpu=(1, 4))        # Request 1 core, limit to 4 cores
# {{/docs-fragment cpu}}

# {{docs-fragment memory}}
# Standard memory units
flyte.Resources(memory="512Mi")    # 512 MiB
flyte.Resources(memory="1Gi")      # 1 GiB
flyte.Resources(memory="2Gi")      # 2 GiB
flyte.Resources(memory="500M")     # 500 MB (decimal)
flyte.Resources(memory="1G")       # 1 GB (decimal)

# Request and limit ranges
flyte.Resources(memory=("1Gi", "4Gi"))  # Request 1 GiB, limit to 4 GiB
# {{/docs-fragment memory}}

# {{docs-fragment gpu}}
# Basic GPU count
flyte.Resources(gpu=1)             # 1 GPU (any available type)
flyte.Resources(gpu=4)             # 4 GPUs

# Specific GPU types with quantity
flyte.Resources(gpu="T4:1")        # 1 NVIDIA T4 GPU
flyte.Resources(gpu="A100:2")      # 2 NVIDIA A100 GPUs
flyte.Resources(gpu="H100:8")      # 8 NVIDIA H100 GPUs
# {{/docs-fragment gpu}}

# {{docs-fragment advanced-gpu}}
# Using the GPU helper function
gpu_config = flyte.GPU(device="A100", quantity=2)
flyte.Resources(gpu=gpu_config)

# GPU with memory partitioning (A100 only)
partitioned_gpu = flyte.GPU(
    device="A100",
    quantity=1,
    partition="1g.5gb"  # 1/7th of A100 with 5GB memory
)
flyte.Resources(gpu=partitioned_gpu)

# A100 80GB with partitioning
large_partition = flyte.GPU(
    device="A100 80G",
    quantity=1,
    partition="7g.80gb"  # Full A100 80GB
)
# {{/docs-fragment advanced-gpu}}

# {{docs-fragment custom}}
# Custom device configuration
custom_device = flyte.Device(
    device="custom_accelerator",
    quantity=2,
    partition="large"
)

resources = flyte.Resources(gpu=custom_device)
# {{/docs-fragment custom}}

# {{docs-fragment tpu}}
# TPU v5p configuration
tpu_config = flyte.TPU(device="V5P", partition="2x2x1")
flyte.Resources(gpu=tpu_config)  # Note: TPUs use the gpu parameter

# TPU v6e configuration
tpu_v6e = flyte.TPU(device="V6E", partition="4x4")
flyte.Resources(gpu=tpu_v6e)
# {{/docs-fragment tpu}}

# {{docs-fragment disk}}
flyte.Resources(disk="10Gi")       # 10 GiB ephemeral storage
flyte.Resources(disk="100Gi")      # 100 GiB ephemeral storage
flyte.Resources(disk="1Ti")        # 1 TiB for large-scale data processing

# Common use cases
flyte.Resources(disk="50Gi")       # ML model training with checkpoints
flyte.Resources(disk="200Gi")      # Large dataset preprocessing
flyte.Resources(disk="500Gi")      # Video/image processing workflows
# {{/docs-fragment disk}}

# {{docs-fragment shm}}
flyte.Resources(shm="1Gi")         # 1 GiB shared memory (/dev/shm)
flyte.Resources(shm="auto")        # Auto-sized shared memory
flyte.Resources(shm="16Gi")        # Large shared memory for distributed training
# {{/docs-fragment shm}}