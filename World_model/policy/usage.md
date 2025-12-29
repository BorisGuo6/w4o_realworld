## World4Omni/policy Usage (RLBench)

### 1) Quick Start

- Run AnyDexGrasp (default PickUpCup task):
```bash
cd /home/boris/workspace/World4Omni/policy
python anydexgrasp_rlbench.py \
  --task_name PickUpCup \
  --seed 1 \
  --ckpt_path /home/boris/workspace/World4Omni/ckpt/checkpoint.tar.18 \
  --save_dir tmp
```

- Other example scripts (optional):
  - `graspnet_rlbench.py`
  - `graspnet_rlbench_icp.py`
  - `graspnet_rlbench_curobo.py`
  - `graspnet_rlbench_openwinebottle.py`

Note: video generation is currently disabled in the script. If needed, re-enable `images_to_video(...)` or use an external tool to stitch frames.

### 2) Built-in tasks (TASK_REGISTRY in anydexgrasp_rlbench.py)

Common tasks and default prompts bundled in `anydexgrasp_rlbench.py`:

```text
OpenWineBottle
PutShoesInBox
TakePlateOffColoredDishRack
TakeFrameOffHanger
PlugChargerInPowerSupply
SlideCabinetOpenAndPlaceCups
StraightenRope
PickUpCup
CloseBox
```

Example:
```bash
python anydexgrasp_rlbench.py --task_name OpenWineBottle --seed 1
```

### 3) List ALL available RLBench tasks (introspection)

Use this to print every task class currently available from `rlbench.tasks`:
```bash
python - << 'PY'
import inspect
from rlbench import tasks as rl_tasks

def is_task_class(name, obj):
    if not inspect.isclass(obj):
        return False
    if name.startswith('_'):
        return False
    # Rough filter: classes defined in rlbench.tasks
    return obj.__module__ == rl_tasks.__name__

task_names = sorted([name for name, obj in vars(rl_tasks).items() if is_task_class(name, obj)])
for n in task_names:
    print(n)
print(f"\nTotal: {len(task_names)} tasks")
PY
```

Note: the exact task set depends on your RLBench version/fork. This script reflects whatever is installed.

### 4) Common arguments

- `--task_name`: task name (see the list above or use the introspection script)
- `--seed`: random seed (affects observations, initial states, etc.)
- `--steps`: number of interpolated poses between start/end (default: 80)
- `--ckpt_path`: AnyDexGrasp checkpoint path (e.g. `/home/boris/workspace/World4Omni/ckpt/checkpoint.tar.18`)
- `--save_dir`: output directory (default: `tmp`)

### 5) Environment tips

- Verified setup: Python 3.9, matching CUDA, PyTorch, and Transformers
- If out of GPU memory:
  - Free large processes (`nvidia-smi`)
  - `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  - Lower image resolution or reduce `--steps`

### 6) Paths & caches

- Checkpoint example: `/home/boris/workspace/World4Omni/ckpt/checkpoint.tar.18`
- Temporary outputs: `/home/boris/workspace/World4Omni/policy/tmp`

Optional mirrors/caches:
```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/home/boris/workspace/World4Omni/hf-cache
```


