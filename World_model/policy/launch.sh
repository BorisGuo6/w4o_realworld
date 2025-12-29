export HF_ENDPOINT=https://hf-mirror.com
export HF_TOKEN=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
export HF_HOME=/home/boris/workspace/World4Omni/hf-cache
export DISPLAY=:99
export OPEN3D_CPU_RENDERING=True

# python graspnet_rlbench_curobo.py
CUDA_VISIBLE_DEVICES=7 python anydexgrasp_rlbench.py --task_name PickUpCup --seed 1 