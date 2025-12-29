GPUS=$1
IMG_CACHE=$2
REAL_SIM=$3
# SAVE_CACHE=$3

cd ~/w4o/feature_based_icp

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH}"
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '/usr/local/cuda' | paste -sd:)

conda activate depthanything
CUDA_VISIBLE_DEVICES=${GPUS} python generate_metric_depth.py --img-path ${IMG_CACHE}
echo "Depth estimation done. Saved to ${IMG_CACHE}"

conda activate geo-aware2

CUDA_VISIBLE_DEVICES=${GPUS} python matching_generation.py --img-path ${IMG_CACHE}
echo "Matching generation done. Saved to point_wise_matching.pkl"
CUDA_VISIBLE_DEVICES=${GPUS} python ft_based_umeyama.py --img-path ${IMG_CACHE} --real-sim ${REAL_SIM}
echo "Transformation done. Saved to transformation.pkl"

cd ~/w4o