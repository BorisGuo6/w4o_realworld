import numpy as np 
from rel import DATA_PATH

from pov.datasets.pov_zarr_dataset import PovZarrDataset
from pov.utils.numpy.common import preallocate_and_concatenate

if __name__ == "__main__":
    ##########################################################################################
    # Hyperparameters
    ##########################################################################################
    exp_dir = DATA_PATH / "0428_9points_zixuan" / "raw_data"
    start_idx = 9
    end_idx = 17
    save_dir = exp_dir.parent / f"{start_idx}_{end_idx}_dp3.zarr"

    ##########################################################################################
    # Main code
    ##########################################################################################

    robot0_all_qpos = []
    point_clouds_arrays = []
    original_actions_arrays = []
    episode_ends_arrays = []

    pov_zarr_dataset = PovZarrDataset(
        save_path=str(save_dir),
    )

    total_step_count = 0
    for i in range(start_idx, end_idx + 1):
        file_path = exp_dir / f"{i}.npz"
        data = np.load(file_path, allow_pickle=True)

        robot0_all_qpos.append(data["proprioceptions"])
        point_clouds_arrays.append(data["pointclouds"])
        original_actions_arrays.append(data["original_actions"])

        total_step_count += len(data["proprioceptions"])
        episode_ends_arrays.append(total_step_count)

    original_actions_arrays = preallocate_and_concatenate(original_actions_arrays, axis=0)
    point_clouds_arrays = preallocate_and_concatenate(point_clouds_arrays, axis=0)
    point_clouds_arrays = np.transpose(point_clouds_arrays, (0, 2, 1))
    robot0_all_qpos = preallocate_and_concatenate(robot0_all_qpos, axis=0)


    pov_zarr_dataset.save_data({
        "data/obs/robot0_all_qpos": np.asarray(robot0_all_qpos, dtype=np.float32),
        "data/obs/point_clouds": np.asarray(point_clouds_arrays, dtype=np.float32),
        "data/actions/original_actions": np.asarray(original_actions_arrays, dtype=np.float32),
        "meta/episode_ends": np.asarray(episode_ends_arrays, dtype=np.int64),
    })

    pov_zarr_dataset.print_structure()
