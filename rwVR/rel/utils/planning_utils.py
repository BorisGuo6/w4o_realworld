import numpy as np


def min_jerk_interpolator_with_alpha(waypt_joint_values_np, planner_timestep, cmd_timestep, alpha=0.2):
    """
    Min Jerk Interpolator with alpha to generate smooth trajectory command values.
    
    Args:
    - waypt_joint_values_np: np.ndarray of shape (n_sparse_wp, n_dof), waypoint joint values.
    - planner_timestep: float, the timestep of the planner, e.g., 1.0/20.0.
    - cmd_timestep: float, the timestep of the command, e.g., 1.0/500.0.
    - alpha: float, tuning parameter for the interpolation interval (0 < alpha <= 1), default is 0.33.
    
    Returns:
    - cmd_joint_values_np: np.ndarray, interpolated joint values for each command timestep.
    """
    
    n_sparse_wp, n_dof = waypt_joint_values_np.shape
    n_steps = int(planner_timestep / cmd_timestep)  # Number of interpolation steps

    # Calculate time fractions for interpolation (t')
    t = np.linspace(0, 1, n_steps)
    t_prime = np.clip(t / alpha, 0, 1)  # Apply alpha scaling and clipping

    # Min jerk interpolation formula using t'
    t_hat = 10 * t_prime**3 - 15 * t_prime**4 + 6 * t_prime**5  

    # Initialize the array for command joint values
    cmd_joint_values_np = []

    # Vectorized interpolation between waypoints
    for i in range(n_sparse_wp - 1):
        start_wp = waypt_joint_values_np[i]
        end_wp = waypt_joint_values_np[i + 1]
        
        # Interpolating values for the current segment using broadcasting
        interpolated_values = (1 - t_hat[:, np.newaxis]) * start_wp + t_hat[:, np.newaxis] * end_wp
        cmd_joint_values_np.append(interpolated_values)

    # Stack all interpolated segments together
    cmd_joint_values_np = np.vstack(cmd_joint_values_np)
    
    return cmd_joint_values_np