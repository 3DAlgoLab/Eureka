@torch.jit.script
def compute_reward(object_rot: Tensor, goal_rot: Tensor, object_angvel: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
    # Constants
    orientation_weight = 0.7
    angvel_weight = 0.3
    orientation_temp = 10.0
    angvel_temp = 5.0
    
    # Compute the orientation error between object and goal rotations
    relative_rot = torch.mul(object_rot, torch.conj(goal_rot))
    orientation_error = 1.0 - torch.abs(relative_rot[:, 3])
    
    # Compute the angular velocity error
    angvel_error = torch.norm(object_angvel, dim=1)

    # Transform the orientation error and the angular velocity error
    orientation_reward = -torch.exp(-orientation_temp * orientation_error)
    angvel_reward = -torch.exp(-angvel_temp * angvel_error)

    # Calculate the total reward
    total_reward = orientation_weight * orientation_reward + angvel_weight * angvel_reward

    # Store individual rewards
    reward_dict = {
        "orientation_reward": orientation_reward,
        "angvel_reward": angvel_reward
    }
  
    return total_reward, reward_dict
