@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the quaternion distance between the current object rotation and the target rotation
    q_dist = quat_distance(object_rot, goal_rot)

    # Inverse the distance and scale it down to better match the range of reinforcement learning algorithms
    scaled_q_dist = -1.0 / (1.0 + q_dist)

    # Reward for reaching the target rotation
    rotation_reward_temperature = 10.0
    rotation_reward = scaled_q_dist * rotation_reward_temperature

    total_reward = rotation_reward

    # Store individual reward components in a dictionary
    reward_dict = {"rotation_reward": rotation_reward}

    return total_reward, reward_dict
