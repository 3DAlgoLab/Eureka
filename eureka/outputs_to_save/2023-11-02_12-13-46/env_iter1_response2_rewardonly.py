@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor, success_threshold: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # The distance between the current object rotation and the target rotation
    rot_distance = torch.norm(object_rot - goal_rot, dim=1)
    
    # Reward function components
    rot_reward = -rot_distance
    
    temperature = 100.0  # Controls sharpness of reward scaling
    reward_normalized = torch.exp(temperature * rot_reward)
    
    # Check success: When distance between current rotation and target rotation is smaller than the threshold
    success = (rot_distance < success_threshold).float()
    
    # Total reward (considering success)
    total_reward = reward_normalized * success

    # Individual reward components dictionary
    reward_components = {"rot_reward": rot_reward, "success": success}
    
    return total_reward, reward_components
