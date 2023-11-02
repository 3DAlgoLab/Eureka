@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor, fingertip_pos: torch.Tensor, object_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute quaternion distance between object_rot and goal_rot
    rot_diff = torch.abs(torch.sum(torch.mul(object_rot, goal_rot), dim=1))
    
    # Increase the temperature variable to further adjust the scale of quaternion_reward
    quaternion_temperature = 15.0
    quaternion_reward = torch.exp(-quaternion_temperature * (1 - rot_diff))
    
    # Calculate the distance between fingertips and object position
    fingertip_object_distance = torch.norm(fingertip_pos - object_pos.unsqueeze(dim=1), dim=-1)
    
    # Encourage the shadow hand to maintain contact with the object
    contact_reward_threshold = 0.03
    contact_reward = torch.exp(-10 * (fingertip_object_distance - contact_reward_threshold))
    contact_reward = torch.mean(contact_reward, dim=1)
    
    # New direction_reward component
    direction_reward = torch.mul(goal_rot, object_rot).sum(dim=1)
    direction_reward = torch.clamp(direction_reward, min=0.0)
    direction_temperature = 5.0
    direction_reward = torch.exp(direction_temperature * direction_reward)

    # Combine quaternion reward, contact reward, and direction_reward components
    reward = quaternion_reward * contact_reward * direction_reward
    
    individual_rewards = {
        "quaternion_reward": quaternion_reward,
        "contact_reward": contact_reward,
        "direction_reward": direction_reward,
    }

    return reward, individual_rewards
