@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor, fingertip_pos: torch.Tensor, object_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute quaternion distance between object_rot and goal_rot
    rot_diff = torch.abs(torch.sum(torch.mul(object_rot, goal_rot), dim=1))
    quaternion_temp = 7.0
    quaternion_reward = torch.exp(-quaternion_temp * (1 - rot_diff))
    
    # Calculate the distance between fingertips and object position
    fingertip_object_distance = torch.norm(fingertip_pos - object_pos.unsqueeze(dim=1), dim=-1)
    
    # Encourage the shadow hand to maintain contact with the object
    contact_reward_threshold = 0.03
    contact_temp = 12.0
    contact_reward = torch.exp(-contact_temp * (fingertip_object_distance - contact_reward_threshold))
    contact_reward = torch.mean(contact_reward, dim=1)
    
    # Strengthen the orientation influence when maintaining contact
    orientation_influence_temp = 3.0
    orientation_influence_reward = torch.exp(orientation_influence_temp * rot_diff)
    combined_orientation_reward = quaternion_reward * contact_reward * orientation_influence_reward
    
    # Combine rewards
    reward = 0.7 * quaternion_reward + 0.3 * combined_orientation_reward
    
    individual_rewards = {
        "quaternion_reward": quaternion_reward,
        "contact_reward": contact_reward,
        "combined_orientation_reward": combined_orientation_reward
    }

    return reward, individual_rewards
