@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor, fingertip_pos: torch.Tensor, object_pos: torch.Tensor, object_angvel: torch.Tensor, target_angvel: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute quaternion distance between object_rot and goal_rot
    rot_diff = torch.abs(torch.sum(torch.mul(object_rot, goal_rot), dim=1))
    quaternion_temp = 25
    quaternion_reward = torch.exp(-quaternion_temp * (1 - rot_diff))
    
    # Calculate the distance between fingertips and object position
    fingertip_object_distance = torch.norm(fingertip_pos - object_pos.unsqueeze(dim=1), dim=-1)
    
    # Encourage the shadow hand to maintain contact with the object
    contact_reward_threshold = 0.03
    contact_reward = torch.exp(-10 * (fingertip_object_distance - contact_reward_threshold))
    contact_reward = torch.mean(contact_reward, dim=1)
    
    # Encourage the agent to achieve the desired angular velocity
    angvel_diff = torch.abs(object_angvel - target_angvel)
    angvel_temp = 10
    angvel_reward = torch.exp(-angvel_temp * angvel_diff)
    angvel_reward = torch.mean(angvel_reward, dim=1)
    
    # Combine quaternion reward, contact reward, and angvel_reward
    reward = quaternion_reward * contact_reward * angvel_reward
    
    individual_rewards = {
        "quaternion_reward": quaternion_reward,
        "contact_reward": contact_reward,
        "angvel_reward": angvel_reward,
    }

    return reward, individual_rewards
