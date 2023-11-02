@torch.jit.script
def quat_conjugate(q: Tensor) -> Tensor:
    return torch.cat((q[:, 0:1], -q[:, 1:2], -q[:, 2:3], -q[:, 3:4]), dim=1)
