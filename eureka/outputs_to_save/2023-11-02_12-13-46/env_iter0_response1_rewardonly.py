@torch.jit.script
def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1[:, 3], q1[:, 0], q1[:, 1], q1[:, 2]
    w2, x2, y2, z2 = q2[:, 3], q2[:, 0], q2[:, 1], q2[:, 2]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2

    return torch.cat((x.unsqueeze(1), y.unsqueeze(1), z.unsqueeze(1), w.unsqueeze(1)), dim=1)