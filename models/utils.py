import torch


def transformation_from_parameters(axisangle, translation, invert=False):
    R = rot_from_axisangle(axisangle)
    t = translation.clone()
    if invert:
        R = R.transpose(1, 2)
        t *= -1
    T = get_translation_matrix(t)
    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)
    return M


def get_translation_matrix(translation_vector):
    T = torch.zeros(translation_vector.shape[0], 4, 4).cuda()
    t = translation_vector.contiguous().view(-1, 3, 1)
    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t
    return T


def rot_from_axisangle(vec):
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)
    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca
    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)
    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC
    rot = torch.zeros((vec.shape[0], 4, 4)).cuda()
    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1
    return rot


def disp_to_depth(disp, min_depth, max_depth):
    min_disp = 1 / max_depth  # 0.01
    max_disp = 1 / min_depth  # 10
    scaled_disp = min_disp + (max_disp - min_disp) * disp  # (10-0.01)*disp+0.01
    depth = 1 / scaled_disp
    return scaled_disp, depth


def robust_l1(pred, target):
    eps = 1e-3
    return torch.sqrt(torch.pow(target - pred, 2) + eps ** 2)
