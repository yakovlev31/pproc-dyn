import torch


Tensor = torch.Tensor


def interpolate(t_eval: Tensor, t: Tensor, z: Tensor, method: str = "nearest") -> Tensor:
    """
    Interpolates values at specified evaluation points.

    Args:
        t_eval (Tensor): The evaluation time points, shape (n,).
        t (Tensor): The trajectory time points, shape (time,).
        z (Tensor): The trajectory values at time points `t`, shape (time, d_z).
        method (str, optional): The interpolation method ('nearest' or 'linear'). Defaults to 'nearest'.

    Returns:
        Tensor: Interpolated values at `t_eval`.
    """
    if method not in {"nearest", "linear"}:
        raise ValueError(f"Interpolation method {method} is not supported.")

    ind_right = torch.searchsorted(t, t_eval)
    ind_left = ind_right - 1
    ind_left.clamp_(min=0)
    ind_right.clamp_(max=len(t) - 1)

    if method == "nearest":
        return _nearest_interpolate(t_eval, t, z, ind_left, ind_right)
    else:  # method == "linear"
        return _linear_interpolate(t_eval, t, z, ind_left, ind_right)


def _nearest_interpolate(t_eval: Tensor, t: Tensor, z: Tensor, ind_left: Tensor, ind_right: Tensor) -> Tensor:
    dist_left = torch.abs(t_eval - t[ind_left])
    dist_right = torch.abs(t_eval - t[ind_right])
    nearer_right = dist_right < dist_left
    return torch.where(nearer_right.unsqueeze(1), z[ind_right], z[ind_left])


def _linear_interpolate(t_eval: Tensor, t: Tensor, z: Tensor, ind_left: Tensor, ind_right: Tensor) -> Tensor:
    t_left = t[ind_left]
    t_right = t[ind_right]
    weight_right = (t_eval - t_left) / (t_right - t_left + 1e-3)
    weight_left = 1 - weight_right
    return weight_left.unsqueeze(1) * z[ind_left] + weight_right.unsqueeze(1) * z[ind_right]
