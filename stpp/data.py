import numpy as np


ndarray = np.ndarray


class Trajectory:
    """Represents a data trajectory defined by time points `t`,
    spatial coordinates `x`, and observations `y`.

    Attributes:
        t (ndarray): Time points of the trajectory, has shape (N,).
        t_ctx (float): Size of the context time interval.
        x (ndarray): Spatial coordinates associated with each time point,
            the shape is (N, d_x), where d_x is the dimensionality of the space.
        y (ndarray): Observations at each time point,
            the shape is (N, D_y), where D_y is the dimensionality of the observations.
    """

    def __init__(
        self,
        t: ndarray,
        t_ctx: float,
        x: ndarray,
        y: ndarray,
    ) -> None:
        if t.ndim != 1:
            raise ValueError("t should be a 1-dimensional array.")
        if x.ndim != 2 or y.ndim != 2:
            raise ValueError("x and y should be 2-dimensional arrays.")
        if not (len(t) == x.shape[0] == y.shape[0]):
            raise ValueError("Length of t must match that of x and y.")

        self.t = t
        self.t_ctx = t_ctx
        self.x = x
        self.y = y

        self.t = self.shift_time(self.t, self.t_ctx)
        self.ctx_mask, self.obs_mask = self.create_time_masks(self.t, t_ref=0)

    def shift_time(self, t: ndarray, t_ctx: float) -> ndarray:
        """Shifts the last time point in [0, t_ctx] to zero.

        Args:
            t: Time points, has shape (N, ).
            t_ctx: Size of the context time interval.

        Returns:
            Time points shifted such that the last point in the context is at zero.
        """
        # mask = t <= t_ctx
        # if sum(mask) == 0:
        #     t_ref = t[0]
        # else:
        #     t_ref = t[mask][-1]
        # return t - t_ref
        return t - t_ctx

    def create_time_masks(self, t: ndarray, t_ref: float) -> tuple[ndarray, ndarray]:
        """Constructs context and observation time masks.

        Args:
            t: Time points, has shape (N, ).
            t_ref: Reference time point. All time points to the left of it correspond
                to the context mask, while all points to the right to the observation mask.

        Returns:
            Context and observation masks.
        """
        eps = 1e-6
        ctx_mask = t <= (t_ref + eps)
        obs_mask = t >= (t_ref - eps)
        return ctx_mask, obs_mask

    def context(self, merge: bool = False) -> tuple[ndarray, ...]:
        """Returns context part of the trajectory.

        Args:
            merge: Whether or not to merge the time points, spactial locations,
                and observations. If True, they are merged, otherwide they are
                returned separaterly.

        Returns:
            Context part of the trajectory.
        """
        t = self.t[self.ctx_mask]
        x = self.x[self.ctx_mask]
        y = self.y[self.ctx_mask]

        if merge:
            return (np.concatenate((t.reshape(-1, 1), x, y), axis=1),)
        else:
            return (t, x, y)

    def observations(self, merge: bool = False) -> tuple[ndarray, ...]:
        """Returns the observations part of the trajectory.

        Args:
            merge: Whether or not to merge the time points, spactial locations,
                and observations. If True, they are merged, otherwide they are
                returned separaterly.

        Returns:
            Observations part of the trajectory.
        """
        t = self.t[self.obs_mask]
        x = self.x[self.obs_mask]
        y = self.y[self.obs_mask]
        if merge:
            return (np.concatenate((t.reshape(-1, 1), x, y), axis=1),)
        else:
            return (t, x, y)
