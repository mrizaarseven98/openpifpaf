import numpy as np

from .utils import scalar_square_add_single
from ..functional import scalar_nonzero_clipped


class Occupancy():
    def __init__(self, shape, reduction, *, min_scale=None):
        assert len(shape) == 3
        if min_scale is None:
            min_scale = reduction
        assert min_scale >= reduction

        self.reduction = reduction
        self.min_scale = min_scale
        self.min_scale_reduced = min_scale / reduction

        self.occupancy = np.zeros((
            shape[0],
            int(shape[1] / reduction),
            int(shape[2] / reduction),
        ), dtype=np.uint8)

    def __len__(self):
        return len(self.occupancy)

    def set(self, f, x, y, s):
        """Setting needs to be centered at the rounded (x, y)."""
        xi = round(x / self.reduction)
        yi = round(y / self.reduction)
        si = round(max(self.min_scale_reduced, s / self.reduction))
        scalar_square_add_single(self.occupancy[f], xi, yi, si, 1)

    def get(self, f, x, y):
        """Getting needs to be done at the floor of (x, y)."""
        xi = x / self.reduction
        yi = y / self.reduction

        # floor is done in scalar_nonzero_clipped below
        return scalar_nonzero_clipped(self.occupancy[f], xi, yi)
