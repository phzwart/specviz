import numpy as np
from scipy.interpolate import griddata


class InterpolatorBase:
    def interpolate(
        self, points: np.ndarray, values: np.ndarray, xi: np.ndarray, yi: np.ndarray
    ) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement interpolate method")

    @property
    def name(self) -> str:
        raise NotImplementedError("Subclasses must implement name property")

    @property
    def description(self) -> str:
        raise NotImplementedError("Subclasses must implement description property")


class NearestInterpolator(InterpolatorBase):
    def interpolate(
        self, points: np.ndarray, values: np.ndarray, xi: np.ndarray, yi: np.ndarray
    ) -> np.ndarray:
        grid_x, grid_y = np.meshgrid(xi, yi)
        return griddata(points, values, (grid_x, grid_y), method="nearest")

    @property
    def name(self) -> str:
        return "Nearest Neighbor"

    @property
    def description(self) -> str:
        return "Nearest neighbor interpolation - assigns value of closest data point"


class LinearInterpolator(InterpolatorBase):
    def interpolate(
        self, points: np.ndarray, values: np.ndarray, xi: np.ndarray, yi: np.ndarray
    ) -> np.ndarray:
        grid_x, grid_y = np.meshgrid(xi, yi)
        return griddata(points, values, (grid_x, grid_y), method="linear")

    @property
    def name(self) -> str:
        return "Linear"

    @property
    def description(self) -> str:
        return "Linear interpolation - assigns value based on linear interpolation"


class CubicInterpolator(InterpolatorBase):
    def interpolate(
        self, points: np.ndarray, values: np.ndarray, xi: np.ndarray, yi: np.ndarray
    ) -> np.ndarray:
        grid_x, grid_y = np.meshgrid(xi, yi)
        return griddata(points, values, (grid_x, grid_y), method="cubic")

    @property
    def name(self) -> str:
        return "Cubic"

    @property
    def description(self) -> str:
        return "Cubic interpolation - assigns value based on cubic interpolation"


class RBFInterpolator(InterpolatorBase):
    def __init__(self, kernel: str):
        self.kernel = kernel

    def interpolate(
        self, points: np.ndarray, values: np.ndarray, xi: np.ndarray, yi: np.ndarray
    ) -> np.ndarray:
        grid_x, grid_y = np.meshgrid(xi, yi)
        return griddata(
            points, values, (grid_x, grid_y), method="rbf", kernel=self.kernel
        )

    @property
    def name(self) -> str:
        return f"RBF ({self.kernel})"

    @property
    def description(self) -> str:
        return f"RBF interpolation - assigns value based on {self.kernel} radial basis function"


# Update the INTERPOLATORS dictionary
INTERPOLATORS = {
    "nearest": NearestInterpolator(),
    "linear": LinearInterpolator(),
    "cubic": CubicInterpolator(),
    "rbf_thin_plate": RBFInterpolator("thin_plate"),
    "rbf_gaussian": RBFInterpolator("gaussian"),
}
