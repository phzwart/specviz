import numpy as np

from specviz.tasks.base_instrument import BaseInstrument


class NumpyInstrument(BaseInstrument):
    def __init__(
        self,
        data_array: np.ndarray,
        wavenumbers: np.ndarray,
        redis_host: str = "localhost",
        redis_port: int = 6379,
    ):
        """
        Initialize NumpyInstrument with pre-loaded data.

        Args:
            data_array (np.ndarray): 3D array of shape (C, Y, X) where:
                C is the number of spectral channels
                Y is the number of y positions
                X is the number of x positions
            wavenumbers (np.ndarray): 1D array of wavenumbers (length C)
            redis_host (str): Redis host address
            redis_port (int): Redis port number
        """
        super().__init__(redis_host=redis_host, redis_port=redis_port)

        if len(data_array.shape) != 3:
            raise ValueError(
                f"Data array must be 3D (C,Y,X), got shape {data_array.shape}"
            )
        if len(wavenumbers) != data_array.shape[0]:
            raise ValueError(
                f"Wavenumbers length ({len(wavenumbers)}) must match spectral dimension ({data_array.shape[0]})"
            )

        # Transpose data to match expected format if necessary
        if data_array.shape[0] != len(wavenumbers):
            print(
                f"Warning: Transposing data array from shape {data_array.shape} to match wavenumbers length {len(wavenumbers)}"
            )
            # Try different permutations to match wavenumbers
            for perm in [(0, 1, 2), (1, 0, 2), (2, 0, 1)]:
                transposed = np.transpose(data_array, perm)
                if transposed.shape[0] == len(wavenumbers):
                    data_array = transposed
                    print(f"Successfully transposed to shape {data_array.shape}")
                    break
            else:
                raise ValueError(
                    "Could not find correct data orientation to match wavenumbers"
                )

        self.data = data_array
        self.wavenumbers = wavenumbers
        self.shape = data_array.shape
        print(f"Initialized NumpyInstrument with data shape (C,Y,X): {self.shape}")
        print(f"Wavenumbers range: {wavenumbers[0]:.2f} to {wavenumbers[-1]:.2f}")

    def perform_measurement(self, measurement: dict) -> dict:
        """Get spectrum from the nearest integer coordinates in the data array"""
        # Round to nearest integer coordinates
        x = int(round(measurement["X"]))
        y = int(round(measurement["Y"]))

        # Truncate coordinates to valid ranges
        original_x, original_y = x, y
        x = max(0, min(x, self.shape[2] - 1))
        y = max(0, min(y, self.shape[1] - 1))

        # Warn if coordinates were truncated
        if x != original_x or y != original_y:
            print(
                f"Warning: Coordinates ({original_x}, {original_y}) truncated to ({x}, {y}) to fit bounds "
                f"[X: 0-{self.shape[2]-1}, Y: 0-{self.shape[1]-1}]"
            )

        print(f"Measuring at ({x}, {y}) - Index: {measurement['hcd_indx']}")

        # Extract spectrum at the specified coordinates
        spectrum = self.data[
            :, y, x
        ].copy()  # Make a copy to ensure we don't modify the original data

        # Verify spectrum shape matches wavenumbers
        if len(spectrum) != len(self.wavenumbers):
            raise ValueError(
                f"Extracted spectrum length {len(spectrum)} doesn't match wavenumbers length {len(self.wavenumbers)}"
            )

        return {
            "hcd_indx": measurement["hcd_indx"],
            "X": x,  # Return the actual (truncated) coordinates used
            "Y": y,
            "wavenumbers": self.wavenumbers.tolist(),
            "spectrum": spectrum.tolist(),
        }


def main():
    """Example usage of NumpyInstrument loading from NPZ file"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run a numpy-based instrument that reads from a 3D array"
    )
    parser.add_argument(
        "npz_file", help="Path to NPZ file containing data_cube and wavenumbers arrays"
    )
    parser.add_argument(
        "--redis-host",
        default="127.0.0.1",
        help="Redis host address (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--redis-port", type=int, default=6379, help="Redis port (default: 6379)"
    )

    args = parser.parse_args()

    try:
        # Load the NPZ file
        print(f"Loading data from {args.npz_file}")
        data = np.load(args.npz_file)
        if "data_cube" not in data or "wavenumbers" not in data:
            raise ValueError(
                "NPZ file must contain 'data_cube' and 'wavenumbers' arrays"
            )

        print(f"Loaded data_cube shape: {data['data_cube'].shape}")
        print(f"Loaded wavenumbers shape: {data['wavenumbers'].shape}")

        # Create instrument instance
        instrument = NumpyInstrument(
            data["data_cube"],
            data["wavenumbers"],
            redis_host=args.redis_host,
            redis_port=args.redis_port,
        )
        instrument.run()

    except Exception as e:
        print(f"Error loading NPZ file: {str(e)}")
        return


if __name__ == "__main__":
    main()
