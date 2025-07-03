import argparse
import json
import signal
import sys
import time

import numpy as np
import redis

from specviz.tasks.base_instrument import BaseInstrument


class RandomInstrument(BaseInstrument):
    def perform_measurement(self, measurement: dict) -> dict:
        """Simulate a measurement at given coordinates"""
        print(
            f"Measuring at ({measurement['X']}, {measurement['Y']}) - Index: {measurement['hcd_indx']}"
        )

        # Generate mock spectrum data
        wavenumbers = np.linspace(100, 1000, 50)  # Reduced points for speed
        spectrum = np.random.normal(1.0, 0.1, 50) * np.exp(
            -((wavenumbers - 500) ** 2) / 10000
        )

        # Simulate brief measurement time
        time.sleep(0.1)  # 100ms delay to simulate measurement

        return {
            "hcd_indx": measurement["hcd_indx"],
            "X": measurement["X"],
            "Y": measurement["Y"],
            "wavenumbers": wavenumbers.tolist(),
            "spectrum": spectrum.tolist(),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Run a mock instrument that updates Redis heartbeat"
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

    instrument = RandomInstrument(
        redis_host=args.redis_host, redis_port=args.redis_port
    )
    instrument.run()


if __name__ == "__main__":
    main()
