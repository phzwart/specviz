from typing import Any, Dict

import os

import dask
import numpy as np
import psutil
import redis
from dask.distributed import Client, progress

from specviz.tasks.interpolators import INTERPOLATORS
from specviz.tasks.redis_logger import RedisLogger


def interpolate_channel(
    points: np.ndarray,
    values: np.ndarray,
    xi: np.ndarray,
    yi: np.ndarray,
    interpolator_key: str,
) -> np.ndarray:
    """Interpolate a single channel"""
    interpolator = INTERPOLATORS[interpolator_key]
    return interpolator.interpolate(points, values, xi, yi)


def run_worker(config, queue):
    """Standalone function to run the worker process"""
    try:
        # Convert lists back to numpy arrays
        config["points"] = np.array(config["points"])
        config["values"] = np.array(config["values"])
        config["xi"] = np.array(config["xi"])
        config["yi"] = np.array(config["yi"])

        # Create new Redis client in worker process
        redis_client = redis.Redis(
            host=config["redis_host"], port=config["redis_port"], decode_responses=True
        )
        worker = InterpolationWorker(redis_client)
        success = worker.run_interpolation(config)
        queue.put(("DONE", success))
    except Exception as e:
        print(f"Worker error: {str(e)}")
        queue.put(("DONE", False))


class InterpolationWorker:
    def __init__(self, redis_client=None):
        self.logger = RedisLogger()
        self.logger.set_redis_client(redis_client)
        self.client = None
        self.pid = os.getpid()

        self.logger.log(
            f"Worker process started (PID: {self.pid})", source="InterpolationWorker"
        )

    def setup_dask_client(self):
        if self.client is None:
            self.client = Client()
            self.logger.log(
                f"Dask client started. Dashboard at {self.client.dashboard_link}",
                source="InterpolationWorker",
            )

    def run_interpolation(self, config: dict[str, Any]) -> bool:
        try:
            # Remove Redis connection details from config before logging
            config_log = {
                k: v for k, v in config.items() if k not in ["redis_host", "redis_port"]
            }
            self.logger.log(
                f"Starting interpolation with config: {config_log}",
                source="InterpolationWorker",
            )

            self.setup_dask_client()

            # Extract configuration
            points = config["points"]
            values = config["values"]
            xi = config["xi"]
            yi = config["yi"]
            interpolator_key = config["interpolator_key"]

            # Create delayed computation for each channel
            n_channels = values.shape[1]
            delayed_results = []

            self.logger.log(
                f"Starting interpolation of {n_channels} channels using {interpolator_key}",
                source="InterpolationWorker",
            )

            for i in range(n_channels):
                result = dask.delayed(interpolate_channel)(
                    points, values[:, i], xi, yi, interpolator_key
                )
                delayed_results.append(result)

            # Compute all channels
            results = dask.compute(*delayed_results)

            # Stack results into 3D array (y, x, channel)
            interpolated_data = np.stack(results, axis=-1)

            self.logger.log(
                f"Interpolation complete. Output shape: {interpolated_data.shape}",
                source="InterpolationWorker",
            )

            return True

        except Exception as e:
            self.logger.log(
                f"Error in interpolation: {str(e)}",
                level="ERROR",
                source="InterpolationWorker",
            )
            return False

        finally:
            if self.client is not None:
                self.client.close()
                self.client = None
