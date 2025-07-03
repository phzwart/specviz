from typing import Dict, Optional, Set, Union

import argparse
import json

import redis


class MeasurementQueue:
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379):
        self.redis_client = redis.Redis(
            host=redis_host, port=redis_port, decode_responses=True
        )
        self.queue_key = "measurement_queue"
        self.queued_set_key = "queued_indices"  # Redis set to track queued indices

    def add_measurement(self, measurement: dict) -> bool:
        """Add measurement to queue if not already queued

        Args:
            measurement: Dictionary containing measurement details (must include hcd_indx)

        Returns:
            bool: Success status
        """
        try:
            hcd_indx = measurement.get("hcd_indx")
            if hcd_indx is None:
                raise ValueError("Measurement must include hcd_indx")

            # Check if index is already queued using Redis SET
            if not self.redis_client.sismember(self.queued_set_key, hcd_indx):
                # Add to queue and tracking set atomically using pipeline
                pipe = self.redis_client.pipeline()
                pipe.lpush(self.queue_key, json.dumps(measurement))
                pipe.sadd(self.queued_set_key, hcd_indx)
                pipe.execute()
                return True
            return False

        except Exception as e:
            print(f"Error adding measurement to queue: {str(e)}")
            return False

    def get_measurement(self, timeout: int = 0) -> Optional[dict]:
        """Get next measurement from queue

        Args:
            timeout: Time to wait for measurement (0 = no wait, None = infinite wait)

        Returns:
            Optional[Dict]: Measurement dictionary or None if queue is empty
        """
        try:
            # Use BRPOP for blocking operation with timeout
            result = self.redis_client.brpop(self.queue_key, timeout=timeout)
            if result is None:
                return None

            # BRPOP returns tuple of (key, value)
            _, measurement_json = result
            measurement = json.loads(measurement_json)

            # Remove from tracking set
            self.redis_client.srem(self.queued_set_key, measurement["hcd_indx"])

            return measurement

        except Exception as e:
            print(f"Error getting measurement from queue: {str(e)}")
            return None

    def queue_length(self) -> int:
        """Get current length of queue

        Returns:
            int: Number of measurements in queue
        """
        try:
            return self.redis_client.llen(self.queue_key)
        except Exception as e:
            print(f"Error getting queue length: {str(e)}")
            return 0

    def get_queued_indices(self) -> set[int]:
        """Get set of currently queued indices

        Returns:
            Set[int]: Set of hcd_indx values currently in queue
        """
        try:
            return {int(x) for x in self.redis_client.smembers(self.queued_set_key)}
        except Exception as e:
            print(f"Error getting queued indices: {str(e)}")
            return set()

    def clear_queue(self) -> bool:
        """Clear all measurements from queue and tracking set

        Returns:
            bool: Success status
        """
        try:
            pipe = self.redis_client.pipeline()
            pipe.delete(self.queue_key)
            pipe.delete(self.queued_set_key)
            pipe.execute()
            return True
        except Exception as e:
            print(f"Error clearing queue: {str(e)}")
            return False


def main():
    # Example usage
    parser = argparse.ArgumentParser(description="Measurement Queue Utility")
    parser.add_argument(
        "--redis-host",
        default="127.0.0.1",
        help="Redis host address (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--redis-port", type=int, default=6379, help="Redis port (default: 6379)"
    )

    args = parser.parse_args()

    queue = MeasurementQueue(redis_host=args.redis_host, redis_port=args.redis_port)

    # Example operations
    print("Adding test measurements...")
    queue.add_measurement({"hcd_indx": 1, "type": "test"})
    print("Adding duplicate measurement...")
    result = queue.add_measurement({"hcd_indx": 1, "type": "test"})
    print(f"Second add successful: {result}")

    print(f"Queue length: {queue.queue_length()}")
    print(f"Queued indices: {queue.get_queued_indices()}")

    print("Getting measurement...")
    measurement = queue.get_measurement(timeout=1)
    print(f"Got measurement: {measurement}")
    print(f"Queued indices after get: {queue.get_queued_indices()}")


if __name__ == "__main__":
    main()
