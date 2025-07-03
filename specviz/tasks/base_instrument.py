import json
import signal
import sys
import time
from abc import ABC, abstractmethod

import redis

from specviz.tasks.measurement_queue import MeasurementQueue


class BaseInstrument(ABC):
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379):
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_client = None
        self.running = True
        self.measurement_queue = MeasurementQueue(
            redis_host=redis_host, redis_port=redis_port
        )

        # Set up signal handling for graceful shutdown
        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGTERM, self.handle_shutdown)

    def connect_redis(self):
        """Establish connection to Redis"""
        try:
            self.redis_client = redis.Redis(
                host=self.redis_host, port=self.redis_port, decode_responses=True
            )
            self.redis_client.ping()
            print(f"Connected to Redis at {self.redis_host}:{self.redis_port}")
            return True
        except Exception as e:
            print(f"Failed to connect to Redis: {str(e)}")
            return False

    def is_collection_active(self) -> bool:
        """Check if collection is currently active"""
        try:
            return self.redis_client.get("collect_data") == "True"
        except:
            return False

    @abstractmethod
    def perform_measurement(self, measurement: dict) -> dict:
        """
        Perform a measurement at given coordinates.
        Must be implemented by subclasses.

        Args:
            measurement (dict): Measurement parameters including X, Y, and hcd_indx

        Returns:
            dict: Measurement results including wavenumbers and spectrum data
        """
        pass

    def handle_shutdown(self, signum, frame):
        """Handle shutdown gracefully"""
        print("\nShutting down instrument...")
        self.running = False
        if self.redis_client:
            self.redis_client.set("instrument_heartbeat", "False")
        sys.exit(0)

    def run(self):
        """Run the instrument main loop"""
        if not self.connect_redis():
            return

        print(f"Starting {self.__class__.__name__}...")
        print("Press Ctrl+C to stop")

        # Set initial heartbeat
        self.redis_client.set("instrument_heartbeat", "True")
        last_status = None

        try:
            while self.running:
                # Keep heartbeat True
                self.redis_client.set("instrument_heartbeat", "True")

                # Check collection status
                is_active = self.is_collection_active()

                # Print status changes
                if is_active != last_status:
                    print(f"Collection {'active' if is_active else 'paused'}")
                    last_status = is_active

                if is_active:
                    # Check queue length
                    if self.measurement_queue.queue_length() == 0:
                        print("Queue empty, pausing collection")
                        self.redis_client.set("collect_data", "False")
                        time.sleep(0.1)
                        continue

                    # Double-check status before getting measurement
                    if self.is_collection_active():
                        measurement = self.measurement_queue.get_measurement(
                            timeout=0.1
                        )
                        if measurement:
                            # Triple-check status before processing
                            if self.is_collection_active():
                                result = self.perform_measurement(measurement)
                                self.redis_client.lpush("xo_actual", json.dumps(result))
                            else:
                                # Put measurement back in queue if collection was paused
                                self.measurement_queue.add_measurement(measurement)
                                print(
                                    "Collection paused during measurement, returning to queue"
                                )
                        time.sleep(0.05)
                else:
                    time.sleep(0.1)  # Longer sleep when paused

                time.sleep(0.01)  # Minimal delay between cycles

        except Exception as e:
            print(f"Error during operation: {str(e)}")
        finally:
            # Ensure heartbeat is set to False when stopping
            self.redis_client.set("instrument_heartbeat", "False")
