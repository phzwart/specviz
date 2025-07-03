from typing import Optional

import json
from datetime import datetime

import redis


class RedisLogger:
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self.log_key = "specviz:logs"  # Stream key for logs
        self.max_length = 1000  # Maximum number of log entries to keep

    def set_redis_client(self, redis_client: Optional[redis.Redis] = None):
        """Update Redis client"""
        self.redis_client = redis_client

    def log(
        self, message: str, level: str = "INFO", source: str = None, **kwargs
    ) -> bool:
        """Log a message to Redis stream

        Args:
            message: The log message
            level: Log level (INFO, WARNING, ERROR, DEBUG)
            source: Source of the log message (e.g., "DatabaseConstructor")
            **kwargs: Additional fields to log

        Returns:
            bool: Success status
        """
        if not self.redis_client:
            return False

        try:
            # Create log entry
            entry = {
                "timestamp": datetime.now().isoformat(),
                "level": level.upper(),
                "message": message,
                "source": source or "Unknown",
            }
            # Add any additional fields
            entry.update(kwargs)

            # Convert all values to strings for Redis
            entry = {k: str(v) for k, v in entry.items()}

            # Add to stream
            self.redis_client.xadd(
                self.log_key, entry, maxlen=self.max_length, approximate=True
            )
            return True

        except Exception as e:
            print(f"Failed to log to Redis: {str(e)}")
            return False

    def get_logs(self, count: int = 100) -> list:
        """Get recent logs from Redis

        Args:
            count: Number of log entries to retrieve

        Returns:
            list: List of log entries
        """
        if not self.redis_client:
            return []

        try:
            logs = self.redis_client.xrevrange(self.log_key, count=count)
            # No need to decode since we're using decode_responses=True
            return [entry[1] for entry in logs]
        except Exception as e:
            print(f"Failed to retrieve logs: {str(e)}")
            return []
