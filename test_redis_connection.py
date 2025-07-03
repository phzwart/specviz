#!/usr/bin/env python3

import sys

import redis


def test_redis_connection():
    try:
        # Create Redis client with same settings as the app
        redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)

        print("Testing Redis connection...")

        # Test ping
        if redis_client.ping():
            print("✓ Redis ping successful")
        else:
            print("✗ Redis ping failed")
            return False

        # Test getting current project
        current_project = redis_client.get("current_project")
        if current_project:
            print(f"✓ Current project: {current_project}")
        else:
            print("✗ No current project found")
            return False

        print("✓ Redis connection test passed")
        return True

    except Exception as e:
        print(f"✗ Redis connection error: {str(e)}")
        return False


if __name__ == "__main__":
    success = test_redis_connection()
    sys.exit(0 if success else 1)
