from typing import Optional

from pathlib import Path

import duckdb
import redis

from specviz.tasks.redis_logger import RedisLogger


class DuckDBConstructor:
    def __init__(self):
        self.logger = RedisLogger()
        self.redis_client = None

    def set_redis_client(self, client):
        """Set Redis client"""
        self.redis_client = client

    def create_database(self, config: dict) -> bool:
        """Create a DuckDB database with initial metadata table

        Args:
            config: Dictionary containing:
                - project_id: str
                - description: str
                - data_path: str
                - redis_client: Optional[redis.Redis]  # Redis client for storing current project

        Returns:
            bool: Success status
        """
        try:
            # Initialize collection status in Redis
            if self.redis_client:
                self.redis_client.set("collect_data", "False")

            self.logger.log(
                f"Creating database for project {config['project_id']}",
                source="DatabaseConstructor",
            )

            # Clean project_id for filename
            clean_project_id = config["project_id"].replace(" ", "_")

            # Ensure directory exists
            data_path = Path(config["data_path"])
            data_path.mkdir(parents=True, exist_ok=True)

            # Create database path
            db_path = data_path / f"{clean_project_id}.duckdb"

            # Connect to database (creates if doesn't exist)
            conn = duckdb.connect(str(db_path))

            # Create metadata table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    project_id VARCHAR,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Create measured_points table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS measured_points (
                    hcd_indx INTEGER PRIMARY KEY,
                    prior_measurements INTEGER DEFAULT 0
                )
            """
            )

            # Insert project metadata
            conn.execute(
                """
                INSERT INTO metadata (project_id, description)
                VALUES (?, ?)
            """,
                [config["project_id"], config["description"]],
            )

            # Commit and close
            conn.commit()
            conn.close()

            # Store database path in Redis if client is available
            if self.redis_client is not None:
                try:
                    self.redis_client.set("current_project", str(db_path))
                except Exception as e:
                    print(f"Warning: Could not set Redis key: {str(e)}")

            self.logger.log(
                f"Successfully created database at {db_path}",
                source="DatabaseConstructor",
            )

            return True

        except Exception as e:
            self.logger.log(
                f"Error creating database: {str(e)}",
                level="ERROR",
                source="DatabaseConstructor",
            )
            return False
