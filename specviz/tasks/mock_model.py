import duckdb
import numpy as np
import pandas as pd
import redis
from tools.dbtools import check_table_exists, read_df_from_db, store_df_in_db
from scipy.spatial.distance import cdist


def create_mock_model(redis_host: str = "localhost", redis_port: int = 6379):
    """Create a mock model based on HCD data statistics"""
    try:
        # Connect to Redis and get database path
        redis_client = redis.Redis(
            host=redis_host, port=redis_port, decode_responses=True
        )

        db_path = redis_client.get("current_project")
        if not db_path:
            raise ValueError("No active project in Redis")

        # Connect to database and read HCD data
        conn = duckdb.connect(db_path)
        if not check_table_exists(conn, "HCD"):
            raise ValueError("HCD table not found")

        df = read_df_from_db(conn, "HCD")

        # Calculate statistics
        mean_x = df["X"].mean()
        mean_y = df["Y"].mean()
        sigma_x = df["X"].std()
        sigma_y = df["Y"].std()

        # Create RBF center and compute values
        sigma = min(sigma_x, sigma_y) / 3.0
        center1 = np.array([[mean_x - sigma * 2, mean_y + sigma * 2]])
        center2 = np.array([[mean_x + sigma * 2, mean_y - sigma * 2]])
        center3 = np.array([[mean_x + sigma * 2, mean_y + sigma * 2]])

        # Get points from HCD data
        points = df[["X", "Y"]].values

        # Compute RBF values for HCD points
        distances1 = cdist(points, center1)
        values1 = np.exp(-(distances1**2) / (2 * sigma**2))

        distances2 = cdist(points, center2)
        values2 = np.exp(-(distances2**2) / (2 * sigma**2))

        distances3 = cdist(points, center3)
        values3 = np.exp(-(distances3**2) / (2 * sigma**2))

        values = 0.33 * (values1 + values2 + values3)
        values = values / np.max(values)

        # Create model DataFrame with hcd_indx instead of coordinates
        model_df = pd.DataFrame(
            {"hcd_indx": np.arange(len(df)), "value": values.ravel()}
        )

        # Store model in database
        store_df_in_db(conn, model_df, "model_0", if_exists="replace", index=False)

        # Create or update conformal thresholds table
        conformal_df = pd.DataFrame(
            {
                "model_name": ["model_0"],
                "threshold_0.01": [0.05],
                "threshold_0.10": [0.20],
                "threshold_0.20": [0.25],
                "threshold_0.25": [0.30],
            }
        )

        if check_table_exists(conn, "conformal_thresholds"):
            existing_df = read_df_from_db(conn, "conformal_thresholds")
            # Remove existing entry for this model if it exists
            existing_df = existing_df[existing_df["model_name"] != "model_0"]
            conformal_df = pd.concat([existing_df, conformal_df], ignore_index=True)

        store_df_in_db(
            conn, conformal_df, "conformal_thresholds", if_exists="replace", index=False
        )

        # Update measured_points table with iteration_0
        if check_table_exists(conn, "measured_points"):
            measured_df = read_df_from_db(conn, "measured_points")
            # Add iteration_0 column based on HCD levels
            measured_df["iteration_0"] = np.where(df["level"] <= 1, 1, 0)
            store_df_in_db(
                conn, measured_df, "measured_points", if_exists="replace", index=False
            )
        else:
            # Create measured_points table with hcd_indx and iteration_0
            measured_df = pd.DataFrame(
                {
                    "hcd_indx": np.arange(len(df)),
                    "prior_measurements": np.zeros(len(df)),
                    "iteration_0": np.where(df["level"] <= 1, 1, 0),
                }
            )
            store_df_in_db(
                conn, measured_df, "measured_points", if_exists="replace", index=False
            )

        # Create or update models table
        if check_table_exists(conn, "models"):
            models_df = read_df_from_db(conn, "models")
            if "model_0" not in models_df["table_name"].values:
                new_model = pd.DataFrame({"table_name": ["model_0"]})
                models_df = pd.concat([models_df, new_model], ignore_index=True)
        else:
            models_df = pd.DataFrame({"table_name": ["model_0"]})

        store_df_in_db(conn, models_df, "models", if_exists="replace", index=False)
        conn.close()

        print(
            f"Created mock model with center ({mean_x:.2f}, {mean_y:.2f}) and sigma {sigma:.2f}"
        )
        return True

    except Exception as e:
        print(f"Error creating mock model: {str(e)}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create a mock model using Redis database path"
    )
    parser.add_argument(
        "--redis-host", default="localhost", help="Redis host (default: localhost)"
    )
    parser.add_argument(
        "--redis-port", type=int, default=6379, help="Redis port (default: 6379)"
    )

    args = parser.parse_args()
    create_mock_model(args.redis_host, args.redis_port)
