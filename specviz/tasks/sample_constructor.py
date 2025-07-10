from typing import Optional

import json

import duckdb
import numpy as np
import pandas as pd
import redis
from hiposa.poisson_tiler import PoissonTiler
from tools.dbtools import store_df_in_db, store_dict_in_db

from specviz.tasks.redis_logger import RedisLogger


class MySamplingGenerator:
    def __init__(self):
        self.conn_file = None
        self.logger = RedisLogger()

    def set_conn(self, conn_file):
        self.conn_file = conn_file

    def set_redis_client(self, redis_client: Optional[redis.Redis] = None):
        self.logger.set_redis_client(redis_client)

    def generate_sampling(self, config):
        try:
            self.logger.log(
                "Starting sampling generation",
                source="SamplingGenerator",
                config=json.dumps(config),
            )

            print("\n=== Starting Sampling Generation ===")
            print(f"Config received: {json.dumps(config, indent=2)}")

            # First build distance sequence
            distance = float(config["initial_distance"])
            distances = [distance]
            level_range = [0]
            done = False

            print(f"\nGenerating distance sequence:")
            print(f"Initial distance: {distance}")

            while not done:
                distance = distance / config["scale"]
                distances.append(distance)
                level_range.append(level_range[-1] + 1)
                if distance < config["minimum_distance"]:
                    done = True
                    distances[-1] = config["minimum_distance"]
                print(f"Level {level_range[-1]}: distance = {distances[-1]}")

            print("\nFinal sequences:")
            print(f"Distances: {distances}")
            print(f"Level range: {level_range}")

            print("\nInitializing PoissonTiler...")
            tiler_obj = PoissonTiler(config["tile_size"], distances)

            region = (
                (config["x_min"], config["x_max"]),
                (config["y_min"], config["y_max"]),
            )
            print(f"Region defined as: {region}")

            print("\nGenerating points...")
            points, levels = tiler_obj.get_points_in_region(region)
            pt_indx = np.arange(len(points))
            levels = np.array(levels, dtype=int)

            print(
                f"Generated {len(points)} points across {len(np.unique(levels))} levels"
            )

            print("\nCreating DataFrames...")
            df_level_scale = pd.DataFrame({"level": level_range, "distance": distances})

            df_points = pd.DataFrame(
                {
                    "index": pt_indx,
                    "X": points[:, 0],
                    "Y": points[:, 1],
                    "level": np.int64(levels),
                }
            )

            config_flags = {
                "current_level": 0,
                "tile_size": config["tile_size"],
                "scale": config["scale"],
                "max_length": config["initial_distance"],
                "min_length": config["minimum_distance"],
                "iteration": 0,
                "x_min": config["x_min"],
                "x_max": config["x_max"],
                "y_min": config["y_min"],
                "y_max": config["y_max"],
            }

            print("\nConnecting to database...")
            conn = duckdb.connect(self.conn_file)

            print("\nStoring DataFrames in database...")
            try:
                store_df_in_db(conn, df_level_scale, "level_scale", "replace")
                print("- Stored level_scale")
                store_df_in_db(conn, df_points, "HCD", "replace")
                print("- Stored HCD")

                # Convert to DataFrame and store
                df_config = pd.DataFrame([config_flags])
                store_df_in_db(conn, df_config, "config_flags", "replace")
                print("- Stored config_flags")

                # Create measured_points table
                df_measured = pd.DataFrame(
                    {"hcd_indx": pt_indx, "prior_measurements": np.zeros_like(pt_indx)}
                )
                store_df_in_db(conn, df_measured, "measured_points", "replace")
                print("- Stored measured_points")

            except Exception as e:
                print(f"\nError storing data in database: {str(e)}")
                raise

            print("\nClosing database connection...")
            conn.close()

            self.logger.log(
                f"Successfully generated sampling with {len(points)} points",
                source="SamplingGenerator",
            )

            print("\n=== Sampling Generation Complete ===")
            return True

        except Exception as e:
            print(f"\n!!! Error in generate_sampling: {str(e)}")
            self.logger.log(
                f"Error generating sampling: {str(e)}",
                level="ERROR",
                source="SamplingGenerator",
            )
            return False
