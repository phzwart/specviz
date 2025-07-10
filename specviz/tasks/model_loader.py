from typing import List, Optional

import threading
from queue import Queue

import duckdb
from tools.dbtools import (
    check_table_exists,
    fetch_dict_from_db,
    read_df_from_db,
)


class ModelLoader:
    def __init__(self):
        self.current_model = None
        self.model_data = None
        self.loading_queue = Queue()
        self.status_queue = Queue()
        self._loading_thread = None

    def get_available_models(self, db_path: str) -> list[str]:
        """Get list of available models from the models table"""
        try:
            conn = duckdb.connect(db_path)
            if not check_table_exists(conn, "models"):
                conn.close()
                return []

            df = read_df_from_db(conn, "models")
            conn.close()

            if "table_name" in df.columns:
                return df["table_name"].tolist()
            return []
        except Exception as e:
            print(f"Error getting models: {str(e)}")
            return []

    def load_model(self, db_path: str, model_name: str):
        """Start asynchronous loading of the selected model"""
        if self._loading_thread and self._loading_thread.is_alive():
            self.status_queue.put(("warning", "Already loading a model"))
            return

        self._loading_thread = threading.Thread(
            target=self._load_model_async, args=(db_path, model_name)
        )
        self._loading_thread.start()

    def _load_model_async(self, db_path: str, model_name: str):
        """Asynchronously load the model data"""
        try:
            conn = duckdb.connect(db_path)
            if not check_table_exists(conn, model_name):
                raise ValueError(f"Model table '{model_name}' not found")

            self.status_queue.put(("info", f"Loading model {model_name}..."))
            df = read_df_from_db(conn, model_name)
            conn.close()

            self.current_model = model_name
            self.model_data = df
            self.loading_queue.put(df)
            self.status_queue.put(
                ("success", f"Model {model_name} loaded successfully")
            )

        except Exception as e:
            self.status_queue.put(("error", f"Error loading model: {str(e)}"))
            self.loading_queue.put(None)

    def get_loading_status(self):
        """Get current loading status if available"""
        try:
            return self.status_queue.get_nowait()
        except:
            return None

    def get_loaded_data(self):
        """Get loaded data if available"""
        try:
            return self.loading_queue.get_nowait()
        except:
            return None
