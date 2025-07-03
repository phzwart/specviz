from typing import Callable, Protocol

import argparse
import math
import os
import threading
from pathlib import Path
from queue import Queue

import dash
import dash_bootstrap_components as dbc
import numpy as np
import redis
from dash import dcc, html
from dash.dependencies import Input, Output, State

from specviz.tasks.database_constructor import DuckDBConstructor
from specviz.tasks.sample_constructor import MySamplingGenerator


class DatabaseConstructor(Protocol):
    def create_database(self, config: dict) -> bool:
        """Create database with given configuration
        Args:
            config: Dictionary with project_id, description, and spatial bounds
        Returns:
            bool: Success status
        """
        ...


class SamplingGenerator(Protocol):
    def generate_sampling(self, config: dict) -> bool:
        """Generate sampling scheme with given configuration
        Args:
            config: Dictionary with sampling parameters
        Returns:
            bool: Success status
        """
        ...


def run_in_thread(func: Callable, args: tuple, result_queue: Queue):
    """Run function in thread and put result in queue"""
    result = func(*args)
    result_queue.put(result)


class SetupSession:
    def __init__(
        self,
        database_constructor: DatabaseConstructor,
        sampling_generator: SamplingGenerator,
        redis_host: str = "localhost",
        redis_port: int = 6379,
    ):
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.database_constructor = database_constructor
        self.sampling_generator = sampling_generator

        # Redis connection setup
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_client = None
        self.redis_status = "Not Connected"
        self.redis_status_color = "danger"
        self._connect_redis()

        # Inject Redis client into constructors
        if hasattr(self.database_constructor, "set_redis_client"):
            self.database_constructor.set_redis_client(self.redis_client)
        if hasattr(self.sampling_generator, "set_redis_client"):
            self.sampling_generator.set_redis_client(self.redis_client)

        self.app.layout = dbc.Container(
            [
                html.H1("Data Collection Session Setup", className="my-4"),
                # Redis Status Alert with Refresh Button
                dbc.Alert(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.H6(
                                            "Redis Connection Status:", className="mb-0"
                                        ),
                                        html.Div(id="redis-status"),
                                    ],
                                    width=10,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Button(
                                            "Refresh",
                                            id="redis-refresh-button",
                                            color="light",
                                            size="sm",
                                            className="float-end",
                                        )
                                    ],
                                    width=2,
                                    className="d-flex align-items-start",
                                ),
                            ]
                        )
                    ],
                    id="redis-alert",
                    color="danger",
                    className="mb-3",
                ),
                # Sample Definition Card
                dbc.Card(
                    [
                        dbc.CardHeader("Sample Definition"),
                        dbc.CardBody(
                            [
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.Label("Data Directory:"),
                                                dbc.Input(
                                                    id="data-path",
                                                    type="text",
                                                    placeholder="Enter path to data directory",
                                                    value=str(
                                                        Path.home()
                                                    ),  # Default to user's home directory
                                                    className="mb-1",
                                                ),
                                                html.Small(
                                                    id="path-validation",
                                                    className="text-muted",
                                                ),
                                            ]
                                        ),
                                    ],
                                    className="mb-3",
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.Label("Project ID:"),
                                                dbc.Input(
                                                    id="project-id",
                                                    type="text",
                                                    placeholder="Enter project ID",
                                                    className="mb-3",
                                                ),
                                            ],
                                            width=6,
                                        ),
                                    ]
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.Label("Sample Description:"),
                                                dbc.Textarea(
                                                    id="sample-description",
                                                    placeholder="Enter sample description",
                                                    className="mb-3",
                                                    style={"height": "100px"},
                                                ),
                                            ]
                                        ),
                                    ]
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                dbc.Button(
                                                    "Generate Database",
                                                    id="generate-database-button",
                                                    color="primary",
                                                    className="w-100",
                                                ),
                                            ],
                                            width=6,
                                        ),
                                    ]
                                ),
                            ]
                        ),
                    ],
                    className="mb-4",
                ),
                # Sampling Scheme Card
                dbc.Card(
                    [
                        dbc.CardHeader("Sampling Scheme"),
                        dbc.CardBody(
                            [
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.Label("X Range:"),
                                                dbc.InputGroup(
                                                    [
                                                        dbc.Input(
                                                            id="x-min",
                                                            type="number",
                                                            value=0,
                                                            placeholder="X min",
                                                        ),
                                                        dbc.Input(
                                                            id="x-max",
                                                            type="number",
                                                            value=128,
                                                            placeholder="X max",
                                                        ),
                                                    ],
                                                    className="mb-3",
                                                ),
                                            ],
                                            width=6,
                                        ),
                                        dbc.Col(
                                            [
                                                html.Label("Y Range:"),
                                                dbc.InputGroup(
                                                    [
                                                        dbc.Input(
                                                            id="y-min",
                                                            type="number",
                                                            value=0,
                                                            placeholder="Y min",
                                                        ),
                                                        dbc.Input(
                                                            id="y-max",
                                                            type="number",
                                                            value=128,
                                                            placeholder="Y max",
                                                        ),
                                                    ],
                                                    className="mb-3",
                                                ),
                                            ],
                                            width=6,
                                        ),
                                    ]
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.Label("Initial Distance:"),
                                                dbc.Input(
                                                    id="initial-distance",
                                                    type="number",
                                                    value=10,
                                                    className="mb-3",
                                                ),
                                            ],
                                            width=4,
                                        ),
                                        dbc.Col(
                                            [
                                                html.Label("Minimum Distance:"),
                                                dbc.Input(
                                                    id="minimum-distance",
                                                    type="number",
                                                    value=2,
                                                    className="mb-3",
                                                ),
                                            ],
                                            width=4,
                                        ),
                                        dbc.Col(
                                            [
                                                html.Label("Scale:"),
                                                dbc.Input(
                                                    id="scale",
                                                    type="number",
                                                    value=math.sqrt(2),
                                                    className="mb-3",
                                                ),
                                            ],
                                            width=4,
                                        ),
                                    ]
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.Label("Tile Size:"),
                                                dbc.Input(
                                                    id="tile-size",
                                                    type="number",
                                                    value=32,
                                                    className="mb-3",
                                                ),
                                            ],
                                            width=4,
                                        ),
                                    ]
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                dbc.Button(
                                                    "Generate Sampling",
                                                    id="generate-sampling-button",
                                                    color="primary",
                                                    className="w-100",
                                                ),
                                            ],
                                            width=6,
                                        ),
                                    ]
                                ),
                            ]
                        ),
                    ]
                ),
                # Status stores
                dcc.Store(id="database-status", data="idle"),
                dcc.Store(id="sampling-status", data="idle"),
                dcc.Interval(id="status-check", interval=500),
                # Add Redis status check interval
                dcc.Interval(id="redis-check", interval=5000),  # Check every 5 seconds
            ],
            fluid=True,
        )

        self._setup_callbacks()

    def _connect_redis(self):
        """Attempt to connect to Redis"""
        try:
            self.redis_client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                socket_connect_timeout=1,
                decode_responses=True,
            )
            self.redis_client.ping()  # Test connection
            self.redis_status = f"Connected to {self.redis_host}:{self.redis_port}"
            self.redis_status_color = "success"

            # Update Redis client in constructors after reconnection
            if hasattr(self.database_constructor, "set_redis_client"):
                self.database_constructor.set_redis_client(self.redis_client)
            if hasattr(self.sampling_generator, "set_redis_client"):
                self.sampling_generator.set_redis_client(self.redis_client)

        except redis.ConnectionError as e:
            self.redis_status = (
                f"Connection Failed: Redis not running on {self.redis_host}:{self.redis_port}. "
                "Please ensure Redis server is started."
            )
            self.redis_status_color = "danger"
            self.redis_client = None

            # Clear Redis client in constructors on connection failure
            if hasattr(self.database_constructor, "set_redis_client"):
                self.database_constructor.set_redis_client(None)
            if hasattr(self.sampling_generator, "set_redis_client"):
                self.sampling_generator.set_redis_client(None)

        except Exception as e:
            self.redis_status = f"Error: {str(e)}"
            self.redis_status_color = "danger"
            self.redis_client = None

            # Clear Redis client in constructors on error
            if hasattr(self.database_constructor, "set_redis_client"):
                self.database_constructor.set_redis_client(None)
            if hasattr(self.sampling_generator, "set_redis_client"):
                self.sampling_generator.set_redis_client(None)

    def _setup_callbacks(self):
        # Update Redis status callback to include the refresh button
        @self.app.callback(
            [Output("redis-status", "children"), Output("redis-alert", "color")],
            [
                Input("redis-check", "n_intervals"),
                Input("redis-refresh-button", "n_clicks"),
            ],  # Add refresh button trigger
        )
        def update_redis_status(n_intervals, n_clicks):
            try:
                if self.redis_client and self.redis_client.ping():
                    # Try to get current project
                    current_project = self.redis_client.get("current_project")
                    status_text = f"Connected to {self.redis_host}:{self.redis_port}"
                    if current_project:
                        status_text += f"\nCurrent Project: {current_project}"
                    else:
                        status_text += "\nNo active project"

                    return (
                        html.Div(
                            [
                                html.Div(
                                    status_text.split("\n")[0]
                                ),  # Connection status
                                html.Small(
                                    status_text.split("\n")[1], className="text-muted"
                                ),  # Project info
                            ]
                        ),
                        "success",
                    )

            except (redis.ConnectionError, redis.TimeoutError):
                self._connect_redis()  # Try to reconnect
            except Exception as e:
                self.redis_status = f"Unexpected Error: {str(e)}"
                self.redis_status_color = "danger"

            # If any error occurred, return the stored status
            return (
                html.Div(
                    [
                        html.Div(self.redis_status),
                        html.Small("No active project", className="text-muted"),
                    ]
                ),
                self.redis_status_color,
            )

        @self.app.callback(
            [
                Output("path-validation", "children"),
                Output("path-validation", "className"),
                Output("generate-database-button", "disabled"),
                Output("generate-database-button", "color"),
                Output("database-status", "data"),
            ],
            [
                Input("data-path", "value"),
                Input("generate-database-button", "n_clicks"),
                Input("status-check", "n_intervals"),
            ],
            [
                State("project-id", "value"),
                State("sample-description", "value"),
                State("database-status", "data"),
            ],
        )
        def handle_path_and_database(
            path, n_clicks, n_intervals, project_id, description, status
        ):
            ctx = dash.callback_context
            trigger = ctx.triggered[0]["prop_id"].split(".")[0]

            # Handle path validation
            if trigger == "data-path":
                if not path:
                    return (
                        "Please enter a path",
                        "text-danger",
                        True,
                        dash.no_update,
                        dash.no_update,
                    )

                try:
                    path = Path(path).resolve()
                    if path.exists() and not path.is_file():
                        return (
                            f"Resolved path: {path}",
                            "text-muted",
                            False,
                            dash.no_update,
                            dash.no_update,
                        )
                    elif path.is_file():
                        return (
                            "Path must be a directory, not a file",
                            "text-danger",
                            True,
                            dash.no_update,
                            dash.no_update,
                        )
                    else:
                        return (
                            f"Directory doesn't exist. Will be created: {path}",
                            "text-warning",
                            False,
                            dash.no_update,
                            dash.no_update,
                        )
                except Exception as e:
                    return (
                        f"Invalid path: {str(e)}",
                        "text-danger",
                        True,
                        dash.no_update,
                        dash.no_update,
                    )

            # Handle database generation
            elif trigger == "generate-database-button" and n_clicks:
                if not all([project_id, path]):
                    return (
                        "Missing required fields",
                        "text-danger",
                        True,
                        "danger",
                        "idle",
                    )

                try:
                    path = Path(path).resolve()
                    path.mkdir(parents=True, exist_ok=True)

                    config = {
                        "project_id": project_id,
                        "description": description or "",
                        "data_path": str(path),
                        "redis_client": self.redis_client,  # Pass Redis client to constructor
                    }

                    self.db_queue = Queue()
                    thread = threading.Thread(
                        target=run_in_thread,
                        args=(
                            self.database_constructor.create_database,
                            (config,),
                            self.db_queue,
                        ),
                    )
                    thread.start()
                    return (
                        f"Creating database in: {path}",
                        "text-warning",
                        True,
                        "warning",
                        "running",
                    )

                except Exception as e:
                    print(f"Error creating directory: {str(e)}")
                    return f"Error: {str(e)}", "text-danger", True, "danger", "idle"

            # Handle status check
            elif trigger == "status-check" and status == "running":
                if hasattr(self, "db_queue") and not self.db_queue.empty():
                    success = self.db_queue.get()
                    path = Path(path).resolve()
                    if success:
                        return (
                            f"Database created in: {path}",
                            "text-success",
                            False,
                            "success",
                            "idle",
                        )
                    else:
                        return (
                            f"Failed to create database in: {path}",
                            "text-danger",
                            True,
                            "danger",
                            "idle",
                        )

                return (
                    "Creating database...",
                    "text-warning",
                    True,
                    "warning",
                    "running",
                )

            # Default case
            return (
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
            )

        @self.app.callback(
            [
                Output("generate-sampling-button", "color"),
                Output("sampling-status", "data"),
            ],
            [
                Input("generate-sampling-button", "n_clicks"),
                Input("status-check", "n_intervals"),
            ],
            [
                State("initial-distance", "value"),
                State("minimum-distance", "value"),
                State("scale", "value"),
                State("tile-size", "value"),
                State("x-min", "value"),
                State("x-max", "value"),
                State("y-min", "value"),
                State("y-max", "value"),
                State("sampling-status", "data"),
            ],
        )
        def handle_sampling_generation(
            n_clicks,
            n_intervals,
            initial_distance,
            minimum_distance,
            scale,
            tile_size,
            x_min,
            x_max,
            y_min,
            y_max,
            status,
        ):
            ctx = dash.callback_context
            trigger = ctx.triggered[0]["prop_id"].split(".")[0]

            if trigger == "generate-sampling-button" and n_clicks:
                if not all(
                    [
                        initial_distance,
                        minimum_distance,
                        scale,
                        tile_size,
                        x_min is not None,
                        x_max is not None,
                        y_min is not None,
                        y_max is not None,
                    ]
                ):
                    return "danger", "idle"

                try:
                    # Get current database path from Redis
                    if not self.redis_client:
                        raise ValueError("Redis not connected")

                    current_db = self.redis_client.get("current_project")
                    if not current_db:
                        raise ValueError("No active project selected")

                    # Set the database connection in the sampling generator
                    self.sampling_generator.set_conn(current_db)

                    config = {
                        "initial_distance": initial_distance,
                        "minimum_distance": minimum_distance,
                        "scale": scale,
                        "tile_size": tile_size,
                        "x_min": x_min,
                        "x_max": x_max,
                        "y_min": y_min,
                        "y_max": y_max,
                    }

                    self.sampling_queue = Queue()
                    thread = threading.Thread(
                        target=run_in_thread,
                        args=(
                            self.sampling_generator.generate_sampling,
                            (config,),
                            self.sampling_queue,
                        ),
                    )
                    thread.start()
                    return "warning", "running"

                except Exception as e:
                    print(f"Error setting up sampling: {str(e)}")
                    return "danger", "idle"

            elif trigger == "status-check" and status == "running":
                if hasattr(self, "sampling_queue") and not self.sampling_queue.empty():
                    success = self.sampling_queue.get()
                    return "success" if success else "danger", "idle"
                return "warning", "running"

            return dash.no_update, status

    def run(self, debug=False, port=8050):
        # Suppress Flask logging if not in debug mode
        if not debug:
            import logging

            log = logging.getLogger("werkzeug")
            log.setLevel(logging.ERROR)

        self.app.run_server(debug=debug, port=port)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Start the Setup Session application")
    parser.add_argument(
        "--redis-host",
        default="127.0.0.1",
        help="Redis host address (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--redis-port", type=int, default=6379, help="Redis port (default: 6379)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8051,
        help="Port to run the Dash app on (default: 8051)",
    )
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")

    args = parser.parse_args()

    # Initialize components
    constructor = DuckDBConstructor()
    generator = MySamplingGenerator()

    # Create and run app
    app = SetupSession(
        constructor, generator, redis_host=args.redis_host, redis_port=args.redis_port
    )

    app.run(debug=args.debug, port=args.port)


if __name__ == "__main__":
    main()
