import argparse
import json
import logging
import time
from threading import Lock

import dash
import dash_bootstrap_components as dbc
import duckdb
import flask
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import redis
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from tools.dbtools import (
    add_column_to_table,
    append_df_to_table,
    check_table_exists,
    read_df_from_db,
    store_df_in_db,
)
from specviz.tasks.measurement_queue import MeasurementQueue

# Suppress Flask logging
log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)


class AcquireData:
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379):
        # Store connection details
        self.redis_host = redis_host
        self.redis_port = redis_port

        # Initialize Redis client
        self.redis_client = redis.Redis(
            host=self.redis_host, port=self.redis_port, decode_responses=True
        )

        # Initialize queue with same connection details
        self.queue = MeasurementQueue(
            redis_host=self.redis_host, redis_port=self.redis_port
        )

        # Initialize data containers
        self.just_measured_indices = set()
        self.hcd_df = None
        self.measured_df = None

        # Add new data containers for spectral data
        self.wavenumber_df = None
        self.measured_data_df = None

        # Create app and layout
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

        self.redis_status = "Not Connected"
        self.redis_status_color = "danger"
        self._connect_redis()

        self.app.layout = dbc.Container(
            [
                html.H1("Data Acquisition", className="my-4"),
                # Load Data Button Row
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Button(
                                    "Load Current Data",
                                    id="load-data-button",
                                    color="primary",
                                    className="mb-3",
                                ),
                                html.Div(id="load-data-status", className="mb-3"),
                            ]
                        )
                    ]
                ),
                # Status Cards Row
                dbc.Row(
                    [
                        # Redis Status Card
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Redis Connection Status"),
                                        dbc.CardBody(
                                            [
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                html.Div(
                                                                    id="redis-status"
                                                                ),
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
                                                        ),
                                                    ]
                                                )
                                            ]
                                        ),
                                    ],
                                    className="mb-3",
                                )
                            ],
                            width=6,
                        ),
                        # Instrument Status Card
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Instrument Health Status"),
                                        dbc.CardBody(
                                            [
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                html.Div(
                                                                    id="instrument-status"
                                                                ),
                                                            ],
                                                            width=10,
                                                        ),
                                                        dbc.Col(
                                                            [
                                                                dbc.Button(
                                                                    "Refresh",
                                                                    id="instrument-refresh-button",
                                                                    color="light",
                                                                    size="sm",
                                                                    className="float-end",
                                                                )
                                                            ],
                                                            width=2,
                                                        ),
                                                    ]
                                                )
                                            ]
                                        ),
                                    ],
                                    className="mb-3",
                                )
                            ],
                            width=6,
                        ),
                    ]
                ),
                # Points Plot
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Measurement Status"),
                                        dbc.CardBody(
                                            [
                                                dcc.Graph(
                                                    id="points-plot",
                                                    style={"height": "600px"},
                                                ),
                                            ]
                                        ),
                                    ]
                                )
                            ]
                        )
                    ],
                    className="mb-3",
                ),
                # Data Acquisition Controls
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Data Acquisition Controls"),
                                        dbc.CardBody(
                                            [
                                                # Fixed height container for all controls
                                                html.Div(
                                                    style={"height": "250px"},
                                                    children=[
                                                        # Queue Controls Row
                                                        dbc.Row(
                                                            [
                                                                dbc.Col(
                                                                    [
                                                                        dbc.Button(
                                                                            "Queue Plan",
                                                                            id="queue-plan-button",
                                                                            color="primary",
                                                                            className="me-2",
                                                                        ),
                                                                        dbc.Button(
                                                                            "Clear Queue",
                                                                            id="clear-queue-button",
                                                                            color="danger",
                                                                            className="me-2",
                                                                        ),
                                                                    ],
                                                                    width=6,
                                                                ),
                                                                dbc.Col(
                                                                    [
                                                                        # Fixed height status container
                                                                        html.Div(
                                                                            id="queue-status",
                                                                            className="d-inline-block",
                                                                            style={
                                                                                "lineHeight": "38px",
                                                                                "height": "38px",
                                                                                "overflow": "hidden",
                                                                            },
                                                                        )
                                                                    ],
                                                                    width=6,
                                                                ),
                                                            ],
                                                            className="mb-3",
                                                        ),
                                                        # Collection Control Row
                                                        dbc.Row(
                                                            [
                                                                dbc.Col(
                                                                    [
                                                                        dbc.Button(
                                                                            "Start Collection",
                                                                            id="collection-control-button",
                                                                            color="success",
                                                                            size="lg",
                                                                            className="w-100 mb-2",
                                                                        ),
                                                                        dbc.Button(
                                                                            "Finalize Collection Iteration",
                                                                            id="finalize-button",
                                                                            color="info",
                                                                            size="lg",
                                                                            className="w-100",
                                                                        ),
                                                                        html.Div(
                                                                            id="finalize-status",
                                                                            className="mt-2",
                                                                        ),
                                                                    ],
                                                                    width=6,
                                                                ),
                                                                dbc.Col(
                                                                    [
                                                                        html.Div(
                                                                            id="collection-status",
                                                                            className="d-inline-block",
                                                                            style={
                                                                                "lineHeight": "48px",
                                                                                "height": "48px",
                                                                                "overflow": "hidden",
                                                                            },
                                                                        )
                                                                    ],
                                                                    width=6,
                                                                ),
                                                            ]
                                                        ),
                                                    ],
                                                )
                                            ],
                                            style={"padding": "1rem"},
                                        ),
                                    ],
                                    style={"height": "300px"},
                                )  # Increased card height from 200px to 300px
                            ]
                        )
                    ]
                ),
                # Update intervals
                dcc.Interval(
                    id="status-check", interval=1000
                ),  # Check status every second
                dcc.Interval(
                    id="plot-check", interval=100, n_intervals=0
                ),  # Update plot every 0.1 seconds
                dcc.Interval(
                    id="queue-check", interval=100, n_intervals=0
                ),  # Queue status update
                dcc.Store(
                    id="collection-state", data=False
                ),  # Store current button state
                dcc.Interval(
                    id="result-check", interval=100, n_intervals=0
                ),  # Check results every 0.1 seconds
            ],
            fluid=True,
        )

        self.df_lock = Lock()

        self._setup_callbacks()

    def _connect_redis(self):
        """Establish connection to Redis"""
        try:
            self.redis_client.ping()
            self.redis_status = f"Connected to {self.redis_client.connection_pool.connection_kwargs['host']}:{self.redis_client.connection_pool.connection_kwargs['port']}"
            self.redis_status_color = "success"
        except redis.ConnectionError as e:
            self.redis_status = (
                f"Connection Failed: Redis not running on {self.redis_client.connection_pool.connection_kwargs['host']}:{self.redis_client.connection_pool.connection_kwargs['port']}. "
                "Please ensure Redis server is started."
            )
            self.redis_status_color = "danger"
        except Exception as e:
            self.redis_status = f"Error: {str(e)}"
            self.redis_status_color = "danger"

    def serpentine_order(self, points: list, binwidth: float = 5.0) -> list:
        """
        Reorder points in a serpentine pattern to optimize motor movement.

        Args:
            points (list): List of dictionaries containing measurement points
                         Each dict should have 'X', 'Y', and 'hcd_indx' keys
            binwidth (float): Width of Y-coordinate bins for serpentine pattern

        Returns:
            list: Reordered points following serpentine pattern
        """
        if not points:
            return points

        # Extract Y coordinates and determine bin edges
        y_coords = [p["Y"] for p in points]
        y_min, y_max = min(y_coords), max(y_coords)

        # Create bins
        num_bins = int(np.ceil((y_max - y_min) / binwidth))
        bins = [[] for _ in range(num_bins)]

        # Assign points to bins based on Y coordinate
        for point in points:
            bin_index = int((point["Y"] - y_min) / binwidth)
            # Handle edge case where point is exactly at y_max
            bin_index = min(bin_index, num_bins - 1)
            bins[bin_index].append(point)

        # Sort each bin by X coordinate, reversing alternate bins
        ordered_points = []
        for i, bin_points in enumerate(bins):
            if bin_points:  # Only process non-empty bins
                # Sort by X coordinate
                sorted_points = sorted(bin_points, key=lambda p: p["X"])
                # Reverse alternate bins for serpentine pattern
                if i % 2 == 1:
                    sorted_points.reverse()
                ordered_points.extend(sorted_points)

        # Print summary of reordering
        print(f"\nReordered {len(points)} points into {num_bins} bins:")
        for i, bin_points in enumerate(bins):
            if bin_points:
                y_range = (y_min + i * binwidth, min(y_min + (i + 1) * binwidth, y_max))
                print(
                    f"Bin {i}: Y range [{y_range[0]:.2f}, {y_range[1]:.2f}], {len(bin_points)} points"
                )

        return ordered_points

    def _setup_callbacks(self):
        @self.app.callback(
            [Output("redis-status", "children"), Output("redis-status", "className")],
            [
                Input("status-check", "n_intervals"),
                Input("redis-refresh-button", "n_clicks"),
            ],
        )
        def update_redis_status(n_intervals, n_clicks):
            try:
                if self.redis_client.ping():
                    # Get current project and collection status
                    current_project = self.redis_client.get("current_project")
                    collection_status = self.redis_client.get("collect_data")

                    status_lines = [
                        f"Connected to {self.redis_client.connection_pool.connection_kwargs['host']}:{self.redis_client.connection_pool.connection_kwargs['port']}"
                    ]

                    if current_project:
                        status_lines.append(f"Current Project: {current_project}")
                    else:
                        status_lines.append("No active project")

                    collection_text = "Collection Status: "
                    if collection_status == "True":
                        collection_text += "Active"
                        collection_class = "text-success"
                    elif collection_status == "False":
                        collection_text += "Paused"
                        collection_class = "text-warning"
                    else:
                        collection_text += "Unknown"
                        collection_class = "text-danger"

                    return (
                        html.Div(
                            [
                                html.Div(status_lines[0]),  # Connection status
                                html.Small(
                                    status_lines[1], className="text-muted d-block"
                                ),  # Project info
                                html.Small(
                                    collection_text,
                                    className=f"{collection_class} d-block",
                                ),  # Collection status
                            ]
                        ),
                        "text-success",
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
                        html.Small("No active project", className="text-muted d-block"),
                        html.Small(
                            "Collection Status: Unknown",
                            className="text-danger d-block",
                        ),
                    ]
                ),
                f"text-{self.redis_status_color}",
            )

        @self.app.callback(
            [
                Output("instrument-status", "children"),
                Output("instrument-status", "className"),
            ],
            [
                Input("status-check", "n_intervals"),
                Input("instrument-refresh-button", "n_clicks"),
            ],
        )
        def update_instrument_status(n_intervals, n_clicks):
            try:
                if not self.redis_client:
                    return "Redis not connected", "text-warning"

                heartbeat = self.redis_client.get("instrument_heartbeat")

                if heartbeat is None:
                    return "Instrument status unknown", "text-warning"

                if heartbeat == "True":
                    return "Instrument online and healthy", "text-success"
                elif heartbeat == "False":
                    return "Instrument offline or error", "text-danger"
                else:
                    return f"Unknown status: {heartbeat}", "text-warning"

            except Exception as e:
                return f"Error checking instrument status: {str(e)}", "text-danger"

        @self.app.callback(
            [
                Output("collection-control-button", "children"),
                Output("collection-control-button", "color"),
                Output("collection-status", "children"),
                Output("collection-status", "className"),
                Output("collection-state", "data"),
            ],
            [
                Input("collection-control-button", "n_clicks"),
                Input("status-check", "n_intervals"),
            ],
            [State("collection-state", "data")],
        )
        def update_collection_control(n_clicks, _, current_state):
            if not self.redis_client:
                return (
                    "Start Collection",
                    "secondary",
                    "Redis Not Connected",
                    "text-danger mt-2",
                    False,
                )

            try:
                # Get current collection status from Redis
                redis_status = self.redis_client.get("collect_data")
                is_collecting = redis_status == "True"

                # If button was clicked, toggle state
                if (
                    dash.callback_context.triggered_id == "collection-control-button"
                    and n_clicks
                ):
                    is_collecting = not is_collecting
                    # Explicitly set string 'True' or 'False' in Redis
                    self.redis_client.set("collect_data", str(is_collecting))
                    print(f"Collection state toggled to: {is_collecting}")

                if is_collecting:
                    return (
                        "Pause Collection",
                        "danger",
                        "Collection Status: Active",
                        "text-success mt-2",
                        True,
                    )
                else:
                    return (
                        "Start Collection",
                        "success",
                        "Collection Status: Paused",
                        "text-warning mt-2",
                        False,
                    )

            except Exception as e:
                print(f"Collection control error: {str(e)}")
                return (
                    "Start Collection",
                    "secondary",
                    f"Error: {str(e)}",
                    "text-danger mt-2",
                    False,
                )

        @self.app.callback(
            [Output("queue-status", "children"), Output("points-plot", "figure")],
            [
                Input("queue-plan-button", "n_clicks"),
                Input("clear-queue-button", "n_clicks"),
                Input("queue-check", "n_intervals"),
                Input("result-check", "n_intervals"),
            ],
            prevent_initial_call=True,
        )
        def handle_queue_operations(
            queue_clicks, clear_clicks, queue_check, result_check
        ):
            triggered_id = dash.callback_context.triggered_id

            try:
                if triggered_id == "queue-plan-button" and queue_clicks:
                    # Get planned points
                    planned_mask = self.measured_df["planned_next"] == 1
                    planned_indices = self.measured_df[planned_mask]["hcd_indx"].values
                    print("\nPlanned indices to queue:", planned_indices.tolist())

                    # Create list of measurement points
                    points_to_queue = []
                    for idx in planned_indices:
                        row = self.hcd_df.loc[idx]
                        points_to_queue.append(
                            {
                                "hcd_indx": int(idx),
                                "X": float(row["X"]),
                                "Y": float(row["Y"]),
                            }
                        )

                    # Reorder points in serpentine pattern
                    ordered_points = self.serpentine_order(points_to_queue)

                    # Add ordered points to queue
                    points_added = 0
                    for point in ordered_points:
                        if self.queue.add_measurement(point):
                            points_added += 1

                    status = html.Div(
                        [
                            html.Div(
                                [
                                    html.Span(
                                        f"Queued: {points_added} points | ",
                                        className="text-success",
                                    ),
                                    html.Span(
                                        f"Queue Length: {self.queue.queue_length()}"
                                    ),
                                ],
                                style={"whiteSpace": "nowrap"},
                            )
                        ],
                        style={"height": "38px"},
                    )
                    return status, self.update_points_plot()

                elif triggered_id == "clear-queue-button" and clear_clicks:
                    # First ensure collection is stopped
                    self.redis_client.set("collect_data", "False")

                    # Wait briefly to ensure collection has stopped
                    time.sleep(0.1)

                    # Clear both queues
                    self.queue.clear_queue()
                    while self.redis_client.rpop("xo_actual"):  # Clear xo_actual queue
                        pass

                    status = html.Div(
                        [
                            html.Div(
                                [
                                    html.Span(
                                        "Queues Cleared | ", className="text-warning"
                                    ),
                                    html.Span("Queue Length: 0"),
                                ],
                                style={"whiteSpace": "nowrap"},
                            )
                        ],
                        style={"height": "38px"},
                    )
                    return status, self.update_points_plot()

                elif triggered_id in ["queue-check", "result-check"]:
                    queue_length = self.queue.queue_length()
                    xo_actual_length = self.redis_client.llen("xo_actual")

                    # Process xo_actual queue aggressively
                    if xo_actual_length > 0:
                        print(
                            f"\nProcessing {xo_actual_length} results from xo_actual queue"
                        )
                        while True:
                            result_json = self.redis_client.rpop("xo_actual")
                            if not result_json:
                                break

                            try:
                                result = json.loads(result_json)
                                idx = result["hcd_indx"]
                                print(
                                    f"\nProcessing result for point {idx} at ({result.get('X', '?')}, {result.get('Y', '?')})"
                                )

                                # Handle spectral data
                                if "wavenumbers" in result and "spectrum" in result:
                                    if (
                                        not result["wavenumbers"]
                                        or not result["spectrum"]
                                    ):
                                        print(
                                            f"Warning: Empty data arrays for point {idx}"
                                        )
                                        continue

                                    if len(result["wavenumbers"]) != len(
                                        result["spectrum"]
                                    ):
                                        print(
                                            f"Warning: Length mismatch for point {idx} - wavenumbers: {len(result['wavenumbers'])}, spectrum: {len(result['spectrum'])}"
                                        )
                                        continue

                                    # Add debug info about the data we're about to store
                                    print(f"Point {idx} data validation:")
                                    print(
                                        f"  Spectrum length: {len(result['spectrum'])}"
                                    )
                                    print(
                                        f"  First few values: {result['spectrum'][:5]}"
                                    )

                                    # Initialize wavenumber DataFrame if not exists
                                    if self.wavenumber_df is None:
                                        self.wavenumber_df = pd.DataFrame(
                                            {"wavenumber": result["wavenumbers"]}
                                        )
                                        print(
                                            f"Created wavenumber DataFrame with {len(self.wavenumber_df)} points"
                                        )

                                    # Add spectrum data with thread-safe operations
                                    intensity_columns = [
                                        f"intensity_{i}"
                                        for i in range(len(result["spectrum"]))
                                    ]
                                    new_row = pd.DataFrame(
                                        {
                                            "hcd_indx": [int(idx)],
                                            **{
                                                col: [val]
                                                for col, val in zip(
                                                    intensity_columns,
                                                    result["spectrum"],
                                                )
                                            },
                                        }
                                    )

                                    print(
                                        f"  Created new row for point {idx} with {len(new_row.columns)} columns"
                                    )
                                    print(
                                        f"  New row hcd_indx value: {new_row['hcd_indx'].values[0]}"
                                    )

                                    # Use a lock to prevent concurrent DataFrame modifications
                                    with self.df_lock:
                                        if self.measured_data_df is None:
                                            self.measured_data_df = new_row
                                            print(
                                                f"  Initialized measured_data_df with point {idx}"
                                            )
                                        else:
                                            # Check if point already exists
                                            exists = (
                                                idx
                                                in self.measured_data_df[
                                                    "hcd_indx"
                                                ].values
                                            )
                                            if exists:
                                                print(
                                                    f"  Warning: Point {idx} already exists in measured_data_df"
                                                )
                                                # Update existing row
                                                mask = (
                                                    self.measured_data_df["hcd_indx"]
                                                    == idx
                                                )
                                                self.measured_data_df.loc[mask] = (
                                                    new_row.iloc[0]
                                                )
                                                print(
                                                    f"  Updated existing row for point {idx}"
                                                )
                                            else:
                                                # Make a copy and add the new row
                                                self.measured_data_df = pd.concat(
                                                    [self.measured_data_df, new_row],
                                                    ignore_index=True,
                                                )

                                                # Verify the addition
                                                if (
                                                    idx
                                                    in self.measured_data_df[
                                                        "hcd_indx"
                                                    ].values
                                                ):
                                                    print(
                                                        f"  Added point {idx} to measured_data_df (now {len(self.measured_data_df)} rows)"
                                                    )
                                                else:
                                                    print(
                                                        f"  ERROR: Failed to add point {idx} to measured_data_df"
                                                    )
                                                    # Try one more time
                                                    self.measured_data_df = pd.concat(
                                                        [
                                                            self.measured_data_df,
                                                            new_row,
                                                        ],
                                                        ignore_index=True,
                                                    )
                                                    print(
                                                        f"  Recovery attempt - point now present: {idx in self.measured_data_df['hcd_indx'].values}"
                                                    )

                                    # Final verification
                                    if (
                                        idx
                                        not in self.measured_data_df["hcd_indx"].values
                                    ):
                                        print(
                                            f"  CRITICAL: Point {idx} missing after all attempts!"
                                        )
                                        print(
                                            f"  Current indices: {sorted(list(self.measured_data_df['hcd_indx'].values))}"
                                        )

                                    # Update tracking after successful addition
                                    self.just_measured_indices.add(idx)
                                    print(f"  Updated tracking for point {idx}")
                                else:
                                    print(
                                        f"Warning: Missing spectral data for point {idx}"
                                    )
                                    continue
                            except Exception as e:
                                print(
                                    f"Error processing result for point {idx}: {str(e)}"
                                )

                    status = html.Div(
                        [
                            html.Div(
                                [
                                    html.Span(
                                        f"Queue Length: {queue_length}",
                                        className="text-success",
                                    ),
                                ],
                                style={"whiteSpace": "nowrap"},
                            )
                        ],
                        style={"height": "38px"},
                    )
                    return status, self.update_points_plot()

            except Exception as e:
                print(f"Queue operation error: {str(e)}")
                return dash.no_update, dash.no_update

            return dash.no_update, dash.no_update

        @self.app.callback(
            [
                Output("load-data-status", "children"),
                Output("load-data-status", "className"),
            ],
            [Input("load-data-button", "n_clicks")],
            prevent_initial_call=True,
        )
        def handle_load_data(n_clicks):
            if not n_clicks:
                return "", ""

            message, status = self.load_data()
            return message, f"text-{status}"

        @self.app.callback(
            [
                Output("finalize-button", "disabled"),
                Output("finalize-button", "color"),
                Output("finalize-status", "children"),
            ],
            [
                Input("finalize-button", "n_clicks"),
                Input("result-check", "n_intervals"),
            ],
            prevent_initial_call=True,
        )
        def handle_finalize(n_clicks, n_intervals):
            try:
                # Check if collection is running
                collection_status = self.redis_client.get("collect_data")
                if collection_status == "True":
                    return (
                        True,
                        "secondary",
                        html.Div(
                            "Cannot finalize while collection is running",
                            className="text-warning",
                        ),
                    )

                # Count points in just_measured_indices
                n_points = len(self.just_measured_indices)

                # If no button click, just update state
                if not n_clicks:
                    if n_points == 0:
                        return True, "secondary", ""
                    else:
                        return (
                            False,
                            "info",
                            html.Div(
                                f"{n_points} points ready to finalize",
                                className="text-info",
                            ),
                        )

                # Handle button click
                ctx = dash.callback_context
                if ctx.triggered[0]["prop_id"].split(".")[0] == "finalize-button":
                    # Verify we have points to finalize
                    if n_points == 0:
                        return (
                            True,
                            "secondary",
                            html.Div("No points to finalize", className="text-warning"),
                        )

                    # Get current project path
                    db_path = self.redis_client.get("current_project")
                    if not db_path:
                        return (
                            True,
                            "secondary",
                            html.Div(
                                "No active project found", className="text-danger"
                            ),
                        )

                    # Connect to database
                    conn = duckdb.connect(db_path)

                    try:
                        # Add debug information
                        print(f"\nFinalizing {n_points} points")
                        print(
                            f"Points to finalize: {sorted(list(self.just_measured_indices))}"
                        )

                        if self.measured_data_df is not None:
                            measured_indices = set(
                                self.measured_data_df["hcd_indx"].unique()
                            )
                            missing_indices = (
                                self.just_measured_indices - measured_indices
                            )
                            if missing_indices:
                                print("\nDETAILED MISSING POINTS ANALYSIS:")
                                print(
                                    f"Total points marked as measured: {len(self.just_measured_indices)}"
                                )
                                print(
                                    f"Points with spectral data: {len(measured_indices)}"
                                )
                                print(
                                    f"Number of missing points: {len(missing_indices)}"
                                )
                                print(
                                    f"Missing points (hcd_indx): {sorted(list(missing_indices))}"
                                )

                                # Print coordinates for missing points
                                print("\nCoordinates of missing points:")
                                for idx in sorted(list(missing_indices)):
                                    if idx in self.hcd_df.index:
                                        point = self.hcd_df.loc[idx]
                                        print(
                                            f"hcd_indx: {idx}, X: {point['X']:.2f}, Y: {point['Y']:.2f}"
                                        )
                                    else:
                                        print(
                                            f"hcd_indx: {idx} - Not found in HCD dataframe"
                                        )
                        else:
                            print(
                                "Warning: No spectral data available in measured_data_df"
                            )

                        # Store wavenumber data if it doesn't exist
                        if self.wavenumber_df is not None:
                            if not check_table_exists(conn, "wavenumbers"):
                                if len(self.wavenumber_df) == 0:
                                    raise ValueError("Empty wavenumber data")
                                store_df_in_db(
                                    conn,
                                    self.wavenumber_df,
                                    "wavenumbers",
                                    if_exists="fail",
                                    index=False,
                                )
                                print("Stored wavenumber data")

                        # Store measured spectral data
                        if self.measured_data_df is not None:
                            if self.wavenumber_df is not None:
                                expected_cols = (
                                    len(self.wavenumber_df) + 1
                                )  # +1 for hcd_indx
                                actual_cols = len(self.measured_data_df.columns)
                                print(
                                    f"Spectral data columns: expected={expected_cols}, actual={actual_cols}"
                                )
                                if actual_cols != expected_cols:
                                    raise ValueError(
                                        f"Column count mismatch: expected {expected_cols}, got {actual_cols}"
                                    )

                            # Verify all points have the same number of intensity values
                            intensity_cols = [
                                col
                                for col in self.measured_data_df.columns
                                if col.startswith("intensity_")
                            ]
                            print(f"Number of intensity columns: {len(intensity_cols)}")

                            append_df_to_table(
                                conn, self.measured_data_df, "measured_data"
                            )
                            print(
                                f"Stored {len(self.measured_data_df)} spectral measurements"
                            )
                            self.measured_data_df = None

                        # Handle iteration columns
                        measured_df = read_df_from_db(conn, "measured_points")
                        iteration_cols = [
                            col
                            for col in measured_df.columns
                            if col.startswith("iteration_")
                        ]
                        next_iter = 1 + (
                            max([int(col.split("_")[1]) for col in iteration_cols])
                            if iteration_cols
                            else 0
                        )

                        # Create new iteration column
                        new_col_name = f"iteration_{next_iter}"
                        new_data = pd.Series(0, index=measured_df.index)
                        new_data[
                            measured_df["hcd_indx"].isin(self.just_measured_indices)
                        ] = 1

                        # Add new column and clear planned_next
                        add_column_to_table(
                            conn, "measured_points", new_col_name, new_data
                        )
                        add_column_to_table(
                            conn,
                            "measured_points",
                            "planned_next",
                            pd.Series(0, index=measured_df.index),
                            overwrite=True,
                        )

                        # Clear just_measured set and reload measured_df
                        self.just_measured_indices.clear()
                        self.measured_df = read_df_from_db(conn, "measured_points")

                        conn.close()
                        return (
                            True,
                            "secondary",
                            html.Div(
                                f"Successfully finalized iteration {next_iter} with {n_points} points",
                                className="text-success",
                            ),
                        )

                    except Exception as e:
                        if "conn" in locals():
                            conn.close()
                        return (
                            True,
                            "secondary",
                            html.Div(f"Error: {str(e)}", className="text-danger"),
                        )

                # Default state based on available points
                if n_points == 0:
                    return True, "secondary", ""
                else:
                    return (
                        False,
                        "info",
                        html.Div(
                            f"{n_points} points ready to finalize",
                            className="text-info",
                        ),
                    )

            except Exception as e:
                print(f"Finalize error: {str(e)}")
                return (
                    True,
                    "secondary",
                    html.Div(f"Error: {str(e)}", className="text-danger"),
                )

    def load_data(self):
        """Load data into memory"""
        try:
            db_path = self.redis_client.get("current_project")
            if not db_path:
                return "No active project found in Redis", "danger"

            conn = duckdb.connect(db_path)

            if not (
                check_table_exists(conn, "HCD")
                and check_table_exists(conn, "measured_points")
            ):
                return f"Required tables not found in {db_path}", "danger"

            self.hcd_df = read_df_from_db(conn, "HCD")
            self.measured_df = read_df_from_db(conn, "measured_points")

            # Verify required columns
            required_hcd_cols = ["X", "Y"]
            required_measured_cols = ["hcd_indx", "planned_next"]

            missing_hcd = [
                col for col in required_hcd_cols if col not in self.hcd_df.columns
            ]
            missing_measured = [
                col
                for col in required_measured_cols
                if col not in self.measured_df.columns
            ]

            if missing_hcd or missing_measured:
                return (
                    f"Missing columns - HCD: {missing_hcd}, measured: {missing_measured}",
                    "danger",
                )

            # Clear just measured points on data load
            self.just_measured_indices.clear()

            conn.close()
            return (
                f"Successfully loaded data: HCD shape={self.hcd_df.shape}, measured shape={self.measured_df.shape}",
                "success",
            )

        except Exception as e:
            return f"Error loading data: {str(e)}", "danger"

    def update_points_plot(self):
        """Generate plot with measured, planned, and just-measured points"""
        try:
            if self.hcd_df is None or self.measured_df is None:
                return go.Figure()

            fig = go.Figure()

            # Get previously measured points (excluding just measured)
            iteration_cols = [
                col
                for col in self.measured_df.columns
                if col.startswith("iteration_") or col == "prior_measurements"
            ]
            measured_mask = self.measured_df[iteration_cols].eq(1).any(axis=1)
            measured_indices = set(self.measured_df[measured_mask]["hcd_indx"].values)
            measured_indices = measured_indices - self.just_measured_indices

            # Add previously measured points (red)
            if measured_indices:
                measured_points = self.hcd_df.loc[
                    self.hcd_df.index.isin(measured_indices)
                ]
                fig.add_trace(
                    go.Scatter(
                        x=measured_points["X"],
                        y=measured_points["Y"],
                        mode="markers",
                        name="Previously Measured",
                        marker=dict(color="red", size=8),
                    )
                )

            # Add just measured points (blue)
            if self.just_measured_indices:
                just_measured_points = self.hcd_df.loc[
                    self.hcd_df.index.isin(self.just_measured_indices)
                ]
                fig.add_trace(
                    go.Scatter(
                        x=just_measured_points["X"],
                        y=just_measured_points["Y"],
                        mode="markers",
                        name="Just Measured",
                        marker=dict(color="#0000FF", size=10),
                    )
                )

            # Add planned points (green)
            planned_mask = self.measured_df["planned_next"] == 1
            planned_indices = self.measured_df[planned_mask]["hcd_indx"].values
            if len(planned_indices) > 0:
                planned_points = self.hcd_df.loc[
                    self.hcd_df.index.isin(planned_indices)
                ]
                fig.add_trace(
                    go.Scatter(
                        x=planned_points["X"],
                        y=planned_points["Y"],
                        mode="markers",
                        name="Planned",
                        marker=dict(color="green", size=8),
                    )
                )

            # Update layout with fixed aspect ratio
            fig.update_layout(
                showlegend=True,
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                margin=dict(l=20, r=20, t=20, b=20),
                plot_bgcolor="white",
                xaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor="LightGray",
                    zeroline=True,
                    zerolinewidth=1,
                    zerolinecolor="LightGray",
                    scaleanchor="y",
                    scaleratio=1,
                ),
                yaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor="LightGray",
                    zeroline=True,
                    zerolinewidth=1,
                    zerolinecolor="LightGray",
                ),
            )

            return fig

        except Exception as e:
            print(f"Error updating plot: {str(e)}")
            return go.Figure()

    def run(self, debug=False, port=8055):
        self.app.run_server(debug=debug, port=port)

    def update_collection_status(self):
        """Update collection status display"""
        try:
            if not self.redis_client:
                return "Redis not connected"

            # Get connection details from Redis client
            redis_host = self.redis_client.connection_pool.connection_kwargs["host"]
            redis_port = self.redis_client.connection_pool.connection_kwargs["port"]
            status = f"Connected to Redis at {redis_host}:{redis_port}"

            is_active = self.redis_client.get("collection_status")
            if is_active == "True":
                status += " | Collection Active"
            elif is_active == "False":
                status += " | Collection Inactive"
            else:
                status += " | Collection Status Unknown"

            return status

        except Exception as e:
            print(f"Error updating collection status: {str(e)}")
            return "Error checking collection status"


def main():
    parser = argparse.ArgumentParser(
        description="Start the Data Acquisition application"
    )
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
        default=8055,
        help="Port to run the Dash app on (default: 8055)",
    )
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")

    args = parser.parse_args()

    app = AcquireData(redis_host=args.redis_host, redis_port=args.redis_port)
    app.run(debug=args.debug, port=args.port)


if __name__ == "__main__":
    main()
