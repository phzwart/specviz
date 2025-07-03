import multiprocessing
import os
import tempfile
from multiprocessing import Pipe, Process
from time import sleep

import dash
import dash_bootstrap_components as dbc
import duckdb
import numpy as np
import pandas as pd
import redis
from consir.workflow.dbtools import (
    add_column_to_table,
    append_df_to_table,
    check_table_exists,
    read_df_from_db,
    store_df_in_db,
)
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from umap import UMAP


def create_wavenumber_mask(wavenumbers, range_string):
    """Create a boolean mask for wavenumber selection based on range string.

    Args:
        wavenumbers (np.ndarray): Array of wavenumbers
        range_string (str): Comma-separated ranges (e.g., "200-400,500-900")

    Returns:
        np.ndarray: Boolean mask for wavenumber selection
    """
    if not range_string.strip():  # If empty string, select all wavenumbers
        return np.ones_like(wavenumbers, dtype=bool)

    # Initialize mask as all False
    final_mask = np.zeros_like(wavenumbers, dtype=bool)

    # Split string into individual ranges
    ranges = range_string.split(",")

    for range_str in ranges:
        try:
            # Split range into start and stop values
            start, stop = map(float, range_str.strip().split("-"))

            # Ensure start < stop
            if start > stop:
                start, stop = stop, start

            # Create mask for this range and combine with OR
            range_mask = (wavenumbers >= start) & (wavenumbers <= stop)
            final_mask = final_mask | range_mask

            print(f"Range {start}-{stop}: Selected {range_mask.sum()} points")

        except ValueError as e:
            print(f"Error parsing range '{range_str}': {str(e)}")
            continue

    print(f"Total points selected: {final_mask.sum()}")
    return final_mask


def umap_runner(
    data, wavenumbers, xy_coords, params, pid, redis_host="localhost", redis_port=6379
):
    """Run UMAP computation and store results in Redis.

    Args:
        data: Input spectral data
        wavenumbers: Wavenumber array
        xy_coords: Spatial coordinates
        params: UMAP parameters dictionary
        pid: Process ID for Redis key generation
        redis_host: Redis host address
        redis_port: Redis port number

    Returns:
        bool: True if successful, False if error occurred
    """
    try:
        print("===>>>>>>  UMAP PROCESS STARTED ===")
        print(f"Data shape: {data.shape}")
        print(f"Wavenumbers: {len(wavenumbers)} points")
        print(f"XY coords shape: {xy_coords.shape}")
        print(f"Parameters: {params}")

        # Connect to Redis
        r = redis.Redis(host=redis_host, port=redis_port)

        # Create Redis keys
        embedding_key = f"temp:umap:{pid}:embedding"
        status_key = f"temp:umap:{pid}:status"

        # Check dimensions
        if len(wavenumbers) != data.shape[1]:
            print(
                f"WARNING: Wavenumber length ({len(wavenumbers)}) doesn't match data columns ({data.shape[1]})"
            )
            wavenumbers = wavenumbers[: data.shape[1]]

        # Create wavenumber mask
        mask = create_wavenumber_mask(wavenumbers, params.get("wavenumber_ranges", ""))

        if mask.sum() == 0:
            print("WARNING: No wavenumbers selected! Using all wavenumbers.")
            mask = np.ones_like(wavenumbers, dtype=bool)

        # Apply mask to data
        selected_data = data[:, mask]
        selected_wavenumbers = wavenumbers[mask]

        print(f"Selected data shape: {selected_data.shape}")
        print(f"Selected wavenumbers: {len(selected_wavenumbers)} points")

        # Add spatial coordinates if weight > 0
        spatial_weight = float(params.get("spatial_weight", 0) or 0)
        spatial_weight = max(0, min(100, spatial_weight))

        if spatial_weight > 0:
            print(f"Adding spatial coordinates with weight: {spatial_weight}")
            weighted_xy = spatial_weight * xy_coords.values
            selected_data = np.hstack([selected_data, weighted_xy])
            print(f"Data shape with spatial coords: {selected_data.shape}")

        # Extract UMAP parameters
        umap_params = {
            "n_neighbors": params["n_neighbors"],
            "min_dist": params["min_dist"],
            "metric": params["metric"],
            "densmap": params["densmap"],
        }

        print(f"Running UMAP with parameters: {umap_params}")

        # Run UMAP
        reducer = UMAP(**umap_params)
        embedding = reducer.fit_transform(selected_data)

        print(f"UMAP embedding shape: {embedding.shape}")

        # Store embedding in Redis
        r.set(embedding_key, embedding.tobytes())
        r.set(status_key, "done")

        print("===>>>>>>  UMAP PROCESS FINISHED ===")
        return True

    except Exception as e:
        error_msg = f"Error in UMAP computation: {str(e)}"
        print(error_msg)
        try:
            r.set(status_key, error_msg)
        except:
            pass
        return False


class ConfigureUMAP:
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379):
        # Initialize basic attributes
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.process = None
        self.current_pid = None
        self.umap_embedding = None

        # Initialize data attributes
        self.spectra = None
        self.wavenumbers = None
        self.measured_coords = None

        # Initialize Redis status
        self.redis_status = "Not Connected"
        self.redis_status_color = "danger"

        # Initialize Redis clients - one for text, one for binary
        self.redis_client = redis.Redis(
            host=self.redis_host,
            port=self.redis_port,
            decode_responses=True,  # For text data
        )

        self.redis_binary = redis.Redis(
            host=self.redis_host,
            port=self.redis_port,
            decode_responses=False,  # For binary data
        )

        # Try initial Redis connection
        self._connect_redis()

        # Create app with unique name
        self.app = dash.Dash(
            f"umap_config_{os.getpid()}", external_stylesheets=[dbc.themes.BOOTSTRAP]
        )

        # Define layout
        self.app.layout = html.Div(
            [
                dcc.Store(id="computation-status", data="idle"),
                html.H1("UMAP Configuration"),
                # Redis Status Card
                dbc.Row(
                    [
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
                            width=12,
                        ),
                    ]
                ),
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
                # UMAP Configuration Card
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader("UMAP Parameters"),
                                        dbc.CardBody(
                                            [
                                                # Wavenumber Range Selection
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                html.Label(
                                                                    "Wavenumber Ranges"
                                                                ),
                                                                dbc.Input(
                                                                    id="wavenumber-ranges",
                                                                    type="text",
                                                                    placeholder="e.g., 200-400,500-900",
                                                                    value="",
                                                                    className="mb-2",
                                                                ),
                                                                html.Small(
                                                                    "Enter comma-separated ranges (leave empty for all)",
                                                                    className="text-muted",
                                                                ),
                                                            ]
                                                        )
                                                    ],
                                                    className="mb-3",
                                                ),
                                                # UMAP Parameters
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                html.Label(
                                                                    "n_neighbors"
                                                                ),
                                                                dbc.Input(
                                                                    id="n-neighbors",
                                                                    type="number",
                                                                    value=15,
                                                                    min=2,
                                                                    step=1,
                                                                ),
                                                            ],
                                                            width=6,
                                                        ),
                                                        dbc.Col(
                                                            [
                                                                html.Label("min_dist"),
                                                                dbc.Input(
                                                                    id="min-dist",
                                                                    type="number",
                                                                    value=0.1,
                                                                    min=0.0,
                                                                    max=1.0,
                                                                    step=0.01,
                                                                ),
                                                            ],
                                                            width=6,
                                                        ),
                                                    ],
                                                    className="mb-3",
                                                ),
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                html.Label("Metric"),
                                                                dcc.Dropdown(
                                                                    id="metric",
                                                                    options=[
                                                                        {
                                                                            "label": "Euclidean",
                                                                            "value": "euclidean",
                                                                        },
                                                                        {
                                                                            "label": "Manhattan",
                                                                            "value": "manhattan",
                                                                        },
                                                                        {
                                                                            "label": "Cosine",
                                                                            "value": "cosine",
                                                                        },
                                                                        {
                                                                            "label": "Correlation",
                                                                            "value": "correlation",
                                                                        },
                                                                    ],
                                                                    value="euclidean",
                                                                ),
                                                            ],
                                                            width=6,
                                                        ),
                                                        dbc.Col(
                                                            [
                                                                html.Label(
                                                                    "Spatial Weight"
                                                                ),
                                                                dbc.Input(
                                                                    id="spatial-weight",
                                                                    type="number",
                                                                    value=0,
                                                                    min=0,
                                                                    max=100,
                                                                    step=1,
                                                                ),
                                                            ],
                                                            width=6,
                                                        ),
                                                    ],
                                                    className="mb-3",
                                                ),
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                dbc.Checkbox(
                                                                    id="densmap",
                                                                    label="Use DensMAP",
                                                                    value=False,
                                                                )
                                                            ]
                                                        )
                                                    ],
                                                    className="mb-3",
                                                ),
                                                # Run Button and Status moved inside card
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                dbc.Button(
                                                                    "Run UMAP",
                                                                    id="run-button",
                                                                    color="primary",
                                                                    className="w-100",
                                                                ),
                                                                html.Div(
                                                                    id="status-output",
                                                                    style={
                                                                        "margin-top": "10px"
                                                                    },
                                                                ),
                                                            ]
                                                        )
                                                    ]
                                                ),
                                            ]
                                        ),
                                    ],
                                    className="mb-3",
                                )
                            ]
                        )
                    ]
                ),
                # Add data store for selections
                dcc.Store(id="selection-store", data=[]),
                # UMAP Results and Spectral Plot Row
                dbc.Row(
                    [
                        # UMAP Plot Column
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            [
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            "UMAP Embedding", width=8
                                                        ),
                                                        dbc.Col(
                                                            [
                                                                dbc.Button(
                                                                    "Clear Selection",
                                                                    id="clear-selection-button",
                                                                    color="secondary",
                                                                    size="sm",
                                                                    className="float-end",
                                                                )
                                                            ],
                                                            width=4,
                                                        ),
                                                    ]
                                                )
                                            ]
                                        ),
                                        dbc.CardBody(
                                            [
                                                dcc.Graph(
                                                    id="umap-scatter",
                                                    config={"displayModeBar": True},
                                                )
                                            ]
                                        ),
                                    ],
                                    className="mb-3",
                                )
                            ],
                            width=6,
                        ),
                        # Spectral Plot Column
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Selected Spectra"),
                                        dbc.CardBody(
                                            [
                                                dcc.Graph(
                                                    id="spectral-plot",
                                                    config={"displayModeBar": True},
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
                # Export Button Row
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Button(
                                    "Export Embedding",
                                    id="export-button",
                                    color="success",
                                    className="mb-3",
                                ),
                                html.Div(id="export-status", className="mb-3"),
                            ],
                            width=12,
                        )
                    ]
                ),
                dcc.Interval(id="process-check", interval=1000, disabled=True),
                dcc.Interval(id="status-check", interval=1000),
            ]
        )

        # Define callbacks
        @self.app.callback(
            [
                Output("status-output", "children"),
                Output("process-check", "disabled"),
                Output("run-button", "children"),
                Output("run-button", "color"),
                Output("run-button", "disabled"),
            ],
            [Input("run-button", "n_clicks"), Input("process-check", "n_intervals")],
            [
                State("n-neighbors", "value"),
                State("min-dist", "value"),
                State("metric", "value"),
                State("densmap", "value"),
                State("spatial-weight", "value"),
                State("wavenumber-ranges", "value"),
            ],
            prevent_initial_call=True,
        )
        def update_status(
            n_clicks,
            n_intervals,
            n_neighbors,
            min_dist,
            metric,
            densmap,
            spatial_weight,
            wavenumber_ranges,
        ):
            ctx = dash.callback_context
            if not ctx.triggered:
                raise PreventUpdate

            trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

            if trigger_id == "run-button" and n_clicks:
                if self.spectra is None:
                    return ("Error: No data loaded", True, "Run UMAP", "danger", False)

                try:
                    params = {
                        "n_neighbors": n_neighbors,
                        "min_dist": min_dist,
                        "metric": metric,
                        "densmap": densmap,
                        "spatial_weight": spatial_weight,
                        "wavenumber_ranges": wavenumber_ranges,
                    }

                    self.process = Process(
                        target=umap_runner,
                        args=(
                            self.spectra,
                            self.wavenumbers,
                            self.measured_coords,
                            params,
                            os.getpid(),
                            self.redis_host,
                            self.redis_port,
                        ),
                    )
                    self.process.start()
                    self.current_pid = self.process.pid
                    return (
                        f"Computing... (PID: {self.current_pid})",
                        False,  # Enable process check immediately
                        "Computing...",
                        "warning",
                        True,
                    )
                except Exception as e:
                    return (f"Error: {str(e)}", True, "Run UMAP", "danger", False)

            elif trigger_id == "process-check" and self.process:
                if not self.process.is_alive():
                    # Check Redis for results
                    status_key = f"temp:umap:{os.getpid()}:status"
                    embedding_key = f"temp:umap:{os.getpid()}:embedding"

                    status = self.redis_client.get(status_key)  # Text client
                    print(f"Redis status: {status}")  # Debug print

                    if status == "done":
                        embedding_bytes = self.redis_binary.get(
                            embedding_key
                        )  # Binary client
                        if embedding_bytes:
                            try:
                                # Use float32 instead of float64
                                self.umap_embedding = np.frombuffer(
                                    embedding_bytes, dtype=np.float32
                                ).reshape(-1, 2)
                                print(
                                    f"Retrieved embedding shape: {self.umap_embedding.shape}"
                                )
                                self._cleanup_redis_keys(os.getpid())
                                self.process = None
                                return (
                                    "Computation complete!",
                                    True,
                                    "Run UMAP",
                                    "success",
                                    False,
                                )
                            except Exception as e:
                                print(f"Error processing embedding: {str(e)}")
                                return (
                                    f"Error processing results: {str(e)}",
                                    True,
                                    "Run UMAP",
                                    "danger",
                                    False,
                                )
                        else:
                            print("No embedding data found in Redis")
                    else:
                        print(f"Unexpected status: {status}")

                    self.process = None
                    return ("Error in computation", True, "Run UMAP", "danger", False)
                return (
                    f"Still computing... (PID: {self.current_pid})",
                    False,
                    "Computing...",
                    "warning",
                    True,
                )

            return (
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
            )

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
                    # Get current project path from Redis
                    current_project = self.redis_client.get("current_project")

                    status_lines = [
                        f"Connected to {self.redis_client.connection_pool.connection_kwargs['host']}:{self.redis_client.connection_pool.connection_kwargs['port']}"
                    ]

                    # Add project path info
                    if current_project:
                        status_lines.append(f"Database: {current_project}")
                    else:
                        status_lines.append("No database selected")

                    # Add UMAP status
                    status_lines.append("Ready for UMAP calculation")

                    return (
                        html.Div(
                            [
                                html.Div(status_lines[0]),  # Connection status
                                html.Small(
                                    status_lines[1], className="text-muted d-block"
                                ),  # Database path
                                html.Small(
                                    status_lines[2], className="text-info d-block"
                                ),  # UMAP status
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
                        html.Small(
                            "No database selected", className="text-muted d-block"
                        ),
                        html.Small(
                            "UMAP status: Not ready", className="text-danger d-block"
                        ),
                    ]
                ),
                f"text-{self.redis_status_color}",
            )

        # Add load data callback
        @self.app.callback(
            [
                Output("load-data-status", "children"),
                Output("load-data-status", "className"),
            ],
            [Input("load-data-button", "n_clicks")],
            prevent_initial_call=True,
        )
        def handle_load_data(n_clicks):
            if n_clicks:
                status, color = self.load_data()
                return status, f"text-{color}"
            return dash.no_update, dash.no_update

        @self.app.callback(
            Output("selection-store", "data"),
            [Input("umap-scatter", "selectedData")],
            [State("selection-store", "data")],
            prevent_initial_call=True,
        )
        def update_selection(selected_data, current_selection):
            current_selection = set(current_selection or [])

            if selected_data:
                new_points = {p["pointIndex"] for p in selected_data["points"]}
                current_selection.update(
                    new_points
                )  # Add new points to existing selection

            return sorted(list(current_selection))

        @self.app.callback(
            Output("umap-scatter", "figure"),
            [Input("selection-store", "data"), Input("status-output", "children")],
            prevent_initial_call=True,
        )
        def update_umap_plot(selected_indices, status):
            if self.umap_embedding is not None:
                selected_indices = selected_indices or []

                # Use data_loaded as uirevision to maintain selection state
                uirevision = "selection" if selected_indices else "load"

                return {
                    "data": [
                        {
                            "type": "scatter",
                            "x": self.umap_embedding[:, 0],
                            "y": self.umap_embedding[:, 1],
                            "mode": "markers",
                            "marker": {
                                "size": 8,
                                "color": [
                                    (
                                        "rgb(30,144,255)"
                                        if i in selected_indices
                                        else "rgb(220,20,60)"
                                    )
                                    for i in range(len(self.umap_embedding))
                                ],
                                "opacity": [
                                    1.0 if i in selected_indices else 0.3
                                    for i in range(len(self.umap_embedding))
                                ],
                            },
                            "name": "UMAP Points",
                        }
                    ],
                    "layout": {
                        "title": "UMAP Embedding",
                        "xaxis": {"title": "UMAP 1"},
                        "yaxis": {"title": "UMAP 2"},
                        "showlegend": False,
                        "hovermode": "closest",
                        "dragmode": "lasso",
                        "margin": {"l": 60, "r": 20, "t": 40, "b": 40},
                        "uirevision": uirevision,  # Key for maintaining selection state
                    },
                }
            return dash.no_update

        @self.app.callback(
            Output("spectral-plot", "figure"),
            [Input("selection-store", "data")],
            prevent_initial_call=True,
        )
        def update_spectral_plot(selected_indices):
            if not selected_indices or self.spectra is None:
                return {
                    "data": [],
                    "layout": {
                        "title": "Select points to see spectral statistics",
                        "xaxis": {"title": "Wavenumber (cm⁻¹)"},
                        "yaxis": {"title": "Intensity"},
                        "showlegend": True,
                    },
                }

            selected_spectra = self.spectra[selected_indices]

            # Calculate statistics
            median = np.median(selected_spectra, axis=0)
            percentile_5 = np.percentile(selected_spectra, 5, axis=0)
            percentile_25 = np.percentile(selected_spectra, 25, axis=0)
            percentile_75 = np.percentile(selected_spectra, 75, axis=0)
            percentile_95 = np.percentile(selected_spectra, 95, axis=0)

            traces = [
                # 5-95 percentile band
                {
                    "type": "scatter",
                    "x": np.concatenate([self.wavenumbers, self.wavenumbers[::-1]]),
                    "y": np.concatenate([percentile_95, percentile_5[::-1]]),
                    "fill": "toself",
                    "fillcolor": "rgba(0,176,246,0.2)",
                    "line": {"color": "rgba(255,255,255,0)"},
                    "name": "5-95 Percentile",
                    "showlegend": True,
                },
                # 25-75 percentile band
                {
                    "type": "scatter",
                    "x": np.concatenate([self.wavenumbers, self.wavenumbers[::-1]]),
                    "y": np.concatenate([percentile_75, percentile_25[::-1]]),
                    "fill": "toself",
                    "fillcolor": "rgba(0,176,246,0.4)",
                    "line": {"color": "rgba(255,255,255,0)"},
                    "name": "25-75 Percentile",
                    "showlegend": True,
                },
                # Median line
                {
                    "type": "scatter",
                    "x": self.wavenumbers,
                    "y": median,
                    "line": {"color": "rgb(0,176,246)", "width": 2},
                    "name": "Median",
                    "showlegend": True,
                },
            ]

            return {
                "data": traces,
                "layout": {
                    "title": f"Selected Spectra (n={len(selected_indices)} spectra)",
                    "xaxis": {"title": "Wavenumber (cm⁻¹)", "autorange": "reversed"},
                    "yaxis": {"title": "Intensity"},
                    "showlegend": True,
                    "hovermode": "closest",
                },
            }

        @self.app.callback(
            [
                Output("umap-scatter", "figure", allow_duplicate=True),
                Output("spectral-plot", "figure", allow_duplicate=True),
            ],
            [Input("status-output", "children")],
            prevent_initial_call=True,
        )
        def update_plots_after_umap(status):
            if status == "Computation complete!" and self.umap_embedding is not None:
                # Create initial UMAP plot
                umap_fig = {
                    "data": [
                        {
                            "type": "scatter",
                            "x": self.umap_embedding[:, 0],
                            "y": self.umap_embedding[:, 1],
                            "mode": "markers",
                            "marker": {
                                "size": 8,
                                "opacity": 0.6,
                                "color": "rgb(220,20,60)",
                            },
                        }
                    ],
                    "layout": {
                        "title": "UMAP Embedding",
                        "xaxis": {"title": "UMAP 1"},
                        "yaxis": {"title": "UMAP 2"},
                        "showlegend": False,
                        "hovermode": "closest",
                        "dragmode": "lasso",
                        "margin": {"l": 60, "r": 20, "t": 40, "b": 40},
                    },
                }

                # Create empty spectral plot
                spectral_fig = {
                    "data": [],
                    "layout": {
                        "title": "Select points to see spectral statistics",
                        "xaxis": {"title": "Wavenumber (cm⁻¹)"},
                        "yaxis": {"title": "Intensity"},
                        "showlegend": True,
                    },
                }

                return umap_fig, spectral_fig
            return dash.no_update, dash.no_update

        @self.app.callback(
            Output("selection-store", "data", allow_duplicate=True),
            [Input("clear-selection-button", "n_clicks")],
            prevent_initial_call=True,
        )
        def clear_selection(n_clicks):
            if n_clicks:
                return []
            return dash.no_update

        @self.app.callback(
            Output("export-status", "children"),
            [Input("export-button", "n_clicks")],
            prevent_initial_call=True,
        )
        def export_embedding(n_clicks):
            if n_clicks and self.umap_embedding is not None:
                try:
                    # Get current project path from Redis
                    db_path = self.redis_client.get("current_project")
                    if not db_path:
                        return html.Div(
                            "No active project found in Redis", className="text-danger"
                        )

                    # Connect to database
                    conn = duckdb.connect(db_path)

                    # Get HCD indices from measured_data table
                    measured_data = read_df_from_db(conn, "measured_data")
                    hcd_indices = measured_data.index.values

                    # Create DataFrame with hcd_indx, UMAP coordinates, and selection column
                    embedding_df = pd.DataFrame(
                        {
                            "hcd_indx": hcd_indices,
                            "umap_x": self.umap_embedding[:, 0],
                            "umap_y": self.umap_embedding[:, 1],
                            "selection": 0,  # Initialize selection column to 0
                        }
                    )

                    # Store DataFrame in database
                    store_df_in_db(conn, embedding_df, "embedding", if_exists="replace")

                    conn.close()

                    return html.Div(
                        "Embedding exported successfully!", className="text-success"
                    )
                except Exception as e:
                    return html.Div(
                        f"Error exporting embedding: {str(e)}", className="text-danger"
                    )
            return dash.no_update

    def run(self, debug=False, port=8056):
        # Suppress Werkzeug logging
        import logging

        log = logging.getLogger("werkzeug")
        log.setLevel(logging.ERROR)

        self.app.run_server(debug=debug, port=port, host="0.0.0.0")

    def _cleanup_redis_keys(self, pid):
        """Clean up Redis keys for a given process ID"""
        try:
            self.redis_client.delete(f"temp:umap:{pid}:embedding")
            self.redis_client.delete(f"temp:umap:{pid}:status")
        except Exception as e:
            print(f"Error cleaning up Redis keys: {str(e)}")

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

    def load_data(self):
        """Load data from DuckDB database"""
        try:
            # Get current project path from Redis
            db_path = self.redis_client.get("current_project")
            if not db_path:
                return "No active project found in Redis", "danger"

            print(f"\n=== Loading data from {db_path} ===")
            conn = duckdb.connect(db_path)

            # Check for required tables
            required_tables = ["measured_data", "wavenumbers", "measured_points", "HCD"]
            for table in required_tables:
                if not check_table_exists(conn, table):
                    conn.close()
                    return f"Required table '{table}' not found", "danger"

            # Load spectral data
            self.wavenumbers = read_df_from_db(conn, "wavenumbers").values.flatten()
            measured_data = read_df_from_db(conn, "measured_data")
            measured_points = read_df_from_db(conn, "measured_points")
            hcd_df = read_df_from_db(conn, "HCD")

            # Extract spectral data
            self.spectra = measured_data.iloc[:, 1:].values  # Skip index column

            # Get coordinates from HCD using measured_points indices
            hcd_indices = measured_points["hcd_indx"].values
            self.measured_coords = hcd_df.loc[hcd_indices, ["X", "Y"]]

            # Print detailed summaries
            print("\n=== Data Summary ===")
            print(f"Wavenumbers:")
            print(f"  - Shape: {self.wavenumbers.shape}")
            print(
                f"  - Range: {self.wavenumbers.min():.1f} to {self.wavenumbers.max():.1f} cm⁻¹"
            )

            print(f"\nSpectral Data:")
            print(f"  - Shape: {self.spectra.shape}")
            print(f"  - Number of spectra: {self.spectra.shape[0]}")
            print(f"  - Points per spectrum: {self.spectra.shape[1]}")
            print(
                f"  - Value range: {self.spectra.min():.2f} to {self.spectra.max():.2f}"
            )

            print(f"\nSpatial Coordinates:")
            print(f"  - Shape: {self.measured_coords.shape}")
            print(
                f"  - X range: {self.measured_coords['X'].min():.2f} to {self.measured_coords['X'].max():.2f}"
            )
            print(
                f"  - Y range: {self.measured_coords['Y'].min():.2f} to {self.measured_coords['Y'].max():.2f}"
            )

            print(f"\nHCD Data:")
            print(f"  - Total points: {len(hcd_df)}")
            print(f"  - Selected points: {len(hcd_indices)}")
            print("==================\n")

            conn.close()
            return "Data loaded successfully", "success"

        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return f"Error: {str(e)}", "danger"


if __name__ == "__main__":
    app = ConfigureUMAP()
    app.run()
