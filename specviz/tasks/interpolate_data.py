from typing import Any, Dict

import argparse
import datetime
import io
import json
import logging
import multiprocessing
import os
import sys
import threading
import time
import traceback
from queue import Empty

import dash
import dash_bootstrap_components as dbc
import dask.array as da
import duckdb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import redis
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from dask.distributed import Client, LocalCluster

from specviz.tasks.dbtools import check_table_exists, read_df_from_db
from specviz.tasks.interpolation_worker import InterpolationWorker, run_worker
from specviz.tasks.interpolators import INTERPOLATORS

# Suppress Flask logging
log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)


def run_interpolation_process(
    points,
    values,
    xi,
    yi,
    interpolator_key,
    name,
    redis_config,
    wavenumbers,
    interpolation_params,
):
    """Standalone process to run interpolation"""
    try:
        print(f"\n=== Interpolation Worker Started (PID: {os.getpid()}) ===")
        print(f"Input shapes:")
        print(f"Points: {points.shape}")
        print(f"Values: {values.shape}")
        print(f"Grid: {len(xi)}x{len(yi)}")

        # Debug input data
        print("\nInput data statistics:")
        print(f"Points range X: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
        print(f"Points range Y: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
        print(f"Grid range X: [{xi.min():.2f}, {xi.max():.2f}]")
        print(f"Grid range Y: [{yi.min():.2f}, {yi.max():.2f}]")
        print(f"Values range: [{values.min():.2f}, {values.max():.2f}]")
        print(f"NaN in points: {np.isnan(points).sum()}")
        print(f"NaN in values: {np.isnan(values).sum()}")
        sys.stdout.flush()

        # Create Redis client without decode_responses for binary data
        redis_client = redis.Redis(
            host=redis_config["host"], port=redis_config["port"], decode_responses=False
        )

        # Get interpolator
        interpolator = INTERPOLATORS[interpolator_key]
        print(f"Using interpolator: {interpolator_key}")
        sys.stdout.flush()

        # Create meshgrid for interpolation
        XI, YI = np.meshgrid(xi, yi)
        print(f"\nMeshgrid shapes:")
        print(f"XI shape: {XI.shape}")
        print(f"YI shape: {YI.shape}")
        sys.stdout.flush()

        # Prepare output array
        output = np.zeros((len(yi), len(xi), values.shape[1]))

        # Interpolate each channel
        print("\nInterpolating channels...")
        for i in range(values.shape[1]):
            if i % 100 == 0:
                print(f"Channel {i}/{values.shape[1]}")
                sys.stdout.flush()

            # Get current channel values
            channel_values = values[:, i]

            # Skip if all values are NaN
            if np.all(np.isnan(channel_values)):
                print(f"Warning: Channel {i} contains all NaN values, skipping")
                continue

            # Remove NaN values for this channel
            valid_mask = ~np.isnan(channel_values)
            if not np.all(valid_mask):
                channel_points = points[valid_mask]
                channel_values = channel_values[valid_mask]
            else:
                channel_points = points

            try:
                # Debug channel data
                if i % 100 == 0:
                    print(f"\nChannel {i} input stats:")
                    print(f"Valid points: {len(channel_points)}")
                    print(
                        f"Points range X: [{channel_points[:, 0].min():.2f}, {channel_points[:, 0].max():.2f}]"
                    )
                    print(
                        f"Points range Y: [{channel_points[:, 1].min():.2f}, {channel_points[:, 1].max():.2f}]"
                    )
                    print(
                        f"Values range: [{channel_values.min():.2f}, {channel_values.max():.2f}]"
                    )
                    sys.stdout.flush()

                # Perform interpolation
                result = interpolator.interpolate(
                    channel_points, channel_values, xi, yi
                )

                # Debug interpolation result
                if i % 100 == 0:
                    print(f"Channel {i} output stats:")
                    print(f"Output shape: {result.shape}")
                    print(
                        f"Output range: [{np.nanmin(result):.2f}, {np.nanmax(result):.2f}]"
                    )
                    print(f"NaN count: {np.isnan(result).sum()}")
                    sys.stdout.flush()

                output[:, :, i] = result

            except Exception as e:
                print(f"Error interpolating channel {i}: {str(e)}")
                traceback.print_exc()
                sys.stdout.flush()
                continue

        print("\nInterpolation complete")
        print(f"Output shape: {output.shape}")
        print(f"Final output stats:")
        print(f"Value range: [{np.nanmin(output):.2f}, {np.nanmax(output):.2f}]")
        print(f"NaN count: {np.isnan(output).sum()}")
        sys.stdout.flush()

        print("\nStoring results...")
        # Store metadata (using a separate Redis client with decode_responses=True)
        metadata_client = redis.Redis(
            host=redis_config["host"], port=redis_config["port"], decode_responses=True
        )

        metadata = {
            "shape": output.shape,
            "wavenumbers": wavenumbers,
            "grid": {"xi": xi.tolist(), "yi": yi.tolist()},
        }
        metadata_client.hset(f"interpolation:{name}", "metadata", json.dumps(metadata))

        # Store channels as binary data
        total_channels = output.shape[2]
        for i in range(total_channels):
            if i % 100 == 0:
                print(f"Storing channel {i}/{total_channels}")
                sys.stdout.flush()
            channel_data = output[:, :, i].tobytes()
            redis_client.hset(
                f"interpolation:{name}", f"channel:{i}".encode(), channel_data
            )

        print("\nInterpolation completed successfully")
        sys.stdout.flush()
        return True

    except Exception as e:
        print(f"\nProcess error: {str(e)}")
        traceback.print_exc()
        sys.stdout.flush()
        return False


class InterpolateData:
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379):
        # Store connection details
        self.redis_host = redis_host
        self.redis_port = redis_port

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

        # Initialize data containers
        self.wavenumber_df = None
        self.measured_data_df = None
        self.interpolated_data = None
        self.wavenumbers = None
        self.current_interpolation_name = None
        self.domain = None

        # Add selected channel storage
        self.selected_channel = 0  # Default to first channel

        # Create app and layout
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

        self.redis_status = "Not Connected"
        self.redis_status_color = "danger"
        self._connect_redis()

        self.worker_process = None
        self.worker_queue = multiprocessing.Queue()
        self.worker_status_thread = None

        self.app.layout = dbc.Container(
            [
                html.H1("Data Interpolation", className="my-4"),
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
                            width=12,
                        ),
                    ]
                ),
                # Load Data Button
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Button(
                                    "Load Data",
                                    id="load-data-button",
                                    color="primary",
                                    className="mb-3",
                                ),
                                html.Div(id="load-data-status", className="mb-3"),
                            ]
                        )
                    ]
                ),
                # Interpolation Controls
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Interpolation Controls"),
                                        dbc.CardBody(
                                            [
                                                # Name input
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                html.Label(
                                                                    "Interpolation Name:"
                                                                ),
                                                                dbc.Input(
                                                                    id="interpolation-name",
                                                                    type="text",
                                                                    placeholder="Enter name for interpolated data",
                                                                    value="interpolation_1",
                                                                    className="mb-2",
                                                                ),
                                                            ],
                                                            width=12,
                                                        ),
                                                    ],
                                                    className="mb-3",
                                                ),
                                                # Interpolation Method Selection
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                html.Label(
                                                                    "Interpolation Method:"
                                                                ),
                                                                dcc.Dropdown(
                                                                    id="interpolation-method",
                                                                    options=[
                                                                        {
                                                                            "label": interp.name,
                                                                            "value": key,
                                                                            "title": interp.description,
                                                                        }
                                                                        for key, interp in INTERPOLATORS.items()
                                                                    ],
                                                                    value="linear",
                                                                    clearable=False,
                                                                    className="mb-2",
                                                                ),
                                                            ],
                                                            width=12,
                                                        ),
                                                    ],
                                                    className="mb-3",
                                                ),
                                                # Grid Setup Row
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                html.Label(
                                                                    "Grid Step Size:"
                                                                ),
                                                                dbc.Input(
                                                                    id="grid-step",
                                                                    type="number",
                                                                    value=1,
                                                                    min=0.001,
                                                                    step="any",
                                                                    className="mb-2",
                                                                ),
                                                            ],
                                                            width=6,
                                                        ),
                                                        dbc.Col(
                                                            [
                                                                html.Div(
                                                                    id="grid-info",
                                                                    className="mt-4",
                                                                )
                                                            ],
                                                            width=6,
                                                        ),
                                                    ],
                                                    className="mb-3",
                                                ),
                                                # Interpolate Button Row
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                dbc.Button(
                                                                    "Run Interpolation",
                                                                    id="interpolate-button",
                                                                    color="success",
                                                                    className="me-2",
                                                                ),
                                                                html.Div(
                                                                    id="interpolation-status",
                                                                    className="mt-2",
                                                                ),
                                                            ],
                                                            width=12,
                                                        ),
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
                # Data Summary Card with Heatmap
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Data Summary"),
                                        dbc.CardBody(
                                            [
                                                html.Div(id="data-summary"),
                                                html.Hr(),
                                                # Channel selection dropdown
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                html.Label(
                                                                    "Select Channel:"
                                                                ),
                                                                dcc.Dropdown(
                                                                    id="channel-selector",
                                                                    options=[],
                                                                    placeholder="Select a wavenumber...",
                                                                    clearable=False,
                                                                    className="mb-2",
                                                                ),
                                                            ],
                                                            width=12,
                                                        ),
                                                    ]
                                                ),
                                                # Heatmap
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                dcc.Graph(
                                                                    id="interpolation-heatmap",
                                                                    config={
                                                                        "displayModeBar": True,
                                                                        "toImageButtonOptions": {
                                                                            "format": "png",  # or 'svg'
                                                                            "filename": "interpolation_heatmap",
                                                                            "height": 600,
                                                                            "width": None,  # Will maintain aspect ratio
                                                                            "scale": 2,  # Increase resolution
                                                                        },
                                                                        "modeBarButtonsToAdd": [
                                                                            "drawline",
                                                                            "drawopenpath",
                                                                            "drawclosedpath",
                                                                            "drawcircle",
                                                                            "drawrect",
                                                                            "eraseshape",
                                                                        ],
                                                                        "modeBarButtonsToRemove": [],
                                                                        "scrollZoom": True,
                                                                    },
                                                                ),
                                                            ],
                                                            width=12,
                                                        ),
                                                    ]
                                                ),
                                                # Download button row
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                html.Hr(),
                                                                html.H5(
                                                                    "Export Data",
                                                                    className="mt-3 mb-2",
                                                                ),
                                                                dbc.Button(
                                                                    "Download Current Channel as NPY",
                                                                    id="download-channel-button",
                                                                    color="primary",
                                                                    className="me-2 mb-2",
                                                                ),
                                                                dbc.Button(
                                                                    "Download All Channels as NPZ",
                                                                    id="download-all-button",
                                                                    color="success",
                                                                    className="mb-2",
                                                                ),
                                                                html.Div(
                                                                    id="download-status",
                                                                    className="mt-2",
                                                                ),
                                                                # Hidden download components
                                                                dcc.Download(
                                                                    id="download-channel-data"
                                                                ),
                                                                dcc.Download(
                                                                    id="download-all-data"
                                                                ),
                                                            ],
                                                            width=12,
                                                        ),
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
                # Update intervals
                dcc.Interval(id="status-check", interval=1000),
            ],
            fluid=True,
        )

        self._setup_callbacks()

    def _connect_redis(self):
        """Establish connection to Redis"""
        try:
            self.redis_client.ping()
            self.redis_status = f"Connected to {self.redis_client.connection_pool.connection_kwargs['host']}:{self.redis_client.connection_pool.connection_kwargs['port']}"
            self.redis_status_color = "success"
        except redis.ConnectionError as e:
            self.redis_status = f"Connection Failed: Redis not running on {self.redis_client.connection_pool.connection_kwargs['host']}:{self.redis_client.connection_pool.connection_kwargs['port']}"
            self.redis_status_color = "danger"
        except Exception as e:
            self.redis_status = f"Error: {str(e)}"
            self.redis_status_color = "danger"

    def _setup_callbacks(self):
        """Set up the Dash callbacks"""
        print("\nSetting up callbacks...")

        # Redis status callback
        print("Setting up redis status callback")

        @self.app.callback(
            [Output("redis-status", "children"), Output("redis-status", "className")],
            [
                Input("status-check", "n_intervals"),
                Input("redis-refresh-button", "n_clicks"),
            ],
            prevent_initial_call=True,
        )
        def update_redis_status(n_intervals, n_clicks):
            print("Redis status callback triggered")
            try:
                if self.redis_client.ping():
                    # Get current project
                    current_project = self.redis_client.get("current_project")

                    status_lines = [
                        f"Connected to {self.redis_client.connection_pool.connection_kwargs['host']}:{self.redis_client.connection_pool.connection_kwargs['port']}"
                    ]

                    if current_project:
                        status_lines.append(f"Current Project: {current_project}")
                    else:
                        status_lines.append("No active project")

                    return (
                        html.Div(
                            [
                                html.Div(status_lines[0]),  # Connection status
                                html.Small(
                                    status_lines[1], className="text-muted d-block"
                                ),  # Project info
                            ]
                        ),
                        "text-success",
                    )

            except (redis.ConnectionError, redis.TimeoutError):
                self._connect_redis()
            except Exception as e:
                self.redis_status = f"Unexpected Error: {str(e)}"
                self.redis_status_color = "danger"

            return (
                html.Div(
                    [
                        html.Div(self.redis_status),
                        html.Small("No active project", className="text-muted d-block"),
                    ]
                ),
                f"text-{self.redis_status_color}",
            )

        # Load data callback
        print("Setting up load data callback")

        @self.app.callback(
            Output("load-data-status", "children"),
            Output("load-data-status", "className"),
            Input("load-data-button", "n_clicks"),
            prevent_initial_call=True,
        )
        def handle_load_data(n_clicks):
            print("Load data callback triggered")
            if not n_clicks:
                return "", ""

            message, status = self.load_data()
            return message, f"text-{status}"

        # Data summary callback
        print("Setting up data summary callback")

        @self.app.callback(
            Output("data-summary", "children"),
            [
                Input("load-data-button", "n_clicks"),
                Input("status-check", "n_intervals"),
            ],
            prevent_initial_call=True,
        )
        def update_data_summary(n_clicks, n_intervals):
            print("Data summary callback triggered")
            if not hasattr(self, "wavenumber_df") or self.wavenumber_df is None:
                return "No data loaded"

            try:
                summary = [
                    html.H6("Spectral Range:", className="mb-2"),
                    html.Div(
                        [
                            f"Wavenumbers: {len(self.wavenumber_df)} points",
                            html.Br(),
                            f"Range: {self.wavenumber_df['wavenumber'].min():.2f} to {self.wavenumber_df['wavenumber'].max():.2f} cm⁻¹",
                        ],
                        className="mb-3",
                    ),
                ]

                # Add domain info only if available
                if hasattr(self, "domain") and self.domain is not None:
                    summary.extend(
                        [
                            html.H6("Spatial Domain:", className="mb-2"),
                            html.Div(
                                [
                                    f"X: [{self.domain['x_min']}, {self.domain['x_max']}]",
                                    html.Br(),
                                    f"Y: [{self.domain['y_min']}, {self.domain['y_max']}]",
                                ],
                                className="mb-3",
                            ),
                        ]
                    )

                # Add measurement info only if available
                if (
                    hasattr(self, "measured_data_df")
                    and self.measured_data_df is not None
                ):
                    summary.extend(
                        [
                            html.H6("Measurements:", className="mb-2"),
                            html.Div(
                                [
                                    f"Total spectra: {len(self.measured_data_df)}",
                                    html.Br(),
                                    f"HCD points: {len(self.hcd_df) if hasattr(self, 'hcd_df') else 'N/A'}",
                                ]
                            ),
                        ]
                    )

                return summary

            except Exception as e:
                print(f"Error in data summary: {str(e)}")
                return f"Error updating summary: {str(e)}"

        # Grid info callback
        print("Setting up grid info callback")

        @self.app.callback(
            Output("grid-info", "children"),
            [
                Input("grid-step", "value"),
                Input("load-data-button", "n_clicks"),
                Input("status-check", "n_intervals"),
            ],
            prevent_initial_call=True,
        )
        def update_grid_info(step, n_clicks, n_intervals):
            print("Grid info callback triggered")
            if not hasattr(self, "domain") or self.domain is None:
                return "No domain information available"

            if not step:
                return "Please enter a grid step size"

            try:
                x_points = len(
                    np.arange(
                        self.domain["x_min"], self.domain["x_max"] + step / 2, step
                    )
                )
                y_points = len(
                    np.arange(
                        self.domain["y_min"], self.domain["y_max"] + step / 2, step
                    )
                )
                total_points = x_points * y_points

                return html.Div(
                    [
                        html.Strong("Grid Size:"),
                        html.Br(),
                        f"X points: {x_points}",
                        html.Br(),
                        f"Y points: {y_points}",
                        html.Br(),
                        f"Total points: {total_points:,}",
                    ]
                )

            except Exception as e:
                print(f"Error in grid info: {str(e)}")
                return f"Error calculating grid: {str(e)}"

        # Interpolation button callback
        print("Setting up interpolation button callback")

        @self.app.callback(
            [
                Output("interpolate-button", "color"),
                Output("interpolate-button", "children"),
                Output("interpolation-status", "children"),
            ],
            [Input("interpolate-button", "n_clicks")],
            [
                State("interpolation-method", "value"),
                State("grid-step", "value"),
                State("interpolation-name", "value"),
            ],
            prevent_initial_call=True,
        )
        def handle_interpolation(n_clicks, method, step, name):
            print("Interpolation button callback triggered")
            if not all([method, step, name]):
                return "danger", "Run Interpolation", "Missing parameters"

            if self.start_interpolation(method, step, name):
                return "warning", "Running...", "Process started"
            else:
                return "danger", "Run Interpolation", "Failed to start"

        # Channel selector options callback - now also sets the value
        print("Setting up channel selector callback")

        @self.app.callback(
            [
                Output("channel-selector", "options"),
                Output("channel-selector", "value"),
            ],
            [Input("status-check", "n_intervals")],
            prevent_initial_call=True,
        )
        def update_channel_selector(n_intervals):
            print("Channel selector callback triggered")
            if not self.current_interpolation_name:
                return [], None

            try:
                metadata_json = self.redis_client.hget(
                    f"interpolation:{self.current_interpolation_name}", "metadata"
                )

                if metadata_json is None:
                    print("No metadata found in Redis")
                    return [], None

                metadata = json.loads(metadata_json)

                options = [
                    {"label": f"{wn:.2f} cm⁻¹", "value": i}
                    for i, wn in enumerate(metadata["wavenumbers"])
                ]

                # Return current selection if it exists and is valid
                if (
                    hasattr(self, "selected_channel")
                    and self.selected_channel is not None
                    and self.selected_channel < len(options)
                ):
                    print(f"Using stored channel selection: {self.selected_channel}")
                    return options, self.selected_channel
                else:
                    # Default to first channel
                    print("Defaulting to first channel")
                    self.selected_channel = 0
                    return options, 0

            except Exception as e:
                print(f"Error in channel selector: {str(e)}")
                traceback.print_exc()
                return [], None

        # Heatmap callback
        print("Setting up heatmap callback")

        @self.app.callback(
            Output("interpolation-heatmap", "figure"),
            [Input("channel-selector", "value")],
            prevent_initial_call=True,
        )
        def update_heatmap(channel_idx):
            print(f"Heatmap callback triggered with channel_idx: {channel_idx}")

            if channel_idx is None:
                print("No channel selected")
                return go.Figure()  # Return an empty figure if no channel is selected

            # Store the selected channel
            self.selected_channel = channel_idx

            try:
                # Get metadata using text client
                metadata_json = self.redis_client.hget(
                    f"interpolation:{self.current_interpolation_name}", "metadata"
                )

                if metadata_json is None:
                    print("No metadata found in Redis")
                    return go.Figure()

                metadata = json.loads(metadata_json)
                y_points, x_points, nchannels = metadata[
                    "shape"
                ]  # Y, X, Channels order

                # Get channel data using binary client
                channel_data = self.redis_binary.hget(
                    f"interpolation:{self.current_interpolation_name}",
                    f"channel:{channel_idx}".encode(),
                )

                if channel_data is None:
                    print(f"No data found for channel {channel_idx}")
                    return go.Figure()

                # Reshape the binary data
                channel_array = np.frombuffer(channel_data, dtype=np.float64).reshape(
                    (y_points, x_points)
                )

                # Create heatmap with explicit value range and square pixels
                fig = go.Figure(
                    data=go.Heatmap(
                        z=channel_array,
                        x=np.linspace(
                            metadata["grid"]["xi"][0],
                            metadata["grid"]["xi"][-1],
                            x_points,
                        ),
                        y=np.linspace(
                            metadata["grid"]["yi"][0],
                            metadata["grid"]["yi"][-1],
                            y_points,
                        ),
                        colorscale="Viridis",
                        zmin=np.nanmin(channel_array),
                        zmax=np.nanmax(channel_array),
                    )
                )

                # Update layout to ensure square pixels and proper static image export
                fig.update_layout(
                    title=f'Interpolated Data - {metadata["wavenumbers"][channel_idx]:.2f} cm⁻¹',
                    xaxis_title="X",
                    yaxis_title="Y",
                    yaxis=dict(
                        scaleanchor="x",
                        scaleratio=1,
                    ),
                    autosize=False,
                    height=600,
                    width=600,
                    margin=dict(l=80, r=20, t=100, b=80),
                    paper_bgcolor="white",
                    plot_bgcolor="white",
                )

                # Configure modebar for better image export
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")

                # Add colorbar title
                fig.update_traces(colorbar=dict(title="Intensity", titleside="right"))

                return fig

            except Exception as e:
                print(f"Error updating heatmap: {str(e)}")
                traceback.print_exc()
                return go.Figure()

        # Download current channel callback
        print("Setting up download channel callback")

        @self.app.callback(
            Output("download-channel-data", "data"),
            Input("download-channel-button", "n_clicks"),
            State("channel-selector", "value"),
            prevent_initial_call=True,
        )
        def download_current_channel(n_clicks, channel_idx):
            if channel_idx is None:
                return None

            try:
                # Get metadata
                metadata_json = self.redis_client.hget(
                    f"interpolation:{self.current_interpolation_name}", "metadata"
                )

                if metadata_json is None:
                    print("No metadata found in Redis")
                    return None

                metadata = json.loads(metadata_json)
                y_points, x_points, nchannels = metadata["shape"]
                wavenumber = metadata["wavenumbers"][channel_idx]

                # Get channel data
                channel_data = self.redis_binary.hget(
                    f"interpolation:{self.current_interpolation_name}",
                    f"channel:{channel_idx}".encode(),
                )

                if channel_data is None:
                    print(f"No data found for channel {channel_idx}")
                    return None

                # Reshape the binary data
                channel_array = np.frombuffer(channel_data, dtype=np.float64).reshape(
                    (y_points, x_points)
                )

                # Create a BytesIO object to store the numpy array
                buffer = io.BytesIO()
                np.save(buffer, channel_array)
                buffer.seek(0)

                # Return the data as a download
                return dcc.send_bytes(
                    buffer.getvalue(),
                    filename=f"interpolated_data_{self.current_interpolation_name}_wavenumber_{wavenumber:.2f}.npy",
                )

            except Exception as e:
                print(f"Error downloading channel data: {str(e)}")
                traceback.print_exc()
                return None

        # Download all channels callback
        print("Setting up download all channels callback")

        @self.app.callback(
            Output("download-all-data", "data"),
            Input("download-all-button", "n_clicks"),
            prevent_initial_call=True,
        )
        def download_all_channels(n_clicks):
            if not self.current_interpolation_name:
                return None

            try:
                # Get metadata
                metadata_json = self.redis_client.hget(
                    f"interpolation:{self.current_interpolation_name}", "metadata"
                )

                if metadata_json is None:
                    print("No metadata found in Redis")
                    return None

                metadata = json.loads(metadata_json)
                y_points, x_points, nchannels = metadata["shape"]
                wavenumbers = metadata["wavenumbers"]

                # Create a dictionary to store all arrays
                data_dict = {
                    "metadata": {
                        "wavenumbers": wavenumbers,
                        "grid_x": metadata["grid"]["xi"],
                        "grid_y": metadata["grid"]["yi"],
                        "shape": metadata["shape"],
                    }
                }

                # Create a BytesIO object to store the numpy arrays
                buffer = io.BytesIO()

                # Always get data for all channels without sampling
                print(f"Saving all {nchannels} channels")
                for channel_idx in range(nchannels):
                    if channel_idx % 100 == 0:
                        print(f"Processing channel {channel_idx}/{nchannels}")

                    channel_data = self.redis_binary.hget(
                        f"interpolation:{self.current_interpolation_name}",
                        f"channel:{channel_idx}".encode(),
                    )

                    if channel_data is not None:
                        channel_array = np.frombuffer(
                            channel_data, dtype=np.float64
                        ).reshape((y_points, x_points))
                        data_dict[f"channel_{channel_idx}"] = channel_array

                # Save the dictionary to the buffer
                print("Compressing data - this may take a while for large datasets...")
                np.savez_compressed(buffer, **data_dict)
                buffer.seek(0)

                # Return the data as a download
                return dcc.send_bytes(
                    buffer.getvalue(),
                    filename=f"interpolated_data_{self.current_interpolation_name}_all_channels.npz",
                )

            except Exception as e:
                print(f"Error downloading all channels: {str(e)}")
                traceback.print_exc()
                return None

        # Download status callback
        @self.app.callback(
            Output("download-status", "children"),
            [
                Input("download-channel-button", "n_clicks"),
                Input("download-all-button", "n_clicks"),
            ],
            prevent_initial_call=True,
        )
        def update_download_status(channel_clicks, all_clicks):
            ctx = dash.callback_context
            if not ctx.triggered:
                return ""

            button_id = ctx.triggered[0]["prop_id"].split(".")[0]

            if button_id == "download-channel-button":
                return html.Div(
                    "Preparing single channel download...", className="text-info"
                )
            elif button_id == "download-all-button":
                return html.Div(
                    "Preparing all channels download (this may take a while)...",
                    className="text-info",
                )

            return ""

        print("Callbacks setup complete")

    def load_data(self):
        """Load data from the current project database"""
        try:
            db_path = self.redis_client.get("current_project")
            if not db_path:
                return "No active project found in Redis", "danger"

            conn = duckdb.connect(db_path)

            # Load wavenumber data
            if check_table_exists(conn, "wavenumbers"):
                self.wavenumber_df = read_df_from_db(conn, "wavenumbers")
            else:
                conn.close()
                return "No wavenumber data found in database", "danger"

            # Load measured spectral data
            if check_table_exists(conn, "measured_data"):
                self.measured_data_df = read_df_from_db(conn, "measured_data")
                # Store HCD indices from measured data
                if "hcd_indx" not in self.measured_data_df.columns:
                    conn.close()
                    return "No HCD index column found in measured data", "danger"
                self.measured_indices = self.measured_data_df["hcd_indx"].values
                print(
                    f"Loaded {len(self.measured_indices)} HCD indices from measured data"
                )
            else:
                conn.close()
                return "No measured data found in database", "danger"

            # Load HCD points and config flags
            if check_table_exists(conn, "HCD") and check_table_exists(
                conn, "config_flags"
            ):
                self.hcd_df = read_df_from_db(conn, "HCD")
                print(f"Loaded {len(self.hcd_df)} HCD points")
                print(
                    f"HCD index range: [{self.hcd_df.index.min()}, {self.hcd_df.index.max()}]"
                )

                config_flags_df = read_df_from_db(conn, "config_flags")
                config_flags = config_flags_df.iloc[0].to_dict()

                self.domain = {
                    "x_min": config_flags["x_min"],
                    "x_max": config_flags["x_max"],
                    "y_min": config_flags["y_min"],
                    "y_max": config_flags["y_max"],
                }
            else:
                conn.close()
                return "Missing HCD points or configuration data", "danger"

            conn.close()

            # Print summary of loaded data
            print("\nData Summary:")
            print("-" * 40)
            print("Spectral Range:")
            print(f"  Wavenumbers: {len(self.wavenumber_df)} points")
            print(
                f"  Range: {self.wavenumber_df['wavenumber'].min():.2f} to {self.wavenumber_df['wavenumber'].max():.2f} cm⁻¹"
            )
            print("\nSpatial Domain:")
            print(f"  X: [{self.domain['x_min']}, {self.domain['x_max']}]")
            print(f"  Y: [{self.domain['y_min']}, {self.domain['y_max']}]")
            print("\nMeasurements:")
            print(f"  Total spectra: {len(self.measured_data_df)}")
            print(f"  HCD points: {len(self.hcd_df)}")
            print(f"  Measured indices: {len(self.measured_indices)}")
            print(
                f"  HCD index range: [{min(self.measured_indices)}, {max(self.measured_indices)}]"
            )

            return "Data loaded successfully", "success"

        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return f"Error: {str(e)}", "danger"

    def _monitor_worker(self):
        """Monitor worker process and handle cleanup"""
        while True:
            if self.worker_process is None:
                break

            if not self.worker_process.is_alive():
                self.worker_process.join()
                self.worker_process = None
                break

            time.sleep(0.1)

    def start_interpolation(self, interpolator_key: str, grid_step: float, name: str):
        """Start interpolation in a separate process"""
        if self.worker_process and self.worker_process.is_alive():
            print("Previous process still running")
            return False

        try:
            # Prepare data using HCD indices from measured data
            filtered_points = self.hcd_df.loc[self.measured_indices][["X", "Y"]].values
            filtered_values = self.measured_data_df.values

            print("\nData preparation:")
            print(
                f"HCD indices range: [{min(self.measured_indices)}, {max(self.measured_indices)}]"
            )
            print(f"Points shape: {filtered_points.shape}")
            print(f"Values shape: {filtered_values.shape}")
            print(f"Coordinate ranges:")
            print(
                f"X: [{filtered_points[:, 0].min():.2f}, {filtered_points[:, 0].max():.2f}]"
            )
            print(
                f"Y: [{filtered_points[:, 1].min():.2f}, {filtered_points[:, 1].max():.2f}]"
            )

            # Store current interpolation name
            self.current_interpolation_name = name

            # Prepare Redis configuration
            redis_config = {
                "host": self.redis_host,
                "port": self.redis_port,
                "decode_responses": True,
            }

            # Empty interpolation params - let the interpolator handle its own defaults
            interpolation_params = {}

            # Start the process with minimal data
            self.worker_process = multiprocessing.Process(
                target=run_interpolation_process,
                args=(
                    filtered_points,
                    filtered_values,
                    np.arange(
                        self.domain["x_min"],
                        self.domain["x_max"] + grid_step / 2,
                        grid_step,
                    ),
                    np.arange(
                        self.domain["y_min"],
                        self.domain["y_max"] + grid_step / 2,
                        grid_step,
                    ),
                    interpolator_key,
                    name,
                    redis_config,
                    self.wavenumber_df["wavenumber"].tolist(),
                    interpolation_params,
                ),
                daemon=False,
            )

            print("\nStarting worker process...")
            print(f"Using interpolator: {interpolator_key}")
            print(f"Grid step size: {grid_step}")
            self.worker_process.start()
            print(f"Worker process started with PID: {self.worker_process.pid}")
            return True

        except Exception as e:
            print(f"Error starting interpolation: {str(e)}")
            traceback.print_exc()
            return False

    def run(self, debug=False, port=8056):
        """Run the Dash application"""
        print(f"Starting server on port {port}")
        self.app.run_server(debug=debug, port=port)


def main():
    parser = argparse.ArgumentParser(
        description="Start the Data Interpolation application"
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
        default=8066,
        help="Port to run the Dash app on (default: 8066)",
    )
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")

    args = parser.parse_args()

    app = InterpolateData(redis_host=args.redis_host, redis_port=args.redis_port)
    app.run(debug=args.debug, port=args.port)


if __name__ == "__main__":
    # This ensures proper multiprocessing behavior
    multiprocessing.freeze_support()
    main()
