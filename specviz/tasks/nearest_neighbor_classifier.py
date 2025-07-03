from typing import Dict, List, Optional, Tuple

import argparse
import pickle
import queue
import time
import traceback
from collections import defaultdict
from multiprocessing import Process, Queue

import dash
import dash_bootstrap_components as dbc
import duckdb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import redis
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from numpy.typing import NDArray
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score, brier_score_loss, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from specviz.tasks.conformal_prediction import OneClassConformalPredictor
from specviz.tasks.dbtools import (
    append_df_to_table,
    check_table_exists,
    fetch_dict_from_db,
    read_df_from_db,
    store_df_in_db,
)


class NearestNeighborClassifierApp:
    """Dashboard for training and applying nearest neighbor classifiers"""

    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379):
        self.redis_host = redis_host
        self.redis_port = redis_port

        # Initialize Redis client
        self.redis_client = redis.Redis(
            host=self.redis_host, port=self.redis_port, decode_responses=True
        )

        self.redis_status = "Not Connected"
        self.redis_status_color = "danger"
        self._connect_redis()

        # Initialize Dash app
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            suppress_callback_exceptions=True,
        )

        # Add process tracking attributes
        self.compute_process = None
        self.result_queue = None

        # Initialize results store
        self.current_figure = None
        self.needs_update = False
        self.computation_finished = False

        # Initialize training attributes
        self.training_results = None

        # Initialize results div with empty plot
        self.results_div = html.Div(
            [
                dcc.Graph(id="accuracy-plot"),
                dcc.Store(id="results-store", data=None),
                dcc.Store(id="current-metric", data="accuracy"),
            ]
        )

        self._setup_layout()
        self._setup_callbacks()

    def _setup_layout(self):
        """Set up the Dash layout"""
        self.app.layout = lambda: dbc.Container(
            [
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
                                                                    id="redis-status",
                                                                    children="Checking Redis connection...",
                                                                ),
                                                                html.Div(
                                                                    id="test-status",
                                                                    children="Test callback not triggered",
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
                # Data Loading Controls
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Data Loading"),
                                        dbc.CardBody(
                                            [
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                dbc.Button(
                                                                    "Load Data",
                                                                    id="load-data-button",
                                                                    color="primary",
                                                                    className="me-2",
                                                                )
                                                            ]
                                                        )
                                                    ]
                                                ),
                                                html.Div(
                                                    id="data-loading-status",
                                                    className="mt-3",
                                                ),
                                            ]
                                        ),
                                    ]
                                )
                            ],
                            width=12,
                        )
                    ],
                    className="mb-3",
                ),
                # Test Set Selection Section
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Test Set Selection"),
                                        dbc.CardBody(
                                            [
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                html.H6(
                                                                    "Measured Points Breakdown by Level",
                                                                    className="mb-3",
                                                                ),
                                                                html.Div(
                                                                    id="level-breakdown",
                                                                    className="mb-3",
                                                                ),
                                                                html.Hr(),
                                                                html.H6(
                                                                    "Calibration Set Selection",
                                                                    className="mb-3",
                                                                ),
                                                                dbc.Row(
                                                                    [
                                                                        dbc.Col(
                                                                            [
                                                                                dbc.Label(
                                                                                    "Minimum Level for Calibration:"
                                                                                ),
                                                                                dcc.Dropdown(
                                                                                    id="calibration-level-select",
                                                                                    options=[],
                                                                                    placeholder="Select minimum level",
                                                                                    className="mb-2",
                                                                                ),
                                                                                dbc.Label(
                                                                                    "Subsample Size:"
                                                                                ),
                                                                                dbc.Input(
                                                                                    id="subsample-size-input",
                                                                                    type="number",
                                                                                    min=1,
                                                                                    placeholder="Leave empty for all points",
                                                                                    className="mb-2",
                                                                                ),
                                                                                dbc.Button(
                                                                                    "Apply Test Set Selection",
                                                                                    id="apply-test-selection-button",
                                                                                    color="success",
                                                                                    className="me-2",
                                                                                ),
                                                                                dbc.Button(
                                                                                    "Clear Selection",
                                                                                    id="clear-test-selection-button",
                                                                                    color="secondary",
                                                                                    className="me-2",
                                                                                ),
                                                                                html.Div(
                                                                                    id="test-selection-status",
                                                                                    className="mt-2",
                                                                                ),
                                                                            ],
                                                                            width=6,
                                                                        ),
                                                                        dbc.Col(
                                                                            [
                                                                                html.H6(
                                                                                    "Current Selection Summary",
                                                                                    className="mb-2",
                                                                                ),
                                                                                html.Div(
                                                                                    id="selection-summary",
                                                                                    className="text-muted",
                                                                                ),
                                                                            ],
                                                                            width=6,
                                                                        ),
                                                                    ]
                                                                ),
                                                            ],
                                                            width=12,
                                                        )
                                                    ]
                                                )
                                            ]
                                        ),
                                    ]
                                )
                            ],
                            width=12,
                        )
                    ],
                    className="mb-3",
                ),
                # Model Configuration Section
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            "Nearest Neighbor Configuration"
                                        ),
                                        dbc.CardBody(
                                            [
                                                dbc.Row(
                                                    [
                                                        # Column 1: (now empty)
                                                        dbc.Col(
                                                            [
                                                                # (No holdout split input here)
                                                            ],
                                                            width=2,
                                                        ),
                                                        # Column 2: Weighted K-NN Parameters
                                                        dbc.Col(
                                                            [
                                                                dbc.Label(
                                                                    "Weighted K-NN Parameters"
                                                                ),
                                                                dbc.InputGroup(
                                                                    [
                                                                        dbc.InputGroupText(
                                                                            "K Min",
                                                                            style={
                                                                                "width": "70px"
                                                                            },
                                                                        ),
                                                                        dbc.Input(
                                                                            id="k-min-input",
                                                                            type="number",
                                                                            value=3,
                                                                            min=1,
                                                                            max=50,
                                                                            step=1,
                                                                            style={
                                                                                "width": "100px"
                                                                            },
                                                                        ),
                                                                    ],
                                                                    className="mb-1",
                                                                ),
                                                                dbc.InputGroup(
                                                                    [
                                                                        dbc.InputGroupText(
                                                                            "K Max",
                                                                            style={
                                                                                "width": "70px"
                                                                            },
                                                                        ),
                                                                        dbc.Input(
                                                                            id="k-max-input",
                                                                            type="number",
                                                                            value=10,
                                                                            min=1,
                                                                            max=50,
                                                                            step=1,
                                                                            style={
                                                                                "width": "100px"
                                                                            },
                                                                        ),
                                                                    ],
                                                                    className="mb-1",
                                                                ),
                                                                dbc.InputGroup(
                                                                    [
                                                                        dbc.InputGroupText(
                                                                            "K Step",
                                                                            style={
                                                                                "width": "70px"
                                                                            },
                                                                        ),
                                                                        dbc.Input(
                                                                            id="k-step-input",
                                                                            type="number",
                                                                            value=1,
                                                                            min=1,
                                                                            max=10,
                                                                            step=1,
                                                                            style={
                                                                                "width": "100px"
                                                                            },
                                                                        ),
                                                                    ],
                                                                    className="mb-1",
                                                                ),
                                                                dbc.InputGroup(
                                                                    [
                                                                        dbc.InputGroupText(
                                                                            "σ Min",
                                                                            style={
                                                                                "width": "70px"
                                                                            },
                                                                        ),
                                                                        dbc.Input(
                                                                            id="sigma-min-input",
                                                                            type="number",
                                                                            value=0.5,
                                                                            min=0.1,
                                                                            max=5.0,
                                                                            step=0.1,
                                                                            style={
                                                                                "width": "100px"
                                                                            },
                                                                        ),
                                                                    ],
                                                                    className="mb-1",
                                                                ),
                                                                dbc.InputGroup(
                                                                    [
                                                                        dbc.InputGroupText(
                                                                            "σ Max",
                                                                            style={
                                                                                "width": "70px"
                                                                            },
                                                                        ),
                                                                        dbc.Input(
                                                                            id="sigma-max-input",
                                                                            type="number",
                                                                            value=2.0,
                                                                            min=0.1,
                                                                            max=10.0,
                                                                            step=0.1,
                                                                            style={
                                                                                "width": "100px"
                                                                            },
                                                                        ),
                                                                    ],
                                                                    className="mb-1",
                                                                ),
                                                                dbc.InputGroup(
                                                                    [
                                                                        dbc.InputGroupText(
                                                                            "σ Step",
                                                                            style={
                                                                                "width": "70px"
                                                                            },
                                                                        ),
                                                                        dbc.Input(
                                                                            id="sigma-step-input",
                                                                            type="number",
                                                                            value=0.5,
                                                                            min=0.1,
                                                                            max=2.0,
                                                                            step=0.1,
                                                                            style={
                                                                                "width": "100px"
                                                                            },
                                                                        ),
                                                                    ],
                                                                    className="mb-1",
                                                                ),
                                                                dbc.InputGroup(
                                                                    [
                                                                        dbc.InputGroupText(
                                                                            "p",
                                                                            style={
                                                                                "width": "70px"
                                                                            },
                                                                        ),
                                                                        dbc.Input(
                                                                            id="p-input",
                                                                            type="number",
                                                                            value=1,
                                                                            min=1,
                                                                            max=10,
                                                                            step=1,
                                                                            style={
                                                                                "width": "100px"
                                                                            },
                                                                        ),
                                                                    ],
                                                                    className="mb-1",
                                                                ),
                                                            ],
                                                            width=6,
                                                        ),
                                                        # Column 3: Feature Space
                                                        dbc.Col(
                                                            [
                                                                dbc.Label(
                                                                    "Feature Space"
                                                                ),
                                                                html.Div(
                                                                    [
                                                                        html.Small(
                                                                            "UMAP Embedding (Fixed)",
                                                                            className="text-muted",
                                                                        ),
                                                                        html.Br(),
                                                                        html.Small(
                                                                            "Euclidean Distance (Fixed)",
                                                                            className="text-muted",
                                                                        ),
                                                                    ],
                                                                    className="mb-1",
                                                                ),
                                                            ],
                                                            width=2,
                                                        ),
                                                        # Column 4: Random State
                                                        dbc.Col(
                                                            [
                                                                dbc.Label(
                                                                    "Random State"
                                                                ),
                                                                dbc.InputGroup(
                                                                    [
                                                                        dbc.InputGroupText(
                                                                            "Seed",
                                                                            style={
                                                                                "width": "70px"
                                                                            },
                                                                        ),
                                                                        dbc.Input(
                                                                            id="random-state-input",
                                                                            type="number",
                                                                            min=0,
                                                                            step=1,
                                                                            value=42,
                                                                            style={
                                                                                "width": "100px"
                                                                            },
                                                                        ),
                                                                    ],
                                                                    className="mb-1",
                                                                ),
                                                            ],
                                                            width=2,
                                                        ),
                                                    ]
                                                ),
                                                # Parameter Summary
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                html.Div(
                                                                    id="parameter-summary",
                                                                    className="mt-3",
                                                                )
                                                            ],
                                                            width=12,
                                                        )
                                                    ]
                                                ),
                                                # Add Run and Kill buttons
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                dbc.Button(
                                                                    "Train KNN Models",
                                                                    id="train-models-button",
                                                                    color="primary",
                                                                    className="me-2",
                                                                ),
                                                                dbc.Button(
                                                                    "Kill Process",
                                                                    id="kill-button",
                                                                    color="danger",
                                                                    disabled=True,
                                                                ),
                                                                html.Div(
                                                                    id="training-status",
                                                                    className="mt-2",
                                                                ),
                                                            ],
                                                            width=12,
                                                        )
                                                    ],
                                                    className="mt-3",
                                                ),
                                            ]
                                        ),
                                    ]
                                )
                            ],
                            width=12,
                        )
                    ]
                ),
                # Results Section
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            [
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            "Model Performance Results",
                                                            width=8,
                                                        ),
                                                        dbc.Col(
                                                            [
                                                                dbc.ButtonGroup(
                                                                    [
                                                                        dbc.Button(
                                                                            "Accuracy",
                                                                            id="accuracy-toggle",
                                                                            color="primary",
                                                                            size="sm",
                                                                        ),
                                                                        dbc.Button(
                                                                            "Brier Score",
                                                                            id="brier-toggle",
                                                                            color="secondary",
                                                                            size="sm",
                                                                        ),
                                                                    ],
                                                                    id="metric-toggle-group",
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
                                                dcc.Graph(id="accuracy-plot"),
                                                dcc.Store(
                                                    id="results-store", data=None
                                                ),
                                                dcc.Store(
                                                    id="current-metric", data="accuracy"
                                                ),
                                            ]
                                        ),
                                    ]
                                )
                            ],
                            width=12,
                        )
                    ],
                    className="mt-3",
                ),
                # Model Selection Section
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Model Selection"),
                                        dbc.CardBody(
                                            [
                                                html.H5(
                                                    "Select Best K Value for Model",
                                                    className="mb-3",
                                                ),
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                dbc.Label("K Value"),
                                                                dcc.Dropdown(
                                                                    id="k-value-select",
                                                                    options=[],
                                                                    placeholder="Select K value",
                                                                ),
                                                            ],
                                                            width=6,
                                                        ),
                                                        dbc.Col(
                                                            [
                                                                dbc.Label("Model Name"),
                                                                dbc.Input(
                                                                    id="model-name-input",
                                                                    type="text",
                                                                    placeholder="Enter model name",
                                                                    className="mb-2",
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
                                                                html.Div(
                                                                    [
                                                                        dbc.Button(
                                                                            "Export Model",
                                                                            id="export-model-button",
                                                                            color="success",
                                                                            className="me-2",
                                                                            disabled=True,
                                                                        ),
                                                                    ]
                                                                ),
                                                                html.Div(
                                                                    id="export-status",
                                                                    className="mt-2",
                                                                ),
                                                                html.Div(
                                                                    id="model-details",
                                                                    className="mt-3",
                                                                ),
                                                            ],
                                                            width=12,
                                                        )
                                                    ]
                                                ),
                                            ]
                                        ),
                                    ]
                                )
                            ],
                            width=12,
                        )
                    ],
                    className="mt-3",
                ),
                # Stores
                dcc.Store(id="data-loaded-store", data=None),
                dcc.Store(id="computation-status", data="idle"),
                dcc.Store(id="test-selection-applied", data=False),
                # Intervals
                dcc.Interval(id="results-interval", interval=100, disabled=False),
                dcc.Interval(id="computation-check", interval=500),
                dcc.Interval(id="status-check", interval=1000),
                dcc.Interval(id="redis-interval", interval=5000, disabled=False),
            ],
            fluid=True,
        )

    def _setup_callbacks(self):
        """Set up the Dash callbacks"""
        print("Setting up callbacks...")

        # Simple test callback
        @self.app.callback(
            Output("test-status", "children"), Input("redis-refresh-button", "n_clicks")
        )
        def test_callback(n_clicks):
            print(f"TEST CALLBACK TRIGGERED: n_clicks={n_clicks}")
            return f"Test callback works! Clicks: {n_clicks or 0}"

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

                    # Add KNN status
                    status_lines.append("Ready for KNN training")

                    return (
                        html.Div(
                            [
                                html.Div(status_lines[0]),  # Connection status
                                html.Small(
                                    status_lines[1], className="text-muted d-block"
                                ),  # Database path
                                html.Small(
                                    status_lines[2], className="text-info d-block"
                                ),  # KNN status
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
                            "KNN status: Not ready", className="text-danger d-block"
                        ),
                    ]
                ),
                f"text-{self.redis_status_color}",
            )

        @self.app.callback(
            [
                Output("data-loading-status", "children"),
                Output("data-loading-status", "className"),
                Output("data-loaded-store", "data"),
            ],
            [Input("load-data-button", "n_clicks")],
            prevent_initial_call=True,
        )
        def load_all_data(n_clicks):
            if not n_clicks:
                return "", "", None

            try:
                db_path = self.redis_client.get("current_project")
                if not db_path:
                    return "No active project found in Redis", "text-danger", None

                conn = duckdb.connect(db_path)

                # Check all required tables exist
                required_tables = ["HCD", "measured_data", "wavenumbers", "embedding"]
                for table in required_tables:
                    if not check_table_exists(conn, table):
                        return (
                            f"Required table '{table}' not found in {db_path}",
                            "text-danger",
                            None,
                        )

                # Load all required tables using dbtools
                hcd_df = read_df_from_db(conn, "HCD")
                measured_data_df = read_df_from_db(conn, "measured_data")
                wavenumbers = read_df_from_db(conn, "wavenumbers")["wavenumber"].values
                embedding_df = read_df_from_db(conn, "embedding")

                # Check for classification column
                if "class" not in embedding_df.columns:
                    # Try to use selection column if available
                    if "selection" in embedding_df.columns:
                        print(
                            "No 'class' column found, using 'selection' column instead"
                        )
                        class_labels = embedding_df["selection"].values
                    else:
                        # Create sample classification based on spatial coordinates
                        print(
                            "No classification data found, creating sample classification based on spatial coordinates"
                        )
                        # Create a simple classification based on spatial position
                        x_coords = hcd_df["X"].values
                        y_coords = hcd_df["Y"].values

                        # Create 2 classes based on spatial position (e.g., left vs right half)
                        x_median = np.median(x_coords)
                        class_labels = (x_coords > x_median).astype(int)

                        # Add the class labels to the embedding dataframe for consistency
                        embedding_df["class"] = class_labels
                        print(
                            f"Created sample classification: {np.unique(class_labels, return_counts=True)}"
                        )
                else:
                    class_labels = embedding_df["class"].values

                # Get HCD indices from measured data
                measured_indices = measured_data_df["hcd_indx"].values

                # Extract spectral data (excluding hcd_indx column)
                spectra = measured_data_df.iloc[:, 1:].values  # Skip hcd_indx column

                # Get corresponding X, Y coordinates from HCD for our indices
                spatial_coords = hcd_df.loc[measured_indices, ["X", "Y"]]

                # Get UMAP coordinates and class labels
                latent_coords = embedding_df[["umap_x", "umap_y"]].values

                # Store as class attributes
                self.measured_indices = measured_indices
                self.spectra = spectra
                self.wavenumbers = wavenumbers
                self.spatial_coords = spatial_coords
                self.latent_coords = latent_coords
                self.embedding_df = embedding_df
                self.measured_data_df = measured_data_df
                self.class_labels = class_labels

                # Get class distribution
                unique_classes, class_counts = np.unique(
                    class_labels, return_counts=True
                )
                class_dist = dict(zip(unique_classes, class_counts))

                conn.close()

                # Compile comprehensive stats
                stats = (
                    f"Successfully loaded all data:\n\n"
                    f"Data Summary:\n"
                    f"- Number of points: {len(measured_indices)}\n"
                    f"- Spectra shape: {spectra.shape}\n"
                    f"- Wavenumber range: {wavenumbers.min():.1f} - {wavenumbers.max():.1f}\n"
                    f"- Spatial coordinates shape: {spatial_coords.shape}\n"
                    f"- UMAP embedding shape: {latent_coords.shape}\n"
                    f"- Class distribution: {class_dist}"
                )

                return stats, "text-success", True

            except Exception as e:
                print(f"Error in load_all_data: {str(e)}")
                return f"Error loading data: {str(e)}", "text-danger", None

        @self.app.callback(
            [
                Output("level-breakdown", "children"),
                Output("calibration-level-select", "options"),
            ],
            [Input("data-loaded-store", "data")],
            prevent_initial_call=True,
        )
        def update_level_breakdown(data_loaded):
            """Update the level breakdown and calibration level options when data is loaded"""
            if not data_loaded:
                return "Load data first to see level breakdown", []

            try:
                # Get level information from HCD for measured points
                db_path = self.redis_client.get("current_project")
                if not db_path:
                    return "No active project found", []

                conn = duckdb.connect(db_path)

                # Get HCD data and measured points
                hcd_df = read_df_from_db(conn, "HCD")
                measured_data_df = read_df_from_db(conn, "measured_data")
                measured_indices = measured_data_df["hcd_indx"].values

                # Get levels for measured points
                measured_levels = hcd_df.loc[measured_indices, "level"].values

                # Get class labels fresh from the database to match current data
                embedding_df = read_df_from_db(conn, "embedding")
                # The embedding dataframe contains all data, and measured_indices are HCD indices
                # We need to get the class labels for the measured points
                # The embedding dataframe index is sequential (0, 1, 2, ...), not HCD indices
                # So we get all class labels and use them directly
                measured_class_labels = embedding_df["class"].values

                # Debug: Check dimensions and data structure
                print(f"measured_levels shape: {measured_levels.shape}")
                print(f"measured_class_labels shape: {measured_class_labels.shape}")
                print(f"measured_indices shape: {measured_indices.shape}")
                print(f"embedding_df columns: {embedding_df.columns.tolist()}")
                print(
                    f"embedding_df index range: {embedding_df.index.min()} to {embedding_df.index.max()}"
                )
                print(
                    f"measured_indices range: {measured_indices.min()} to {measured_indices.max()}"
                )

                # Count points per level
                unique_levels, level_counts = np.unique(
                    measured_levels, return_counts=True
                )
                level_stats = dict(zip(unique_levels, level_counts))

                # Create breakdown display with class distribution
                breakdown_items = []
                for level in sorted(unique_levels):
                    count = level_stats[level]
                    # Get class distribution for this level
                    # Find which measured points are at this level
                    level_mask = measured_levels == level
                    print(
                        f"Level {level}: level_mask shape: {level_mask.shape}, sum: {level_mask.sum()}"
                    )
                    # Get the class labels for these measured points
                    # level_mask is a boolean mask over measured_levels
                    # Use the same mask to index into measured_class_labels
                    level_class_labels = measured_class_labels[level_mask]
                    unique_classes, class_counts = np.unique(
                        level_class_labels, return_counts=True
                    )
                    class_dist_str = ", ".join(
                        [
                            f"Class {c}: {n}"
                            for c, n in zip(unique_classes, class_counts)
                        ]
                    )
                    breakdown_items.append(
                        html.Li(f"Level {level}: {count} points ({class_dist_str})")
                    )

                breakdown = html.Ul(breakdown_items)

                # Create dropdown options for calibration level selection
                # Include all levels that have measured points
                dropdown_options = [
                    {"label": f"Level {level} ({count} points)", "value": level}
                    for level, count in sorted(level_stats.items())
                ]

                conn.close()

                return breakdown, dropdown_options

            except Exception as e:
                print(f"Error in update_level_breakdown: {str(e)}")
                return f"Error: {str(e)}", []

        @self.app.callback(
            [
                Output("selection-summary", "children"),
                Output("test-selection-status", "children"),
                Output("test-selection-status", "className"),
                Output("test-selection-applied", "data"),
            ],
            [Input("apply-test-selection-button", "n_clicks")],
            [
                State("calibration-level-select", "value"),
                State("subsample-size-input", "value"),
                State("data-loaded-store", "data"),
            ],
            prevent_initial_call=True,
        )
        def apply_test_selection(
            n_clicks, calibration_level, subsample_size, data_loaded
        ):
            """Apply the test set selection based on level and subsample size"""
            if not n_clicks or not data_loaded:
                raise PreventUpdate

            if calibration_level is None:
                return (
                    "Please select a calibration level",
                    "Please select a calibration level",
                    "text-warning",
                    False,
                )

            try:
                # Get level information from HCD for measured points
                db_path = self.redis_client.get("current_project")
                if not db_path:
                    return (
                        "No active project found",
                        "No active project found",
                        "text-danger",
                        False,
                    )

                conn = duckdb.connect(db_path)

                # Get HCD data and measured points
                hcd_df = read_df_from_db(conn, "HCD")
                measured_data_df = read_df_from_db(conn, "measured_data")
                measured_indices = measured_data_df["hcd_indx"].values

                # Get levels for measured points
                measured_levels = hcd_df.loc[measured_indices, "level"].values

                # Define training set: points below the selected calibration level
                training_mask = measured_levels < calibration_level
                training_indices = measured_indices[training_mask]
                training_levels = measured_levels[training_mask]

                # Define calibration set: points at the selected calibration level
                calibration_mask = measured_levels == calibration_level
                calibration_indices = measured_indices[calibration_mask]
                calibration_levels = measured_levels[calibration_mask]

                # Apply subsampling to calibration set if specified
                if subsample_size and subsample_size > 0:
                    if len(calibration_indices) <= subsample_size:
                        selected_calibration_indices = calibration_indices
                        remaining_calibration_indices = np.array([])
                        subsample_note = (
                            f" (all {len(calibration_indices)} points used)"
                        )
                    else:
                        # Random subsampling (calibration set is only one level, so no stratification needed)
                        selected_calibration_indices = np.random.choice(
                            calibration_indices, size=subsample_size, replace=False
                        )
                        remaining_calibration_indices = np.setdiff1d(
                            calibration_indices, selected_calibration_indices
                        )
                        subsample_note = f" (subsampled from {len(calibration_indices)} to {len(selected_calibration_indices)} points, {len(remaining_calibration_indices)} points added to training)"
                else:
                    selected_calibration_indices = calibration_indices
                    remaining_calibration_indices = np.array([])
                    subsample_note = ""

                # Update class attributes with training data (including remaining calibration points)
                if len(remaining_calibration_indices) > 0:
                    # Add remaining calibration points to training set
                    combined_training_indices = np.concatenate(
                        [training_indices, remaining_calibration_indices]
                    )
                else:
                    combined_training_indices = training_indices

                self.training_indices = combined_training_indices

                # Get corresponding data for training indices
                training_mask = np.isin(measured_indices, combined_training_indices)
                self.training_spectra = self.spectra[training_mask]
                self.training_spatial_coords = self.spatial_coords.iloc[training_mask]
                self.training_latent_coords = self.latent_coords[training_mask]
                self.training_class_labels = self.class_labels[training_mask]

                # Update class attributes with calibration data
                self.calibration_indices = selected_calibration_indices

                # Get corresponding data for calibration indices
                calibration_mask = np.isin(
                    measured_indices, selected_calibration_indices
                )
                self.calibration_spectra = self.spectra[calibration_mask]
                self.calibration_spatial_coords = self.spatial_coords.iloc[
                    calibration_mask
                ]
                self.calibration_latent_coords = self.latent_coords[calibration_mask]
                self.calibration_class_labels = self.class_labels[calibration_mask]

                # Create summary
                training_levels = hcd_df.loc[combined_training_indices, "level"].values
                unique_training_levels, training_level_counts = np.unique(
                    training_levels, return_counts=True
                )

                calibration_levels = hcd_df.loc[
                    selected_calibration_indices, "level"
                ].values
                unique_calibration_levels, calibration_level_counts = np.unique(
                    calibration_levels, return_counts=True
                )

                training_items = []
                for level in sorted(unique_training_levels):
                    count = training_level_counts[unique_training_levels == level][0]
                    training_items.append(f"Level {level}: {count} points")

                calibration_items = []
                for level in sorted(unique_calibration_levels):
                    count = calibration_level_counts[
                        unique_calibration_levels == level
                    ][0]
                    calibration_items.append(f"Level {level}: {count} points")

                summary = html.Div(
                    [
                        html.Strong(
                            f"Training Set: {len(combined_training_indices)} points"
                        ),
                        html.Br(),
                        html.Small("Training: " + ", ".join(training_items)),
                        html.Br(),
                        html.Strong(
                            f"Calibration Set: {len(selected_calibration_indices)} points{subsample_note}",
                            className="text-info",
                        ),
                        html.Br(),
                        html.Small(
                            "Calibration: " + ", ".join(calibration_items),
                            className="text-info",
                        ),
                    ]
                )

                status = f"Successfully selected {len(combined_training_indices)} training points (level <{calibration_level} + remaining calibration) and {len(selected_calibration_indices)} calibration points (level {calibration_level})"

                conn.close()

                return summary, status, "text-success", True

            except Exception as e:
                print(f"Error in apply_test_selection: {str(e)}")
                return f"Error: {str(e)}", f"Error: {str(e)}", "text-danger", False

        @self.app.callback(
            [
                Output("selection-summary", "children", allow_duplicate=True),
                Output("test-selection-status", "children", allow_duplicate=True),
                Output("test-selection-status", "className", allow_duplicate=True),
                Output("test-selection-applied", "data", allow_duplicate=True),
            ],
            [Input("clear-test-selection-button", "n_clicks")],
            prevent_initial_call=True,
        )
        def clear_test_selection(n_clicks):
            """Clear the test set selection and return to using full dataset"""
            if not n_clicks:
                raise PreventUpdate

            # Remove calibration data attributes
            if hasattr(self, "calibration_indices"):
                del self.calibration_indices
            if hasattr(self, "calibration_spectra"):
                del self.calibration_spectra
            if hasattr(self, "calibration_spatial_coords"):
                del self.calibration_spatial_coords
            if hasattr(self, "calibration_latent_coords"):
                del self.calibration_latent_coords
            if hasattr(self, "calibration_class_labels"):
                del self.calibration_class_labels

            # Remove training data attributes
            if hasattr(self, "training_indices"):
                del self.training_indices
            if hasattr(self, "training_spectra"):
                del self.training_spectra
            if hasattr(self, "training_spatial_coords"):
                del self.training_spatial_coords
            if hasattr(self, "training_latent_coords"):
                del self.training_latent_coords
            if hasattr(self, "training_class_labels"):
                del self.training_class_labels

            return (
                "No selection applied",
                "Test selection cleared - using full dataset",
                "text-info",
                False,
            )

        @self.app.callback(
            Output("parameter-summary", "children"),
            [
                Input("k-min-input", "value"),
                Input("k-max-input", "value"),
                Input("k-step-input", "value"),
                Input("sigma-min-input", "value"),
                Input("sigma-max-input", "value"),
                Input("sigma-step-input", "value"),
                Input("p-input", "value"),
                Input("random-state-input", "value"),
                Input("data-loaded-store", "data"),
                Input("test-selection-applied", "data"),
            ],
        )
        def update_parameter_summary(
            k_min,
            k_max,
            k_step,
            sigma_min,
            sigma_max,
            sigma_step,
            p,
            random_state,
            data_loaded,
            test_selection_applied,
        ):
            """Update the parameter summary text"""
            if not data_loaded:
                return "Load data first to see configuration"
            # Validate numeric inputs
            if any(
                v is None
                for v in [k_min, k_max, k_step, sigma_min, sigma_max, sigma_step, p]
            ):
                return "Error: Missing required parameters"
            # Determine which dataset will be used
            if hasattr(self, "calibration_latent_coords") and hasattr(
                self, "calibration_class_labels"
            ):
                dataset_info = (
                    f"Calibration dataset ({len(self.calibration_class_labels)} points)"
                )
            else:
                dataset_info = f"Full dataset ({len(self.class_labels)} points)"
            summary = [
                html.H6("Current Configuration:", className="mb-2"),
                html.Ul(
                    [
                        html.Li(f"Dataset: {dataset_info}"),
                        html.Li(f"K Range: {k_min} to {k_max} (step {k_step})"),
                        html.Li(
                            f"σ Range: {sigma_min} to {sigma_max} (step {sigma_step})"
                        ),
                        html.Li(f"p: {p}"),
                        html.Li(f"Random State: {random_state}"),
                        html.Li(f"Feature Space: UMAP Embedding (Fixed)"),
                        html.Li(f"Distance Metric: Euclidean (Fixed)"),
                        html.Li(
                            f"Total Combinations: {(k_max - k_min + 1) // k_step * int((sigma_max - sigma_min) / sigma_step + 1)}"
                        ),
                    ]
                ),
            ]
            return summary

        @self.app.callback(
            [
                Output("training-status", "children"),
                Output("kill-button", "disabled"),
                Output("train-models-button", "color"),
                Output("computation-status", "data"),
            ],
            [
                Input("train-models-button", "n_clicks"),
                Input("kill-button", "n_clicks"),
            ],
            [
                State("k-min-input", "value"),
                State("k-max-input", "value"),
                State("k-step-input", "value"),
                State("sigma-min-input", "value"),
                State("sigma-max-input", "value"),
                State("sigma-step-input", "value"),
                State("p-input", "value"),
                State("random-state-input", "value"),
                State("computation-status", "data"),
            ],
            prevent_initial_call=True,
        )
        def handle_training_controls(
            train_clicks,
            kill_clicks,
            k_min,
            k_max,
            k_step,
            sigma_min,
            sigma_max,
            sigma_step,
            p,
            random_state,
            status,
        ):
            if not hasattr(self, "spatial_coords") or not hasattr(self, "class_labels"):
                return "Please load data first", True, "danger", "idle"
            ctx = dash.callback_context
            if not ctx.triggered:
                raise PreventUpdate
            trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
            if trigger_id == "kill-button" and kill_clicks:
                if hasattr(self, "compute_process") and self.compute_process:
                    self.compute_process.terminate()
                    self.compute_process = None
                    self.result_queue = None
                    return "Process killed", True, "primary", "idle"
                return "No process to kill", True, "primary", "idle"
            if status == "computing":
                raise PreventUpdate
            # Clear previous results when starting new training
            if hasattr(self, "training_results"):
                del self.training_results
            # Check if training and calibration data is available, otherwise use full dataset
            if hasattr(self, "training_spatial_coords") and hasattr(
                self, "training_class_labels"
            ):
                # Use training data for model training
                spatial_coords = self.training_spatial_coords.values
                spectra = self.training_spectra
                class_labels = self.training_class_labels
                data_source = "training"

                # Use calibration set for conformal calibration
                if hasattr(self, "calibration_spatial_coords") and hasattr(
                    self, "calibration_class_labels"
                ):
                    calibration_spatial_coords = self.calibration_spatial_coords.values
                    calibration_spectra = self.calibration_spectra
                    calibration_class_labels = self.calibration_class_labels
                    use_predefined_calibration = True
                else:
                    # Fallback to random split if no calibration set defined
                    calibration_spatial_coords = None
                    calibration_spectra = None
                    calibration_class_labels = None
                    use_predefined_calibration = False
            else:
                # Use full dataset with random split
                spatial_coords = self.spatial_coords.values
                spectra = self.spectra
                class_labels = self.class_labels
                data_source = "full"
                calibration_spatial_coords = None
                calibration_spectra = None
                calibration_class_labels = None
                use_predefined_calibration = False

            # Prepare parameters for training
            training_params = {
                "k_min": k_min,
                "k_max": k_max,
                "k_step": k_step,
                "sigma_min": sigma_min,
                "sigma_max": sigma_max,
                "sigma_step": sigma_step,
                "p": p,
                "holdout_size": 0.2,  # Only used if no predefined calibration
                "random_state": random_state,
                "spatial_coords": spatial_coords,
                "spectra": spectra,
                "class_labels": class_labels,
                "calibration_spatial_coords": calibration_spatial_coords,
                "calibration_spectra": calibration_spectra,
                "calibration_class_labels": calibration_class_labels,
                "use_predefined_calibration": use_predefined_calibration,
                "data_source": data_source,
            }
            # Start the computation process
            data_info = f"using {data_source} dataset ({len(class_labels)} points)"
            print(f"\nStarting Weighted KNN training process... {data_info}")
            self.result_queue = Queue()
            self.compute_process = Process(
                target=run_knn_training_process,
                args=(training_params, self.result_queue),
            )
            self.compute_process.start()
            print(f"Process started with PID: {self.compute_process.pid}")

            return (
                f"Training started (PID: {self.compute_process.pid}) - {data_info}",
                False,
                "warning",
                "computing",
            )

        @self.app.callback(
            [
                Output("accuracy-plot", "figure"),
                Output("results-store", "data"),
                Output("train-models-button", "children"),
                Output("train-models-button", "disabled"),
                Output("train-models-button", "color", allow_duplicate=True),
                Output("computation-status", "data", allow_duplicate=True),
                Output("kill-button", "disabled", allow_duplicate=True),
                Output("training-status", "children", allow_duplicate=True),
                Output("k-value-select", "options"),
                Output("current-metric", "data"),
            ],
            [
                Input("results-interval", "n_intervals"),
                Input("accuracy-toggle", "n_clicks"),
                Input("brier-toggle", "n_clicks"),
            ],
            [State("computation-status", "data"), State("current-metric", "data")],
            prevent_initial_call=True,
        )
        def update_display(
            n_intervals, acc_clicks, brier_clicks, status, current_metric
        ):
            ctx = dash.callback_context
            trigger_id = (
                ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None
            )

            # Handle metric toggle
            if trigger_id in ["accuracy-toggle", "brier-toggle"]:
                if hasattr(self, "training_results") and self.training_results:
                    metric = "accuracy" if trigger_id == "accuracy-toggle" else "brier"
                    fig = create_accuracy_plot(self.training_results, metric)
                    return [fig] + [dash.no_update] * 9
                return [dash.no_update] * 10

            # Handle training completion
            if status != "computing":
                raise PreventUpdate

            # Check if we have the necessary attributes for training
            if not hasattr(self, "result_queue") or self.result_queue is None:
                # This is normal when training hasn't started yet
                return [dash.no_update] * 10

            if not hasattr(self, "compute_process") or self.compute_process is None:
                # This is normal when training hasn't started yet
                return [dash.no_update] * 10

            if not self.compute_process.is_alive():
                print("Process has completed or terminated")
                self.compute_process = None
                self.result_queue = None
                return [dash.no_update] * 10

            try:
                try:
                    result = self.result_queue.get_nowait()

                    if isinstance(result, dict) and "error" in result:
                        print(f"Error in training: {result['error']}")
                        return [dash.no_update] * 8 + [[]] + [dash.no_update]

                    training_results = result
                    print(f"Processing training results")

                    self.training_results = training_results

                    # Create figure with current metric
                    fig = create_accuracy_plot(
                        training_results, current_metric or "accuracy"
                    )

                    # Get K and sigma combinations for dropdown
                    result_keys = sorted(list(training_results.keys()))
                    k_options = [
                        {
                            "label": f'K={result["k"]}, σ={result["sigma"]:.2f}, p={result["p"]}',
                            "value": key,
                        }
                        for key, result in [
                            (k, training_results[k]) for k in result_keys
                        ]
                    ]

                    # Clean up
                    self.compute_process = None
                    self.result_queue = None

                    return [
                        fig,  # figure
                        str(time.time()),  # store data
                        "Train KNN Models",  # button text
                        False,  # button disabled
                        "primary",  # button color
                        "idle",  # computation status
                        True,  # kill button disabled
                        "Training completed successfully",  # status text
                        k_options,  # k value options
                        current_metric or "accuracy",  # current metric
                    ]

                except queue.Empty:
                    return [dash.no_update] * 10

            except Exception as e:
                print(f"Error in update_display: {str(e)}")
                self.compute_process = None
                self.result_queue = None
                return [
                    dash.no_update,  # figure
                    dash.no_update,  # store data
                    "Train KNN Models",  # button text
                    False,  # button disabled
                    "primary",  # button color
                    "idle",  # computation status
                    True,  # kill button disabled
                    f"Error: {str(e)}",  # status text
                    [],  # k options
                    current_metric or "accuracy",  # current metric
                ]

        @self.app.callback(
            [
                Output("export-model-button", "disabled", allow_duplicate=True),
                Output("model-details", "children"),
            ],
            [Input("k-value-select", "value")],
            prevent_initial_call=True,
        )
        def handle_model_selection(result_key):
            if result_key is None:
                return True, None

            if (
                not hasattr(self, "training_results")
                or result_key not in self.training_results
            ):
                return True, "Selected model not found in training results"

            result = self.training_results[result_key]

            # Compute conformal thresholds for different coverage levels
            error_rates = [0.01, 0.05, 0.10, 0.20, 0.25]
            thresholds = []

            try:
                print("Computing conformal thresholds for selected model...")
                print(f"Calibration data shape: {result['calibration_proba'].shape}")
                print(f"Calibration labels shape: {result['y_calibration'].shape}")

                # Check class distribution in calibration set
                unique_labels, label_counts = np.unique(
                    result["y_calibration"], return_counts=True
                )
                print(
                    f"Calibration class distribution: {dict(zip(unique_labels, label_counts))}"
                )

                # Check if we have enough samples of each class
                n_classes = len(unique_labels)
                min_samples_per_class = min(label_counts)
                print(f"Minimum samples per class: {min_samples_per_class}")

                if min_samples_per_class < 2:
                    print("Warning: Very few samples per class in holdout set")

                # Create dot plot for class 1 calibration scores
                class_1_mask = result["y_calibration"] == 1
                class_1_probs = result["calibration_proba"][class_1_mask, 1]
                class_0_mask = result["y_calibration"] == 0
                class_0_probs = result["calibration_proba"][class_0_mask, 1]

                print(f"\nClass 1 calibration scores ({len(class_1_probs)} points):")
                print(f"  Min: {class_1_probs.min():.4f}")
                print(f"  Max: {class_1_probs.max():.4f}")
                print(f"  Mean: {class_1_probs.mean():.4f}")
                print(f"  Std: {class_1_probs.std():.4f}")
                print(
                    f"  Percentiles: 25%={np.percentile(class_1_probs, 25):.4f}, 50%={np.percentile(class_1_probs, 50):.4f}, 75%={np.percentile(class_1_probs, 75):.4f}, 95%={np.percentile(class_1_probs, 95):.4f}, 99%={np.percentile(class_1_probs, 99):.4f}"
                )

                print(f"\nClass 0 calibration scores ({len(class_0_probs)} points):")
                print(f"  Min: {class_0_probs.min():.4f}")
                print(f"  Max: {class_0_probs.max():.4f}")
                print(f"  Mean: {class_0_probs.mean():.4f}")
                print(f"  Std: {class_0_probs.std():.4f}")

                # Create simple ASCII dot plot
                print(f"\nClass 1 probability distribution (dot plot):")
                print("0.0" + "=" * 50 + "1.0")
                for i, prob in enumerate(sorted(class_1_probs)):
                    pos = int(prob * 50)
                    marker = (
                        "●" if i < len(class_1_probs) - 1 else "●"
                    )  # Use filled circle
                    print(" " * pos + marker)
                print("0.0" + "=" * 50 + "1.0")

                # Print raw scores as lists for copy-paste verification
                print(
                    f"\nClass 1 scores (sorted): {[float(x) for x in sorted(class_1_probs)]}"
                )
                print(
                    f"Class 0 scores (sorted): {[float(x) for x in sorted(class_0_probs)]}"
                )

                for alpha in error_rates:
                    confidence_level = 1 - alpha
                    print(
                        f"  Using confidence_level={confidence_level:.3f} for alpha={alpha:.2f}"
                    )
                    try:
                        predictor = OneClassConformalPredictor(
                            confidence_level=confidence_level
                        )
                        predictor.calibrate(
                            result["calibration_proba"], result["y_calibration"]
                        )
                        # Get threshold for class 1 (or all classes)
                        thresholds_dict = predictor.get_thresholds()
                        print(f"  All thresholds: {thresholds_dict}")
                        class_threshold = thresholds_dict.get(1, None)
                        print(f"  Class 1 threshold: {class_threshold}")
                    except Exception as e:
                        print(f"  Error in conformal prediction: {str(e)}")
                        import traceback

                        traceback.print_exc()
                        class_threshold = None

                    # Handle case where threshold is None (likely due to insufficient class samples)
                    if class_threshold is None or np.isnan(class_threshold):
                        # If no samples of class 1, threshold should be 0 (never predict class 1)
                        # If all samples are class 1, threshold should be 1 (always predict class 1)
                        if 1 not in unique_labels:
                            class_threshold = 0.0  # No class 1 samples
                            print(
                                f"  α={alpha:.2f}: threshold=0.0 (no class 1 samples)"
                            )
                        elif len(unique_labels) == 1 and unique_labels[0] == 1:
                            class_threshold = 1.0  # All samples are class 1
                            print(
                                f"  α={alpha:.2f}: threshold=1.0 (all samples are class 1)"
                            )
                        else:
                            class_threshold = 0.5  # Fallback
                            print(f"  α={alpha:.2f}: threshold=0.5 (fallback)")
                    else:
                        # Ensure threshold is a valid number
                        class_threshold = float(class_threshold)
                        print(f"  α={alpha:.2f}: threshold={class_threshold:.4f}")

                    # Store the actual threshold value (no inversion needed)
                    thresholds.append(class_threshold)
                    print(f"  α={alpha:.2f}: storing threshold={class_threshold:.4f}")
            except Exception as e:
                print(f"Error computing conformal thresholds: {str(e)}")
                thresholds = [None] * len(error_rates)

            # Create model details with conformal thresholds table
            details = html.Div(
                [
                    html.H6(
                        f"Model Details: K={result['k']}, σ={result['sigma']:.2f}, p={result['p']}"
                    ),
                    html.Ul(
                        [
                            html.Li(
                                f"Calibration Accuracy: {result['calibration_accuracy']:.4f}"
                            ),
                            html.Li(f"Brier Score: {result['brier_score']:.4f}"),
                            html.Li(f"Feature Space: Spatial Coordinates (X, Y)"),
                            html.Li(f"Distance Metric: Euclidean"),
                            html.Li(f"K: {result['k']}"),
                            html.Li(f"σ: {result['sigma']:.2f}"),
                            html.Li(f"p: {result['p']}"),
                        ]
                    ),
                    # Conformal thresholds table
                    html.H6("Class 1 Conformal Thresholds:", className="mt-3"),
                    dbc.Table(
                        # Header
                        [
                            html.Thead(
                                html.Tr(
                                    [
                                        html.Th("Error Rate (α)"),
                                        html.Th("Coverage (1-α)"),
                                        html.Th("Threshold"),
                                    ]
                                )
                            )
                        ]
                        +
                        # Body
                        [
                            html.Tbody(
                                [
                                    html.Tr(
                                        [
                                            html.Td(f"{alpha:.2%}"),
                                            html.Td(f"{(1-alpha):.2%}"),
                                            html.Td(
                                                f"{threshold:.4f}"
                                                if threshold is not None
                                                and not np.isnan(threshold)
                                                else "N/A"
                                            ),
                                        ]
                                    )
                                    for alpha, threshold in zip(error_rates, thresholds)
                                ]
                            )
                        ],
                        bordered=True,
                        hover=True,
                        striped=True,
                        className="mt-3",
                    ),
                ]
            )

            return False, details

        @self.app.callback(
            [
                Output("export-status", "children"),
                Output("export-model-button", "disabled"),
            ],
            [Input("export-model-button", "n_clicks")],
            [State("model-name-input", "value"), State("k-value-select", "value")],
            prevent_initial_call=True,
        )
        def handle_model_export(n_clicks, model_name, result_key):
            if not n_clicks:
                raise PreventUpdate

            if not model_name:
                return "Please enter a model name", True

            # Check if name contains only valid characters
            if not model_name.replace("_", "").isalnum():
                return (
                    "Model name can only contain letters, numbers, and underscores",
                    True,
                )

            try:
                # Get the database path from Redis
                db_path = self.redis_client.get("current_project")
                if not db_path:
                    return "No active project found in Redis", True

                # Get the selected model result
                result = self.training_results[result_key]

                # Connect to database
                conn = duckdb.connect(db_path)

                try:
                    # Check if table already exists
                    table_name = f"knn_model_{model_name}"
                    if check_table_exists(conn, table_name):
                        return f"Error: Table '{table_name}' already exists", True

                    # Get all HCD indices and compute probabilities for ALL HCD points
                    hcd_df = read_df_from_db(conn, "HCD")
                    all_hcd_indices = (
                        hcd_df.index.values
                    )  # These are the actual HCD indices
                    print(f"HCD table has {len(hcd_df)} rows")

                    # Get spatial coordinates for ALL HCD points
                    all_hcd_spatial_coords = hcd_df[["X", "Y"]].values
                    print(
                        f"Using spatial coordinates for all {len(all_hcd_spatial_coords)} HCD points"
                    )

                    # Get the trained model
                    model = result["model"]

                    # Compute probabilities for ALL HCD points using the trained model
                    print(
                        f"Computing probabilities for {len(all_hcd_spatial_coords)} HCD points..."
                    )
                    all_probs = model.predict_proba(all_hcd_spatial_coords)
                    n_classes = all_probs.shape[1]
                    print(f"Probability matrix shape: {all_probs.shape}")

                    # Create export dataframe: hcd_indx, value (prob_class_1)
                    # Match the ensemble classifier format
                    export_df = pd.DataFrame(
                        {
                            "hcd_indx": all_hcd_indices,
                            "value": all_probs[
                                :, 1
                            ],  # Use class 1 probability as the 'value'
                        }
                    )

                    # Export predictions to database
                    store_df_in_db(conn, export_df, table_name)

                    # Conformal calibration on holdout set
                    error_rates = [0.01, 0.05, 0.10, 0.20, 0.25]
                    thresholds = []

                    # Check class distribution in calibration set
                    unique_labels, label_counts = np.unique(
                        result["y_calibration"], return_counts=True
                    )
                    print(
                        f"Export: Calibration class distribution: {dict(zip(unique_labels, label_counts))}"
                    )

                    # Create dot plot for class 1 calibration scores (export version)
                    class_1_mask = result["y_calibration"] == 1
                    class_1_probs = result["calibration_proba"][class_1_mask, 1]
                    class_0_mask = result["y_calibration"] == 0
                    class_0_probs = result["calibration_proba"][class_0_mask, 1]

                    print(
                        f"\nExport: Class 1 calibration scores ({len(class_1_probs)} points):"
                    )
                    print(f"  Min: {class_1_probs.min():.4f}")
                    print(f"  Max: {class_1_probs.max():.4f}")
                    print(f"  Mean: {class_1_probs.mean():.4f}")
                    print(f"  Std: {class_1_probs.std():.4f}")
                    print(
                        f"  Percentiles: 25%={np.percentile(class_1_probs, 25):.4f}, 50%={np.percentile(class_1_probs, 50):.4f}, 75%={np.percentile(class_1_probs, 75):.4f}, 95%={np.percentile(class_1_probs, 95):.4f}, 99%={np.percentile(class_1_probs, 99):.4f}"
                    )

                    print(
                        f"\nExport: Class 0 calibration scores ({len(class_0_probs)} points):"
                    )
                    print(f"  Min: {class_0_probs.min():.4f}")
                    print(f"  Max: {class_0_probs.max():.4f}")
                    print(f"  Mean: {class_0_probs.mean():.4f}")
                    print(f"  Std: {class_0_probs.std():.4f}")

                    # Create simple ASCII dot plot
                    print(f"\nExport: Class 1 probability distribution (dot plot):")
                    print("0.0" + "=" * 50 + "1.0")
                    for i, prob in enumerate(sorted(class_1_probs)):
                        pos = int(prob * 50)
                        marker = (
                            "●" if i < len(class_1_probs) - 1 else "●"
                        )  # Use filled circle
                        print(" " * pos + marker)
                    print("0.0" + "=" * 50 + "1.0")

                    # Print raw scores as lists for copy-paste verification
                    print(
                        f"\nExport: Class 1 scores (sorted): {[float(x) for x in sorted(class_1_probs)]}"
                    )
                    print(
                        f"\nExport: Class 0 scores (sorted): {[float(x) for x in sorted(class_0_probs)]}"
                    )

                    for alpha in error_rates:
                        confidence_level = 1 - alpha
                        print(
                            f"Export: Using confidence_level={confidence_level:.3f} for alpha={alpha:.2f}"
                        )
                        try:
                            predictor = OneClassConformalPredictor(
                                confidence_level=confidence_level
                            )
                            predictor.calibrate(
                                result["calibration_proba"], result["y_calibration"]
                            )
                            # Get threshold for class 1 (or all classes)
                            thresholds_dict = predictor.get_thresholds()
                            class_threshold = thresholds_dict.get(1, None)
                            print(f"Export: Class 1 threshold: {class_threshold}")
                        except Exception as e:
                            print(f"Export: Error in conformal prediction: {str(e)}")
                            import traceback

                            traceback.print_exc()
                            class_threshold = None

                        # Handle case where threshold is None (likely due to insufficient class samples)
                        if class_threshold is None or np.isnan(class_threshold):
                            # If no samples of class 1, threshold should be 0 (never predict class 1)
                            # If all samples are class 1, threshold should be 1 (always predict class 1)
                            if 1 not in unique_labels:
                                class_threshold = 0.0  # No class 1 samples
                                print(
                                    f"Export: α={alpha:.2f}: threshold=0.0 (no class 1 samples)"
                                )
                            elif len(unique_labels) == 1 and unique_labels[0] == 1:
                                class_threshold = 1.0  # All samples are class 1
                                print(
                                    f"Export: α={alpha:.2f}: threshold=1.0 (all samples are class 1)"
                                )
                            else:
                                class_threshold = 0.5  # Fallback
                                print(
                                    f"Export: α={alpha:.2f}: threshold=0.5 (fallback)"
                                )
                        else:
                            # Ensure threshold is a valid number
                            class_threshold = float(class_threshold)
                            print(
                                f"Export: α={alpha:.2f}: threshold={class_threshold:.4f}"
                            )

                        # Store the actual threshold value (no inversion needed)
                        thresholds.append(class_threshold)
                        print(
                            f"Export: α={alpha:.2f}: storing threshold={class_threshold:.4f}"
                        )

                    # Save thresholds to conformal_thresholds table
                    thresholds_df = pd.DataFrame(
                        {
                            "model_name": table_name,
                            "threshold_0.01": thresholds[0],
                            "threshold_0.05": thresholds[1],
                            "threshold_0.10": thresholds[2],
                            "threshold_0.20": thresholds[3],
                            "threshold_0.25": thresholds[4],
                        },
                        index=[0],
                    )
                    if not check_table_exists(conn, "conformal_thresholds"):
                        store_df_in_db(conn, thresholds_df, "conformal_thresholds")
                    else:
                        append_df_to_table(conn, thresholds_df, "conformal_thresholds")

                    # Register model in models table
                    if check_table_exists(conn, "models"):
                        models_df = read_df_from_db(conn, "models")
                        if table_name not in models_df["table_name"].values:
                            new_model = pd.DataFrame({"table_name": [table_name]})
                            models_df = pd.concat(
                                [models_df, new_model], ignore_index=True
                            )
                    else:
                        models_df = pd.DataFrame({"table_name": [table_name]})
                    store_df_in_db(
                        conn, models_df, "models", if_exists="replace", index=False
                    )

                    return (
                        f"Successfully exported KNN model to table '{table_name}', updated conformal thresholds, "
                        "and registered in models table"
                    ), True
                finally:
                    conn.close()
            except Exception as e:
                print(f"Error in handle_model_export: {str(e)}")
                import traceback

                traceback.print_exc()
                return f"Error exporting model: {str(e)}", True

        @self.app.callback(
            [Output("accuracy-toggle", "color"), Output("brier-toggle", "color")],
            [Input("current-metric", "data")],
        )
        def update_toggle_colors(current_metric):
            if current_metric == "accuracy":
                return "primary", "secondary"
            else:  # brier
                return "secondary", "primary"

    def _connect_redis(self):
        """Establish connection to Redis and check status"""
        try:
            print(
                f"Attempting to connect to Redis at {self.redis_host}:{self.redis_port}"
            )
            if self.redis_client.ping():
                self.redis_status = "Connected"
                self.redis_status_color = "success"
                print(
                    f"Successfully connected to Redis at {self.redis_host}:{self.redis_port}"
                )
            else:
                self.redis_status = "Connection Failed"
                self.redis_status_color = "danger"
                print("Redis ping failed")
        except Exception as e:
            self.redis_status = f"Connection Error: {str(e)}"
            self.redis_status_color = "danger"
            print(f"Redis connection error: {str(e)}")

    def run(self, debug: bool = False, port: int = 8062):
        """Run the Dash server"""
        try:
            import logging

            log = logging.getLogger("werkzeug")
            log.setLevel(logging.ERROR)
            self.app.run_server(debug=debug, port=port)

        finally:
            # Clean up connections
            if hasattr(self, "redis_client"):
                self.redis_client.close()


def run_knn_training_process(params, result_queue):
    """Run KNN training in a separate process"""
    try:
        print("\n=== Starting Weighted KNN Training ===")
        print(f"Data source: {params.get('data_source', 'full')} dataset")
        print(f"Total points: {len(params['class_labels'])}")
        unique_labels, label_counts = np.unique(
            params["class_labels"], return_counts=True
        )
        print(
            f"Class distribution in training data: {dict(zip(unique_labels, label_counts))}"
        )
        print(f"K Min: {params['k_min']}")
        print(f"K Max: {params['k_max']}")
        print(f"K Step: {params['k_step']}")
        print(f"σ Min: {params['sigma_min']}")
        print(f"σ Max: {params['sigma_max']}")
        print(f"σ Step: {params['sigma_step']}")
        print(f"p: {params['p']}")
        print(f"Holdout size: {params['holdout_size']}")
        print(f"Feature space: UMAP Embedding")
        print(f"Distance metric: Euclidean")
        print(f"Normalize features: False (never normalized)")

        # Use spatial coordinates as features (never normalized)
        X = params["spatial_coords"]
        y = params["class_labels"]
        random_state = params["random_state"]
        use_predefined_calibration = params.get("use_predefined_calibration", False)

        if use_predefined_calibration:
            # Use predefined calibration set from level selection
            X_train = X  # Use all training data for model training
            y_train = y
            X_calibration = params["calibration_spatial_coords"]
            y_calibration = params["calibration_class_labels"]
            print(f"Using predefined calibration set: {len(X_calibration)} points")
        else:
            # Fallback to random split
            holdout_size = params["holdout_size"]
            from sklearn.model_selection import train_test_split

            X_train, X_calibration, y_train, y_calibration = train_test_split(
                X, y, test_size=holdout_size, random_state=random_state, stratify=y
            )
            print(f"Using random calibration split: {len(X_calibration)} points")

        # Debug: Check class distribution in train and calibration sets
        train_unique, train_counts = np.unique(y_train, return_counts=True)
        cal_unique, cal_counts = np.unique(y_calibration, return_counts=True)
        print(
            f"Training set class distribution: {dict(zip(train_unique, train_counts))}"
        )
        print(
            f"Calibration set class distribution: {dict(zip(cal_unique, cal_counts))}"
        )

        results = {}

        # For inference on all HCD points, we need all HCD spatial coords
        # We'll assume params['all_hcd_spatial_coords'] and params['all_hcd_indices'] are provided
        # If not, fallback to using X (calibration set only)
        all_hcd_spatial_coords = params.get("all_hcd_spatial_coords", X)
        all_hcd_indices = params.get("all_hcd_indices", np.arange(X.shape[0]))

        # Generate parameter ranges
        k_values = range(params["k_min"], params["k_max"] + 1, params["k_step"])
        sigma_values = np.arange(
            params["sigma_min"],
            params["sigma_max"] + params["sigma_step"],
            params["sigma_step"],
        )

        total_combinations = len(k_values) * len(sigma_values)
        current_combination = 0

        for k in k_values:
            for sigma in sigma_values:
                current_combination += 1
                print(
                    f"Training Weighted KNN {current_combination}/{total_combinations}: k={k}, σ={sigma:.2f}, p={params['p']}"
                )

                # Create and train model
                model = WeightedKNNClassifier(k=k, sigma=sigma, p=params["p"])
                model.fit(X_train, y_train)

                # Evaluate on calibration set
                y_calibration_proba = model.predict_proba(X_calibration)
                y_calibration_pred = np.argmax(y_calibration_proba, axis=1)
                from sklearn.metrics import accuracy_score, brier_score_loss

                calibration_accuracy = accuracy_score(y_calibration, y_calibration_pred)
                brier_score = (
                    brier_score_loss(y_calibration, y_calibration_proba[:, 1])
                    if y_calibration_proba.shape[1] > 1
                    else 0.0
                )

                # Inference on ALL HCD points
                all_probs = model.predict_proba(all_hcd_spatial_coords)

                # Store everything needed for export and conformal calibration
                result_key = f"k{k}_sigma{sigma:.2f}_p{params['p']}"
                results[result_key] = {
                    "model": model,
                    "k": k,
                    "sigma": sigma,
                    "p": params["p"],
                    "calibration_accuracy": calibration_accuracy,
                    "brier_score": brier_score,
                    "X_train": X_train,
                    "y_train": y_train,
                    "X_calibration": X_calibration,
                    "y_calibration": y_calibration,
                    "calibration_proba": y_calibration_proba,
                    "all_hcd_indices": all_hcd_indices,
                    "all_hcd_spatial_coords": all_hcd_spatial_coords,
                    "all_probs": all_probs,
                }
                print(
                    f"  K={k}, σ={sigma:.2f}, p={params['p']}: Calibration Acc={calibration_accuracy:.4f}, Brier={brier_score:.4f}"
                )

        print("\n=== Training Complete ===")
        result_queue.put(results)

    except Exception as e:
        print(f"\nError in KNN training: {str(e)}")
        result_queue.put({"error": str(e)})


def create_accuracy_plot(training_results, metric="accuracy"):
    """Create a plot of accuracies or Brier scores for different K and sigma combinations"""
    # Extract K and sigma values from result keys
    k_values = []
    sigma_values = []
    holdout_accuracies = []
    brier_scores = []
    cv_means = []
    cv_stds = []

    for result_key, result in training_results.items():
        k_values.append(result["k"])
        sigma_values.append(result["sigma"])
        holdout_accuracies.append(result["calibration_accuracy"])
        brier_scores.append(result["brier_score"])
        # cv_means and cv_stds are not used anymore, but keep for compatibility
        cv_means.append(result.get("cv_mean", 0.0))
        cv_stds.append(result.get("cv_std", 0.0))

    # Choose the metric to plot
    if metric == "brier":
        color_values = brier_scores
        metric_name = "Brier Score"
        colorbar_title = "Brier Score"
        hover_text = [
            f"K={k}<br>σ={σ:.2f}<br>Brier={brier:.4f}"
            for k, σ, brier in zip(k_values, sigma_values, brier_scores)
        ]
    else:  # accuracy
        color_values = holdout_accuracies
        metric_name = "Holdout Accuracy"
        colorbar_title = "Holdout Accuracy"
        hover_text = [
            f"K={k}<br>σ={σ:.2f}<br>Acc={acc:.3f}"
            for k, σ, acc in zip(k_values, sigma_values, holdout_accuracies)
        ]

    # Create figure with subplots
    fig = go.Figure()

    # Create a heatmap-like scatter plot
    fig.add_trace(
        go.Scatter(
            x=k_values,
            y=sigma_values,
            mode="markers",
            marker=dict(
                size=15,
                color=color_values,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title=colorbar_title),
            ),
            text=hover_text,
            hoverinfo="text",
            name=metric_name,
        )
    )

    fig.update_layout(
        title=f"Weighted K-NN Performance: {metric_name} vs K and σ",
        xaxis_title="K (Number of Neighbors)",
        yaxis_title="σ (Kernel Width)",
        showlegend=False,
        margin=dict(l=50, r=50, t=50, b=50),
    )

    return fig


class WeightedKNNClassifier:
    """Custom K-NN classifier with weighted distances based on kernel function"""

    def __init__(self, k: int = 5, sigma: float = 1.0, p: int = 1):
        """
        Initialize the weighted K-NN classifier

        Args:
            k: Number of nearest neighbors (excluding self)
            sigma: Kernel width parameter
            p: Control parameter (integer >= 1)
        """
        self.k = k
        self.sigma = sigma
        self.p = max(1, int(p))  # Ensure p is integer >= 1
        self.X_train = None
        self.y_train = None
        self.n_classes = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the classifier with training data"""
        self.X_train = X.copy()
        self.y_train = y.copy()
        self.n_classes = len(np.unique(y))
        return self

    def _weighted_distance(self, distances: np.ndarray) -> np.ndarray:
        """Calculate weighted distances using the kernel function"""
        # Weight function: 1/(1+(d/sigma)^(2p))
        return 1.0 / (1.0 + (distances / self.sigma) ** (2 * self.p))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities based on weighted sums per class"""
        if self.X_train is None:
            raise ValueError("Classifier must be fitted before prediction")

        n_samples = X.shape[0]
        proba = np.zeros((n_samples, self.n_classes))

        for i in range(n_samples):
            # Calculate distances to all training points
            distances = cdist([X[i]], self.X_train, metric="euclidean")[0]

            # Find k nearest neighbors (excluding self if present)
            sorted_indices = np.argsort(distances)

            # If the closest point is very close (likely self), exclude it
            if distances[sorted_indices[0]] < 1e-10:
                neighbor_indices = sorted_indices[1 : self.k + 1]
            else:
                neighbor_indices = sorted_indices[: self.k]

            neighbor_distances = distances[neighbor_indices]
            neighbor_labels = self.y_train[neighbor_indices]

            # Calculate weights using kernel function
            weights = self._weighted_distance(neighbor_distances)

            # Sum weights per class (no normalization here)
            for class_idx in range(self.n_classes):
                class_mask = neighbor_labels == class_idx
                proba[i, class_idx] = np.sum(weights[class_mask])

            # Normalize at the end to get probabilities
            total_weight = np.sum(proba[i, :])
            if total_weight > 0:
                proba[i, :] = proba[i, :] / total_weight

        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels"""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def get_params(self, deep=True):
        """Get parameters for this estimator"""
        return {"k": self.k, "sigma": self.sigma, "p": self.p}

    def set_params(self, **params):
        """Set parameters for this estimator"""
        for key, value in params.items():
            if key == "k":
                self.k = value
            elif key == "sigma":
                self.sigma = value
            elif key == "p":
                self.p = max(1, int(value))
        return self


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Nearest Neighbor Classifier app"
    )
    parser.add_argument("--port", type=int, default=8062, help="Port to run the app on")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    args = parser.parse_args()

    app = NearestNeighborClassifierApp()
    app.run(debug=args.debug, port=args.port)
