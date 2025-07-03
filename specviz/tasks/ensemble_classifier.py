from typing import Dict, List, Optional, Tuple

import argparse
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

from specviz.tasks.conformal_prediction import OneClassConformalPredictor
from specviz.tasks.dbtools import (
    append_df_to_table,
    check_table_exists,
    fetch_dict_from_db,
    read_df_from_db,
    store_df_in_db,
)
from specviz.tasks.parameter_scan import (
    evaluate_models_at_coordinates,
    run_parameter_scan,
)


class EnsembleClassifierApp:
    """Dashboard for training and applying ensemble classifiers"""

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
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

        # Add process tracking attributes
        self.compute_process = None
        self.result_queue = None

        # Initialize results store
        self.current_figure = None
        self.needs_update = False
        self.computation_finished = False

        # Initialize results div with empty plot
        self.results_div = html.Div(
            [dcc.Graph(id="accuracy-boxplot"), dcc.Store(id="results-store", data=None)]
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
                # Model Configuration Section
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Model Configuration"),
                                        dbc.CardBody(
                                            [
                                                dbc.Row(
                                                    [
                                                        # Column 1: Data Split Parameters
                                                        dbc.Col(
                                                            [
                                                                dbc.Label("Data Split"),
                                                                dbc.InputGroup(
                                                                    [
                                                                        dbc.InputGroupText(
                                                                            "Cal",
                                                                            style={
                                                                                "width": "70px"
                                                                            },
                                                                        ),
                                                                        dbc.Input(
                                                                            id="cal-size-input",
                                                                            type="number",
                                                                            min=0.05,
                                                                            max=0.4,
                                                                            step=0.01,
                                                                            value=0.2,
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
                                                                            "Test",
                                                                            style={
                                                                                "width": "70px"
                                                                            },
                                                                        ),
                                                                        dbc.Input(
                                                                            id="test-size-input",
                                                                            type="number",
                                                                            min=0.05,
                                                                            max=0.4,
                                                                            step=0.01,
                                                                            value=0.2,
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
                                                                            "Min N",
                                                                            style={
                                                                                "width": "70px"
                                                                            },
                                                                        ),
                                                                        dbc.Input(
                                                                            id="min-samples-input",
                                                                            type="number",
                                                                            min=3,
                                                                            max=100,
                                                                            step=1,
                                                                            value=5,
                                                                            style={
                                                                                "width": "100px"
                                                                            },
                                                                        ),
                                                                    ],
                                                                    className="mb-1",
                                                                ),
                                                            ],
                                                            width=3,
                                                        ),
                                                        # Column 2: Training Parameters
                                                        dbc.Col(
                                                            [
                                                                dbc.Label("Training"),
                                                                dbc.InputGroup(
                                                                    [
                                                                        dbc.InputGroupText(
                                                                            "RFF",
                                                                            style={
                                                                                "width": "70px"
                                                                            },
                                                                        ),
                                                                        dbc.Input(
                                                                            id="n-features-input",
                                                                            type="number",
                                                                            value=1024,
                                                                            min=64,
                                                                            max=2048,
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
                                                                            "Batch",
                                                                            style={
                                                                                "width": "70px"
                                                                            },
                                                                        ),
                                                                        dbc.Input(
                                                                            id="batch-size-input",
                                                                            type="number",
                                                                            value=32,
                                                                            min=32,
                                                                            max=128,
                                                                            step=32,
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
                                                            width=3,
                                                        ),
                                                        # Column 3: Optimization
                                                        dbc.Col(
                                                            [
                                                                dbc.Label(
                                                                    "Optimization"
                                                                ),
                                                                dbc.InputGroup(
                                                                    [
                                                                        dbc.InputGroupText(
                                                                            "Splits",
                                                                            style={
                                                                                "width": "70px"
                                                                            },
                                                                        ),
                                                                        dbc.Input(
                                                                            id="n-splits-input",
                                                                            type="number",
                                                                            min=3,
                                                                            max=10,
                                                                            step=1,
                                                                            value=5,
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
                                                                            "Epochs",
                                                                            style={
                                                                                "width": "70px"
                                                                            },
                                                                        ),
                                                                        dbc.Input(
                                                                            id="max-epochs-input",
                                                                            type="number",
                                                                            min=10,
                                                                            max=1000,
                                                                            step=10,
                                                                            value=200,
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
                                                                            "1e",
                                                                            style={
                                                                                "width": "70px"
                                                                            },
                                                                        ),
                                                                        dbc.Input(
                                                                            id="learning-rate-input",
                                                                            type="number",
                                                                            min=-4,
                                                                            max=-1,
                                                                            step=0.1,
                                                                            value=-3,
                                                                            style={
                                                                                "width": "100px"
                                                                            },
                                                                        ),
                                                                    ],
                                                                    className="mb-1",
                                                                ),
                                                            ],
                                                            width=3,
                                                        ),
                                                        # Column 4: Length Scales
                                                        dbc.Col(
                                                            [
                                                                dbc.Label(
                                                                    "Length Scales"
                                                                ),
                                                                dbc.InputGroup(
                                                                    [
                                                                        dbc.InputGroupText(
                                                                            "Min",
                                                                            style={
                                                                                "width": "70px"
                                                                            },
                                                                        ),
                                                                        dbc.Input(
                                                                            id="length-scale-min-input",
                                                                            type="number",
                                                                            min=0.25,
                                                                            max=1.0,
                                                                            step=0.05,
                                                                            value=0.25,
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
                                                                            "Max",
                                                                            style={
                                                                                "width": "70px"
                                                                            },
                                                                        ),
                                                                        dbc.Input(
                                                                            id="length-scale-max-input",
                                                                            type="number",
                                                                            min=1.25,
                                                                            max=5.0,
                                                                            step=0.05,
                                                                            value=2.0,
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
                                                                            "Num",
                                                                            style={
                                                                                "width": "70px"
                                                                            },
                                                                        ),
                                                                        dbc.Input(
                                                                            id="n-length-scales-input",
                                                                            type="number",
                                                                            min=3,
                                                                            max=20,
                                                                            step=1,
                                                                            value=5,
                                                                            style={
                                                                                "width": "100px"
                                                                            },
                                                                        ),
                                                                    ],
                                                                    className="mb-1",
                                                                ),
                                                            ],
                                                            width=3,
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
                                                # Add Run and Kill buttons inside Model Configuration
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                dbc.Button(
                                                                    "Run Parameter Scan",
                                                                    id="run-scan-button",
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
                                                                    id="scan-status",
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
                dbc.Row([dbc.Col([self.results_div], width=12)], className="mt-3"),
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
                                                    "Select Length Scale Range for Model Ensemble",
                                                    className="mb-3",
                                                ),
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                dbc.Label(
                                                                    "Lower Length Scale"
                                                                ),
                                                                dcc.Dropdown(
                                                                    id="lower-scale-select",
                                                                    options=[],
                                                                    placeholder="Select lower bound",
                                                                ),
                                                            ],
                                                            width=6,
                                                        ),
                                                        dbc.Col(
                                                            [
                                                                dbc.Label(
                                                                    "Upper Length Scale"
                                                                ),
                                                                dcc.Dropdown(
                                                                    id="upper-scale-select",
                                                                    options=[],
                                                                    placeholder="Select upper bound",
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
                                                                dbc.Label("Model Name"),
                                                                dbc.Input(
                                                                    id="model-name-input",
                                                                    type="text",
                                                                    placeholder="Enter model name",
                                                                    className="mb-2",
                                                                ),
                                                                html.Div(
                                                                    [
                                                                        dbc.Button(
                                                                            "Evaluate Selected Models",
                                                                            id="evaluate-models-button",
                                                                            color="primary",
                                                                            className="me-2",
                                                                            disabled=True,
                                                                        ),
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
                                                                    id="evaluation-results",
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
                # Intervals
                dcc.Interval(id="results-interval", interval=100),
                dcc.Interval(id="computation-check", interval=500),
            ],
            fluid=True,
        )

    def _setup_callbacks(self):
        """Set up the Dash callbacks"""

        @self.app.callback(
            [Output("redis-status", "children"), Output("redis-status", "className")],
            [
                Input("results-interval", "n_intervals"),
                Input("redis-refresh-button", "n_clicks"),
            ],
        )
        def update_redis_status(n_intervals, n_clicks):
            try:
                if self.redis_client.ping():
                    current_project = self.redis_client.get("current_project")
                    status_lines = [f"Connected to {self.redis_host}:{self.redis_port}"]

                    if current_project:
                        status_lines.append(f"Current Project: {current_project}")
                    else:
                        status_lines.append("No active project")

                    return (
                        html.Div(
                            [
                                html.Div(status_lines[0]),
                                html.Small(
                                    status_lines[1], className="text-muted d-block"
                                ),
                            ]
                        ),
                        "text-success",
                    )

            except Exception as e:
                self._connect_redis()

            return (
                html.Div(
                    [
                        html.Div(self.redis_status),
                        html.Small("No active project", className="text-muted d-block"),
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
                required_tables = [
                    "HCD",
                    "measured_data",
                    "wavenumbers",
                    "embedding",
                    "level_scale",
                ]
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
                level_scale_df = read_df_from_db(conn, "level_scale")

                # Check for classification column
                if "class" not in embedding_df.columns:
                    return (
                        "No 'class' column found in embedding data",
                        "text-danger",
                        None,
                    )

                # Get HCD indices from measured data
                measured_indices = measured_data_df["hcd_indx"].values

                # Get the maximum level for these points from HCD
                max_level = hcd_df.loc[measured_indices, "level"].max()

                # Get the corresponding distance from level_scale
                base_length_scale = level_scale_df.loc[
                    level_scale_df["level"] == max_level, "distance"
                ].iloc[0]

                # Extract spectral data (excluding hcd_indx column)
                spectra = measured_data_df.iloc[:, 1:].values  # Skip hcd_indx column

                # Get corresponding X, Y coordinates from HCD for our indices
                spatial_coords = hcd_df.loc[measured_indices, ["X", "Y"]]

                # Get UMAP coordinates and class labels
                latent_coords = embedding_df[["umap_x", "umap_y"]].values
                class_labels = embedding_df["class"].values

                # Store as class attributes
                self.measured_indices = measured_indices
                self.spectra = spectra
                self.wavenumbers = wavenumbers
                self.spatial_coords = spatial_coords
                self.latent_coords = latent_coords
                self.base_length_scale = base_length_scale
                self.embedding_df = embedding_df
                self.measured_data_df = measured_data_df
                self.class_labels = class_labels  # Store class labels

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
                    f"- Base length scale: {base_length_scale:.3f}\n"
                    f"- Class distribution: {class_dist}"
                )

                return stats, "text-success", True

            except Exception as e:
                print(f"Error in load_all_data: {str(e)}")  # Debug print
                return f"Error loading data: {str(e)}", "text-danger", None

        @self.app.callback(
            Output("parameter-summary", "children"),
            [
                Input("n-features-input", "value"),
                Input("n-splits-input", "value"),
                Input("max-epochs-input", "value"),
                Input("batch-size-input", "value"),
                Input("learning-rate-input", "value"),
                Input("length-scale-min-input", "value"),
                Input("length-scale-max-input", "value"),
                Input("n-length-scales-input", "value"),
                Input("cal-size-input", "value"),
                Input("test-size-input", "value"),
                Input("min-samples-input", "value"),
                Input("random-state-input", "value"),
                Input("data-loaded-store", "data"),
            ],
        )
        def update_parameter_summary(
            n_features,
            n_splits,
            max_epochs,
            batch_size,
            learning_rate,
            length_scale_min,
            length_scale_max,
            n_length_scales,
            cal_size,
            test_size,
            min_samples,
            random_state,
            data_loaded,
        ):
            """Update the parameter summary text"""
            if not data_loaded or not hasattr(self, "base_length_scale"):
                return "Load data first to see length scale values"

            # Validate numeric inputs
            if any(v is None for v in [cal_size, test_size, min_samples]):
                return "Error: Missing required parameters"

            learning_rate_value = 10**learning_rate

            # Calculate the length scale multipliers
            multipliers = np.linspace(
                length_scale_min, length_scale_max, n_length_scales
            )

            # Calculate actual length scales
            length_scales = multipliers * self.base_length_scale

            summary = [
                html.H6("Current Configuration:", className="mb-2"),
                html.Ul(
                    [
                        html.Li(f"Random Fourier Features: {n_features}"),
                        html.Li(
                            f"Data Split: Cal={cal_size:.2f}, Test={test_size:.2f}, Min Samples={min_samples}"
                        ),
                        html.Li(
                            f"Training: Splits={n_splits}, Epochs={max_epochs}, Batch={batch_size}"
                        ),
                        html.Li(f"Learning Rate: {learning_rate_value:.1e}"),
                        html.Li(f"Random State: {random_state}"),
                        html.Li(
                            [
                                "Length Scales: ",
                                html.Br(),
                                html.Small(f"Base scale: {self.base_length_scale:.3f}"),
                                html.Br(),
                                html.Small(
                                    f"Multipliers: {', '.join([f'{m:.2f}' for m in multipliers])}"
                                ),
                                html.Br(),
                                html.Small(
                                    f"Final scales: {', '.join([f'{ls:.3f}' for ls in length_scales])}"
                                ),
                            ]
                        ),
                    ]
                ),
            ]

            return summary

        @self.app.callback(
            [
                Output("scan-status", "children"),
                Output("kill-button", "disabled"),
                Output("run-scan-button", "color"),
                Output("computation-status", "data"),
            ],
            [Input("run-scan-button", "n_clicks"), Input("kill-button", "n_clicks")],
            [
                State("n-features-input", "value"),
                State("n-splits-input", "value"),
                State("max-epochs-input", "value"),
                State("batch-size-input", "value"),
                State("learning-rate-input", "value"),
                State("length-scale-min-input", "value"),
                State("length-scale-max-input", "value"),
                State("n-length-scales-input", "value"),
                State("cal-size-input", "value"),
                State("test-size-input", "value"),
                State("min-samples-input", "value"),
                State("random-state-input", "value"),
                State("computation-status", "data"),
            ],
            prevent_initial_call=True,
        )
        def handle_parameter_scan(
            run_clicks,
            kill_clicks,
            n_features,
            n_splits,
            max_epochs,
            batch_size,
            learning_rate,
            length_scale_min,
            length_scale_max,
            n_length_scales,
            cal_size,
            test_size,
            min_samples,
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

            # Clear previous results and thresholds when starting new scan
            if hasattr(self, "scan_results"):
                del self.scan_results
            if hasattr(self, "current_thresholds"):
                del self.current_thresholds
            if hasattr(self, "selected_models"):
                del self.selected_models

            # Calculate length scales
            multipliers = np.linspace(
                length_scale_min, length_scale_max, n_length_scales
            )
            length_scales = multipliers * self.base_length_scale

            # Prepare parameters for scan
            scan_params = {
                "X": self.spatial_coords.values,
                "y": self.class_labels,
                "n_features": n_features,
                "n_splits": n_splits,
                "max_epochs": max_epochs,
                "batch_size": batch_size,
                "learning_rate": 10**learning_rate,
                "length_scales": length_scales.tolist(),
                "cal_size": cal_size,
                "test_size": test_size,
                "min_samples_per_class": min_samples,
                "random_state": random_state,
            }

            # Start the computation process
            print("\nStarting computation process...")
            self.result_queue = Queue()
            self.compute_process = Process(
                target=run_parameter_scan_process, args=(scan_params, self.result_queue)
            )
            self.compute_process.start()
            print(f"Process started with PID: {self.compute_process.pid}")

            return (
                f"Process started (PID: {self.compute_process.pid})",
                False,
                "warning",
                "computing",
            )

        @self.app.callback(
            [
                Output("accuracy-boxplot", "figure"),
                Output("results-store", "data"),
                Output("run-scan-button", "children"),
                Output("run-scan-button", "disabled"),
                Output("run-scan-button", "color", allow_duplicate=True),
                Output("computation-status", "data", allow_duplicate=True),
                Output("kill-button", "disabled", allow_duplicate=True),
                Output("scan-status", "children", allow_duplicate=True),
                Output("lower-scale-select", "options", allow_duplicate=True),
                Output("upper-scale-select", "options", allow_duplicate=True),
                Output("evaluate-models-button", "disabled"),
            ],
            [Input("results-interval", "n_intervals")],
            [State("computation-status", "data")],
            prevent_initial_call=True,
        )
        def update_display(n_intervals, status):
            if status != "computing":
                raise PreventUpdate

            if not hasattr(self, "result_queue") or self.result_queue is None:
                print("Warning: result_queue is not initialized")
                return [dash.no_update] * 11

            if not hasattr(self, "compute_process"):
                print("Warning: compute_process is not initialized")
                return [dash.no_update] * 11

            if self.compute_process and not self.compute_process.is_alive():
                print("Process has completed or terminated")
                self.compute_process = None
                self.result_queue = None
                return [dash.no_update] * 11

            try:
                try:
                    result = self.result_queue.get_nowait()

                    if isinstance(result, dict) and "error" in result:
                        print(f"Error in scan: {result['error']}")
                        return [dash.no_update] * 8 + [[]] * 2 + [True]

                    results_list, calibration_data = result
                    print(f"Processing {len(results_list)} results")

                    self.scan_results = results_list
                    self.calibration_data = calibration_data

                    # Create figure
                    fig = create_accuracy_boxplot(results_list)

                    # Get scales
                    length_scales = sorted(list({r.length_scale for r in results_list}))
                    scale_options = [
                        {"label": f"{ls:.3f}", "value": ls} for ls in length_scales
                    ]

                    # Clean up
                    self.compute_process = None
                    self.result_queue = None

                    return [
                        fig,  # figure
                        str(time.time()),  # store data
                        "Run Parameter Scan",  # button text
                        False,  # button disabled
                        "primary",  # button color
                        "idle",  # computation status
                        True,  # kill button disabled
                        "Scan completed successfully",  # status text
                        scale_options,  # lower scale options
                        scale_options,  # upper scale options
                        False,  # evaluate button disabled
                    ]

                except queue.Empty:
                    return [dash.no_update] * 11

            except Exception as e:
                print(f"Error in update_display: {str(e)}")
                self.compute_process = None
                self.result_queue = None
                return [
                    dash.no_update,  # figure
                    dash.no_update,  # store data
                    "Run Parameter Scan",  # button text
                    False,  # button disabled
                    "primary",  # button color
                    "idle",  # computation status
                    True,  # kill button disabled
                    f"Error: {str(e)}",  # status text
                    [],  # lower scale options
                    [],  # upper scale options
                    True,  # evaluate button disabled
                ]

        @self.app.callback(
            [
                Output("lower-scale-select", "options", allow_duplicate=True),
                Output("upper-scale-select", "options", allow_duplicate=True),
            ],
            [Input("computation-status", "data")],
            prevent_initial_call=True,
        )
        def update_scale_selections(status):
            if status == "idle" and hasattr(self, "scan_results"):
                # Get just the results list (not the tuple)
                results_list = (
                    self.scan_results[0]
                    if isinstance(self.scan_results, tuple)
                    else self.scan_results
                )
                scales = sorted(list({r.length_scale for r in results_list}))
                options = [
                    {"label": f"λ={scale:.4f}", "value": scale} for scale in scales
                ]
                return options, options
            return [], []

        @self.app.callback(
            [
                Output("evaluate-models-button", "disabled", allow_duplicate=True),
                Output("evaluation-results", "children"),
                Output("export-model-button", "disabled", allow_duplicate=True),
            ],
            [
                Input("lower-scale-select", "value"),
                Input("upper-scale-select", "value"),
                Input("evaluate-models-button", "n_clicks"),
            ],
            [State("computation-status", "data")],
            prevent_initial_call=True,
        )
        def handle_model_evaluation(lower_scale, upper_scale, n_clicks, status):
            print(f"\nEvaluating models... n_clicks: {n_clicks}")

            if not n_clicks or status != "idle":
                return (
                    not (lower_scale is not None and upper_scale is not None),
                    None,
                    True,
                )

            if lower_scale is None or upper_scale is None:
                return True, None, True

            if lower_scale > upper_scale:
                return (
                    True,
                    "Lower bound must be less than or equal to upper bound",
                    True,
                )

            try:
                print(f"Selected scales: {lower_scale:.4f} to {upper_scale:.4f}")

                # Get models within the selected range
                results_list = (
                    self.scan_results[0]
                    if isinstance(self.scan_results, tuple)
                    else self.scan_results
                )
                selected_models = [
                    result
                    for result in results_list
                    if lower_scale <= result.length_scale <= upper_scale
                ]
                print(f"Found {len(selected_models)} models in range")

                # Store selected models as class attribute
                self.selected_models = selected_models

                # Get calibration coordinates and evaluate models
                cal_coordinates = self.calibration_data["X"]
                cal_predictions_df = evaluate_models_at_coordinates(
                    coordinates=cal_coordinates, scan_results=selected_models
                )
                print("Generated predictions for calibration set")

                # Get calibration labels
                cal_true_labels = self.calibration_data["y"]

                # Compute conformal thresholds for different coverage levels for class 1
                error_rates = [0.01, 0.05, 0.10, 0.20, 0.25]
                thresholds = []

                print("Computing conformal thresholds...")
                for alpha in error_rates:
                    predictor = OneClassConformalPredictor(confidence_level=alpha)
                    predictor.calibrate(
                        cal_probabilities=cal_predictions_df[
                            [
                                f"prob_class_{i}"
                                for i in range(selected_models[0].n_classes)
                            ]
                        ].values,
                        cal_labels=cal_true_labels,
                    )
                    # Get threshold for class 1
                    class_threshold = predictor.get_thresholds().get(1, None)
                    thresholds.append(class_threshold)
                    print(f"  α={alpha:.2f}: threshold={class_threshold:.4f}")

                # Store thresholds as class attribute
                self.current_thresholds = thresholds

                # Create evaluation results summary
                summary_div = html.Div(
                    [
                        html.H5("Model Evaluation Results"),
                        html.P(
                            f"Selected {len(selected_models)} models with length scales "
                            f"between {lower_scale:.4f} and {upper_scale:.4f}"
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
                                                    else "N/A"
                                                ),
                                            ]
                                        )
                                        for alpha, threshold in zip(
                                            error_rates, thresholds
                                        )
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

                print("Evaluation complete")
                return False, summary_div, False

            except Exception as e:
                print(f"Error in handle_model_evaluation: {str(e)}")
                traceback.print_exc()
                return True, f"Error evaluating models: {str(e)}", True

        @self.app.callback(
            [
                Output("export-status", "children"),
                Output("export-model-button", "disabled"),
            ],
            [Input("export-model-button", "n_clicks")],
            [State("model-name-input", "value")],
            prevent_initial_call=True,
        )
        def handle_model_export(n_clicks, model_name):
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

                # Construct table name - prefix with 'model_' to avoid conflicts
                table_name = f"model_{model_name}"

                # Connect to database
                conn = duckdb.connect(db_path)

                try:
                    # Check if table already exists
                    if check_table_exists(conn, table_name):
                        return f"Error: Table '{table_name}' already exists", True

                    # Get HCD indices and class 1 probabilities
                    hcd_df = read_df_from_db(conn, "HCD")
                    hcd_coordinates = hcd_df[["X", "Y"]].values
                    predictions_df = evaluate_models_at_coordinates(
                        coordinates=hcd_coordinates, scan_results=self.selected_models
                    )

                    # Create export dataframe
                    export_df = pd.DataFrame(
                        {
                            "hcd_indx": np.arange(len(hcd_coordinates)),
                            "value": predictions_df["prob_class_1"],
                        }
                    )

                    # Export predictions to database
                    store_df_in_db(conn, export_df, table_name)

                    # Create thresholds dataframe using stored thresholds
                    thresholds_df = pd.DataFrame(
                        {
                            "model_name": table_name,
                            "threshold_0.01": self.current_thresholds[0],
                            "threshold_0.05": self.current_thresholds[1],
                            "threshold_0.10": self.current_thresholds[2],
                            "threshold_0.20": self.current_thresholds[3],
                            "threshold_0.25": self.current_thresholds[4],
                        },
                        index=[0],
                    )

                    # Add to conformal_thresholds table
                    if not check_table_exists(conn, "conformal_thresholds"):
                        store_df_in_db(conn, thresholds_df, "conformal_thresholds")
                    else:
                        append_df_to_table(conn, thresholds_df, "conformal_thresholds")

                    # Update models table
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
                        f"Successfully exported model to table '{table_name}', updated conformal thresholds, "
                        "and registered in models table"
                    ), True

                finally:
                    conn.close()

            except Exception as e:
                print(f"Error in handle_model_export: {str(e)}")
                traceback.print_exc()
                return f"Error exporting model: {str(e)}", True

    def _connect_redis(self):
        """Establish connection to Redis and check status"""
        try:
            if self.redis_client.ping():
                self.redis_status = "Connected"
                self.redis_status_color = "success"
                print(
                    f"Successfully connected to Redis at {self.redis_host}:{self.redis_port}"
                )
            else:
                self.redis_status = "Connection Failed"
                self.redis_status_color = "danger"
        except Exception as e:
            self.redis_status = f"Connection Error: {str(e)}"
            self.redis_status_color = "danger"

    def run(self, debug: bool = False, port: int = 8060):
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


def run_parameter_scan_process(params, result_queue):
    try:
        print("\n=== Starting Parameter Scan ===")
        print("Preparing X and y data...")
        print(f"Input shapes - X: {params['X'].shape}, y: {params['y'].shape}")
        print("\nParameters:")
        print(f"- Number of features: {params['n_features']}")
        print(f"- Number of splits: {params['n_splits']}")
        print(f"- Batch size: {params['batch_size']}")
        print(f"- Learning rate: {params['learning_rate']}")
        print(f"- Max epochs: {params['max_epochs']}")
        print(
            f"- Length scales: {len(params['length_scales'])} values from "
            f"{min(params['length_scales']):.3f} to {max(params['length_scales']):.3f}"
        )
        print(f"- Cal/Test sizes: {params['cal_size']:.2f}/{params['test_size']:.2f}")
        print(f"- Min samples per class: {params['min_samples_per_class']}")
        print(f"- Random state: {params['random_state']}")

        # Single call to run_parameter_scan - it handles all parallelization internally
        results, calibration_data = run_parameter_scan(
            X=params["X"],
            y=params["y"],
            length_scales=params["length_scales"],
            n_splits=params["n_splits"],
            cal_size=params["cal_size"],
            test_size=params["test_size"],
            min_samples_per_class=params["min_samples_per_class"],
            n_features=params["n_features"],
            max_epochs=params["max_epochs"],
            batch_size=params["batch_size"],
            learning_rate=params["learning_rate"],
            random_state=params["random_state"],
            verbose=True,  # So we can see progress
        )

        print("\n=== Scan Complete ===")
        print(f"Generated {len(results)} models")

        result_queue.put((results, calibration_data))

    except Exception as e:
        print(f"\nError in parameter scan: {str(e)}")
        result_queue.put({"error": str(e)})


def create_accuracy_boxplot(results_list):
    """Create a boxplot of accuracies for each length scale"""
    import plotly.graph_objects as go

    # Group results by length scale
    length_scales = sorted(list({r.length_scale for r in results_list}))
    accuracies = {ls: [] for ls in length_scales}

    for result in results_list:
        accuracies[result.length_scale].append(result.val_acc)

    # Create figure
    fig = go.Figure()

    fig.add_trace(
        go.Box(
            x=[ls for ls in length_scales for _ in accuracies[ls]],
            y=[acc for ls in length_scales for acc in accuracies[ls]],
            name="Validation Accuracy",
        )
    )

    fig.update_layout(
        title="Model Validation Accuracy vs Length Scale",
        xaxis_title="Length Scale",
        yaxis_title="Validation Accuracy",
        showlegend=False,
        margin=dict(l=50, r=50, t=50, b=50),
    )

    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Ensemble Classifier app")
    parser.add_argument("--port", type=int, default=8058, help="Port to run the app on")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    args = parser.parse_args()

    app = EnsembleClassifierApp()
    app.run(debug=args.debug, port=args.port)
