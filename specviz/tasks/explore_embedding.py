import dash
import dash_bootstrap_components as dbc
import duckdb
import numpy as np
import pandas as pd
import redis
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from tools.dbtools import check_table_exists, read_df_from_db, store_df_in_db
from specviz.tasks.spatial_latent_space_explorer import SpatialLatentSpaceExplorer


class ExploreEmbedding:
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

        # Initialize default scales
        self.default_spatial_scale = None
        self.default_latent_scale = None

        # Initialize Dash app
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
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
                # Controls Row
                dbc.Row(
                    [
                        # Selection and Diffusion Controls
                        dbc.Col(
                            [
                                # Load and Clear Buttons
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                dbc.Button(
                                                    "Load Current Data",
                                                    id="load-data-button",
                                                    color="primary",
                                                    className="me-2 mb-3",
                                                ),
                                                dbc.Button(
                                                    "Clear Selection",
                                                    id="clear-selection-button",
                                                    color="secondary",
                                                    className="me-2 mb-3",
                                                ),
                                            ]
                                        )
                                    ]
                                ),
                                # Data Source Selection
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.Label("Spectral Data Source:"),
                                                dcc.Dropdown(
                                                    id="data-source",
                                                    options=[
                                                        {"label": "Original Data (measured_data)", "value": "measured_data"},
                                                        {"label": "Baseline Corrected Data (baseline_corrected_data)", "value": "baseline_corrected_data"},
                                                        {"label": "Baseline Data (baseline_data)", "value": "baseline_data"},
                                                    ],
                                                    value="measured_data",
                                                    className="mb-2",
                                                ),
                                                html.Small(
                                                    "Select which spectral data to use for analysis",
                                                    className="text-muted",
                                                ),
                                            ]
                                        )
                                    ],
                                    className="mb-3",
                                ),
                                # Selection Mode
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                dbc.RadioItems(
                                                    id="selection-mode",
                                                    options=[
                                                        {
                                                            "label": "Select in UMAP",
                                                            "value": "umap_add",
                                                        },
                                                        {
                                                            "label": "Select in Spatial",
                                                            "value": "spatial_add",
                                                        },
                                                        {
                                                            "label": "Remove from UMAP",
                                                            "value": "umap_remove",
                                                        },
                                                        {
                                                            "label": "Remove from Spatial",
                                                            "value": "spatial_remove",
                                                        },
                                                    ],
                                                    value="umap_add",
                                                    inline=True,
                                                    className="mb-3",
                                                ),
                                            ]
                                        )
                                    ]
                                ),
                                # Diffusion Controls
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.Label("Latent Diffusion:"),
                                                dbc.Input(
                                                    type="number",
                                                    id="latent-diffusion-value",
                                                    value=self.default_latent_scale,
                                                    step=0.05,
                                                    style={"width": "100px"},
                                                ),
                                                dbc.Button(
                                                    "Latent Diffuse",
                                                    id="latent-diffuse-button",
                                                    color="primary",
                                                    size="sm",
                                                    className="ms-2",
                                                ),
                                            ],
                                            width=6,
                                        ),
                                        dbc.Col(
                                            [
                                                html.Label("Spatial Diffusion:"),
                                                dbc.Input(
                                                    type="number",
                                                    id="spatial-diffusion-value",
                                                    value=self.default_spatial_scale,
                                                    step=0.1,
                                                    style={"width": "100px"},
                                                ),
                                                dbc.Button(
                                                    "Spatial Diffuse",
                                                    id="spatial-diffuse-button",
                                                    color="primary",
                                                    size="sm",
                                                    className="ms-2",
                                                ),
                                            ],
                                            width=6,
                                        ),
                                    ],
                                    className="mb-3",
                                ),
                                html.Div(id="load-data-status", className="mb-3"),
                            ],
                            width=6,
                        ),
                        # Normalization Controls
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Spectral Normalization"),
                                        dbc.CardBody(
                                            [
                                                # Wavenumber Range Selection
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                dbc.Label(
                                                                    "Wavenumber Ranges"
                                                                ),
                                                                dbc.Input(
                                                                    id="normalization-wavenumber-ranges",
                                                                    type="text",
                                                                    placeholder="e.g., 200-400,500-900",
                                                                    value="",
                                                                ),
                                                                dbc.FormText(
                                                                    "Enter ranges as start-end, separate multiple ranges with commas"
                                                                ),
                                                            ],
                                                            width=12,
                                                        ),
                                                    ],
                                                    className="mb-3",
                                                ),
                                                # Normalization Method Selection
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                dbc.Label(
                                                                    "Normalization Method"
                                                                ),
                                                                dbc.RadioItems(
                                                                    id="normalization-method",
                                                                    options=[
                                                                        {
                                                                            "label": "No Normalization",
                                                                            "value": "none",
                                                                        },
                                                                        {
                                                                            "label": "Vector Normalize",
                                                                            "value": "vector",
                                                                        },
                                                                        {
                                                                            "label": "Min-Max Normalize",
                                                                            "value": "minmax",
                                                                        },
                                                                    ],
                                                                    value="none",
                                                                    className="mb-3",
                                                                ),
                                                            ],
                                                            width=12,
                                                        ),
                                                    ]
                                                ),
                                                # Normalize Button
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                dbc.Button(
                                                                    "Normalize",
                                                                    id="normalize-button",
                                                                    color="primary",
                                                                    className="w-100",
                                                                ),
                                                            ],
                                                            width=12,
                                                        ),
                                                    ]
                                                ),
                                                # Status message
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                html.Div(
                                                                    id="normalization-status",
                                                                    className="mt-2",
                                                                )
                                                            ],
                                                            width=12,
                                                        ),
                                                    ]
                                                ),
                                            ]
                                        ),
                                    ]
                                )
                            ]
                        ),
                    ]
                ),
                # Plots Row
                dbc.Row(
                    [
                        # UMAP Plot
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader("UMAP Projection"),
                                        dbc.CardBody(
                                            [
                                                dcc.Graph(
                                                    id="umap-plot",
                                                    config={
                                                        "displayModeBar": True,
                                                        "modeBarButtonsToAdd": [
                                                            "lasso2d"
                                                        ],
                                                    },
                                                )
                                            ]
                                        ),
                                    ]
                                )
                            ],
                            width=6,
                        ),
                        # Spatial Plot
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Spatial Distribution"),
                                        dbc.CardBody(
                                            [
                                                dcc.Graph(
                                                    id="spatial-plot",
                                                    config={
                                                        "displayModeBar": True,
                                                        "modeBarButtonsToAdd": [
                                                            "lasso2d"
                                                        ],
                                                    },
                                                )
                                            ]
                                        ),
                                    ]
                                )
                            ],
                            width=6,
                        ),
                    ]
                ),
                # Spectral Plot Row
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
                                                            "Selected Spectra",
                                                            width="auto",
                                                        ),
                                                        dbc.Col(
                                                            [
                                                                dbc.Switch(
                                                                    id="spectral-view-toggle",
                                                                    label="Show Individual Spectra",
                                                                    value=False,
                                                                    className="float-end",
                                                                )
                                                            ]
                                                        ),
                                                    ]
                                                )
                                            ]
                                        ),
                                        dbc.CardBody([dcc.Graph(id="spectral-plot")]),
                                    ]
                                )
                            ],
                            width=12,
                        )
                    ],
                    className="mt-3",
                ),
                # Export Classification Row
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Button(
                                    "Export Classification",
                                    id="export-classification-button",
                                    color="success",
                                    className="mt-3 mb-3",
                                ),
                                html.Div(id="export-status", className="mt-2"),
                            ],
                            width=12,
                        )
                    ]
                ),
                # Stores
                dcc.Store(
                    id="default-scales-store",
                    data={
                        "spatial": self.default_spatial_scale,
                        "latent": self.default_latent_scale,
                    },
                ),
                dcc.Store(id="selection-store", data=[]),
                dcc.Store(id="data-loaded-store", data=False),
                dcc.Store(id="normalized-data-store", data=None),
                # Interval for Redis status
                dcc.Interval(id="status-check", interval=1000),
            ],
            fluid=True,
        )

    def _setup_callbacks(self):
        """Set up the Dash callbacks"""

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
                Output("load-data-status", "children"),
                Output("load-data-status", "className"),
                Output("data-loaded-store", "data"),
            ],  # Add output for data loaded store
            [Input("load-data-button", "n_clicks")],
            [State("data-source", "value")],
            prevent_initial_call=True,
        )
        def handle_load_data(n_clicks, data_source):
            if n_clicks:
                status, color = self.load_data(data_source)
                return status, f"text-{color}", True
            return dash.no_update, dash.no_update, dash.no_update

        @self.app.callback(
            [Output("umap-plot", "figure"), Output("spatial-plot", "figure")],
            [Input("data-loaded-store", "data"), Input("selection-store", "data")],
            prevent_initial_call=True,
        )
        def update_plots(data_loaded, selected_indices):
            if not data_loaded or not hasattr(self, "latent_coords"):
                return self._empty_umap_plot(), self._empty_spatial_plot()

            selected_indices = selected_indices or []

            # Use data_loaded as uirevision - it will change when data is reloaded
            # but remain constant for selection updates
            uirevision = "selection" if selected_indices else "load"

            # Create UMAP plot
            umap_fig = {
                "data": [
                    {
                        "type": "scatter",
                        "x": self.latent_coords[:, 0],
                        "y": self.latent_coords[:, 1],
                        "mode": "markers",
                        "marker": {
                            "size": 5,
                            "color": [
                                (
                                    "rgb(30,144,255)"
                                    if i in selected_indices
                                    else "rgb(220,20,60)"
                                )
                                for i in range(len(self.latent_coords))
                            ],
                            "opacity": [
                                1.0 if i in selected_indices else 0.3
                                for i in range(len(self.latent_coords))
                            ],
                        },
                        "name": "UMAP Points",
                    }
                ],
                "layout": {
                    "title": "UMAP Projection",
                    "xaxis": {"title": "UMAP 1"},
                    "yaxis": {"title": "UMAP 2"},
                    "dragmode": "lasso",
                    "hovermode": "closest",
                    "uirevision": uirevision,  # Changes on data load
                },
            }

            # Create Spatial plot
            spatial_fig = {
                "data": [
                    {
                        "type": "scatter",
                        "x": self.spatial_coords["X"],
                        "y": self.spatial_coords["Y"],
                        "mode": "markers",
                        "marker": {
                            "size": 5,
                            "color": [
                                (
                                    "rgb(30,144,255)"
                                    if i in selected_indices
                                    else "rgb(220,20,60)"
                                )
                                for i in range(len(self.spatial_coords))
                            ],
                            "opacity": [
                                1.0 if i in selected_indices else 0.3
                                for i in range(len(self.spatial_coords))
                            ],
                        },
                        "name": "Spatial Points",
                    }
                ],
                "layout": {
                    "title": "Spatial Distribution",
                    "xaxis": {
                        "title": "X Position",
                        "scaleanchor": "y",
                        "scaleratio": 1,
                    },
                    "yaxis": {"title": "Y Position"},
                    "dragmode": "lasso",
                    "hovermode": "closest",
                    "uirevision": uirevision,  # Changes on data load
                },
            }

            return umap_fig, spatial_fig

        @self.app.callback(
            Output("selection-store", "data"),
            [
                Input("umap-plot", "selectedData"),
                Input("spatial-plot", "selectedData"),
                Input("clear-selection-button", "n_clicks"),
                Input("selection-mode", "value"),
                Input("data-loaded-store", "data"),
                Input("latent-diffuse-button", "n_clicks"),
                Input("spatial-diffuse-button", "n_clicks"),
            ],
            [
                State("selection-store", "data"),
                State("latent-diffusion-value", "value"),
                State("spatial-diffusion-value", "value"),
            ],
            prevent_initial_call=True,
        )
        def update_selection(
            umap_selected,
            spatial_selected,
            clear_clicks,
            mode,
            data_loaded,
            latent_diffuse_clicks,
            spatial_diffuse_clicks,
            current_selection,
            latent_diffusion_value,
            spatial_diffusion_value,
        ):
            ctx = dash.callback_context
            if not ctx.triggered:
                return dash.no_update

            trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

            if trigger_id == "clear-selection-button":
                return []

            if trigger_id == "data-loaded-store":
                return []

            if trigger_id == "latent-diffuse-button":
                if not current_selection:
                    return []
                return self._compute_latent_diffusion(
                    current_selection, latent_diffusion_value
                )

            if trigger_id == "spatial-diffuse-button":
                if not current_selection:
                    return []
                return self._compute_spatial_diffusion(
                    current_selection, spatial_diffusion_value
                )

            current_selection = set(current_selection or [])

            # Handle selection based on mode
            if mode in ["umap_add", "umap_remove"] and umap_selected:
                new_points = {p["pointIndex"] for p in umap_selected["points"]}
                if mode == "umap_add":
                    current_selection.update(new_points)
                else:  # umap_remove
                    # Only remove points that are actually in the current selection
                    current_selection.difference_update(new_points)

            elif mode in ["spatial_add", "spatial_remove"] and spatial_selected:
                new_points = {p["pointIndex"] for p in spatial_selected["points"]}
                if mode == "spatial_add":
                    current_selection.update(new_points)
                else:  # spatial_remove
                    # Only remove points that are actually in the current selection
                    current_selection.difference_update(new_points)

            return sorted(list(current_selection))

        @self.app.callback(
            [
                Output("spatial-diffusion-value", "value"),
                Output("latent-diffusion-value", "value"),
            ],
            [Input("data-loaded-store", "data")],
            prevent_initial_call=True,
        )
        def update_diffusion_values(data_loaded):
            if not data_loaded:
                return None, None
            return self.default_spatial_scale, self.default_latent_scale

        @self.app.callback(
            Output("spectral-plot", "figure"),
            [
                Input("selection-store", "data"),
                Input("data-loaded-store", "data"),
                Input("normalized-data-store", "data"),
                Input("spectral-view-toggle", "value"),
            ],
            prevent_initial_call=True,
        )
        def update_spectral_plot(
            selected_indices, data_loaded, normalized_data, show_individual
        ):
            if not data_loaded or not hasattr(self, "spectra") or not selected_indices:
                return {
                    "data": [],
                    "layout": {
                        "title": "Select points to see spectral statistics",
                        "xaxis": {"title": "Wavenumber (cm⁻¹)"},
                        "yaxis": {"title": "Intensity"},
                        "showlegend": True,
                    },
                }

            # Determine which data to use
            if normalized_data is not None:
                working_spectra = np.array(normalized_data["spectra"])
                working_wavenumbers = np.array(normalized_data["wavenumbers"])
                title_prefix = f"Normalized Spectra ({normalized_data['method']}"
                if normalized_data["ranges"]:
                    title_prefix += f", ranges: {normalized_data['ranges']}"
                title_prefix += ")"
            else:
                working_spectra = self.spectra
                working_wavenumbers = self.wavenumbers
                title_prefix = "Raw Spectra"

            # Validate indices are within bounds
            valid_indices = [i for i in selected_indices if i < len(working_spectra)]
            if len(valid_indices) != len(selected_indices):
                print(
                    f"Warning: {len(selected_indices) - len(valid_indices)} indices were out of bounds"
                )

            if not valid_indices:
                return {
                    "data": [],
                    "layout": {
                        "title": "No valid points selected",
                        "xaxis": {"title": "Wavenumber (cm⁻¹)"},
                        "yaxis": {"title": "Intensity"},
                        "showlegend": True,
                    },
                }

            # Get selected spectra using validated indices
            selected_spectra = working_spectra[valid_indices]

            if show_individual:
                # Plot individual spectra
                traces = []
                for i, spectrum in enumerate(selected_spectra):
                    traces.append(
                        {
                            "type": "scatter",
                            "x": working_wavenumbers,
                            "y": spectrum,
                            "mode": "lines",
                            "line": {"color": "rgba(0,176,246,0.1)"},
                            "showlegend": False,
                            "hoverinfo": "skip",
                        }
                    )

                # Add mean spectrum on top
                mean_spectrum = np.mean(selected_spectra, axis=0)
                traces.append(
                    {
                        "type": "scatter",
                        "x": working_wavenumbers,
                        "y": mean_spectrum,
                        "mode": "lines",
                        "line": {"color": "rgb(0,176,246)", "width": 2},
                        "name": "Mean",
                        "showlegend": True,
                    }
                )

            else:
                # Statistical view
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
                        "x": np.concatenate(
                            [working_wavenumbers, working_wavenumbers[::-1]]
                        ),
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
                        "x": np.concatenate(
                            [working_wavenumbers, working_wavenumbers[::-1]]
                        ),
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
                        "x": working_wavenumbers,
                        "y": median,
                        "line": {"color": "rgb(0,176,246)", "width": 2},
                        "name": "Median",
                        "showlegend": True,
                    },
                ]

            # Create the plot
            fig = {
                "data": traces,
                "layout": {
                    "title": f"{title_prefix} (n={len(selected_indices)} spectra)",
                    "xaxis": {
                        "title": "Wavenumber (cm⁻¹)",
                        "autorange": "reversed",  # Reverse x-axis for wavenumbers
                    },
                    "yaxis": {"title": "Intensity"},
                    "showlegend": True,
                    "hovermode": "closest",
                },
            }

            return fig

        @self.app.callback(
            [
                Output("normalized-data-store", "data"),
                Output("normalization-status", "children"),
                Output("normalization-status", "className"),
            ],
            [Input("normalize-button", "n_clicks")],
            [
                State("normalization-method", "value"),
                State("normalization-wavenumber-ranges", "value"),
            ],
            prevent_initial_call=True,
        )
        def handle_normalization(n_clicks, method, ranges):
            if not n_clicks:
                return None, "", ""

            if not hasattr(self, "spectra") or self.spectra is None:
                return None, "No data loaded", "text-danger"

            # Call the normalization function
            normalized_spectra, working_wavenumbers, status, status_type = (
                normalize_spectra(self.spectra, self.wavenumbers, method, ranges)
            )

            if status_type == "danger" or normalized_spectra is None:
                return None, status, f"text-{status_type}"

            # Store the results
            result = {
                "spectra": normalized_spectra.tolist(),
                "wavenumbers": working_wavenumbers.tolist(),
                "method": method,
                "ranges": ranges,
            }

            return result, status, f"text-{status_type}"

        @self.app.callback(
            [Output("export-status", "children"), Output("export-status", "className")],
            [Input("export-classification-button", "n_clicks")],
            [State("selection-store", "data")],
            prevent_initial_call=True,
        )
        def handle_export_classification(n_clicks, selected_indices):
            if not n_clicks or not hasattr(self, "redis_client"):
                return "", ""

            try:
                # Get current project
                current_project = self.redis_client.get("current_project")
                if not current_project:
                    return "No active project", "text-danger"

                # Connect to DuckDB
                conn = duckdb.connect(current_project)

                # Read all necessary tables
                embedding_df = read_df_from_db(conn, "embedding")
                measured_data_df = read_df_from_db(conn, "measured_data")

                # Create classification array
                classification = np.zeros(len(self.latent_coords), dtype=int)
                if selected_indices:
                    classification[selected_indices] = 1

                # Update the class column in the existing DataFrame
                embedding_df["class"] = classification

                # Store back in DuckDB, overwriting the existing table
                store_df_in_db(conn, embedding_df, "embedding", if_exists="replace")

                conn.close()

                return (
                    f"Classification exported successfully with {len(selected_indices)} selected points",
                    "text-success",
                )

            except Exception as e:
                print(f"Error in export: {str(e)}")  # Debug print
                return f"Error exporting classification: {str(e)}", "text-danger"

    def _empty_umap_plot(self):
        """Helper method for empty UMAP plot"""
        return {
            "data": [],
            "layout": {
                "title": "Load data to see UMAP projection",
                "xaxis": {"title": "UMAP 1"},
                "yaxis": {"title": "UMAP 2"},
            },
        }

    def _empty_spatial_plot(self):
        """Helper method for empty spatial plot"""
        return {
            "data": [],
            "layout": {
                "title": "Load data to see spatial distribution",
                "xaxis": {"title": "X Position"},
                "yaxis": {"title": "Y Position"},
            },
        }

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

    def load_data(self, data_source="measured_data"):
        """Load data into memory with proper alignment"""
        try:
            db_path = self.redis_client.get("current_project")
            if not db_path:
                return "No active project found in Redis", "danger"

            conn = duckdb.connect(db_path)

            # Check all required tables exist
            required_tables = [
                "HCD",
                "wavenumbers",
                "embedding",
                "level_scale",
            ]
            
            # Add the selected data source to required tables
            if data_source not in required_tables:
                required_tables.append(data_source)
            
            for table in required_tables:
                if not check_table_exists(conn, table):
                    return f"Required table '{table}' not found in {db_path}", "danger"

            # Load all required tables using dbtools
            hcd_df = read_df_from_db(conn, "HCD")
            spectral_data_df = read_df_from_db(conn, data_source)
            wavenumbers = read_df_from_db(conn, "wavenumbers")["wavenumber"].values
            embedding_df = read_df_from_db(conn, "embedding")
            level_scale_df = read_df_from_db(conn, "level_scale")

            # Get HCD indices from spectral data
            measured_indices = spectral_data_df["hcd_indx"].values

            # Get the maximum level for these points from HCD
            max_level = hcd_df.loc[measured_indices, "level"].max()

            # Get the corresponding distance from level_scale
            default_spatial_scale = level_scale_df.loc[
                level_scale_df["level"] == max_level, "distance"
            ].iloc[0]

            # Calculate and round latent scale
            latent_coords = embedding_df[["umap_x", "umap_y"]].values
            from sklearn.metrics.pairwise import euclidean_distances

            pairwise_distances = euclidean_distances(
                latent_coords[:1000], latent_coords[:1000]
            )
            default_latent_scale = (
                np.median(pairwise_distances[pairwise_distances > 0]) * 0.1
            )

            # Extract spectral data (excluding hcd_indx column)
            spectra = spectral_data_df.iloc[:, 1:].values  # Skip hcd_indx column

            # Get corresponding X, Y coordinates from HCD for our indices
            spatial_coords = hcd_df.loc[measured_indices, ["X", "Y"]]

            # Get UMAP coordinates
            latent_coords = embedding_df[["umap_x", "umap_y"]].values

            # Store the default scales rounded to 1 decimal place
            self.default_spatial_scale = round(float(default_spatial_scale), 1)
            self.default_latent_scale = round(float(default_latent_scale), 1)

            print(
                f"Calculated scales - Spatial: {self.default_spatial_scale}, Latent: {self.default_latent_scale}"
            )
            print(f"Data source: {data_source}")

            # Store aligned data
            self.measured_indices = measured_indices
            self.spectra = spectra
            self.wavenumbers = wavenumbers
            self.spatial_coords = spatial_coords
            self.latent_coords = latent_coords

            conn.close()

            # Update stores using dcc.Store.update() pattern
            if hasattr(self, "app"):
                # Force store update by setting to None first
                for store in self.app.layout().children:
                    if isinstance(store, dcc.Store):
                        if store.id == "default-scales-store":
                            store.data = {
                                "spatial": self.default_spatial_scale,
                                "latent": self.default_latent_scale,
                            }
                        elif store.id == "data-loaded-store":
                            store.data = True

            status = (
                f"Successfully loaded data from {data_source}:\n"
                f"- Number of points: {len(measured_indices)}\n"
                f"- Spectra shape: {spectra.shape}\n"
                f"- Wavenumber range: {wavenumbers.min():.1f} - {wavenumbers.max():.1f}\n"
                f"- UMAP embedding shape: {latent_coords.shape}\n"
                f"- Default spatial scale: {self.default_spatial_scale:.1f}\n"
                f"- Default latent scale: {self.default_latent_scale:.1f}"
            )
            return status, "success"

        except Exception as e:
            print(f"Error in load_data: {str(e)}")  # Debug print
            return f"Error loading data: {str(e)}", "danger"

    def _compute_latent_diffusion(
        self, selected_indices: list, distance: float
    ) -> list:
        """Compute latent space diffusion for selected points."""
        from sklearn.metrics.pairwise import euclidean_distances

        selected_set = set(selected_indices)
        all_indices = set(range(len(self.latent_coords)))

        # Return early if no points to process
        if distance >= 0 and len(all_indices - selected_set) == 0:
            return sorted(selected_set)
        if distance < 0 and len(selected_set) == 0:
            return []

        if distance >= 0:
            # Growth mode
            unselected_indices = list(all_indices - selected_set)
            selected_points = self.latent_coords[list(selected_set)]
            unselected_points = self.latent_coords[unselected_indices]

            distances = euclidean_distances(unselected_points, selected_points)
            within_distance = np.any(distances <= abs(distance), axis=1)
            new_indices = set(np.array(unselected_indices)[within_distance])

            return sorted(selected_set | new_indices)
        else:
            # Shrink mode
            selected_points = self.latent_coords[list(selected_set)]
            unselected_indices = list(all_indices - selected_set)

            if (
                not unselected_indices
            ):  # If no unselected points, return original selection
                return sorted(selected_set)

            unselected_points = self.latent_coords[unselected_indices]

            distances = euclidean_distances(selected_points, unselected_points)
            within_distance = np.any(distances <= abs(distance), axis=1)
            points_to_remove = set(np.array(list(selected_set))[within_distance])

            return sorted(selected_set - points_to_remove)

    def _compute_spatial_diffusion(
        self, selected_indices: list, distance: float
    ) -> list:
        """Compute spatial diffusion for selected points."""
        from sklearn.metrics.pairwise import euclidean_distances

        selected_set = set(selected_indices)
        all_indices = set(range(len(self.spatial_coords)))

        # Return early if no points to process
        if distance >= 0 and len(all_indices - selected_set) == 0:
            return sorted(selected_set)
        if distance < 0 and len(selected_set) == 0:
            return []

        if distance >= 0:
            # Growth mode
            unselected_indices = list(all_indices - selected_set)
            selected_points = self.spatial_coords.iloc[list(selected_set)].values
            unselected_points = self.spatial_coords.iloc[unselected_indices].values

            distances = euclidean_distances(unselected_points, selected_points)
            within_distance = np.any(distances <= abs(distance), axis=1)
            new_indices = set(np.array(unselected_indices)[within_distance])

            return sorted(selected_set | new_indices)
        else:
            # Shrink mode
            selected_points = self.spatial_coords.iloc[list(selected_set)].values
            unselected_indices = list(all_indices - selected_set)

            if (
                not unselected_indices
            ):  # If no unselected points, return original selection
                return sorted(selected_set)

            unselected_points = self.spatial_coords.iloc[unselected_indices].values

            distances = euclidean_distances(selected_points, unselected_points)
            within_distance = np.any(distances <= abs(distance), axis=1)
            points_to_remove = set(np.array(list(selected_set))[within_distance])

            return sorted(selected_set - points_to_remove)

    def run(self, debug: bool = False, port: int = 8050):
        try:
            import logging

            log = logging.getLogger("werkzeug")
            log.setLevel(logging.ERROR)
            self.app.run_server(debug=debug, port=port, host="0.0.0.0")

        finally:
            # Clean up connections
            if hasattr(self, "redis_client"):
                self.redis_client.close()


def normalize_spectra(
    spectra: np.ndarray, wavenumbers: np.ndarray, method: str, ranges: str = None
) -> tuple:
    """Normalize spectral data according to specified parameters.

    Args:
        spectra (np.ndarray): Input spectral data of shape (n_spectra, n_wavenumbers)
        wavenumbers (np.ndarray): Wavenumber values
        method (str): Normalization method ('none', 'vector', or 'minmax')
        ranges (str, optional): Comma-separated wavenumber ranges (e.g., "200-400,500-900")

    Returns:
        tuple: (normalized_spectra, working_wavenumbers, status_message, status_type)
    """
    try:
        # Create mask for normalization range if provided
        if ranges and ranges.strip():
            norm_mask = np.zeros(len(wavenumbers), dtype=bool)
            for range_str in ranges.split(","):
                start, end = map(float, range_str.strip().split("-"))
                # Ensure start < end
                if start > end:
                    start, end = end, start
                norm_mask |= (wavenumbers >= start) & (wavenumbers <= end)

            if not np.any(norm_mask):
                return (
                    None,
                    None,
                    "No wavenumbers selected in specified ranges",
                    "danger",
                )

            # Extract the range-specific data for normalization calculation
            norm_spectra = spectra[:, norm_mask]
        else:
            norm_spectra = spectra
            norm_mask = slice(None)  # Use all wavenumbers

        # Apply normalization using range-specific data but normalize full spectrum
        if method == "none":
            normalized_spectra = spectra
            status = "No normalization applied"

        elif method == "vector":
            # Calculate L2 norm using only the selected range
            norms = np.linalg.norm(norm_spectra, axis=1, keepdims=True)
            # Handle zero-norm spectra
            norms[norms == 0] = 1.0
            # Apply normalization to full spectrum
            normalized_spectra = spectra / norms
            status = "Vector normalization applied"

        elif method == "minmax":
            # Calculate min/max using only the selected range
            min_vals = norm_spectra.min(axis=1, keepdims=True)
            max_vals = norm_spectra.max(axis=1, keepdims=True)
            # Handle constant spectra
            diff = max_vals - min_vals
            diff[diff == 0] = 1.0
            # Apply normalization to full spectrum
            normalized_spectra = (spectra - min_vals) / diff
            status = "Min-max normalization applied"

        else:
            return None, None, f"Unknown normalization method: {method}", "danger"

        # Add details to status message
        if ranges and ranges.strip():
            status += f" (normalization range: {ranges})"

        return normalized_spectra, wavenumbers, status, "success"

    except Exception as e:
        return None, None, f"Error during normalization: {str(e)}", "danger"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the Embedding Explorer app")
    parser.add_argument("--port", type=int, default=8050, help="Port to run the app on")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    args = parser.parse_args()

    explorer = ExploreEmbedding()
    explorer.run(debug=args.debug, port=args.port)
