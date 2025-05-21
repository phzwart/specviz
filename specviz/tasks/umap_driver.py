from typing import Dict, Optional, Protocol

import multiprocessing
import sys
import time
from multiprocessing import Process, Queue

import dash
import dash_bootstrap_components as dbc
import numpy as np
import umap
from dash import Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate

# Set start method only if not already set
try:
    if sys.platform == "darwin":  # macOS
        multiprocessing.set_start_method("spawn")
    else:  # Linux/Windows can use the default
        multiprocessing.set_start_method("fork")
except RuntimeError:
    # A start method has already been set, we'll use whatever it is
    pass


class UMAPExportConnector(Protocol):
    def export_embedding(self, embedding: np.ndarray, params: dict) -> bool:
        """
        Export UMAP embedding with its parameters
        Args:
            embedding: The UMAP embedding array (n_samples, 2)
            params: Dictionary of UMAP parameters (n_neighbors, min_dist, metric, densmap)
        Returns:
            bool: Success status
        """
        ...


def run_umap_process(data, params, wavenumber_indices, result_queue):
    """Function to run in separate process"""
    # Convert data to numpy array if it isn't already
    data = np.asarray(data)
    wavenumber_indices = np.asarray(wavenumber_indices)

    # Select only specified wavenumbers
    data_subset = data[:, wavenumber_indices]

    # Create UMAP parameters dict without the wavenumber selection
    umap_params = {
        "n_neighbors": params["n_neighbors"],
        "min_dist": params["min_dist"],
        "metric": params["metric"],
        "densmap": params["densmap"],
    }

    reducer = umap.UMAP(**umap_params)
    embedding = reducer.fit_transform(data_subset)
    result_queue.put(embedding)


def create_umap_app(
    input_data=None,
    wavenumbers=None,
    positions=None,
    reverse_wavenumbers=False,
    export_connector: Optional[UMAPExportConnector] = None,
):
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    # Store the input data, wavenumbers, positions and reverse option
    if input_data is not None:
        app.input_data = input_data
    else:
        app.input_data = np.random.random((1000, 100))

    if wavenumbers is not None:
        app.wavenumbers = wavenumbers
    else:
        app.wavenumbers = np.linspace(0, 100, app.input_data.shape[1])

    if positions is not None:
        app.positions = positions
    else:
        # Create dummy positions if none provided
        app.positions = np.random.random((app.input_data.shape[0], 2))

    app.reverse_wavenumbers = reverse_wavenumbers
    app.export_connector = export_connector
    app.current_params = None  # Store current UMAP parameters

    # Add stores at the top of the layout
    app.layout = dbc.Container(
        [
            dcc.Store(id="computation-status", data="idle"),
            dcc.Store(id="selection-store", data=[]),  # Add selection store
            dcc.Store(id="umap-params-store"),
            dcc.Interval(id="computation-check", interval=500),
            html.H1("UMAP Configuration and Visualization", className="mb-4"),
            dbc.Card(
                [
                    dbc.CardHeader("Data Information"),
                    dbc.CardBody([html.P(f"Data shape: {app.input_data.shape}")]),
                ],
                className="mb-3",
            ),
            # UMAP Configuration section
            dbc.Card(
                [
                    dbc.CardHeader("UMAP Configuration"),
                    dbc.CardBody(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.Label("Number of Neighbors:"),
                                            dcc.Input(
                                                id="n-neighbors",
                                                type="number",
                                                value=15,
                                                min=2,
                                                className="form-control",
                                            ),
                                        ]
                                    ),
                                    dbc.Col(
                                        [
                                            html.Label("Min Distance:"),
                                            dcc.Input(
                                                id="min-dist",
                                                type="number",
                                                value=0.1,
                                                min=0.0,
                                                step=0.1,
                                                className="form-control",
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
                                            html.Label("Distance Metric:"),
                                            dcc.Dropdown(
                                                id="distance-metric",
                                                options=[
                                                    {
                                                        "label": "Euclidean",
                                                        "value": "euclidean",
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
                                                className="form-control",
                                            ),
                                        ]
                                    ),
                                    dbc.Col(
                                        [
                                            html.Label("DensMAP:"),
                                            dcc.Dropdown(
                                                id="densmap",
                                                options=[
                                                    {"label": "True", "value": True},
                                                    {"label": "False", "value": False},
                                                ],
                                                value=False,
                                                className="form-control",
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
                                            html.Label("Wavenumber Ranges:"),
                                            dcc.Input(
                                                id="wavenumber-ranges",
                                                type="text",
                                                placeholder="e.g., 600-1200,1500-1600",
                                                value="",
                                                className="form-control",
                                            ),
                                            html.Small(
                                                "Leave empty to use all wavenumbers. Format: start-end,start-end",
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
                                            dbc.Button(
                                                "Run UMAP",
                                                id="run-button",
                                                color="primary",
                                                className="me-2",
                                            ),
                                            dbc.Button(
                                                "Export Embedding",
                                                id="export-button",
                                                color="success",
                                                disabled=True,  # Initially disabled
                                            ),
                                        ]
                                    ),
                                ],
                                className="mt-3",
                            ),
                        ]
                    ),
                ],
                className="mb-3",
            ),
            # Results section
            dbc.Card(
                [
                    dbc.CardHeader("Results"),
                    dbc.CardBody(
                        [
                            dbc.Row(
                                [
                                    dbc.Col([dcc.Graph(id="umap-plot")], width=6),
                                    dbc.Col([dcc.Graph(id="spectrum-plot")], width=6),
                                ]
                            ),
                            dbc.Row(
                                [dbc.Col([dcc.Graph(id="positions-plot")], width=12)]
                            ),
                        ]
                    ),
                ]
            ),
        ]
    )

    @app.callback(
        [
            Output("umap-plot", "figure"),
            Output("export-button", "disabled"),
            Output("umap-params-store", "data"),
            Output("computation-status", "data"),
            Output("run-button", "color"),
        ],
        [Input("run-button", "n_clicks"), Input("computation-check", "n_intervals")],
        [
            State("n-neighbors", "value"),
            State("min-dist", "value"),
            State("distance-metric", "value"),
            State("densmap", "value"),
            State("wavenumber-ranges", "value"),
            State("computation-status", "data"),
        ],
    )
    def handle_umap_computation(
        n_clicks,
        n_intervals,
        n_neighbors,
        min_dist,
        metric,
        densmap,
        wavenumber_ranges,
        status,
    ):
        ctx = dash.callback_context
        trigger = ctx.triggered[0]["prop_id"].split(".")[0]

        if trigger == "run-button" and n_clicks:
            # Get wavenumber indices
            wavenumber_indices = parse_wavenumber_ranges(
                wavenumber_ranges, app.wavenumbers
            )

            # Store all parameters including wavenumber selection
            params = {
                "n_neighbors": n_neighbors,
                "min_dist": min_dist,
                "metric": metric,
                "densmap": densmap,
                "selected_wavenumbers": wavenumber_indices.tolist(),  # Store for reference
            }
            app.current_params = params

            # Start the computation process
            app.result_queue = Queue()
            app.compute_process = Process(
                target=run_umap_process,
                args=(app.input_data, params, wavenumber_indices, app.result_queue),
            )
            app.compute_process.start()

            return dash.no_update, True, params, "computing", "warning"

        elif trigger == "computation-check" and status == "computing":
            # Check if computation is done
            try:
                if not hasattr(app, "result_queue"):
                    return dash.no_update, True, dash.no_update, "idle", "primary"

                if not app.result_queue.empty():
                    # Get the result
                    embedding = app.result_queue.get_nowait()
                    app.embedding = embedding
                    app.compute_process.join()  # Clean up the process

                    figure = {
                        "data": [
                            {
                                "type": "scatter",
                                "x": embedding[:, 0],
                                "y": embedding[:, 1],
                                "mode": "markers",
                                "marker": {"size": 5},
                            }
                        ],
                        "layout": {
                            "title": "UMAP Projection",
                            "xaxis": {"title": "UMAP1"},
                            "yaxis": {
                                "title": "UMAP2",
                                "scaleanchor": "x",
                                "scaleratio": 1,
                            },
                            "dragmode": "lasso",
                        },
                    }
                    return (
                        figure,
                        False,
                        dash.no_update,
                        "idle",
                        "success",
                    )  # Green on success

                return dash.no_update, True, dash.no_update, "computing", "warning"

            except Exception as e:
                print(f"Error during UMAP computation: {str(e)}")
                return (
                    dash.no_update,
                    True,
                    dash.no_update,
                    "idle",
                    "danger",
                )  # Red on error

        return dash.no_update, True, dash.no_update, status, "primary"

    @app.callback(
        [Output("export-button", "n_clicks"), Output("export-button", "color")],
        [Input("export-button", "n_clicks")],
        [State("umap-params-store", "data")],
    )
    def handle_export(n_clicks, params):
        if not n_clicks:
            raise PreventUpdate

        if not hasattr(app, "embedding"):
            print("Warning: No embedding to export")
            return None, "danger"

        if app.export_connector is None:
            print("Warning: No export connector configured")
            return None, "danger"

        try:
            success = app.export_connector.export_embedding(app.embedding, params)
            print(f"Export {'successful' if success else 'failed'}")
            return None, "success" if success else "danger"
        except Exception as e:
            print(f"Error during export: {str(e)}")
            return None, "danger"

    @app.callback(
        Output("spectrum-plot", "figure"), [Input("umap-plot", "selectedData")]
    )
    def display_spectrum_stats(selectedData):
        if selectedData is None or len(selectedData["points"]) == 0:
            # Show an empty plot with a message when no points are selected
            return {
                "data": [],
                "layout": {
                    "title": "Use lasso tool to select points",
                    "xaxis": {
                        "title": "Wavenumber",
                        "autorange": (
                            "reversed" if app.reverse_wavenumbers else "normal"
                        ),
                    },
                    "yaxis": {"title": "Intensity"},
                    "annotations": [
                        {
                            "text": "Select points in the UMAP plot using the lasso tool",
                            "xref": "paper",
                            "yref": "paper",
                            "showarrow": False,
                            "x": 0.5,
                            "y": 0.5,
                        }
                    ],
                },
            }

        # Get indices of selected points
        selected_indices = [p["pointIndex"] for p in selectedData["points"]]

        # Get the corresponding spectra
        selected_spectra = app.input_data[selected_indices]

        # Calculate percentiles
        percentiles = np.percentile(selected_spectra, [5, 25, 50, 75, 95], axis=0)

        traces = []

        # Add 5-95 percentile band (lighter shade)
        traces.append(
            {
                "type": "scatter",
                "x": app.wavenumbers,
                "y": percentiles[4],  # 95th percentile
                "mode": "lines",
                "line": {"width": 0},
                "showlegend": False,
            }
        )
        traces.append(
            {
                "type": "scatter",
                "x": app.wavenumbers,
                "y": percentiles[0],  # 5th percentile
                "mode": "lines",
                "line": {"width": 0},
                "fill": "tonexty",
                "fillcolor": "rgba(0,0,255,0.1)",
                "name": "5-95 percentile range",
            }
        )

        # Add 25-75 percentile band (darker shade)
        traces.append(
            {
                "type": "scatter",
                "x": app.wavenumbers,
                "y": percentiles[3],  # 75th percentile
                "mode": "lines",
                "line": {"width": 0},
                "showlegend": False,
            }
        )
        traces.append(
            {
                "type": "scatter",
                "x": app.wavenumbers,
                "y": percentiles[1],  # 25th percentile
                "mode": "lines",
                "line": {"width": 0},
                "fill": "tonexty",
                "fillcolor": "rgba(0,0,255,0.2)",
                "name": "25-75 percentile range",
            }
        )

        # Add median line
        traces.append(
            {
                "type": "scatter",
                "x": app.wavenumbers,
                "y": percentiles[2],
                "mode": "lines",
                "line": {"color": "rgba(0,0,255,1)", "width": 2},
                "name": "Median",
            }
        )

        figure = {
            "data": traces,
            "layout": {
                "title": f"Statistical summary of {len(selected_indices)} selected spectra",
                "xaxis": {
                    "title": "Wavenumber",
                    "autorange": "reversed" if app.reverse_wavenumbers else "normal",
                },
                "yaxis": {"title": "Intensity"},
                "showlegend": True,
                "hovermode": "closest",
            },
        }

        return figure

    @app.callback(
        Output("positions-plot", "figure"), [Input("umap-plot", "selectedData")]
    )
    def update_positions_plot(selectedData):
        # Create base scatter plot with all points in grey
        traces = [
            {
                "type": "scatter",
                "x": app.positions[:, 0],
                "y": app.positions[:, 1],
                "mode": "markers",
                "marker": {"size": 8, "color": "lightgrey", "opacity": 0.6},
                "name": "Unselected Points",
            }
        ]

        # If points are selected, add them as a new trace with different color
        if selectedData and len(selectedData["points"]) > 0:
            selected_indices = [p["pointIndex"] for p in selectedData["points"]]
            selected_positions = app.positions[selected_indices]

            traces.append(
                {
                    "type": "scatter",
                    "x": selected_positions[:, 0],
                    "y": selected_positions[:, 1],
                    "mode": "markers",
                    "marker": {"size": 8, "color": "blue", "opacity": 0.8},
                    "name": "Selected Points",
                }
            )

        figure = {
            "data": traces,
            "layout": {
                "title": "Spatial Distribution",
                "xaxis": {"title": "X Position"},
                "yaxis": {
                    "title": "Y Position",
                    "scaleanchor": "x",  # Lock the aspect ratio
                    "scaleratio": 1,  # 1:1 ratio
                },
                "showlegend": True,
                "hovermode": "closest",
            },
        }

        return figure

    @app.callback(
        Output("selection-store", "data"), [Input("umap-plot", "selectedData")]
    )
    def store_selection(selectedData):
        if selectedData is None:
            return []
        return [p["pointIndex"] for p in selectedData["points"]]

    return app


def parse_wavenumber_ranges(range_string: str, wavenumbers: np.ndarray) -> np.ndarray:
    """Parse wavenumber range string and return indices of selected wavenumbers

    Format: "600-1200,1500-1600"
    """
    # Convert wavenumbers to numpy array if it isn't already
    wavenumbers = np.asarray(wavenumbers)

    if not range_string or not range_string.strip():
        return np.arange(len(wavenumbers))

    ranges = []
    for range_str in range_string.split(","):
        try:
            start, end = map(float, range_str.strip().split("-"))
            # Ensure we're using numpy boolean array
            mask = np.logical_and(wavenumbers >= start, wavenumbers <= end)
            ranges.append(mask)
        except ValueError:
            print(f"Warning: Invalid range format: {range_str}")
            continue

    if not ranges:
        return np.arange(len(wavenumbers))

    # Combine all masks with OR operation
    final_mask = np.zeros_like(wavenumbers, dtype=bool)
    for mask in ranges:
        final_mask = np.logical_or(final_mask, mask)

    return np.where(final_mask)[0]
