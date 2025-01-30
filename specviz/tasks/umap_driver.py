import dash
import dash_bootstrap_components as dbc
import numpy as np
import umap
from dash import Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate


def create_umap_app(
    input_data=None, wavenumbers=None, positions=None, reverse_wavenumbers=False
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

    # Layout definition with added positions plot
    app.layout = dbc.Container(
        [
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
                            dbc.Button(
                                "Run UMAP",
                                id="run-button",
                                color="primary",
                                className="mt-3",
                            ),
                        ]
                    ),
                ],
                className="mb-3",
            ),
            # Results section with three plots
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
        Output("umap-plot", "figure"),
        [Input("run-button", "n_clicks")],
        [
            State("n-neighbors", "value"),
            State("min-dist", "value"),
            State("distance-metric", "value"),
            State("densmap", "value"),
        ],
    )
    def run_umap(n_clicks, n_neighbors, min_dist, metric, densmap):
        if n_clicks is None:
            raise PreventUpdate

        reducer = umap.UMAP(
            n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, densmap=densmap
        )

        embedding = reducer.fit_transform(app.input_data)
        app.embedding = embedding

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
                "yaxis": {"title": "UMAP2"},
                "dragmode": "lasso",  # Enable lasso selection
            },
        }

        return figure

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
                "yaxis": {"title": "Y Position"},
                "showlegend": True,
                "hovermode": "closest",
            },
        }

        return figure

    return app
