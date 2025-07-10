from typing import Dict, Union

import argparse
import sqlite3

import dash
import dash_bootstrap_components as dbc
import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import redis
from dash import dcc, html
from dash.dependencies import Input, Output, State

from tools.dbtools import (
    add_column_to_table,
    append_df_to_table,
    check_table_exists,
    fetch_dict_from_db,
    read_df_from_db,
    store_df_in_db,
)
from specviz.tasks.model_loader import ModelLoader


class MeasurementPlanner:
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379):
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

        # Get initial level value
        initial_level = self._get_initial_level()

        # Redis setup
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_client = None
        self.redis_status = "Not Connected"
        self.redis_status_color = "danger"
        self._connect_redis()

        self.model_loader = ModelLoader()

        self.app.layout = dbc.Container(
            [
                html.H1("Measurement Planner", className="my-4"),
                # Redis Status Alert
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
                # Level Selection and Plot
                dbc.Row(
                    [
                        # Controls Column
                        dbc.Col(
                            [
                                # Level Selection Card
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            [
                                                "Level Selection",
                                                dbc.Button(
                                                    "Refresh",
                                                    id="plot-refresh-button",
                                                    color="light",
                                                    size="sm",
                                                    className="float-end",
                                                ),
                                            ]
                                        ),
                                        dbc.CardBody(
                                            [
                                                dbc.Label("Maximum Level:"),
                                                dcc.Slider(
                                                    id="level-slider",
                                                    min=0,
                                                    max=10,
                                                    step=1,
                                                    value=initial_level,
                                                    marks=None,
                                                ),
                                                html.Div(
                                                    id="points-count", className="mt-3"
                                                ),
                                            ]
                                        ),
                                    ],
                                    className="mb-3",
                                ),  # Add margin bottom
                                # Model Selection Card
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Model Selection"),
                                        dbc.CardBody(
                                            [
                                                dbc.Select(
                                                    id="model-select",
                                                    options=[
                                                        {
                                                            "label": "No Model",
                                                            "value": "",
                                                        }
                                                    ],
                                                    value="",
                                                    placeholder="Select a model...",
                                                ),
                                                html.Div(
                                                    id="model-status", className="mt-2"
                                                ),
                                                dbc.Spinner(
                                                    html.Div(
                                                        id="model-loading-spinner"
                                                    ),
                                                    size="sm",
                                                    color="primary",
                                                ),
                                                dbc.Label(
                                                    "Threshold Mode:", className="mt-3"
                                                ),
                                                dbc.RadioItems(
                                                    id="threshold-mode",
                                                    options=[
                                                        {
                                                            "label": "Manual",
                                                            "value": "manual",
                                                        },
                                                        {
                                                            "label": "Conformal",
                                                            "value": "conformal",
                                                        },
                                                    ],
                                                    value="manual",
                                                    inline=True,
                                                    className="mb-2",
                                                ),
                                                html.Div(
                                                    [
                                                        dbc.Label(
                                                            "Value Cutoff:",
                                                            className="mt-2",
                                                        ),
                                                        dcc.Slider(
                                                            id="value-cutoff-slider",
                                                            min=0,
                                                            max=1,
                                                            step=0.01,
                                                            value=0,
                                                            marks=None,
                                                            tooltip={
                                                                "placement": "bottom",
                                                                "always_visible": True,
                                                            },
                                                        ),
                                                    ],
                                                    id="manual-threshold-div",
                                                ),
                                                html.Div(
                                                    [
                                                        dbc.Label(
                                                            "Conformal Threshold:",
                                                            className="mt-2",
                                                        ),
                                                        dbc.Select(
                                                            id="conformal-threshold-select",
                                                            options=[
                                                                {
                                                                    "label": "1%",
                                                                    "value": "0.01",
                                                                },
                                                                {
                                                                    "label": "5%",
                                                                    "value": "0.05",
                                                                },
                                                                {
                                                                    "label": "10%",
                                                                    "value": "0.10",
                                                                },
                                                                {
                                                                    "label": "20%",
                                                                    "value": "0.20",
                                                                },
                                                                {
                                                                    "label": "25%",
                                                                    "value": "0.25",
                                                                },
                                                            ],
                                                            value="0.01",
                                                        ),
                                                    ],
                                                    id="conformal-threshold-div",
                                                ),
                                            ]
                                        ),
                                    ]
                                ),
                                # Random Draw Card (new)
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Random Draw Settings"),
                                        dbc.CardBody(
                                            [
                                                dbc.Label("Maximum Points to Draw:"),
                                                dbc.Input(
                                                    id="max-draw-input",
                                                    type="number",
                                                    min=1,
                                                    step=1,
                                                    value=100,
                                                    style={"width": "100%"},
                                                ),
                                                dbc.Button(
                                                    "Random Draw",
                                                    id="random-draw-button",
                                                    color="primary",
                                                    className="mt-3 w-100",
                                                ),
                                                html.Div(
                                                    id="draw-status", className="mt-2"
                                                ),
                                            ]
                                        ),
                                    ],
                                    className="mt-3",
                                ),
                                # Export Selection Button
                                dbc.Button(
                                    "Export Selection",
                                    id="export-selection-button",
                                    color="primary",
                                    className="mt-3 w-100",
                                    disabled=False,
                                ),
                                html.Div(id="export-status", className="mt-2"),
                            ],
                            width=3,
                        ),
                        # Plot
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Point Distribution"),
                                        dbc.CardBody(
                                            [
                                                dcc.Graph(
                                                    id="point-plot",
                                                    config={
                                                        "doubleClick": "reset",
                                                        "displayModeBar": True,
                                                    },
                                                )
                                            ]
                                        ),
                                    ]
                                )
                            ],
                            width=9,
                        ),
                    ]
                ),
                # Store for preserving zoom state
                dcc.Store(id="zoom-state", data=None),
                # Update interval
                dcc.Interval(id="redis-check", interval=5000),  # Redis status check
                dcc.Interval(
                    id="model-status-check", interval=500
                ),  # Model status check
            ],
            fluid=True,
        )

        self._setup_callbacks()

    def _connect_redis(self):
        try:
            self.redis_client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                socket_connect_timeout=1,
                decode_responses=True,
            )
            self.redis_client.ping()
            self.redis_status = f"Connected to {self.redis_host}:{self.redis_port}"
            self.redis_status_color = "success"
        except redis.ConnectionError as e:
            self.redis_status = (
                f"Connection Failed: Redis not running on {self.redis_host}:{self.redis_port}. "
                "Please ensure Redis server is started."
            )
            self.redis_status_color = "danger"
            self.redis_client = None
        except Exception as e:
            self.redis_status = f"Error: {str(e)}"
            self.redis_status_color = "danger"
            self.redis_client = None

    def _setup_callbacks(self):
        @self.app.callback(
            [Output("redis-status", "children"), Output("redis-alert", "color")],
            [
                Input("redis-check", "n_intervals"),
                Input("redis-refresh-button", "n_clicks"),
            ],
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
            Output("zoom-state", "data"),
            [
                Input("point-plot", "relayoutData"),
                Input("point-plot", "restyleData"),
            ],  # Add restyle data to capture double-click
            [State("zoom-state", "data")],
        )
        def store_zoom_state(relayoutData, restyleData, current_state):
            # Check if this is a double-click reset event
            if relayoutData and "autosize" in relayoutData:
                return None  # Clear zoom state on reset

            if not relayoutData:
                return current_state

            zoom_data = {}
            # Capture both direct range updates and autorange events
            range_keys = [
                "xaxis.range[0]",
                "xaxis.range[1]",
                "yaxis.range[0]",
                "yaxis.range[1]",
                "xaxis.autorange",
                "yaxis.autorange",
            ]

            for key in range_keys:
                if key in relayoutData:
                    zoom_data[key] = relayoutData[key]

            # If we have new zoom data, return it; otherwise keep current state
            return zoom_data if zoom_data else current_state

        @self.app.callback(
            [
                Output("point-plot", "figure"),
                Output("points-count", "children"),
                Output("level-slider", "max"),
                Output("level-slider", "marks"),
                Output("level-slider", "value"),
            ],
            [
                Input("level-slider", "value"),
                Input("plot-refresh-button", "n_clicks"),
                Input("model-select", "value"),
                Input("value-cutoff-slider", "value"),
                Input("random-draw-button", "n_clicks"),
            ],
            [State("zoom-state", "data"), State("max-draw-input", "value")],
        )
        def update_plot_and_slider(
            level_value,
            refresh_clicks,
            selected_model,
            cutoff_value,
            draw_clicks,
            zoom_state,
            max_draw,
        ):
            if level_value is None:
                level_value = 0

            ctx = dash.callback_context
            triggered_id = (
                ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None
            )

            try:
                if not self.redis_client:
                    raise ValueError("Redis not connected")

                db_path = self.redis_client.get("current_project")
                if not db_path:
                    raise ValueError("No active project")

                conn = duckdb.connect(db_path)
                if not check_table_exists(conn, "HCD"):
                    raise ValueError("HCD table not found")

                # Read HCD data for coordinates and levels
                df = read_df_from_db(conn, "HCD")
                df["hcd_indx"] = np.arange(len(df))  # Add index if not present

                # Get measured points from measured_points table
                measured_mask = pd.Series(False, index=df.index)
                measured_max_level = 0
                if check_table_exists(conn, "measured_points"):
                    measured_df = read_df_from_db(conn, "measured_points")
                    iteration_cols = [
                        col
                        for col in measured_df.columns
                        if col.startswith("iteration_")
                    ]
                    if iteration_cols:
                        measured_mask = measured_df[iteration_cols].eq(1).any(axis=1)
                        if measured_mask.any():
                            measured_max_level = df[measured_mask]["level"].max()
                            print(
                                f"Maximum level in measured points: {measured_max_level}"
                            )

                # Get candidate points
                candidate_points = df[~measured_mask]
                available_levels = sorted(candidate_points["level"].unique())

                # Find nearest available level above measured_max_level
                if not available_levels:
                    print("No available levels found")
                    adjusted_level = measured_max_level + 1
                else:
                    # Only adjust level if refresh button was clicked
                    if triggered_id == "plot-refresh-button":
                        # Filter for levels above measured_max_level
                        higher_levels = [
                            l for l in available_levels if l > measured_max_level
                        ]
                        if higher_levels:
                            adjusted_level = min(higher_levels)
                            print(
                                f"Selected next available level above measured: {adjusted_level}"
                            )
                        else:
                            adjusted_level = measured_max_level + 1
                            print(
                                f"No higher levels available, using measured_max_level + 1: {adjusted_level}"
                            )
                    else:
                        # Keep current level_value
                        adjusted_level = level_value

                # Update max_level and marks - ensure marks keys are strings
                max_level = (
                    max(available_levels)
                    if available_levels
                    else measured_max_level + 1
                )
                marks = (
                    {str(i): str(i) for i in sorted(available_levels)}
                    if available_levels
                    else {"0": "0"}
                )

                print(f"Available levels: {available_levels}")
                print(
                    f"Current level: {adjusted_level}, Max measured level: {measured_max_level}"
                )

                # Use adjusted level for filtering
                filtered_df = candidate_points[
                    candidate_points["level"] <= adjusted_level
                ]

                # Get model values if a model is selected
                color_values = filtered_df["level"]  # Default coloring by level
                color_label = "Level"

                if selected_model:
                    model_df = read_df_from_db(conn, selected_model)
                    # Join with filtered_df using hcd_indx
                    filtered_df = filtered_df.merge(
                        model_df[["hcd_indx", "value"]], on="hcd_indx", how="left"
                    )

                    # Apply cutoff filter
                    filtered_df = filtered_df[filtered_df["value"] >= cutoff_value]
                    color_values = filtered_df["value"]
                    color_label = f"Model Value ({selected_model})"

                conn.close()

                # Create figure with two traces
                fig = go.Figure()

                # Add measured points (red)
                if measured_mask.any():
                    fig.add_trace(
                        go.Scatter(
                            x=df[measured_mask]["X"].tolist(),
                            y=df[measured_mask]["Y"].tolist(),
                            mode="markers",
                            name="Measured",
                            marker=dict(color="red", size=5),
                            showlegend=True,
                        )
                    )

                # Add candidate points with model values if available
                if selected_model:
                    fig.add_trace(
                        go.Scatter(
                            x=filtered_df["X"].tolist(),
                            y=filtered_df["Y"].tolist(),
                            mode="markers",
                            name="Candidates",
                            marker=dict(
                                color=filtered_df[
                                    "value"
                                ].tolist(),  # Color by model value
                                colorscale="Viridis",
                                showscale=True,
                                colorbar=dict(title=f"Model Value ({selected_model})"),
                                size=filtered_df["level"].map(
                                    lambda x: 6 + x * 2
                                ),  # Size by level
                                sizemode="diameter",
                                sizeref=1,
                            ),
                            text=[
                                f"Level: {l}<br>Probability: {v:.3f}"
                                for l, v in zip(
                                    filtered_df["level"], filtered_df["value"]
                                )
                            ],
                            hovertemplate="%{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>",
                            showlegend=True,
                        )
                    )
                else:
                    # If no model selected, color by level
                    fig.add_trace(
                        go.Scatter(
                            x=filtered_df["X"].tolist(),
                            y=filtered_df["Y"].tolist(),
                            mode="markers",
                            name="Candidates",
                            marker=dict(
                                color=filtered_df["level"].tolist(),
                                colorscale="Viridis",
                                showscale=True,
                                colorbar=dict(title="Level"),
                                size=8,
                            ),
                            text=[f"Level: {l}" for l in filtered_df["level"]],
                            hovertemplate="%{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>",
                            showlegend=True,
                        )
                    )

                # Add random draw points if button was clicked
                if triggered_id == "random-draw-button" and not filtered_df.empty:
                    n_points = (
                        min(max_draw, len(filtered_df))
                        if max_draw
                        else len(filtered_df)
                    )
                    random_indices = np.random.choice(
                        filtered_df.index, size=n_points, replace=False
                    )
                    random_selection = filtered_df.loc[random_indices]

                    hover_text = [f"Level: {l}" for l in random_selection["level"]]
                    if selected_model:
                        hover_text = [
                            f"Level: {l}<br>Probability: {v:.3f}"
                            for l, v in zip(
                                random_selection["level"], random_selection["value"]
                            )
                        ]

                    fig.add_trace(
                        go.Scatter(
                            x=random_selection["X"].tolist(),
                            y=random_selection["Y"].tolist(),
                            mode="markers",
                            name="Random Draw",
                            marker=dict(
                                color="pink",
                                size=4,
                                symbol="circle",
                            ),
                            text=hover_text,
                            hovertemplate="%{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>",
                            showlegend=True,
                        )
                    )

                # Update layout
                fig.update_layout(
                    title=f"Points Distribution (Level ≤ {adjusted_level}, {len(measured_mask[measured_mask])} measured, {len(filtered_df)} candidates)",
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    margin=dict(l=20, r=20, t=40, b=20),
                    yaxis=dict(
                        scaleanchor="x",
                        scaleratio=1,
                    ),
                    showlegend=True,
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                )

                # Apply zoom state or use full dataset limits
                if zoom_state:
                    if any(
                        key.endswith("autorange") and zoom_state[key]
                        for key in zoom_state
                    ):
                        fig.update_layout(xaxis_autorange=True, yaxis_autorange=True)
                    else:
                        fig.update_layout(
                            xaxis_range=[
                                zoom_state.get("xaxis.range[0]"),
                                zoom_state.get("xaxis.range[1]"),
                            ],
                            yaxis_range=[
                                zoom_state.get("yaxis.range[0]"),
                                zoom_state.get("yaxis.range[1]"),
                            ],
                        )
                else:
                    x_range = [df["X"].min(), df["X"].max()]
                    y_range = [df["Y"].min(), df["Y"].max()]
                    fig.update_layout(
                        xaxis_range=x_range,
                        yaxis_range=y_range,
                        xaxis_autorange=False,
                        yaxis_autorange=False,
                    )

                # Update axes styling
                fig.update_xaxes(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor="LightGray",
                    zeroline=True,
                    zerolinewidth=2,
                    zerolinecolor="Gray",
                )

                fig.update_yaxes(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor="LightGray",
                    zeroline=True,
                    zerolinewidth=2,
                    zerolinecolor="Gray",
                )

                point_count = html.Div(
                    [
                        html.Div(
                            f"Showing {len(measured_mask[measured_mask])} measured points and {len(filtered_df)} candidates"
                        ),
                        html.Div(
                            f"Maximum measured level: {measured_max_level}",
                            className="text-muted",
                        ),
                        html.Div(
                            f"Available levels: {', '.join(map(str, available_levels))}",
                            className="text-muted small",
                        ),
                    ]
                )

                return fig.to_dict(), point_count, max_level, marks, adjusted_level

            except Exception as e:
                print(f"Error in update_plot_and_slider: {str(e)}")
                empty_fig = px.scatter(title="No data available")
                return empty_fig.to_dict(), f"Error: {str(e)}", 0, {"0": "0"}, 0

        @self.app.callback(
            Output("model-select", "options"),
            [Input("plot-refresh-button", "n_clicks")],
        )
        def update_model_options(n_clicks):
            try:
                if not self.redis_client:
                    raise ValueError("Redis not connected")

                db_path = self.redis_client.get("current_project")
                if not db_path:
                    raise ValueError("No active project")

                models = self.model_loader.get_available_models(db_path)
                options = [{"label": "No Model", "value": ""}]  # Add empty option
                options.extend([{"label": model, "value": model} for model in models])
                return options

            except Exception as e:
                print(f"Error updating model options: {str(e)}")
                return [{"label": "No Model", "value": ""}]

        @self.app.callback(
            [
                Output("model-status", "children"),
                Output("model-status", "className"),
                Output("model-loading-spinner", "children"),
            ],
            [
                Input("model-select", "value"),
                Input("model-status-check", "n_intervals"),
            ],
        )
        def handle_model_loading(selected_model, n_intervals):
            ctx = dash.callback_context
            trigger = ctx.triggered[0]["prop_id"].split(".")[0]

            # Create base table structure
            def create_threshold_table(thresholds=None):
                return dbc.Table(
                    [
                        html.Thead(
                            [html.Tr([html.Th("Percentile"), html.Th("Threshold")])]
                        ),
                        html.Tbody(
                            [
                                html.Tr(
                                    [
                                        html.Td("1%"),
                                        html.Td(
                                            f"{thresholds['threshold_0.01'].iloc[0]:.3f}"
                                            if thresholds is not None
                                            else "-"
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td("10%"),
                                        html.Td(
                                            f"{thresholds['threshold_0.10'].iloc[0]:.3f}"
                                            if thresholds is not None
                                            else "-"
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td("20%"),
                                        html.Td(
                                            f"{thresholds['threshold_0.20'].iloc[0]:.3f}"
                                            if thresholds is not None
                                            else "-"
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td("25%"),
                                        html.Td(
                                            f"{thresholds['threshold_0.25'].iloc[0]:.3f}"
                                            if thresholds is not None
                                            else "-"
                                        ),
                                    ]
                                ),
                            ]
                        ),
                    ],
                    bordered=True,
                    size="sm",
                    hover=True,
                )

            if trigger == "model-select":
                try:
                    if not selected_model:  # No model selected
                        return create_threshold_table(), "mt-3", False

                    if not self.redis_client:
                        raise ValueError("Redis not connected")

                    db_path = self.redis_client.get("current_project")
                    if not db_path:
                        raise ValueError("No active project")

                    # Read conformal thresholds
                    conn = duckdb.connect(db_path)
                    if check_table_exists(conn, "conformal_thresholds"):
                        thresholds_df = read_df_from_db(conn, "conformal_thresholds")
                        model_thresholds = thresholds_df[
                            thresholds_df["model_name"] == selected_model
                        ]

                        if not model_thresholds.empty:
                            return (
                                create_threshold_table(model_thresholds),
                                "mt-3",
                                False,
                            )

                    conn.close()
                    self.model_loader.load_model(db_path, selected_model)
                    return create_threshold_table(), "text-info", True

                except Exception as e:
                    return f"Error: {str(e)}", "text-danger", False

            # Check for status updates
            status = self.model_loader.get_loading_status()
            if status:
                status_type, message = status
                className = {
                    "info": "text-info",
                    "success": "text-success",
                    "error": "text-danger",
                    "warning": "text-warning",
                }.get(status_type, "text-info")

                return message, className, False

            return dash.no_update, dash.no_update, dash.no_update

        @self.app.callback(
            [
                Output("manual-threshold-div", "style"),
                Output("conformal-threshold-div", "style"),
            ],
            [Input("threshold-mode", "value")],
        )
        def toggle_threshold_mode(mode):
            if mode == "manual":
                return {"display": "block"}, {"display": "none"}
            else:
                return {"display": "none"}, {"display": "block"}

        @self.app.callback(
            Output("value-cutoff-slider", "value"),
            [
                Input("threshold-mode", "value"),
                Input("conformal-threshold-select", "value"),
                Input("model-select", "value"),
            ],
            [State("value-cutoff-slider", "value")],
        )
        def update_threshold_value(
            mode, conformal_value, selected_model, current_slider_value
        ):
            if not selected_model or mode == "manual":
                return current_slider_value

            try:
                if not self.redis_client:
                    raise ValueError("Redis not connected")

                db_path = self.redis_client.get("current_project")
                if not db_path:
                    raise ValueError("No active project")

                conn = duckdb.connect(db_path)
                thresholds_df = read_df_from_db(conn, "conformal_thresholds")
                model_thresholds = thresholds_df[
                    thresholds_df["model_name"] == selected_model
                ]

                if not model_thresholds.empty:
                    threshold_col = f"threshold_{conformal_value}"
                    return model_thresholds[threshold_col].iloc[0]

                conn.close()
                return current_slider_value

            except Exception as e:
                print(f"Error updating threshold value: {str(e)}")
                return current_slider_value

        @self.app.callback(
            [
                Output("export-status", "children"),
                Output("export-selection-button", "color"),
                Output("export-selection-button", "disabled"),
            ],
            [Input("export-selection-button", "n_clicks")],
            [
                State("level-slider", "value"),
                State("model-select", "value"),
                State("value-cutoff-slider", "value"),
                State("random-draw-button", "n_clicks"),
                State("max-draw-input", "value"),
            ],
        )
        def export_selection(
            n_clicks, level_value, selected_model, cutoff_value, draw_clicks, max_draw
        ):
            if not n_clicks:  # Don't run on initial load
                return None, "primary", False

            try:
                if not self.redis_client:
                    raise ValueError("Redis not connected")

                db_path = self.redis_client.get("current_project")
                if not db_path:
                    raise ValueError("No active project")

                conn = duckdb.connect(db_path)

                # Read HCD data
                df = read_df_from_db(conn, "HCD")
                df["hcd_indx"] = np.arange(len(df))  # Ensure we have indices

                # Initialize selection with level filter
                selection_mask = df["level"] <= level_value

                # Apply model filter if selected
                if selected_model:
                    model_df = read_df_from_db(conn, selected_model)
                    # Merge with main df to get values
                    df = df.merge(
                        model_df[["hcd_indx", "value"]], on="hcd_indx", how="left"
                    )
                    selection_mask &= df["value"] >= cutoff_value

                # Get measured points status
                if check_table_exists(conn, "measured_points"):
                    measured_df = read_df_from_db(conn, "measured_points")
                    iteration_cols = [
                        col
                        for col in measured_df.columns
                        if col.startswith("iteration_")
                    ]
                    if iteration_cols:
                        measured_mask = measured_df[iteration_cols].eq(1).any(axis=1)
                        selection_mask &= ~measured_mask

                # Get filtered dataset
                filtered_df = df[selection_mask]

                # If we have a random draw, use only those points
                if draw_clicks:
                    n_points = (
                        min(max_draw, len(filtered_df))
                        if max_draw
                        else len(filtered_df)
                    )
                    random_indices = np.random.choice(
                        filtered_df.index, size=n_points, replace=False
                    )
                    selected_indices = filtered_df.loc[random_indices][
                        "hcd_indx"
                    ].tolist()
                else:
                    return (
                        html.Div(
                            "Please make a random draw first", className="text-warning"
                        ),
                        "warning",
                        False,
                    )

                # Create or update measured_points table
                if not check_table_exists(conn, "measured_points"):
                    measured_df = pd.DataFrame(
                        {"hcd_indx": df["hcd_indx"], "planned_next": 0}
                    )
                    store_df_in_db(conn, measured_df, "measured_points", index=False)
                else:
                    # Reset all planned_next values
                    measured_df = read_df_from_db(conn, "measured_points")
                    measured_df["planned_next"] = 0

                # Update planned_next for selected points
                measured_df.loc[
                    measured_df["hcd_indx"].isin(selected_indices), "planned_next"
                ] = 1

                # Store updated measured_points
                store_df_in_db(
                    conn,
                    measured_df,
                    "measured_points",
                    if_exists="replace",
                    index=False,
                )

                n_selected = len(selected_indices)
                conn.close()

                if n_selected == 0:
                    return (
                        html.Div(
                            "No points selected for export", className="text-warning"
                        ),
                        "warning",
                        False,
                    )

                return (
                    html.Div(
                        f"Successfully exported {n_selected} randomly drawn points",
                        className="text-success",
                    ),
                    "success",
                    False,
                )

            except Exception as e:
                if "conn" in locals():
                    conn.close()
                return (
                    html.Div(f"Error during export: {str(e)}", className="text-danger"),
                    "danger",
                    False,
                )

    def _get_initial_level(self) -> int:
        """Calculate initial level value based on measured points"""
        try:
            print("\n=== Calculating Initial Level ===")
            if not self.redis_client:
                print("No Redis client - defaulting to level 0")
                return 0

            db_path = self.redis_client.get("current_project")
            if not db_path:
                print("No active project - defaulting to level 0")
                return 0

            conn = duckdb.connect(db_path)
            print(f"Connected to database: {db_path}")

            # Check if measured_points table exists
            if not check_table_exists(conn, "measured_points"):
                print("No measured_points table found - defaulting to level 0")
                conn.close()
                return 0

            # Get measured points
            measured_df = read_df_from_db(conn, "measured_points")
            hcd_df = read_df_from_db(conn, "HCD")
            print(f"Found {len(measured_df)} entries in measured_points table")

            # Get indices of measured points
            iteration_cols = [
                col for col in measured_df.columns if col.startswith("iteration_")
            ]
            if iteration_cols:
                measured_mask = measured_df[iteration_cols].eq(1).any(axis=1)
                measured_indices = measured_df[measured_mask]["hcd_indx"].tolist()
                print(f"Found {len(measured_indices)} measured points")

                # Get levels of measured points
                measured_levels = hcd_df[hcd_df["hcd_indx"].isin(measured_indices)][
                    "level"
                ]

                if not measured_levels.empty:
                    # Set initial level to max measured level + 1
                    initial_level = int(measured_levels.max() + 1)
                    print(f"Max measured level: {measured_levels.max()}")
                    print(f"Setting initial level to: {initial_level}")
                    conn.close()
                    return initial_level

            print("No measured points found - defaulting to level 0")
            conn.close()
            return 0

        except Exception as e:
            print(f"Error calculating initial level: {str(e)}")
            return 0

    def run(self, debug=False, port=8053):
        self.app.run_server(debug=debug, port=port)


def main():
    parser = argparse.ArgumentParser(
        description="Start the Measurement Planner application"
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
        default=8053,
        help="Port to run the Dash app on (default: 8053)",
    )
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")

    args = parser.parse_args()

    app = MeasurementPlanner(redis_host=args.redis_host, redis_port=args.redis_port)
    app.run(debug=args.debug, port=args.port)


if __name__ == "__main__":
    main()
