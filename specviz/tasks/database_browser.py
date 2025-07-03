from typing import Dict, Union

import argparse

import dash
import dash_bootstrap_components as dbc
import duckdb
import pandas as pd
import redis
from dash import dcc, html
from dash.dependencies import Input, Output, State

from specviz.tasks.dbtools import check_table_exists, read_df_from_db


class DatabaseBrowser:
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379):
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

        # Redis setup
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_client = None
        self.redis_status = "Not Connected"
        self.redis_status_color = "danger"
        self._connect_redis()

        self.app.layout = dbc.Container(
            [
                html.H1("Database Browser", className="my-4"),
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
                                        html.Div(
                                            id="current-experiment", className="mt-2"
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
                                    className="d-flex align-items-start",
                                ),
                            ]
                        )
                    ],
                    id="redis-alert",
                    color="danger",
                    className="mb-3",
                ),
                # Database Content
                dbc.Card(
                    [
                        dbc.CardHeader("Database Tables"),
                        dbc.CardBody(
                            [
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                dcc.Dropdown(
                                                    id="table-selector",
                                                    placeholder="Select a table to inspect",
                                                    className="mb-3",
                                                ),
                                                # Add refresh button for random rows
                                                dbc.Button(
                                                    "Draw Random Rows",
                                                    id="draw-rows-button",
                                                    color="secondary",
                                                    size="sm",
                                                    className="mb-3",
                                                ),
                                                # Table preview will appear right below
                                                html.Div(
                                                    id="table-preview", className="mt-3"
                                                ),
                                            ],
                                            width=8,
                                        ),
                                        # Summary stats in a collapsible card on the right
                                        dbc.Col(
                                            [
                                                dbc.Button(
                                                    "Show Table Details",
                                                    id="collapse-button",
                                                    className="mb-3",
                                                    color="secondary",
                                                    n_clicks=0,
                                                ),
                                                dbc.Collapse(
                                                    html.Div(id="table-list"),
                                                    id="collapse",
                                                    is_open=False,
                                                ),
                                            ],
                                            width=4,
                                        ),
                                    ]
                                ),
                            ]
                        ),
                    ]
                ),
                # Update interval
                dcc.Interval(id="redis-check", interval=5000),
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
            [
                Output("redis-status", "children"),
                Output("redis-alert", "color"),
                Output("current-experiment", "children"),
            ],
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
                    experiment_text = (
                        f"Current Experiment: {current_project}"
                        if current_project
                        else "No active experiment"
                    )
                    return status_text, "success", experiment_text
                else:
                    self._connect_redis()
                    return self.redis_status, "danger", "No connection"
            except Exception as e:
                return f"Error: {str(e)}", "danger", "Error checking status"

        @self.app.callback(
            [Output("collapse", "is_open"), Output("collapse-button", "children")],
            [Input("collapse-button", "n_clicks")],
            [State("collapse", "is_open")],
        )
        def toggle_collapse(n_clicks, is_open):
            if n_clicks:
                return not is_open, (
                    "Hide Table Details" if not is_open else "Show Table Details"
                )
            return is_open, "Show Table Details"

        @self.app.callback(
            [Output("table-selector", "options"), Output("table-list", "children")],
            [
                Input("redis-check", "n_intervals"),
                Input("redis-refresh-button", "n_clicks"),
            ],
        )
        def update_table_list(n_intervals, n_clicks):
            try:
                if not self.redis_client:
                    return [], html.Div("No Redis connection", className="text-danger")

                db_path = self.redis_client.get("current_project")
                if not db_path:
                    return [], html.Div(
                        "No active experiment", className="text-warning"
                    )

                conn = duckdb.connect(db_path)

                # Get list of tables
                tables = conn.execute("SHOW TABLES").fetchall()
                table_names = [t[0] for t in tables]

                # Create dropdown options
                dropdown_options = [
                    {"label": name, "value": name} for name in table_names
                ]

                # Create table info cards with more compact styling
                table_info = []
                for table_name in table_names:
                    # Get column information
                    columns = conn.execute(f"DESCRIBE {table_name}").fetchall()
                    column_info = [f"{col[0]} ({col[1]})" for col in columns]

                    # Get row count
                    row_count = conn.execute(
                        f"SELECT COUNT(*) FROM {table_name}"
                    ).fetchone()[0]

                    table_info.append(
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    html.H6(
                                        table_name, className="mb-0"
                                    ),  # Smaller header
                                    className="py-2",  # Less padding
                                ),
                                dbc.CardBody(
                                    [
                                        html.P(
                                            f"Rows: {row_count}", className="mb-2 small"
                                        ),  # Smaller text
                                        html.P("Columns:", className="mb-1 small"),
                                        html.Ul(
                                            [
                                                html.Li(col, className="small")
                                                for col in column_info
                                            ],
                                            className="mb-0",
                                        ),  # Remove bottom margin
                                    ],
                                    className="py-2",
                                ),  # Less padding
                            ],
                            className="mb-2",
                        )  # Less margin between cards
                    )

                conn.close()
                return dropdown_options, table_info

            except Exception as e:
                return [], html.Div(f"Error: {str(e)}", className="text-danger")

        @self.app.callback(
            Output("table-preview", "children"),
            [Input("table-selector", "value"), Input("draw-rows-button", "n_clicks")],
        )
        def show_table_preview(selected_table, n_clicks):
            if not selected_table:
                return html.Div()

            try:
                db_path = self.redis_client.get("current_project")
                if not db_path:
                    return html.Div("No active experiment", className="text-warning")

                conn = duckdb.connect(db_path)

                # Get total row count
                row_count = conn.execute(
                    f"SELECT COUNT(*) FROM {selected_table}"
                ).fetchone()[0]

                # If table has more than 20 rows, select random sample
                if row_count > 20:
                    df = conn.execute(
                        f"""
                        SELECT * FROM {selected_table} 
                        USING SAMPLE 20 ROWS
                    """
                    ).df()
                    preview_text = (
                        f"Showing 20 random rows out of {row_count} total rows"
                    )
                else:
                    df = conn.execute(f"SELECT * FROM {selected_table}").df()
                    preview_text = f"Showing all {row_count} rows"

                conn.close()

                return [
                    html.H4(f"Preview of {selected_table}"),
                    html.P(preview_text, className="text-muted"),
                    dbc.Table.from_dataframe(
                        df,
                        striped=True,
                        bordered=True,
                        hover=True,
                        responsive=True,
                        className="mt-3",
                    ),
                ]

            except Exception as e:
                return html.Div(f"Error: {str(e)}", className="text-danger")

    def run(self, debug=False, port=8054):
        self.app.run_server(debug=debug, port=port)


def main():
    parser = argparse.ArgumentParser(
        description="Start the Database Browser application"
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
        default=8054,
        help="Port to run the Dash app on (default: 8054)",
    )
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")

    args = parser.parse_args()

    app = DatabaseBrowser(redis_host=args.redis_host, redis_port=args.redis_port)
    app.run(debug=args.debug, port=args.port)


if __name__ == "__main__":
    main()
