import argparse
from datetime import datetime

import dash
import dash_bootstrap_components as dbc
import redis
from dash import dcc, html
from dash.dependencies import Input, Output, State

from specviz.tasks.redis_logger import RedisLogger


class LogViewer:
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379):
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

        # Redis setup
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_client = None
        self.redis_status = "Not Connected"
        self.redis_status_color = "danger"
        self.logger = RedisLogger()
        self._connect_redis()

        self.app.layout = dbc.Container(
            [
                html.H1("SpecViz Log Viewer", className="my-4"),
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
                                    width=8,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Input(
                                            id="update-interval",
                                            type="number",
                                            value=1000,
                                            min=100,
                                            max=10000,
                                            step=100,
                                            size="sm",
                                            placeholder="Update interval (ms)",
                                        )
                                    ],
                                    width=2,
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
                # Log Display
                dbc.Card(
                    [
                        dbc.CardHeader(
                            [
                                dbc.Row(
                                    [
                                        dbc.Col("Log Stream", width=8),
                                        dbc.Col(
                                            [
                                                dbc.Switch(
                                                    id="auto-scroll",
                                                    label="Auto-scroll",
                                                    value=True,
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
                                html.Div(
                                    id="log-stream",
                                    style={
                                        "maxHeight": "600px",
                                        "overflowY": "auto",
                                        "fontFamily": "monospace",
                                    },
                                )
                            ]
                        ),
                    ]
                ),
                # Update intervals
                dcc.Interval(id="redis-check", interval=5000),  # Redis status check
                dcc.Interval(id="log-update", interval=1000),  # Log update interval
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
            self.logger.set_redis_client(self.redis_client)
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
        # Update interval callback
        @self.app.callback(
            Output("log-update", "interval"), Input("update-interval", "value")
        )
        def update_interval(value):
            return value if value is not None else 1000

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
                    return self.redis_status, "success"
            except:
                self._connect_redis()
            return self.redis_status, self.redis_status_color

        @self.app.callback(
            Output("log-stream", "children"), Input("log-update", "n_intervals")
        )
        def update_logs(n):
            if not self.redis_client:
                return html.Div("No Redis connection", className="text-danger")

            try:
                logs = self.logger.get_logs(count=100)  # Get last 100 logs

                log_entries = []
                for log in reversed(logs):  # Show newest last
                    # Format timestamp
                    timestamp = datetime.fromisoformat(log["timestamp"])
                    formatted_time = timestamp.strftime("%Y-%m-%d %H:%M:%S")

                    # Create log entry with proper styling
                    level_colors = {
                        "INFO": "text-info",
                        "WARNING": "text-warning",
                        "ERROR": "text-danger",
                        "DEBUG": "text-secondary",
                    }

                    entry = html.Div(
                        [
                            html.Span(f"{formatted_time} ", className="text-muted"),
                            html.Span(
                                f"[{log['level']}] ",
                                className=level_colors.get(log["level"], "text-muted"),
                            ),
                            html.Span(f"{log['source']}: ", className="text-primary"),
                            html.Span(log["message"]),
                        ],
                        className="mb-1",
                    )

                    log_entries.append(entry)

                return log_entries

            except Exception as e:
                return html.Div(
                    f"Error fetching logs: {str(e)}", className="text-danger"
                )

    def run(self, debug=False, port=8052):
        self.app.run_server(debug=debug, port=port)


def main():
    parser = argparse.ArgumentParser(description="Start the Log Viewer application")
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
        default=8052,
        help="Port to run the Dash app on (default: 8052)",
    )
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")

    args = parser.parse_args()

    app = LogViewer(redis_host=args.redis_host, redis_port=args.redis_port)
    app.run(debug=args.debug, port=args.port)


if __name__ == "__main__":
    main()
