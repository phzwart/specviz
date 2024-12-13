import uuid

import dash
import numpy as np
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate


class BaseExplorer:
    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.app = self._create_base_app()

    def _create_base_app(self):
        app = dash.Dash(
            f"{self.__class__.__name__}_{self.session_id}",  # Unique name per instance
            suppress_callback_exceptions=True,
            update_title=None,
        )
        # Prevent caching
        app.index_string = """
        <!DOCTYPE html>
        <html>
            <head>
                <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
                <meta http-equiv="Pragma" content="no-cache">
                <meta http-equiv="Expires" content="0">
                {%metas%}
                <title>{%title%}</title>
                {%favicon%}
                {%css%}
            </head>
            <body>
                {%app_entry%}
                <footer>
                    {%config%}
                    {%scripts%}
                    {%renderer%}
                </footer>
            </body>
        </html>
        """
        return app

    def run(self, debug=False, port=8050):
        """Run the Dash application

        Args:
            debug (bool): Whether to run in debug mode
            port (int): Port number to run on
        """
        import socket

        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        print(f"Dashboard running at:")
        print(f"    Local URL: http://127.0.0.1:{port}/")
        print(f"    Network URL: http://{ip_address}:{port}/")

        self.app.run_server(debug=debug, port=port, host="0.0.0.0")
