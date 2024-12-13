import socket
import uuid
from threading import Thread

import dash
from dash import dcc, html
from dash.dependencies import Input, Output

from .latent_space_explorer import LatentSpaceExplorer
from .spatial_latent_space_explorer import SpatialLatentSpaceExplorer


class CombinedExplorer:
    def __init__(
        self,
        sample_ids,
        latent_coordinates,
        spectra,
        wavelengths,
        spatial_coordinates=None,
        spatial_heatmap=None,
        max_points=100000,
        base_port=8050,
    ):
        """
        Initialize Combined Explorer

        Args:
            sample_ids: Sample identifiers
            latent_coordinates: Latent space coordinates (N,2)
            spectra: Spectral data (N,M)
            wavelengths: Wavelength values (M,)
            spatial_coordinates: Optional physical space coordinates (N,2)
            spatial_heatmap: Optional tuple of (values, extent) for background heatmap
            max_points: Maximum number of points to display
            base_port: Base port number (default: 8050)
        """
        # Generate unique session ID
        self.session_id = str(uuid.uuid4())

        # Store ports for microservices
        self.main_port = base_port
        self.latent_port = base_port + 1
        self.spatial_port = base_port + 2

        # Create apps with unique names and no caching
        self.latent_explorer = LatentSpaceExplorer(
            sample_ids=sample_ids,
            latent_coordinates=latent_coordinates,
            spectra=spectra,
            wavelengths=wavelengths,
            max_points=max_points,
        )

        self.spatial_explorer = None
        if spatial_coordinates is not None:
            self.spatial_explorer = SpatialLatentSpaceExplorer(
                sample_ids=sample_ids,
                latent_coordinates=latent_coordinates,
                spatial_coordinates=spatial_coordinates,
                spectra=spectra,
                wavelengths=wavelengths,
                max_points=max_points,
                spatial_heatmap=spatial_heatmap,
            )

        # Create main app
        self.app = dash.Dash(
            f"CombinedExplorer_{self.session_id}",
            suppress_callback_exceptions=True,
            update_title=None,
        )

        # Set up the app layout and callbacks
        self.app.layout = self._create_layout()
        self._setup_callbacks()

        # Add no-cache headers
        self.app.index_string = """
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

        # Add properties to store selections
        self.selected_points = None
        self.selection_callback = None

        # Pass the callback to child explorers
        self.latent_explorer.set_selection_callback(self._update_selection)
        if self.spatial_explorer:
            self.spatial_explorer.set_selection_callback(self._update_selection)

    def run(self, debug=False):
        """Run the combined explorer"""
        # Start microservices in separate threads
        Thread(
            target=self.latent_explorer.run,
            kwargs={"port": self.latent_port, "debug": debug},
            daemon=True,
        ).start()

        if self.spatial_explorer:
            Thread(
                target=self.spatial_explorer.run,
                kwargs={"port": self.spatial_port, "debug": debug},
                daemon=True,
            ).start()

        # Get host information
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        print(f"Combined Explorer running at:")
        print(f"    Local URL: http://127.0.0.1:{self.main_port}/")
        print(f"    Network URL: http://{ip_address}:{self.main_port}/")

        # Run main app
        self.app.run_server(debug=debug, port=self.main_port, host="0.0.0.0")

    def _create_layout(self):
        """Create the app layout"""
        return html.Div(
            [
                html.H1("Spectra Explorer"),
                dcc.Tabs(
                    [
                        dcc.Tab(
                            label="Latent Space Explorer",
                            children=[
                                html.Iframe(
                                    id="latent-iframe",
                                    style={
                                        "width": "100%",
                                        "height": "1000px",
                                        "border": "none",
                                    },
                                )
                            ],
                        ),
                        dcc.Tab(
                            label="Spatial Latent Space Explorer",
                            children=[
                                html.Iframe(
                                    id="spatial-iframe",
                                    style={
                                        "width": "100%",
                                        "height": "1000px",
                                        "border": "none",
                                    },
                                )
                            ],
                            disabled=self.spatial_explorer is None,
                        ),
                    ]
                ),
            ]
        )

    def _setup_callbacks(self):
        """Set up the app callbacks"""

        @self.app.callback(
            [Output("latent-iframe", "src"), Output("spatial-iframe", "src")],
            [Input("latent-iframe", "id")],  # Dummy input to trigger on load
        )
        def update_iframes(_):
            hostname = socket.gethostname()
            ip_address = socket.gethostbyname(hostname)

            # Add session ID to URLs to prevent caching
            latent_url = (
                f"http://{ip_address}:{self.latent_port}?session={self.session_id}"
            )
            spatial_url = (
                f"http://{ip_address}:{self.spatial_port}?session={self.session_id}"
                if self.spatial_explorer
                else ""
            )

            return latent_url, spatial_url

    def _update_selection(self, selected_points):
        """Callback to update the selected points"""
        self.selected_points = selected_points
        if self.selection_callback:
            self.selection_callback(selected_points)

    def on_selection(self, callback):
        """Register a callback to be called when selection changes

        Args:
            callback: Function that takes selected points as argument
        """
        self.selection_callback = callback

    def get_selected_points(self):
        """Get currently selected points

        Returns:
            List of selected point indices or None if no selection
        """
        return self.selected_points
