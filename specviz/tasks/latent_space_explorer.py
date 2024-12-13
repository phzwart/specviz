import numpy as np
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate

from .base_explorer import BaseExplorer


class LatentSpaceExplorer(BaseExplorer):
    def __init__(
        self, sample_ids, latent_coordinates, spectra, wavelengths, max_points=50000
    ):
        """
        Initialize the Latent Space Explorer with optional downsampling

        Args:
            sample_ids: Sample identifiers
            latent_coordinates: Latent space coordinates (N,2)
            spectra: Spectral data (N,M)
            wavelengths: Wavelength values (M,)
            max_points: Maximum number of points to display (default: 5000)
        """
        super().__init__()  # Initialize base explorer first

        # Downsample if necessary
        n_samples = len(sample_ids)
        if n_samples > max_points:
            # Calculate stride for downsampling
            stride = n_samples // max_points
            print(
                f"Downsampling {n_samples} points to ~{max_points} points (stride={stride})"
            )

            # Downsample all arrays
            self.sample_ids = sample_ids[::stride]
            self.latent_coordinates = latent_coordinates[::stride]
            self.spectra = spectra[::stride]
            self.wavelengths = wavelengths
        else:
            self.sample_ids = sample_ids
            self.latent_coordinates = latent_coordinates
            self.spectra = spectra
            self.wavelengths = wavelengths

        # Pre-compute statistics for all data
        self.mean_spectrum = np.mean(self.spectra, axis=0)
        self.std_spectrum = np.std(self.spectra, axis=0)

        # Pre-compute scatter plot data
        self.scatter_data = {
            "x": self.latent_coordinates[:, 0],
            "y": self.latent_coordinates[:, 1],
        }

        # Add storage for last selection
        self.last_selection = None

        # Set up the app layout and callbacks
        self.app.layout = self._create_layout()
        self._setup_callbacks(self.app)

    def _create_layout(self):
        """Create the app layout"""
        return html.Div(
            [
                html.H1("Latent Space Explorer"),
                html.Div(
                    [
                        html.Div(
                            [
                                dcc.Graph(
                                    id="latent-scatter",
                                    figure=self._create_scatter(),
                                    style={"height": "600px"},
                                )
                            ],
                            style={"width": "48%", "display": "inline-block"},
                        ),
                        html.Div(
                            [
                                dcc.Graph(
                                    id="spectrum-plot",
                                    figure=self._create_spectrum(),
                                    style={"height": "600px"},
                                )
                            ],
                            style={"width": "48%", "display": "inline-block"},
                        ),
                    ]
                ),
                html.Div(
                    [  # Control Panel
                        html.H4("Display Options"),
                        html.Div(
                            [
                                html.Label("Spectrum Display:"),
                                dcc.RadioItems(
                                    id="display-mode",
                                    options=[
                                        {"label": "Show All Spectra", "value": "all"},
                                        {
                                            "label": "Show Median with Quartiles",
                                            "value": "statistics",
                                        },
                                    ],
                                    value="all",
                                    inline=True,
                                ),
                            ],
                            style={"padding": "10px"},
                        ),
                        html.Div(
                            [
                                html.Label("Selection Mode:"),
                                dcc.RadioItems(
                                    id="click-mode",
                                    options=[
                                        {"label": "Single Point", "value": "single"},
                                        {"label": "Area Click", "value": "area"},
                                        {
                                            "label": "Box/Lasso Select",
                                            "value": "selection",
                                        },
                                    ],
                                    value="selection",
                                    inline=True,
                                ),
                            ],
                            style={"padding": "10px"},
                        ),
                        html.Div(
                            [
                                html.Label("Click Radius:"),
                                dcc.Input(
                                    id="click-radius",
                                    type="number",
                                    value=0.05,
                                    step=0.01,
                                    min=0.01,
                                    max=1.0,
                                ),
                            ],
                            style={"padding": "10px"},
                        ),
                    ],
                    style={"padding": "20px"},
                ),
            ]
        )

    def _create_scatter(self):
        """Pre-create scatter plot with basic layout"""
        fig = go.Figure()
        fig.add_trace(
            go.Scattergl(
                x=self.scatter_data["x"],
                y=self.scatter_data["y"],
                mode="markers",
                marker=dict(size=8, color="blue", opacity=0.6),
                name="Samples",
                showlegend=False,
                selectedpoints=[],
            )
        )

        fig.update_layout(
            xaxis_title="Latent Dimension 1",
            yaxis_title="Latent Dimension 2",
            title="Latent Space",
            dragmode="select",
            uirevision=True,
            selectionrevision=True,
            showlegend=False,
            clickmode="event+select",
            modebar=dict(add=["select2d", "lasso2d", "pan"]),
            selectdirection="any",
            selections=[
                dict(type="rect", line=dict(color="rgba(255,255,255,0.5)"), opacity=0.3)
            ],
        )
        return fig

    def _create_spectrum(self):
        """Pre-create spectrum plot with basic layout"""
        fig = go.Figure()

        fig.update_layout(
            xaxis_title="Wavelength",
            yaxis_title="Intensity",
            title="Select points in latent space to view spectra",
            uirevision=True,  # Preserve UI state on updates
        )
        return fig

    def _setup_callbacks(self, app):
        @app.callback(
            [Output("spectrum-plot", "figure"), Output("latent-scatter", "figure")],
            [
                Input("latent-scatter", "selectedData"),
                Input("latent-scatter", "clickData"),
                Input("display-mode", "value"),
                Input("click-mode", "value"),
                Input("click-radius", "value"),
            ],
            prevent_initial_call=True,
        )
        def update_plots(
            selected_data, click_data, display_mode, click_mode, click_radius
        ):
            # Initialize selected indices
            selected_indices = []

            # Update last_selection if we have new selection data
            if selected_data and selected_data["points"]:
                self.last_selection = selected_data

            # Handle selection based on mode
            if click_mode == "area" and click_data and click_data["points"]:
                # Area selection from click
                click_point = click_data["points"][0]
                x, y = click_point["x"], click_point["y"]
                distances = np.sqrt(
                    (self.latent_coordinates[:, 0] - x) ** 2
                    + (self.latent_coordinates[:, 1] - y) ** 2
                )
                selected_indices.extend(np.where(distances <= click_radius)[0])

            elif click_mode == "single" and click_data and click_data["points"]:
                # Single point selection
                selected_indices.append(click_data["points"][0]["pointIndex"])

            elif click_mode == "selection":
                # Use last selection if available for selection mode
                if self.last_selection and self.last_selection["points"]:
                    selected_indices.extend(
                        point["pointIndex"] for point in self.last_selection["points"]
                    )
                # Also check current selection
                elif selected_data and selected_data["points"]:
                    selected_indices.extend(
                        point["pointIndex"] for point in selected_data["points"]
                    )

            if not selected_indices:
                raise PreventUpdate

            # Remove duplicates and sort
            selected_indices = sorted(list(set(selected_indices)))
            selected_spectra = self.spectra[selected_indices]

            # Create spectrum plot
            spectrum_fig = go.Figure()

            if display_mode == "all":
                for idx, spectrum in zip(selected_indices, selected_spectra):
                    spectrum_fig.add_trace(
                        go.Scattergl(
                            x=self.wavelengths,
                            y=spectrum,
                            name=f"Sample {self.sample_ids[idx]}",
                            opacity=0.7,
                            line=dict(width=1),
                            showlegend=False,
                        )
                    )
                title_suffix = f"Selected Spectra (n={len(selected_indices)})"
            else:  # statistics mode
                # Calculate all quantiles
                median = np.median(selected_spectra, axis=0)
                q05 = np.percentile(selected_spectra, 5, axis=0)
                q25 = np.percentile(selected_spectra, 25, axis=0)
                q75 = np.percentile(selected_spectra, 75, axis=0)
                q95 = np.percentile(selected_spectra, 95, axis=0)

                # Plot median
                spectrum_fig.add_trace(
                    go.Scattergl(
                        x=self.wavelengths,
                        y=median,
                        name="Median",
                        line=dict(color="blue", width=2),
                        showlegend=True,
                    )
                )

                # Add 25-75 percentile range
                spectrum_fig.add_trace(
                    go.Scatter(
                        x=np.concatenate([self.wavelengths, self.wavelengths[::-1]]),
                        y=np.concatenate([q75, q25[::-1]]),
                        fill="toself",
                        fillcolor="rgba(0,0,255,0.2)",
                        line=dict(color="rgba(255,255,255,0)"),
                        name="25-75th Percentile",
                        showlegend=True,
                    )
                )

                # Add 5-95 percentile range
                spectrum_fig.add_trace(
                    go.Scatter(
                        x=np.concatenate([self.wavelengths, self.wavelengths[::-1]]),
                        y=np.concatenate([q95, q05[::-1]]),
                        fill="toself",
                        fillcolor="rgba(0,0,255,0.1)",
                        line=dict(color="rgba(255,255,255,0)"),
                        name="5-95th Percentile",
                        showlegend=True,
                    )
                )

                title_suffix = f"Statistical View (n={len(selected_indices)})"

            spectrum_fig.update_layout(
                xaxis_title="Wavelength",
                yaxis_title="Intensity",
                title=f"Spectra - {title_suffix}",
                uirevision="static",
                showlegend=(display_mode == "statistics"),
            )

            # Create scatter plot
            scatter_fig = go.Figure()

            # Plot all points
            scatter_fig.add_trace(
                go.Scattergl(
                    x=self.scatter_data["x"],
                    y=self.scatter_data["y"],
                    mode="markers",
                    marker=dict(size=8, color="blue", opacity=0.6),
                    name="Samples",
                    showlegend=False,
                )
            )

            # Add selected points
            scatter_fig.add_trace(
                go.Scattergl(
                    x=self.latent_coordinates[selected_indices, 0],
                    y=self.latent_coordinates[selected_indices, 1],
                    mode="markers",
                    marker=dict(size=12, color="red", opacity=1.0),
                    name="Selected",
                    showlegend=False,
                )
            )

            # Set dragmode based on mode
            dragmode = "select"
            if click_mode == "selection":
                dragmode = "lasso"
            elif click_mode in ["single", "area"]:
                dragmode = False  # Disable selection tools for click modes

            scatter_fig.update_layout(
                xaxis_title="Latent Dimension 1",
                yaxis_title="Latent Dimension 2",
                title="Latent Space",
                dragmode=dragmode,
                uirevision=True,
                showlegend=False,
                clickmode="event+select",
                modebar=dict(add=["select2d", "lasso2d", "pan"]),
                selectdirection="any",
            )

            return spectrum_fig, scatter_fig

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
