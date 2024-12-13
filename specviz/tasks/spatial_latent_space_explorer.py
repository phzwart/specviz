import numpy as np
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate

from .base_explorer import BaseExplorer


class SpatialLatentSpaceExplorer(BaseExplorer):
    def __init__(
        self,
        sample_ids,
        latent_coordinates,
        spatial_coordinates,
        spectra,
        wavelengths,
        max_points=100000,
        spatial_heatmap=None,
    ):
        """Initialize the Spatial Latent Space Explorer"""
        super().__init__()  # Initialize base explorer first

        # Downsample if necessary
        n_samples = len(sample_ids)
        if n_samples > max_points:
            stride = n_samples // max_points
            print(
                f"Downsampling {n_samples} points to ~{max_points} points (stride={stride})"
            )

            self.sample_ids = sample_ids[::stride]
            self.latent_coordinates = latent_coordinates[::stride]
            self.spatial_coordinates = spatial_coordinates[::stride]
            self.spectra = spectra[::stride]
            self.wavelengths = wavelengths
        else:
            self.sample_ids = sample_ids
            self.latent_coordinates = latent_coordinates
            self.spatial_coordinates = spatial_coordinates
            self.spectra = spectra
            self.wavelengths = wavelengths

        # Pre-compute statistics
        self.mean_spectrum = np.mean(self.spectra, axis=0)
        self.std_spectrum = np.std(self.spectra, axis=0)

        # Store scatter plot data
        self.latent_data = {
            "x": self.latent_coordinates[:, 0],
            "y": self.latent_coordinates[:, 1],
        }
        self.spatial_data = {
            "x": self.spatial_coordinates[:, 0],
            "y": self.spatial_coordinates[:, 1],
        }

        # Store last selections
        self.last_latent_selection = None
        self.last_spatial_selection = None

        # Add storage for current selection
        self.current_selection = None

        # Store heatmap data if provided
        self.spatial_heatmap = spatial_heatmap

        # Set up the app layout and callbacks
        self.app.layout = self._create_layout()
        self._setup_callbacks(self.app)

    def _create_layout(self):
        """Create the app layout"""
        return html.Div(
            [
                html.H1("Spatial Latent Space Explorer"),
                html.Div(
                    [  # Top row
                        html.Div(
                            [
                                dcc.Graph(
                                    id="latent-scatter",
                                    figure=self._create_scatter("latent"),
                                    style={"height": "400px"},
                                )
                            ],
                            style={"width": "48%", "display": "inline-block"},
                        ),
                        html.Div(
                            [
                                dcc.Graph(
                                    id="spectrum-plot",
                                    figure=self._create_spectrum(),
                                    style={"height": "400px"},
                                )
                            ],
                            style={"width": "48%", "display": "inline-block"},
                        ),
                    ]
                ),
                html.Div(
                    [  # Bottom row
                        html.Div(
                            [  # Control Panel
                                html.H4("Display Options"),
                                html.Div(
                                    [
                                        html.Label("Spectrum Display:"),
                                        dcc.RadioItems(
                                            id="display-mode",
                                            options=[
                                                {"label": "Show All", "value": "all"},
                                                {
                                                    "label": "Show Quants",
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
                                        html.Label("Selection Tool:"),
                                        dcc.RadioItems(
                                            id="tool-mode",
                                            options=[
                                                {"label": "Box Select", "value": "box"},
                                                {
                                                    "label": "Lasso Select",
                                                    "value": "lasso",
                                                },
                                            ],
                                            value="box",
                                            inline=True,
                                        ),
                                    ],
                                    style={"padding": "10px"},
                                ),
                                html.Div(
                                    [
                                        html.Label("Scatter Points:"),
                                        dcc.RadioItems(
                                            id="scatter-mode",
                                            options=[
                                                {
                                                    "label": "Show All Points",
                                                    "value": "all",
                                                },
                                                {
                                                    "label": "Selected Only",
                                                    "value": "selected",
                                                },
                                            ],
                                            value="all",
                                            inline=True,
                                        ),
                                    ],
                                    style={"padding": "10px"},
                                ),
                            ],
                            style={
                                "width": "48%",
                                "display": "inline-block",
                                "vertical-align": "top",
                            },
                        ),
                        html.Div(
                            [
                                dcc.Graph(
                                    id="spatial-scatter",
                                    figure=self._create_scatter("spatial"),
                                    style={"height": "400px"},
                                )
                            ],
                            style={"width": "48%", "display": "inline-block"},
                        ),
                    ]
                ),
            ]
        )

    def _create_scatter(self, plot_type, scatter_mode="all", selected_indices=None):
        """Create scatter plot with basic layout"""
        data = self.latent_data if plot_type == "latent" else self.spatial_data
        title = "Latent Space" if plot_type == "latent" else "Spatial Coordinates"
        x_title = "Latent Dimension 1" if plot_type == "latent" else "X Position"
        y_title = "Latent Dimension 2" if plot_type == "latent" else "Y Position"

        fig = go.Figure()

        # Add heatmap for spatial plot if available
        if plot_type == "spatial" and self.spatial_heatmap is not None:
            values, extent = self.spatial_heatmap
            fig.add_trace(
                go.Heatmap(
                    z=values,
                    x=np.linspace(extent[0], extent[1], values.shape[1]),
                    y=np.linspace(extent[2], extent[3], values.shape[0]),
                    colorscale="Viridis",
                    showscale=True,
                )
            )

        # Add scatter points based on mode
        if scatter_mode == "all" or plot_type == "latent":
            fig.add_trace(
                go.Scattergl(
                    x=data["x"],
                    y=data["y"],
                    mode="markers",
                    marker=dict(size=8, color="blue", opacity=0.6),
                    name="Samples",
                    showlegend=False,
                    selectedpoints=[],
                )
            )

        # Add selected points if any
        if selected_indices is not None:
            fig.add_trace(
                go.Scattergl(
                    x=data["x"][selected_indices],
                    y=data["y"][selected_indices],
                    mode="markers",
                    marker=dict(size=12, color="red", opacity=1.0),
                    name="Selected",
                    showlegend=False,
                )
            )

        layout_dict = {
            "xaxis_title": x_title,
            "yaxis_title": y_title,
            "title": title,
            "dragmode": "select",
            "uirevision": True,
            "showlegend": False,
            "clickmode": "event+select",
            "selectionrevision": True,
        }

        if plot_type == "spatial":
            layout_dict.update(
                {
                    "yaxis": {"scaleanchor": "x", "scaleratio": 1},
                }
            )

        fig.update_layout(**layout_dict)
        return fig

    def _create_spectrum(self):
        """Create spectrum plot with basic layout"""
        fig = go.Figure()
        fig.update_layout(
            xaxis_title="Wavelength",
            yaxis_title="Intensity",
            title="Select points to view spectra",
            uirevision=True,
        )
        return fig

    def _setup_callbacks(self, app):
        @app.callback(
            [
                Output("spectrum-plot", "figure"),
                Output("latent-scatter", "figure"),
                Output("spatial-scatter", "figure"),
            ],
            [
                Input("latent-scatter", "selectedData"),
                Input("spatial-scatter", "selectedData"),
                Input("display-mode", "value"),
                Input("tool-mode", "value"),
                Input("scatter-mode", "value"),
            ],
            prevent_initial_call=True,
        )
        def update_plots(
            latent_selected, spatial_selected, display_mode, tool_mode, scatter_mode
        ):
            # Update current selection if we have new selection data
            if latent_selected and latent_selected["points"]:
                self.current_selection = latent_selected
            elif spatial_selected and spatial_selected["points"]:
                self.current_selection = spatial_selected

            # If we have no current selection, prevent update
            if not self.current_selection:
                raise PreventUpdate

            selected_indices = sorted(
                point["pointIndex"] for point in self.current_selection["points"]
            )

            # Create plots
            spectrum_fig = go.Figure()

            if display_mode == "all":
                for idx, spectrum in zip(
                    selected_indices, self.spectra[selected_indices]
                ):
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
                median = np.median(self.spectra[selected_indices], axis=0)
                q05 = np.percentile(self.spectra[selected_indices], 5, axis=0)
                q25 = np.percentile(self.spectra[selected_indices], 25, axis=0)
                q75 = np.percentile(self.spectra[selected_indices], 75, axis=0)
                q95 = np.percentile(self.spectra[selected_indices], 95, axis=0)

                spectrum_fig.add_trace(
                    go.Scattergl(
                        x=self.wavelengths,
                        y=median,
                        name="Median",
                        line=dict(color="blue", width=2),
                    )
                )

                spectrum_fig.add_trace(
                    go.Scatter(
                        x=np.concatenate([self.wavelengths, self.wavelengths[::-1]]),
                        y=np.concatenate([q75, q25[::-1]]),
                        fill="toself",
                        fillcolor="rgba(0,0,255,0.2)",
                        line=dict(color="rgba(255,255,255,0)"),
                        name="25-75th Percentile",
                    )
                )

                spectrum_fig.add_trace(
                    go.Scatter(
                        x=np.concatenate([self.wavelengths, self.wavelengths[::-1]]),
                        y=np.concatenate([q95, q05[::-1]]),
                        fill="toself",
                        fillcolor="rgba(0,0,255,0.1)",
                        line=dict(color="rgba(255,255,255,0)"),
                        name="5-95th Percentile",
                    )
                )

                title_suffix = f"Statistical View (n={len(selected_indices)})"

            spectrum_fig.update_layout(
                xaxis_title="Wavelength",
                yaxis_title="Intensity",
                title=f"Spectra - {title_suffix}",
                showlegend=(display_mode == "statistics"),
            )

            latent_fig = self._create_scatter("latent", scatter_mode, selected_indices)
            spatial_fig = self._create_scatter(
                "spatial", scatter_mode, selected_indices
            )

            # Set selection tool
            dragmode = "select" if tool_mode == "box" else "lasso"
            latent_fig.update_layout(dragmode=dragmode)
            spatial_fig.update_layout(dragmode=dragmode)

            return spectrum_fig, latent_fig, spatial_fig

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
