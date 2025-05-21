from typing import Callable, List, Optional, Protocol, Tuple

from abc import ABC, abstractmethod

import dash  # Add this import for PreventUpdate
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback_context, dcc, html


# Define the protocol for database connectors
class DatabaseConnector(Protocol):
    def export_selection(self, selected_indices: list[int]) -> None: ...


class BaseExplorer(ABC):
    @abstractmethod
    def run(self, debug: bool = False, port: int = 8050):
        pass


class SpatialLatentSpaceExplorer(BaseExplorer):
    def __init__(
        self,
        sample_ids: np.ndarray,
        latent_coordinates: np.ndarray,
        spatial_coordinates: np.ndarray,
        spectra: np.ndarray,
        wavelengths: np.ndarray,
        max_points: int = 100000,
        spatial_heatmap: Optional[
            tuple[np.ndarray, tuple[float, float, float, float]]
        ] = None,
        external_selection_callback: Optional[Callable] = None,
        database_connector: Optional[DatabaseConnector] = None,  # New parameter
    ):
        """Initialize the explorer with data and optional parameters."""
        self.app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

        # Store the external callback if provided
        self.external_selection_callback = external_selection_callback

        # Store the database connector
        self.database_connector = database_connector

        # Downsample if necessary
        if len(sample_ids) > max_points:
            stride = len(sample_ids) // max_points
            self.sample_ids = sample_ids[::stride]
            self.latent_coords = latent_coordinates[::stride]
            self.spatial_coords = spatial_coordinates[::stride]
            self.spectra = spectra[::stride]
        else:
            self.sample_ids = sample_ids
            self.latent_coords = latent_coordinates
            self.spatial_coords = spatial_coordinates
            self.spectra = spectra

        self.wavelengths = wavelengths
        self.spatial_heatmap = spatial_heatmap

        # Precompute spectral statistics
        self.spectral_stats = self._compute_spectral_statistics()

        # Store for selection state
        self.selected_indices = []

        # Initialize the app layout and callbacks
        self._create_layout()
        self._setup_callbacks()

    def _compute_spectral_statistics(self) -> dict:
        """Compute statistical summaries of the spectra."""
        return {
            "median": np.median(self.spectra, axis=0),
            "q05": np.percentile(self.spectra, 5, axis=0),
            "q25": np.percentile(self.spectra, 25, axis=0),
            "q75": np.percentile(self.spectra, 75, axis=0),
            "q95": np.percentile(self.spectra, 95, axis=0),
        }

    def _create_scatter_plot(
        self,
        coordinates: np.ndarray,
        selected_indices: list[int] = None,
        title: str = "",
        mode: str = "markers",
        dragmode: str = None,
        equal_aspect: bool = False,  # New parameter to control aspect ratio
    ) -> go.Figure:
        """Create a scatter plot with optional selection highlighting."""
        fig = go.Figure()

        # Create base scatter plot with all points in gray
        fig.add_trace(
            go.Scatter(
                x=coordinates[:, 0],
                y=coordinates[:, 1],
                mode=mode,
                marker=dict(color="gray", size=5, opacity=0.5),
                name="All Points",
                selectedpoints=[],  # Prevent automatic selection highlighting
                unselected=dict(marker=dict(opacity=0.5)),  # Force unselected style
                selected=dict(
                    marker=dict(opacity=0.5)
                ),  # Force selected style to match unselected
                showlegend=False,
            )
        )

        # Add selected points as a single trace
        if selected_indices:
            selected_coords = coordinates[selected_indices]
            fig.add_trace(
                go.Scatter(
                    x=selected_coords[:, 0],
                    y=selected_coords[:, 1],
                    mode=mode,
                    marker=dict(color="red", size=8, opacity=1.0),
                    name="Selected Points",
                    showlegend=False,
                    hoverinfo="skip",  # Prevent duplicate hover info
                )
            )

        layout_updates = {
            "title": title,
            "dragmode": dragmode,
            "hovermode": "closest",
            "modebar": dict(add=["lasso2d"]),
            "uirevision": "constant",
            "showlegend": False,
            "selectionrevision": False,  # Disable Plotly's selection revision
        }

        # Add equal aspect ratio for spatial plot
        if equal_aspect:
            layout_updates["yaxis"] = dict(
                scaleanchor="x",  # Lock the aspect ratio
                scaleratio=1,  # 1:1 ratio
            )

        fig.update_layout(**layout_updates)

        return fig

    def _create_spectrum_plot(
        self, selected_indices: list[int] = None, display_mode: str = "all"
    ) -> go.Figure:
        """Create a spectrum plot showing either all spectra or statistics."""
        fig = go.Figure()

        if not selected_indices:
            selected_indices = list(range(len(self.spectra)))

        if display_mode == "all":
            # Instead of individual traces, use a single trace with all data
            fig.add_trace(
                go.Scatter(
                    x=np.tile(self.wavelengths, len(selected_indices)),
                    y=self.spectra[selected_indices].flatten(),
                    mode="lines",
                    line=dict(color="blue", width=1),
                    opacity=0.3,
                    showlegend=False,
                )
            )
        else:
            # Use faster numpy operations for statistics
            selected_spectra = self.spectra[selected_indices]
            median = np.median(selected_spectra, axis=0)
            # Compute all percentiles at once
            percentiles = np.percentile(selected_spectra, [5, 25, 75, 95], axis=0)
            q05, q25, q75, q95 = percentiles

            # Add 95% confidence band
            fig.add_trace(
                go.Scatter(
                    x=self.wavelengths,
                    y=q95,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=True,
                    name="95% Band",
                    fillcolor="rgba(0,0,255,0.1)",
                    fill=None,
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=self.wavelengths,
                    y=q05,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    fillcolor="rgba(0,0,255,0.1)",
                    fill="tonexty",
                    name="5% Band",
                )
            )

            # Add 75% confidence band
            fig.add_trace(
                go.Scatter(
                    x=self.wavelengths,
                    y=q75,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=True,
                    name="75% Band",
                    fillcolor="rgba(0,0,255,0.2)",
                    fill=None,
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=self.wavelengths,
                    y=q25,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    fillcolor="rgba(0,0,255,0.2)",
                    fill="tonexty",
                    name="25% Band",
                )
            )

            # Add median line
            fig.add_trace(
                go.Scatter(
                    x=self.wavelengths,
                    y=median,
                    mode="lines",
                    line=dict(color="blue", width=2),
                    name="Median",
                    showlegend=True,
                )
            )

        fig.update_layout(
            title="Spectral View",
            xaxis_title="Wavelength",
            yaxis_title="Intensity",
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )

        return fig

    def _create_layout(self):
        """Create the app layout."""
        self.app.layout = dbc.Container(
            [
                html.H1("Spatial Latent Space Explorer", className="mb-4"),
                # Store for selection state
                dcc.Store(id="selection-store", data=[]),
                # Control Panel
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                # First row: Selection controls and export button
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                dbc.RadioItems(
                                                                    options=[
                                                                        {
                                                                            "label": "Select in Latent Space",
                                                                            "value": "latent",
                                                                        },
                                                                        {
                                                                            "label": "Select in Spatial View",
                                                                            "value": "spatial",
                                                                        },
                                                                    ],
                                                                    value="latent",
                                                                    id="selection-source",
                                                                    inline=True,
                                                                    className="mb-2",
                                                                ),
                                                            ],
                                                            width=6,
                                                        ),
                                                        dbc.Col(
                                                            [
                                                                dbc.Button(
                                                                    "Clear Selection",
                                                                    id="clear-selection-button",
                                                                    color="secondary",
                                                                    size="sm",
                                                                    className="me-2",
                                                                ),
                                                                dbc.Button(
                                                                    "Export Selection",
                                                                    id="export-selection-button",
                                                                    color="success",
                                                                    size="sm",
                                                                ),
                                                            ],
                                                            width=6,
                                                        ),
                                                    ],
                                                    className="mb-2",
                                                ),
                                                # Second row: Latent Diffusion controls
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                html.Label(
                                                                    "Latent Diffusion:"
                                                                ),
                                                                dbc.Input(
                                                                    type="number",
                                                                    id="latent-diffusion-value",
                                                                    value=0.1,
                                                                    step=0.1,
                                                                    style={
                                                                        "width": "100px"
                                                                    },
                                                                ),
                                                            ],
                                                            width=6,
                                                        ),
                                                        dbc.Col(
                                                            [
                                                                html.Br(),  # Add space to align with input
                                                                dbc.Button(
                                                                    "Latent Diffuse",
                                                                    id="latent-diffuse-button",
                                                                    color="primary",
                                                                    size="sm",
                                                                ),
                                                            ],
                                                            width=6,
                                                        ),
                                                    ],
                                                    className="mb-2",
                                                ),
                                                # Third row: Spatial Diffusion controls
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                html.Label(
                                                                    "Spatial Diffusion:"
                                                                ),
                                                                dbc.Input(
                                                                    type="number",
                                                                    id="spatial-diffusion-value",
                                                                    value=1.0,
                                                                    step=0.5,
                                                                    style={
                                                                        "width": "100px"
                                                                    },
                                                                ),
                                                            ],
                                                            width=6,
                                                        ),
                                                        dbc.Col(
                                                            [
                                                                html.Br(),  # Add space to align with input
                                                                dbc.Button(
                                                                    "Spatial Diffuse",
                                                                    id="spatial-diffuse-button",
                                                                    color="primary",
                                                                    size="sm",
                                                                ),
                                                            ],
                                                            width=6,
                                                        ),
                                                    ],
                                                    className="mb-2",
                                                ),
                                                # Display mode for spectra
                                                dbc.RadioItems(
                                                    id="spectrum-display-mode",
                                                    options=[
                                                        {
                                                            "label": " All Spectra",
                                                            "value": "all",
                                                        },
                                                        {
                                                            "label": " Statistics",
                                                            "value": "stats",
                                                        },
                                                    ],
                                                    value="all",
                                                    inline=True,
                                                    className="mb-2",
                                                ),
                                            ]
                                        )
                                    ]
                                )
                            ]
                        )
                    ],
                    className="mb-4",
                ),
                # Plots
                dbc.Row(
                    [
                        dbc.Col(dcc.Graph(id="latent-plot"), width=6),
                        dbc.Col(dcc.Graph(id="spectrum-plot"), width=6),
                    ],
                    className="mb-4",
                ),
                dbc.Row([dbc.Col(dcc.Graph(id="spatial-plot"), width=12)]),
            ]
        )

    def _compute_latent_diffusion(
        self, selected_indices: list[int], distance: float
    ) -> list[int]:
        """
        Compute latent space diffusion for selected points.

        Args:
            selected_indices: Current selection
            distance: Distance threshold (positive for growth, negative for shrink)

        Returns:
            Updated list of selected indices
        """
        from sklearn.metrics.pairwise import euclidean_distances

        # Convert to numpy array for easier indexing
        selected_set = set(selected_indices)
        all_indices = set(range(len(self.latent_coords)))

        # Compute distances between all points
        distances = euclidean_distances(self.latent_coords)

        if distance >= 0:
            # Growth mode: Find unselected points near selected points
            unselected_indices = list(all_indices - selected_set)
            selected_points = self.latent_coords[list(selected_set)]
            unselected_points = self.latent_coords[unselected_indices]

            # Compute distances between selected and unselected points
            distances = euclidean_distances(unselected_points, selected_points)

            # Find points within threshold
            within_distance = np.any(distances <= abs(distance), axis=1)
            new_indices = set(np.array(unselected_indices)[within_distance])

            # Add to selection
            return sorted(selected_set | new_indices)
        else:
            # Shrink mode: Remove selected points that are near unselected points
            unselected_indices = list(all_indices - selected_set)
            selected_points = self.latent_coords[list(selected_set)]
            unselected_points = self.latent_coords[unselected_indices]

            # Compute distances between selected and unselected points
            distances = euclidean_distances(selected_points, unselected_points)

            # Find points within threshold
            within_distance = np.any(distances <= abs(distance), axis=1)
            points_to_remove = set(np.array(list(selected_set))[within_distance])

            # Remove from selection
            return sorted(selected_set - points_to_remove)

    def _compute_spatial_diffusion(
        self, selected_indices: list[int], distance: float
    ) -> list[int]:
        """
        Compute spatial diffusion for selected points.

        Args:
            selected_indices: Current selection
            distance: Distance threshold (positive for growth, negative for shrink)

        Returns:
            Updated list of selected indices
        """
        from sklearn.metrics.pairwise import euclidean_distances

        # Convert to numpy array for easier indexing
        selected_set = set(selected_indices)
        all_indices = set(range(len(self.spatial_coords)))

        if distance >= 0:
            # Growth mode: Find unselected points near selected points
            unselected_indices = list(all_indices - selected_set)
            selected_points = self.spatial_coords[list(selected_set)]
            unselected_points = self.spatial_coords[unselected_indices]

            # Compute distances between selected and unselected points
            distances = euclidean_distances(unselected_points, selected_points)

            # Find points within threshold
            within_distance = np.any(distances <= abs(distance), axis=1)
            new_indices = set(np.array(unselected_indices)[within_distance])

            # Add to selection
            return sorted(selected_set | new_indices)
        else:
            # Shrink mode: Remove selected points that are near unselected points
            unselected_indices = list(all_indices - selected_set)
            selected_points = self.spatial_coords[list(selected_set)]
            unselected_points = self.spatial_coords[unselected_indices]

            # Compute distances between selected and unselected points
            distances = euclidean_distances(selected_points, unselected_points)

            # Find points within threshold
            within_distance = np.any(distances <= abs(distance), axis=1)
            points_to_remove = set(np.array(list(selected_set))[within_distance])

            # Remove from selection
            return sorted(selected_set - points_to_remove)

    def _setup_callbacks(self):
        """Set up all callbacks for interactivity."""

        @self.app.callback(
            Output("selection-store", "data"),
            [
                Input("latent-plot", "selectedData"),
                Input("spatial-plot", "selectedData"),
                Input("clear-selection-button", "n_clicks"),
                Input("latent-diffuse-button", "n_clicks"),
                Input("spatial-diffuse-button", "n_clicks"),
                Input("selection-source", "value"),
            ],
            [
                State("selection-store", "data"),
                State("latent-diffusion-value", "value"),
                State("spatial-diffusion-value", "value"),
            ],
        )
        def update_selection_store(
            latent_selection,
            spatial_selection,
            clear_clicks,
            latent_diffuse_clicks,
            spatial_diffuse_clicks,
            selection_source,
            current_selection,
            latent_diffusion_value,
            spatial_diffusion_value,
        ):
            """Update selection store based on all possible triggers"""
            ctx = callback_context

            if not ctx.triggered:
                return current_selection or []

            trigger = ctx.triggered[0]["prop_id"].split(".")[0]
            print(f"Trigger: {trigger}")

            # Handle clear button click
            if trigger == "clear-selection-button":
                return []

            # Handle diffusion
            if trigger == "latent-diffuse-button":
                if not current_selection:
                    return []
                return self._compute_latent_diffusion(
                    current_selection, latent_diffusion_value
                )

            if trigger == "spatial-diffuse-button":
                if not current_selection:
                    return []
                return self._compute_spatial_diffusion(
                    current_selection, spatial_diffusion_value
                )

            # Handle selection updates
            if trigger in ["latent-plot", "spatial-plot"]:
                # Only process selection if it comes from the active source
                if (trigger == "latent-plot" and selection_source != "latent") or (
                    trigger == "spatial-plot" and selection_source != "spatial"
                ):
                    return current_selection or []

                selection = (
                    latent_selection if trigger == "latent-plot" else spatial_selection
                )

                # Only update selection if there are new points selected
                if selection and "points" in selection:
                    # Only consider points from the main trace (curveNumber 0)
                    new_indices = {
                        p["pointIndex"]
                        for p in selection["points"]
                        if p["curveNumber"] == 0
                    }
                    current_set = set(current_selection or [])
                    truly_new_indices = new_indices - current_set
                    updated_selection = list(current_set | truly_new_indices)
                    return sorted(updated_selection)

            # Maintain current selection if no new selection is made
            return current_selection or []

        @self.app.callback(
            [
                Output("latent-plot", "figure"),
                Output("spatial-plot", "figure"),
                Output("spectrum-plot", "figure"),
            ],
            [
                Input("selection-store", "data"),
                Input("spectrum-display-mode", "value"),
                Input("selection-source", "value"),
            ],
        )
        def update_plots(selected_indices, display_mode, selection_source):
            """Update all plots based on selection state"""
            print("\nDEBUG: update_plots called")
            print(f"Selected indices: {selected_indices}")
            print(f"Display mode: {display_mode}")
            print(f"Selection source: {selection_source}")

            # Create latent space plot
            latent_fig = self._create_scatter_plot(
                self.latent_coords,
                selected_indices,
                "Latent Space",
                dragmode="lasso" if selection_source == "latent" else None,
                equal_aspect=False,  # Latent space doesn't need equal aspect
            )

            # Create spatial plot
            spatial_fig = self._create_scatter_plot(
                self.spatial_coords,
                selected_indices,
                "Spatial View",
                dragmode="lasso" if selection_source == "spatial" else None,
                equal_aspect=True,  # Spatial plot needs equal aspect
            )

            # Create spectrum plot
            spectrum_fig = self._create_spectrum_plot(selected_indices, display_mode)

            return latent_fig, spatial_fig, spectrum_fig

        @self.app.callback(
            [
                Output("export-selection-button", "n_clicks"),
                Output("export-selection-button", "color"),
            ],  # Add color output
            Input("export-selection-button", "n_clicks"),
            State("selection-store", "data"),
        )
        def handle_export(n_clicks, selected_indices):
            """Handle export button clicks"""
            if not n_clicks:
                raise dash.exceptions.PreventUpdate()

            print(f"DEBUG: Export button clicked")
            print(f"DEBUG: n_clicks = {n_clicks}")
            print(f"DEBUG: selected_indices = {selected_indices}")

            if self.database_connector is None:
                print("Warning: No database connected")
                return None, "danger"  # Red color when no connector
            else:
                try:
                    success = self.database_connector.export_selection(selected_indices)
                    print(
                        f"Successfully exported {len(selected_indices)} selected points"
                    )
                    # Return green if success is True, red if False
                    return None, "success" if success else "danger"
                except Exception as e:
                    print(f"Error exporting selection: {str(e)}")
                    return None, "danger"  # Red color on error

    def run(self, debug: bool = False, port: int = 8050):
        """Run the Dash app."""
        import logging
        import socket

        # Suppress Werkzeug logging
        log = logging.getLogger("werkzeug")
        log.setLevel(logging.ERROR)

        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)

        print(f"Dashboard running at:")
        print(f"  Local: http://127.0.0.1:{port}/")
        print(f"  Network: http://{local_ip}:{port}/")

        self.app.run_server(debug=debug, port=port, host="0.0.0.0")
