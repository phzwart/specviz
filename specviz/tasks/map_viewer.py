import dash
import dash_bootstrap_components as dbc
import einops
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html
from matplotlib.path import Path


class HyperspectralViewer:
    def __init__(self, hyperspectral_data, latent_coords, wavenumbers):
        """
        Initialize the viewer with data

        Parameters:
        -----------
        hyperspectral_data : np.ndarray (C, Y, X)
        latent_coords : np.ndarray (2, Y, X)
        wavenumbers : np.ndarray (C,)
        """
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

        # Store data
        self.hyperspectral_data = hyperspectral_data
        self.latent_coords = latent_coords
        self.wavenumbers = wavenumbers

        # Reshape latent coordinates for scatter plot
        self.latent_flat = einops.rearrange(latent_coords, "d y x -> (y x) d")

        # Create layout
        self.app.layout = self._create_layout()

        # Setup callbacks
        self._setup_callbacks()

    def _create_layout(self):
        """Create the app layout"""
        return dbc.Container(
            [
                # Store for selected indices
                dcc.Store(id="selection-store", data=[]),
                html.H1("Hyperspectral Data Viewer"),
                # Selection controls row
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
                                            "label": "Select in Heatmap",
                                            "value": "heatmap",
                                        },
                                    ],
                                    value="latent",
                                    id="selection-source",
                                    inline=True,
                                    className="mb-2",
                                ),
                                dbc.Checklist(
                                    options=[
                                        {
                                            "label": "Additive Selection",
                                            "value": "additive",
                                        }
                                    ],
                                    value=[],
                                    id="selection-mode",
                                    switch=True,
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
                                    className="ms-2",
                                ),
                            ],
                            width=6,
                        ),
                    ],
                    className="mb-3",
                ),
                # Top row: Latent Space and Spectral Plot
                dbc.Row(
                    [
                        dbc.Col(dcc.Graph(id="latent-plot"), width=6),
                        dbc.Col(dcc.Graph(id="spectral-plot"), width=6),
                    ]
                ),
                # Bottom row: Heatmap and wavenumber slider
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dcc.Graph(id="heatmap"),
                                dcc.Slider(
                                    id="wavenumber-slider",
                                    min=0,
                                    max=len(self.wavenumbers) - 1,
                                    value=0,
                                    marks={
                                        i: f"{self.wavenumbers[i]:.0f}"
                                        for i in range(
                                            0,
                                            len(self.wavenumbers),
                                            len(self.wavenumbers) // 5,
                                        )
                                    },
                                ),
                            ]
                        )
                    ]
                ),
            ]
        )

    def _setup_callbacks(self):
        """Setup all callbacks"""

        def points_inside_lasso(x_points, y_points, lasso_x, lasso_y):
            """Helper function to find points inside a lasso selection"""
            points = np.column_stack([x_points, y_points])
            lasso_points = np.column_stack([lasso_x, lasso_y])

            # Create path from lasso points
            path = Path(lasso_points)

            # Find points inside the path
            return path.contains_points(points)

        @self.app.callback(
            Output("selection-store", "data"),
            [
                Input("latent-plot", "selectedData"),
                Input("heatmap", "selectedData"),
                Input("selection-mode", "value"),
                Input("clear-selection-button", "n_clicks"),
                Input("selection-source", "value"),
            ],
            [State("selection-store", "data")],
        )
        def update_selection(
            latent_selection,
            heatmap_selection,
            selection_mode,
            clear_clicks,
            selection_source,
            current_selection,
        ):
            """Update selection store based on either plot's selection"""
            ctx = dash.callback_context

            if not ctx.triggered:
                return current_selection or []

            trigger = ctx.triggered[0]["prop_id"].split(".")[0]

            # Handle clear button click
            if trigger == "clear-selection-button":
                return []

            # Only process selection if it comes from the active source
            if (trigger == "latent-plot" and selection_source != "latent") or (
                trigger == "heatmap" and selection_source != "heatmap"
            ):
                return current_selection or []

            selection = (
                latent_selection if trigger == "latent-plot" else heatmap_selection
            )

            if trigger == "heatmap" and selection and "lassoPoints" in selection:
                lasso = selection["lassoPoints"]
                Y, X = self.hyperspectral_data.shape[1:]

                # Create coordinate grids
                y_coords, x_coords = np.mgrid[0:Y, 0:X]
                x_flat = x_coords.ravel()
                y_flat = y_coords.ravel()

                # Find points inside lasso
                inside = points_inside_lasso(x_flat, y_flat, lasso["x"], lasso["y"])
                new_indices = np.where(inside)[0]

                if "additive" in (selection_mode or []):
                    combined_indices = list(set(current_selection + list(new_indices)))
                    return sorted(combined_indices)
                else:
                    return sorted(list(new_indices))

            elif selection and "points" in selection and len(selection["points"]) > 0:
                new_indices = [p["pointIndex"] for p in selection["points"]]

                if "additive" in (selection_mode or []):
                    combined_indices = list(set(current_selection + new_indices))
                    return sorted(combined_indices)
                else:
                    return new_indices

            return current_selection or []

        @self.app.callback(
            [
                Output("latent-plot", "figure"),
                Output("spectral-plot", "figure"),
                Output("heatmap", "figure"),
            ],
            [
                Input("selection-store", "data"),
                Input("wavenumber-slider", "value"),
                Input("selection-source", "value"),
            ],
        )
        def update_plots(selected_indices, wavenumber_idx, selection_source):
            """Update all plots based on selection and current wavenumber"""

            # Create latent space plot
            latent_fig = go.Figure(
                go.Scatter(
                    x=self.latent_flat[:, 0],
                    y=self.latent_flat[:, 1],
                    mode="markers",
                    marker=dict(color="gray", size=5, opacity=0.5),
                )
            )

            # Add selected points as a separate trace
            if selected_indices:
                selected_points = self.latent_flat[selected_indices]
                latent_fig.add_trace(
                    go.Scatter(
                        x=selected_points[:, 0],
                        y=selected_points[:, 1],
                        mode="markers",
                        marker=dict(color="red", size=8, opacity=1.0),
                        showlegend=False,
                    )
                )

            latent_fig.update_layout(
                dragmode="lasso" if selection_source == "latent" else None,
                title="Latent Space",
                uirevision="constant",
                hovermode="closest",
            )

            # Create spectral plot
            spectral_fig = go.Figure()
            if selected_indices:
                spectra = einops.rearrange(self.hyperspectral_data, "c y x -> c (y x)")[
                    :, selected_indices
                ]
                for spectrum in spectra.T:
                    spectral_fig.add_trace(
                        go.Scatter(
                            x=self.wavenumbers,
                            y=spectrum,
                            mode="lines",
                            line=dict(width=1, color="blue"),
                            showlegend=False,
                        )
                    )
            spectral_fig.update_layout(
                title="Spectral View",
                xaxis_title="Wavenumber",
                yaxis_title="Intensity",
                showlegend=False,
            )

            # Create heatmap
            Y, X = self.hyperspectral_data.shape[1:]
            aspect_ratio = Y / X  # Calculate the aspect ratio from data dimensions

            heatmap_fig = go.Figure(
                go.Heatmap(
                    z=self.hyperspectral_data[wavenumber_idx],
                    colorscale="Viridis",
                    showscale=True,
                    name="Full Data",
                    # Set origin to lower left
                    y=list(range(Y)),  # Explicitly set y coordinates
                    yaxis="y",
                )
            )

            if selected_indices:
                mask = np.zeros((Y, X))
                y_indices = [idx // X for idx in selected_indices]
                x_indices = [idx % X for idx in selected_indices]
                mask[y_indices, x_indices] = self.hyperspectral_data[
                    wavenumber_idx, y_indices, x_indices
                ]

                heatmap_fig.add_trace(
                    go.Heatmap(
                        z=mask,
                        colorscale=[[0, "rgba(255,0,0,0)"], [1, "rgba(255,0,0,0.7)"]],
                        showscale=False,
                        name="Selection",
                        y=list(range(Y)),  # Explicitly set y coordinates
                        yaxis="y",
                    )
                )

            heatmap_fig.update_layout(
                dragmode="lasso" if selection_source == "heatmap" else None,
                title=f"Heatmap (Wavenumber: {self.wavenumbers[wavenumber_idx]:.0f})",
                yaxis=dict(
                    scaleanchor="x",  # Lock the aspect ratio
                    scaleratio=aspect_ratio,  # Set the correct ratio
                    autorange=True,  # This ensures origin at bottom
                ),
            )

            # Configure the modebar to show lasso selection tool
            heatmap_fig.update_layout(modebar=dict(add=["lasso2d"]))

            return latent_fig, spectral_fig, heatmap_fig

    def run(self, debug=True, port=8050):
        """Run the Dash app"""
        self.app.run(debug=debug, port=port)
