import multiprocessing
import os
import tempfile
import argparse
import sys
from multiprocessing import Pipe, Process
from time import sleep

import dash
import dash_bootstrap_components as dbc
import duckdb
import numpy as np
import pandas as pd
import redis
from pybaselines import Baseline
from tools.dbtools import (
    add_column_to_table,
    append_df_to_table,
    check_table_exists,
    read_df_from_db,
    store_df_in_db,
)
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate


def apply_baseline_correction(data, wavenumbers, method, **kwargs):
    """
    Apply baseline correction to spectral data
    
    Args:
        data: Input spectral data (n_samples, n_features)
        wavenumbers: Wavenumber array
        method: Baseline correction method
        **kwargs: Additional parameters for the baseline method
    
    Returns:
        tuple: (corrected_data, baseline_data)
    """
    baseline_fitter = Baseline(wavenumbers)
    
    # Clean kwargs to remove any dictionary values that might cause issues
    cleaned_kwargs = {}
    for key, value in kwargs.items():
        if value is not None and not isinstance(value, dict):
            cleaned_kwargs[key] = value
    
            # Dictionary mapping method names to their function calls and accepted parameters
        method_configs = {
            "imodpoly": {
                "func": baseline_fitter.imodpoly,
                "params": ["poly_order", "tol", "max_iter", "use_original", "mask_initial_peaks", "num_std"]
            },
            "quantile": {
                "func": baseline_fitter.quant_reg,
                "params": ["poly_order", "quantile", "tol", "max_iter", "eps"]
            },
            "rubberband": {
                "func": baseline_fitter.rubberband,
                "params": ["segments", "lam", "diff_order", "smooth_half_window"]
            },
            "pspline_arpls": {
                "func": baseline_fitter.pspline_arpls,
                "params": ["lam", "num_knots", "spline_degree", "diff_order", "max_iter", "tol"]
            },
            "interp_pts": {
                "func": baseline_fitter.interp_pts,
                "params": ["baseline_points", "interp_method"]
            },
        }
    
    corrected_data = np.zeros_like(data)
    baseline_data = np.zeros_like(data)
    
    # Track failures to avoid processing all spectra if method is fundamentally broken
    consecutive_failures = 0
    max_consecutive_failures = 5
    
    for i in range(data.shape[0]):
        try:
            # Validate the data for this spectrum
            spectrum_data = data[i]
            if np.any(np.isnan(spectrum_data)) or np.any(np.isinf(spectrum_data)):
                print(f"Warning: Spectrum {i} contains NaN or infinite values, skipping")
                corrected_data[i] = spectrum_data
                baseline_data[i] = np.zeros_like(spectrum_data)
                continue
            
                        # Ensure data is in the right format (float64) and handle any problematic values
            spectrum_data = np.asarray(spectrum_data, dtype=np.float64)
            
            # Replace any infinite values with NaN, then replace NaN with 0
            spectrum_data = np.where(np.isinf(spectrum_data), np.nan, spectrum_data)
            spectrum_data = np.where(np.isnan(spectrum_data), 0.0, spectrum_data)
            
            # Additional validation: ensure data is finite and numeric
            if not np.all(np.isfinite(spectrum_data)):
                print(f"Warning: Spectrum {i} contains non-finite values after cleaning")
                spectrum_data = np.where(np.isfinite(spectrum_data), spectrum_data, 0.0)
            
            # Ensure data is not all zeros or all the same value
            if np.all(spectrum_data == 0) or np.all(spectrum_data == spectrum_data[0]):
                print(f"Warning: Spectrum {i} has no variation, adding small noise")
                spectrum_data = spectrum_data + np.random.normal(0, 1e-10, len(spectrum_data))
            
            # Debug: print data info for first few spectra
            if i < 3:
                print(f"Spectrum {i} data info:")
                print(f"  Shape: {spectrum_data.shape}")
                print(f"  Type: {type(spectrum_data)}")
                print(f"  Min: {np.min(spectrum_data)}")
                print(f"  Max: {np.max(spectrum_data)}")
                print(f"  Mean: {np.mean(spectrum_data)}")
                print(f"  Sample values: {spectrum_data[:5]}")
                print(f"  Has NaN: {np.any(np.isnan(spectrum_data))}")
                print(f"  Has Inf: {np.any(np.isinf(spectrum_data))}")
                
            if method in method_configs:
                # Get the function and accepted parameters for this method
                config = method_configs[method]
                func = config["func"]
                accepted_params = config["params"]
                
                # Filter kwargs to only include accepted parameters
                method_kwargs = {k: v for k, v in cleaned_kwargs.items() if k in accepted_params}
                
                # Additional safety check: remove any dictionary values and None values
                # Also convert parameters to appropriate types
                final_kwargs = {}
                for k, v in method_kwargs.items():
                    if v is not None and not isinstance(v, dict):
                        # Convert to appropriate type based on parameter
                        if k in ["poly_order", "max_iter", "num_knots", "spline_degree", "half_window", "segments", "diff_order", "smooth_half_window"]:
                            # These should be integers
                            try:
                                final_kwargs[k] = int(float(v))
                            except (ValueError, TypeError):
                                print(f"Warning: Skipping non-numeric parameter {k}={v}")
                        else:
                            # These should be floats
                            try:
                                final_kwargs[k] = float(v)
                            except (ValueError, TypeError):
                                print(f"Warning: Skipping non-numeric parameter {k}={v}")
                
                # Call the function with filtered parameters
                try:
                    # Additional check: ensure data doesn't contain problematic values
                    if np.any(np.isnan(spectrum_data)) or np.any(np.isinf(spectrum_data)):
                        print(f"Warning: Spectrum {i} contains problematic values, skipping")
                        corrected_data[i] = spectrum_data
                        baseline_data[i] = np.zeros_like(spectrum_data)
                        continue
                    
                    # Check for any non-numeric values
                    if not np.issubdtype(spectrum_data.dtype, np.number):
                        print(f"Warning: Spectrum {i} is not numeric, skipping")
                        corrected_data[i] = spectrum_data
                        baseline_data[i] = np.zeros_like(spectrum_data)
                        continue
                    
                    # For morphological method, ensure we don't pass any dictionary parameters
                    if method == "morphological":
                        # Only pass half_window, explicitly avoid window_kwargs and kwargs
                        mor_kwargs = {}
                        if "half_window" in final_kwargs:
                            half_window_val = final_kwargs["half_window"]
                            # Ensure half_window is not too large for the data
                            max_half_window = len(spectrum_data) // 4  # Use at most 1/4 of data length
                            if half_window_val > max_half_window:
                                print(f"  Warning: half_window {half_window_val} too large for data length {len(spectrum_data)}, reducing to {max_half_window}")
                                half_window_val = max_half_window
                            mor_kwargs["half_window"] = half_window_val
                        # Ensure data is float64 and properly formatted
                        spectrum_data = np.asarray(spectrum_data, dtype=np.float64)
                        
                        # Additional validation for morphological method
                        if len(spectrum_data) < 10:
                            print(f"  Warning: Data too short for morphological method, skipping")
                            corrected_data[i] = spectrum_data
                            baseline_data[i] = np.zeros_like(spectrum_data)
                            continue
                        
                        # Ensure data is finite and positive (morphological methods work better with positive data)
                        if np.any(spectrum_data < 0):
                            print(f"  Warning: Data contains negative values, shifting to positive")
                            spectrum_data = spectrum_data - np.min(spectrum_data)
                        
                        result = func(spectrum_data, **mor_kwargs)
                    else:
                        # Try calling with explicit parameter names instead of **final_kwargs
                        if method == "polynomial":
                            result = func(spectrum_data, poly_order=final_kwargs.get('poly_order', 2))
                        elif method == "modpoly":
                            result = func(spectrum_data, poly_order=final_kwargs.get('poly_order', 2))
                        elif method == "morphological":
                            result = func(spectrum_data, half_window=final_kwargs.get('half_window', 50))
                        else:
                            # Fall back to **final_kwargs for other methods
                            result = func(spectrum_data, **final_kwargs)
                except Exception as e:
                    import traceback
                    print(f"Detailed error for spectrum {i}, method {method}:")
                    print(f"  Function: {func.__name__}")
                    print(f"  Data shape: {spectrum_data.shape}")
                    print(f"  Data type: {type(spectrum_data)}")
                    print(f"  Data sample: {spectrum_data[:5]}")
                    print(f"  Parameters: {final_kwargs}")
                    print(f"  Error: {str(e)}")
                    print(f"  Error type: {type(e)}")
                    print(f"  Full traceback:")
                    print(traceback.format_exc())
                    
                    # Try fallback to polynomial if modpoly fails
                    if method == "modpoly":
                        print(f"  Trying polynomial fallback for spectrum {i}")
                        try:
                            fallback_result = baseline_fitter.poly(spectrum_data, poly_order=3)
                            if isinstance(fallback_result, tuple):
                                baseline, params = fallback_result
                                corrected = spectrum_data - baseline
                            else:
                                baseline = fallback_result
                                corrected = spectrum_data - baseline
                            corrected_data[i] = corrected
                            baseline_data[i] = baseline
                            print(f"  Fallback successful for spectrum {i}")
                            continue
                        except Exception as e2:
                            print(f"  Fallback also failed for spectrum {i}: {str(e2)}")
                    
                    raise e
                
                # Handle different return types automatically
                if isinstance(result, tuple):
                    if len(result) == 2:
                        # For pybaselines, first element is baseline, second is params dict
                        baseline, params = result
                        corrected = data[i] - baseline
                    else:
                        # Unexpected tuple length
                        print(f"Warning: Unexpected return tuple length for {method}")
                        corrected, baseline = data[i], np.zeros_like(data[i])
                else:
                    # Single return value - assume it's the baseline
                    baseline = result
                    corrected = data[i] - baseline
                    
            else:
                print(f"Unknown method: {method}")
                corrected, baseline = data[i], np.zeros_like(data[i])
                
            corrected_data[i] = corrected
            baseline_data[i] = baseline
            
            # Reset failure counter on success
            consecutive_failures = 0
            
        except Exception as e:
            consecutive_failures += 1
            print(f"Error processing spectrum {i} with method {method}: {str(e)}")
            print(f"Parameters passed: {cleaned_kwargs}")
            
            # If we've had too many consecutive failures, stop processing
            if consecutive_failures >= max_consecutive_failures:
                print(f"Stopping processing after {consecutive_failures} consecutive failures.")
                print(f"Method '{method}' appears to be incompatible with this data.")
                print(f"Consider trying a different baseline correction method.")
                # Fill remaining spectra with original data
                for j in range(i, data.shape[0]):
                    corrected_data[j] = data[j]
                    baseline_data[j] = np.zeros_like(data[j])
                break
            
            corrected_data[i] = data[i]
            baseline_data[i] = np.zeros_like(data[i])
    
    return corrected_data, baseline_data, consecutive_failures


def apply_median_filter(data, window_size):
    """
    Apply median filter to spectral data
    
    Args:
        data: Input spectral data (n_samples, n_features)
        window_size: Size of the median filter window (must be odd)
    
    Returns:
        filtered_data: Median filtered spectral data
    """
    from scipy.signal import medfilt
    
    # Ensure window_size is odd
    if window_size % 2 == 0:
        window_size += 1
    
    # Apply median filter to each spectrum
    filtered_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        filtered_data[i] = medfilt(data[i], window_size)
    
    return filtered_data


def apply_normalization(data, method, **kwargs):
    """
    Apply normalization to spectral data
    
    Args:
        data: Input spectral data (n_samples, n_features)
        method: Normalization method
        **kwargs: Additional parameters for normalization
    
    Returns:
        normalized_data: Normalized spectral data
    """
    normalized_data = np.zeros_like(data)
    
    for i in range(data.shape[0]):
        try:
            if method == "none":
                normalized_data[i] = data[i]
            elif method == "min_max":
                min_val = np.min(data[i])
                max_val = np.max(data[i])
                if max_val > min_val:
                    normalized_data[i] = (data[i] - min_val) / (max_val - min_val)
                else:
                    normalized_data[i] = data[i]
            elif method == "z_score":
                mean_val = np.mean(data[i])
                std_val = np.std(data[i])
                if std_val > 0:
                    normalized_data[i] = (data[i] - mean_val) / std_val
                else:
                    normalized_data[i] = data[i]
            elif method == "snv":
                mean_val = np.mean(data[i])
                std_val = np.std(data[i])
                if std_val > 0:
                    normalized_data[i] = (data[i] - mean_val) / std_val
                else:
                    normalized_data[i] = data[i]
            elif method == "area":
                area = np.trapz(data[i])
                if area > 0:
                    normalized_data[i] = data[i] / area
                else:
                    normalized_data[i] = data[i]
            elif method == "vector":
                norm = np.linalg.norm(data[i])
                if norm > 0:
                    normalized_data[i] = data[i] / norm
                else:
                    normalized_data[i] = data[i]
            else:
                print(f"Unknown normalization method: {method}")
                normalized_data[i] = data[i]
                
        except Exception as e:
            print(f"Error normalizing spectrum {i} with method {method}: {str(e)}")
            normalized_data[i] = data[i]
    
    return normalized_data


def baseline_correction_runner(
    data, wavenumbers, params, pid, redis_host="localhost", redis_port=6379
):
    """Run baseline correction and store results in Redis.

    Args:
        data: Input spectral data
        wavenumbers: Wavenumber array
        params: Baseline correction parameters dictionary
        pid: Process ID for Redis key generation
        redis_host: Redis host address
        redis_port: Redis port number

    Returns:
        bool: True if successful, False if error occurred
    """
    try:
        print("===>>>>>>  BASELINE CORRECTION PROCESS STARTED ===")
        print(f"Data shape: {data.shape}")
        print(f"Wavenumbers: {len(wavenumbers)} points")
        print(f"Parameters: {params}")

        # Check and fix dimension mismatch
        if len(wavenumbers) != data.shape[1]:
            print(f"WARNING: Wavenumber length ({len(wavenumbers)}) doesn't match data columns ({data.shape[1]})")
            if len(wavenumbers) > data.shape[1]:
                wavenumbers = wavenumbers[:data.shape[1]]
                print(f"Truncated wavenumbers to {len(wavenumbers)} points")
            else:
                data = data[:, :len(wavenumbers)]
                print(f"Truncated data to {data.shape[1]} columns")
            print(f"Adjusted - Data shape: {data.shape}, Wavenumbers: {len(wavenumbers)} points")

        # Connect to Redis
        r = redis.Redis(host=redis_host, port=redis_port)

        # Create Redis keys
        corrected_key = f"temp:baseline:{pid}:corrected"
        baseline_key = f"temp:baseline:{pid}:baseline"
        status_key = f"temp:baseline:{pid}:status"

        # Extract parameters
        baseline_method = params.get("baseline_method", "polynomial")
        normalization_method = params.get("normalization_method", "none")
        median_filter_window = params.get("median_filter_window", None)
        
        # Get baseline parameters based on method - use the same smart filtering as apply_baseline_correction
        baseline_params = {}
        
        # Dictionary mapping method names to their accepted parameters (same as in apply_baseline_correction)
        method_configs = {
            "imodpoly": ["poly_order", "tol", "max_iter", "use_original", "mask_initial_peaks", "num_std"],
            "quantile": ["poly_order", "quantile", "tol", "max_iter", "eps"],
            "rubberband": ["segments", "lam", "diff_order", "smooth_half_window"],
            "pspline_arpls": ["lam", "num_knots", "spline_degree", "diff_order", "max_iter", "tol"],
            "interp_pts": ["baseline_points", "interp_method"],
        }
        
        if baseline_method in method_configs:
            accepted_params = method_configs[baseline_method]
            
            # Default values for common parameters
            defaults = {
                "poly_order": 2,
                "lambda_param": 1000.0,
                "frac": 0.1,
                "quantile": 0.1,
                "half_window": 50,
                "segments": 1,
                "rubberband_lam": 1000.0,
                "num_knots": 100,
                "spline_degree": 3,
                "num_std": 2.0,
                "max_iter": 100,
                "tol": 0.001
            }
            
            # Only include parameters that are accepted by this method
            for param in accepted_params:
                value = params.get(param, defaults.get(param))
                if value is not None and not isinstance(value, dict):
                    baseline_params[param] = value
            
            # Special mapping for loess method
            if baseline_method == "loess" and "frac" in params:
                baseline_params["fraction"] = params["frac"]
            
            # Special mapping for rubberband method
            if baseline_method == "rubberband" and "rubberband_lam" in params:
                baseline_params["lam"] = params["rubberband_lam"]
            
            # Special mapping for methods that use 'lam' instead of 'lambda_param'
            lam_methods = ["mpls", "arpls", "asls", "iasls", "psalsa", "drpls", "iarpls", "aspls", 
                          "pspline_asls", "pspline_iasls", "pspline_arpls", "pspline_iarpls", 
                          "pspline_psalsa", "pspline_drpls", "pspline_aspls"]
            if baseline_method in lam_methods and "lambda_param" in params:
                baseline_params["lam"] = params["lambda_param"]
        
        print(f"Applying baseline correction with method: {baseline_method}")
        print(f"Baseline parameters: {baseline_params}")
        
        # Additional safety check: ensure all values are numeric
        safe_baseline_params = {}
        for k, v in baseline_params.items():
            try:
                # Convert to appropriate type based on parameter
                if k in ["poly_order", "max_iter", "num_knots", "spline_degree", "half_window", "segments", "diff_order", "smooth_half_window"]:
                    # These should be integers
                    int_val = int(float(v))
                    safe_baseline_params[k] = int_val
                else:
                    # These should be floats
                    float_val = float(v)
                    safe_baseline_params[k] = float_val
            except (ValueError, TypeError):
                print(f"Warning: Skipping non-numeric parameter {k}={v} (type: {type(v)})")
        
        print(f"Safe baseline parameters: {safe_baseline_params}")

        # Apply median filter if specified
        working_data = data.copy()
        if median_filter_window is not None and median_filter_window > 1:
            print(f"Applying median filter with window size: {median_filter_window}")
            working_data = apply_median_filter(data, median_filter_window)
            print(f"Data shape after median filtering: {working_data.shape}")

        # Apply baseline correction
        corrected_data, baseline_data, consecutive_failures = apply_baseline_correction(
            working_data, wavenumbers, baseline_method, **safe_baseline_params
        )

        print(f"Applied baseline correction. Corrected data shape: {corrected_data.shape}")

        # Apply normalization if specified
        if normalization_method != "none":
            print(f"Applying normalization with method: {normalization_method}")
            corrected_data = apply_normalization(corrected_data, normalization_method)
            print(f"Applied normalization. Final data shape: {corrected_data.shape}")

        # Store results in Redis
        r.set(corrected_key, corrected_data.tobytes())
        r.set(baseline_key, baseline_data.tobytes())
        
        # Also store baseline data in Redis for immediate access
        baseline_key_data = f"temp:baseline:{pid}:baseline_data"
        r.set(baseline_key_data, baseline_data.tobytes())
        
        # Check if we had failures and set appropriate status
        max_consecutive_failures = 5  # Define this here
        if consecutive_failures >= max_consecutive_failures:
            error_msg = f"Baseline correction failed: Method '{baseline_method}' is incompatible with this data. Consider trying a different method."
            r.set(status_key, error_msg)
            print("===>>>>>>  BASELINE CORRECTION PROCESS FAILED ===")
            return False
        else:
            r.set(status_key, "done")
            print("===>>>>>>  BASELINE CORRECTION PROCESS FINISHED ===")
            return True

    except Exception as e:
        error_msg = f"Error in baseline correction: {str(e)}"
        print(error_msg)
        try:
            r.set(status_key, error_msg)
        except:
            pass
        return False


class BaselineCorrectionApp:
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379):
        # Initialize basic attributes
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.data = None
        self.wavenumbers = None
        self.xy_coords = None
        self.process = None
        self.current_pid = None
        
        # Initialize Redis client
        self.redis_client = redis.Redis(host=redis_host, port=redis_port)

        # Initialize Dash app
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.app.title = "Baseline Correction Tool"
        self._setup_layout()
        self._setup_callbacks()

    def _setup_layout(self):
        """Setup the Dash app layout"""
        self.app.layout = html.Div(
            [
                html.H1("Baseline Correction Tool"),
                # Redis Status Card (matches configure_umap)
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Redis Connection Status"),
                                        dbc.CardBody(
                                            [
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                html.Div(id="redis-status"),
                                                                html.Div([
                                                                    html.Span("Current Project: ", style={"fontWeight": "bold"}),
                                                                    html.Span(id="current-project", className="text-primary"),
                                                                ], className="mt-2"),
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
                                                        ),
                                                    ]
                                                ),
                                                dcc.Interval(id="status-check", interval=5000),
                                            ]
                                        ),
                                    ],
                                    className="mb-3",
                                )
                            ],
                            width=12,
                        ),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H1("Baseline Correction Tool", className="text-center mb-4"),
                                html.Hr(),
                            ],
                            width=12,
                        )
                    ]
                ),
                
                # Data Loading Section
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Data Loading"),
                                        dbc.CardBody(
                                            [
                                                dbc.Button(
                                                    "Load Data from Database",
                                                    id="load-data-button",
                                                    color="primary",
                                                    className="mb-3",
                                                ),
                                                html.Div(id="load-data-status"),
                                            ]
                                        ),
                                    ]
                                )
                            ],
                            width=12,
                        )
                    ],
                    className="mb-4",
                ),

                # Baseline Correction Parameters
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Baseline Correction Parameters"),
                                        dbc.CardBody(
                                            [
                                                # Preprocessing Section
                                                html.H6("Preprocessing", className="mb-3"),
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                html.Label("Median Filter Window Size:"),
                                                                dcc.Input(
                                                                    id="median-filter-window",
                                                                    type="number",
                                                                    value=5,
                                                                    min=3,
                                                                    max=51,
                                                                    step=2,
                                                                    className="form-control mb-3",
                                                                    placeholder="Odd number (3-51)",
                                                                ),
                                                                html.Small("Set to 1 to disable median filtering", className="text-muted"),
                                                            ],
                                                            width=4,
                                                        ),
                                                    ]
                                                ),
                                                html.Hr(),
                                                html.H6("Processing", className="mb-3"),
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                html.Label("Baseline Method:"),
                                                                dcc.Dropdown(
                                                                    id="baseline-method",
                                                                    options=[
                                                                        {"label": "Iterative Modified Polynomial", "value": "imodpoly"},
                                                                        {"label": "Quantile", "value": "quantile"},
                                                                        {"label": "Rubberband", "value": "rubberband"},
                                                                        {"label": "PSpline ARPLS", "value": "pspline_arpls"},
                                                                        {"label": "Interpolate Points (Manual)", "value": "interp_pts"},
                                                                    ],
                                                                    value="imodpoly",
                                                                    className="mb-3",
                                                                ),
                                                            ],
                                                            width=6,
                                                        ),
                                                        dbc.Col(
                                                            [
                                                                html.Label("Normalization Method:"),
                                                                dcc.Dropdown(
                                                                    id="normalization-method",
                                                                    options=[
                                                                        {"label": "None", "value": "none"},
                                                                        {"label": "Min-Max", "value": "min_max"},
                                                                        {"label": "Z-Score", "value": "z_score"},
                                                                        {"label": "SNV", "value": "snv"},
                                                                        {"label": "Area", "value": "area"},
                                                                        {"label": "Vector", "value": "vector"},
                                                                    ],
                                                                    value="none",
                                                                    className="mb-3",
                                                                ),
                                                            ],
                                                            width=6,
                                                        ),
                                                    ]
                                                ),
                                                
                                                # Iterative Modified Polynomial parameters
                                                html.Div(
                                                    id="imodpoly-params",
                                                    children=[
                                                        dbc.Row(
                                                            [
                                                                dbc.Col(
                                                                    [
                                                                        html.Label("Polynomial Order:"),
                                                                        dcc.Input(
                                                                            id="poly-order",
                                                                            type="number",
                                                                            value=2,
                                                                            min=1,
                                                                            max=10,
                                                                            className="form-control",
                                                                        ),
                                                                    ],
                                                                    width=4,
                                                                ),
                                                                dbc.Col(
                                                                    [
                                                                        html.Label("Tolerance:"),
                                                                        dcc.Input(
                                                                            id="tol",
                                                                            type="number",
                                                                            value=0.001,
                                                                            min=0.0001,
                                                                            max=0.1,
                                                                            step=0.0001,
                                                                            className="form-control",
                                                                        ),
                                                                    ],
                                                                    width=4,
                                                                ),
                                                                dbc.Col(
                                                                    [
                                                                        html.Label("Max Iterations:"),
                                                                        dcc.Input(
                                                                            id="max-iter",
                                                                            type="number",
                                                                            value=100,
                                                                            min=10,
                                                                            max=1000,
                                                                            className="form-control",
                                                                        ),
                                                                    ],
                                                                    width=4,
                                                                ),
                                                            ],
                                                            className="mb-3",
                                                        ),
                                                        dbc.Row(
                                                            [
                                                                dbc.Col(
                                                                    [
                                                                        html.Label("Use Original:"),
                                                                        dcc.Dropdown(
                                                                            id="use-original",
                                                                            options=[
                                                                                {"label": "True", "value": True},
                                                                                {"label": "False", "value": False},
                                                                            ],
                                                                            value=True,
                                                                            className="form-control",
                                                                        ),
                                                                    ],
                                                                    width=4,
                                                                ),
                                                                dbc.Col(
                                                                    [
                                                                        html.Label("Mask Initial Peaks:"),
                                                                        dcc.Dropdown(
                                                                            id="mask-initial-peaks",
                                                                            options=[
                                                                                {"label": "True", "value": True},
                                                                                {"label": "False", "value": False},
                                                                            ],
                                                                            value=False,
                                                                            className="form-control",
                                                                        ),
                                                                    ],
                                                                    width=4,
                                                                ),
                                                                dbc.Col(
                                                                    [
                                                                        html.Label("Number of Std Devs:"),
                                                                        dcc.Input(
                                                                            id="num-std",
                                                                            type="number",
                                                                            value=2.0,
                                                                            min=0.1,
                                                                            max=10.0,
                                                                            step=0.1,
                                                                            className="form-control",
                                                                        ),
                                                                    ],
                                                                    width=4,
                                                                ),
                                                            ],
                                                            className="mb-3",
                                                        ),
                                                    ]
                                                ),
                                                
                                                # Quantile parameters
                                                html.Div(
                                                    id="quantile-params",
                                                    children=[
                                                        dbc.Row(
                                                            [
                                                                dbc.Col(
                                                                    [
                                                                        html.Label("Polynomial Order:"),
                                                                        dcc.Input(
                                                                            id="quantile-poly-order",
                                                                            type="number",
                                                                            value=2,
                                                                            min=1,
                                                                            max=10,
                                                                            className="form-control",
                                                                        ),
                                                                    ],
                                                                    width=3,
                                                                ),
                                                                dbc.Col(
                                                                    [
                                                                        html.Label("Quantile:"),
                                                                        dcc.Input(
                                                                            id="quantile",
                                                                            type="number",
                                                                            value=0.1,
                                                                            min=0.01,
                                                                            max=0.5,
                                                                            step=0.01,
                                                                            className="form-control",
                                                                        ),
                                                                    ],
                                                                    width=3,
                                                                ),
                                                                dbc.Col(
                                                                    [
                                                                        html.Label("Tolerance:"),
                                                                        dcc.Input(
                                                                            id="quantile-tol",
                                                                            type="number",
                                                                            value=0.001,
                                                                            min=0.0001,
                                                                            max=0.1,
                                                                            step=0.0001,
                                                                            className="form-control",
                                                                        ),
                                                                    ],
                                                                    width=3,
                                                                ),
                                                                dbc.Col(
                                                                    [
                                                                        html.Label("Max Iterations:"),
                                                                        dcc.Input(
                                                                            id="quantile-max-iter",
                                                                            type="number",
                                                                            value=100,
                                                                            min=10,
                                                                            max=1000,
                                                                            className="form-control",
                                                                        ),
                                                                    ],
                                                                    width=3,
                                                                ),
                                                            ],
                                                            className="mb-3",
                                                        ),
                                                        dbc.Row(
                                                            [
                                                                dbc.Col(
                                                                    [
                                                                        html.Label("Epsilon:"),
                                                                        dcc.Input(
                                                                            id="eps",
                                                                            type="number",
                                                                            value=1e-8,
                                                                            min=1e-10,
                                                                            max=1e-5,
                                                                            step=1e-9,
                                                                            className="form-control",
                                                                        ),
                                                                    ],
                                                                    width=4,
                                                                ),
                                                            ],
                                                            className="mb-3",
                                                        ),
                                                    ],
                                                    style={"display": "none"},
                                                ),
                                                
                                                # Rubberband parameters
                                                html.Div(
                                                    id="rubberband-params",
                                                    children=[
                                                        dbc.Row(
                                                            [
                                                                dbc.Col(
                                                                    [
                                                                        html.Label("Segments:"),
                                                                        dcc.Input(
                                                                            id="segments",
                                                                            type="number",
                                                                            value=1,
                                                                            min=1,
                                                                            max=10,
                                                                            className="form-control",
                                                                        ),
                                                                    ],
                                                                    width=3,
                                                                ),
                                                                dbc.Col(
                                                                    [
                                                                        html.Label("Lambda:"),
                                                                        dcc.Input(
                                                                            id="rubberband-lam",
                                                                            type="number",
                                                                            value=1000.0,
                                                                            step=100.0,
                                                                            className="form-control",
                                                                        ),
                                                                    ],
                                                                    width=3,
                                                                ),
                                                                dbc.Col(
                                                                    [
                                                                        html.Label("Diff Order:"),
                                                                        dcc.Input(
                                                                            id="diff-order",
                                                                            type="number",
                                                                            value=2,
                                                                            min=1,
                                                                            max=4,
                                                                            className="form-control",
                                                                        ),
                                                                    ],
                                                                    width=3,
                                                                ),
                                                                dbc.Col(
                                                                    [
                                                                        html.Label("Smooth Half Window:"),
                                                                        dcc.Input(
                                                                            id="smooth-half-window",
                                                                            type="number",
                                                                            value=1,
                                                                            min=1,
                                                                            max=100,
                                                                            className="form-control",
                                                                        ),
                                                                    ],
                                                                    width=3,
                                                                ),
                                                            ],
                                                            className="mb-3",
                                                        ),
                                                    ],
                                                    style={"display": "none"},
                                                ),
                                                
                                                # PSpline ARPLS parameters
                                                html.Div(
                                                    id="pspline-params",
                                                    children=[
                                                        dbc.Row(
                                                            [
                                                                dbc.Col(
                                                                    [
                                                                        html.Label("Lambda:"),
                                                                        dcc.Input(
                                                                            id="pspline-lam",
                                                                            type="number",
                                                                            value=1000.0,
                                                                            step=100.0,
                                                                            className="form-control",
                                                                        ),
                                                                    ],
                                                                    width=4,
                                                                ),
                                                                dbc.Col(
                                                                    [
                                                                        html.Label("Number of Knots:"),
                                                                        dcc.Input(
                                                                            id="num-knots",
                                                                            type="number",
                                                                            value=100,
                                                                            min=10,
                                                                            max=1000,
                                                                            className="form-control",
                                                                        ),
                                                                    ],
                                                                    width=4,
                                                                ),
                                                                dbc.Col(
                                                                    [
                                                                        html.Label("Spline Degree:"),
                                                                        dcc.Input(
                                                                            id="spline-degree",
                                                                            type="number",
                                                                            value=3,
                                                                            min=1,
                                                                            max=5,
                                                                            className="form-control",
                                                                        ),
                                                                    ],
                                                                    width=4,
                                                                ),
                                                            ],
                                                            className="mb-3",
                                                        ),
                                                        dbc.Row(
                                                            [
                                                                dbc.Col(
                                                                    [
                                                                        html.Label("Diff Order:"),
                                                                        dcc.Input(
                                                                            id="pspline-diff-order",
                                                                            type="number",
                                                                            value=2,
                                                                            min=1,
                                                                            max=4,
                                                                            className="form-control",
                                                                        ),
                                                                    ],
                                                                    width=3,
                                                                ),
                                                                dbc.Col(
                                                                    [
                                                                        html.Label("Max Iterations:"),
                                                                        dcc.Input(
                                                                            id="pspline-max-iter",
                                                                            type="number",
                                                                            value=100,
                                                                            min=10,
                                                                            max=1000,
                                                                            className="form-control",
                                                                        ),
                                                                    ],
                                                                    width=3,
                                                                ),
                                                                dbc.Col(
                                                                    [
                                                                        html.Label("Tolerance:"),
                                                                        dcc.Input(
                                                                            id="pspline-tol",
                                                                            type="number",
                                                                            value=0.001,
                                                                            min=0.0001,
                                                                            max=0.1,
                                                                            step=0.0001,
                                                                            className="form-control",
                                                                        ),
                                                                    ],
                                                                    width=3,
                                                                ),
                                                                dbc.Col(
                                                                    [
                                                                        html.Label("Predefined Knot Sets:"),
                                                                        dcc.Dropdown(
                                                                            id="knot-presets",
                                                                            options=[
                                                                                {"label": "Custom", "value": "custom"},
                                                                                {"label": "Uniform (5 knots)", "value": "uniform_5"},
                                                                                {"label": "Uniform (10 knots)", "value": "uniform_10"},
                                                                                {"label": "Uniform (20 knots)", "value": "uniform_20"},
                                                                                {"label": "Dense (50 knots)", "value": "dense_50"},
                                                                                {"label": "Sparse (3 knots)", "value": "sparse_3"},
                                                                                {"label": "Logarithmic (10 knots)", "value": "log_10"},
                                                                            ],
                                                                            value="uniform_10",
                                                                            className="form-control",
                                                                        ),
                                                                    ],
                                                                    width=3,
                                                                ),
                                                            ],
                                                            className="mb-3",
                                                        ),
                                                        dbc.Row(
                                                            [
                                                                dbc.Col(
                                                                    [
                                                                        html.Label("Custom Knot Positions:"),
                                                                        dcc.Textarea(
                                                                            id="knot-positions",
                                                                            placeholder="Enter custom knot positions as comma-separated values (e.g., 0, 0.25, 0.5, 0.75, 1.0)",
                                                                            value="",
                                                                            rows=3,
                                                                            className="form-control",
                                                                        ),
                                                                        html.Small("Leave empty to use predefined set", className="text-muted"),
                                                                    ],
                                                                    width=12,
                                                                ),
                                                            ],
                                                            className="mb-3",
                                                        ),
                                                    ],
                                                    style={"display": "none"},
                                                ),
                                                
                                                # Interpolate Points parameters
                                                html.Div(
                                                    id="interp-params",
                                                    children=[
                                                        dbc.Row(
                                                            [
                                                                dbc.Col(
                                                                    [
                                                                        html.Label("Interpolation Method:"),
                                                                        dcc.Dropdown(
                                                                            id="interp-method",
                                                                            options=[
                                                                                {"label": "Linear", "value": "linear"},
                                                                                {"label": "Cubic", "value": "cubic"},
                                                                                {"label": "Quadratic", "value": "quadratic"},
                                                                                {"label": "Nearest", "value": "nearest"},
                                                                                {"label": "Previous", "value": "previous"},
                                                                                {"label": "Next", "value": "next"},
                                                                            ],
                                                                            value="linear",
                                                                            className="form-control",
                                                                        ),
                                                                    ],
                                                                    width=6,
                                                                ),
                                                                dbc.Col(
                                                                    [
                                                                        html.Label("Flat Region Wavenumbers:"),
                                                                        dcc.Textarea(
                                                                            id="baseline-points",
                                                                            placeholder="Enter wavenumbers where baseline should be flat (e.g., 100,200,300,400)",
                                                                            value="",
                                                                            rows=4,
                                                                            className="form-control",
                                                                        ),
                                                                        html.Small("Format: comma-separated wavenumbers where baseline should be flat", className="text-muted"),
                                                                    ],
                                                                    width=6,
                                                                ),
                                                            ],
                                                            className="mb-3",
                                                        ),
                                                        dbc.Row(
                                                            [
                                                                dbc.Col(
                                                                    [
                                                                        html.Label("Preset Flat Regions:"),
                                                                        dcc.Dropdown(
                                                                            id="flat-region-presets",
                                                                            options=[
                                                                                {"label": "Custom Wavenumbers", "value": "custom"},
                                                                                {"label": "Auto-Detect Minima", "value": "auto_minima"},
                                                                                {"label": "Quarter Points", "value": "quarter_points"},
                                                                                {"label": "Third Points", "value": "third_points"},
                                                                                {"label": "Start and End Only", "value": "start_end"},
                                                                            ],
                                                                            value="start_end",
                                                                            className="form-control",
                                                                        ),
                                                                    ],
                                                                    width=12,
                                                                ),
                                                            ],
                                                            className="mb-3",
                                                        ),
                                                    ],
                                                    style={"display": "none"},
                                                ),
                                            ]
                                        ),
                                    ]
                                )
                            ],
                            width=12,
                        )
                    ],
                    className="mb-4",
                ),

                # Run Button and Status
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Button(
                                    "Run Baseline Correction",
                                    id="run-button",
                                    color="success",
                                    size="lg",
                                    className="w-100 mb-3",
                                ),
                                html.Div(id="status-output", className="alert alert-info"),
                                dcc.Interval(id="process-check", interval=1000, disabled=True),
                            ],
                            width=12,
                        )
                    ],
                    className="mb-4",
                ),

                # Spatial Visualization Section
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Spatial Distribution"),
                                        dbc.CardBody(
                                            [
                                                dcc.Graph(
                                                    id="spatial-plot", 
                                                    style={"height": "500px"},
                                                    config={
                                                        "displayModeBar": True,
                                                        "modeBarButtonsToAdd": ["lasso2d"]
                                                    }
                                                ),
                                            ]
                                        ),
                                    ]
                                )
                            ],
                            width=6,
                        ),
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Selected Spectrum"),
                                        dbc.CardBody(
                                            [
                                                dcc.Graph(
                                                    id="spectrum-plot", 
                                                    style={"height": "500px"}
                                                ),
                                            ]
                                        ),
                                    ]
                                )
                            ],
                            width=6,
                        ),
                    ],
                    className="mb-4",
                ),
                


                # Export Section
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Export Results"),
                                        dbc.CardBody(
                                            [
                                                dbc.Button(
                                                    "Export Corrected Data",
                                                    id="export-button",
                                                    color="primary",
                                                    className="me-2",
                                                ),
                                                html.Div(id="export-status"),
                                            ]
                                        ),
                                    ]
                                )
                            ],
                            width=12,
                        )
                    ],
                    className="mb-4",
                ),

                # Hidden stores
                dcc.Store(id="selection-store", data=[]),
            ]
        )

    def _setup_callbacks(self):
        """Setup all Dash callbacks"""
        
        @self.app.callback(
            [
                Output("status-output", "children"),
                Output("process-check", "disabled"),
                Output("run-button", "children"),
                Output("run-button", "color"),
                Output("run-button", "disabled"),
            ],
            [Input("run-button", "n_clicks"), Input("process-check", "n_intervals")],
            [
                State("median-filter-window", "value"),
                State("baseline-method", "value"),
                State("normalization-method", "value"),
                # IModPoly parameters
                State("poly-order", "value"),
                State("tol", "value"),
                State("max-iter", "value"),
                State("use-original", "value"),
                State("mask-initial-peaks", "value"),
                State("num-std", "value"),
                # Quantile parameters
                State("quantile-poly-order", "value"),
                State("quantile", "value"),
                State("quantile-tol", "value"),
                State("quantile-max-iter", "value"),
                State("eps", "value"),
                # Rubberband parameters
                State("segments", "value"),
                State("rubberband-lam", "value"),
                State("diff-order", "value"),
                State("smooth-half-window", "value"),
                # PSpline parameters
                State("pspline-lam", "value"),
                State("num-knots", "value"),
                State("spline-degree", "value"),
                State("pspline-diff-order", "value"),
                State("pspline-max-iter", "value"),
                State("pspline-tol", "value"),
                State("knot-presets", "value"),
                State("knot-positions", "value"),
                # Interpolate Points parameters
                State("interp-method", "value"),
                State("baseline-points", "value"),
                State("flat-region-presets", "value"),
            ],
            prevent_initial_call=True,
        )
        def update_status(
            n_clicks,
            n_intervals,
            median_filter_window,
            baseline_method,
            normalization_method,
            # IModPoly parameters
            poly_order,
            tol,
            max_iter,
            use_original,
            mask_initial_peaks,
            num_std,
            # Quantile parameters
            quantile_poly_order,
            quantile,
            quantile_tol,
            quantile_max_iter,
            eps,
            # Rubberband parameters
            segments,
            rubberband_lam,
            diff_order,
            smooth_half_window,
            # PSpline parameters
            pspline_lam,
            num_knots,
            spline_degree,
            pspline_diff_order,
            pspline_max_iter,
            pspline_tol,
            knot_presets,
            knot_positions,
            # Interpolate Points parameters
            interp_method,
            baseline_points,
            flat_region_presets,
        ):
            ctx = dash.callback_context
            if not ctx.triggered:
                raise PreventUpdate

            trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

            if trigger_id == "run-button" and n_clicks:
                if self.data is None:
                    return ("Error: No data loaded", True, "Run Baseline Correction", "danger", False)

                try:
                    params = {
                        "median_filter_window": median_filter_window,
                        "baseline_method": baseline_method,
                        "normalization_method": normalization_method,
                    }
                    
                    # Add method-specific parameters
                    if baseline_method == "imodpoly":
                        params.update({
                            "poly_order": poly_order,
                            "tol": tol,
                            "max_iter": max_iter,
                            "use_original": use_original,
                            "mask_initial_peaks": mask_initial_peaks,
                            "num_std": num_std,
                        })
                    elif baseline_method == "quantile":
                        params.update({
                            "poly_order": quantile_poly_order,
                            "quantile": quantile,
                            "tol": quantile_tol,
                            "max_iter": quantile_max_iter,
                            "eps": eps,
                        })
                    elif baseline_method == "rubberband":
                        params.update({
                            "segments": segments,
                            "lam": rubberband_lam,
                            "diff_order": diff_order,
                            "smooth_half_window": smooth_half_window,
                        })
                    elif baseline_method == "pspline_arpls":
                        params.update({
                            "lam": pspline_lam,
                            "num_knots": num_knots,
                            "spline_degree": spline_degree,
                            "diff_order": pspline_diff_order,
                            "max_iter": pspline_max_iter,
                            "tol": pspline_tol,
                        })
                        
                        # Handle knot positions
                        if knot_positions and knot_positions.strip():
                            # Use custom knot positions
                            try:
                                knot_pos_list = [float(x.strip()) for x in knot_positions.split(",")]
                                params["knot_positions"] = knot_pos_list
                            except ValueError:
                                print(f"Warning: Invalid knot positions format: {knot_positions}")
                        else:
                            # Use predefined knot sets
                            knot_presets_map = {
                                "uniform_5": [0.0, 0.25, 0.5, 0.75, 1.0],
                                "uniform_10": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                                "uniform_20": [i/20.0 for i in range(21)],
                                "dense_50": [i/50.0 for i in range(51)],
                                "sparse_3": [0.0, 0.5, 1.0],
                                "log_10": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                            }
                            
                            if knot_presets in knot_presets_map:
                                params["knot_positions"] = knot_presets_map[knot_presets]
                                print(f"Using predefined knot set: {knot_presets}")
                            else:
                                print(f"Warning: Unknown knot preset: {knot_presets}")
                    elif baseline_method == "interp_pts":
                        params.update({
                            "interp_method": interp_method,
                        })
                        
                        # Handle baseline points
                        if baseline_points and baseline_points.strip():
                            # Use custom wavenumbers for flat regions
                            try:
                                # Parse wavenumbers from text format
                                wavenumbers_text = baseline_points.strip()
                                wavenumber_list = [float(x.strip()) for x in wavenumbers_text.split(",")]
                                
                                # Find the closest wavenumber indices in the data
                                wavenumber_indices = []
                                for target_wavenumber in wavenumber_list:
                                    # Find the closest wavenumber in the data
                                    closest_idx = np.argmin(np.abs(self.wavenumbers - target_wavenumber))
                                    wavenumber_indices.append(closest_idx)
                                
                                # Get the actual wavenumbers from the data
                                actual_wavenumbers = [self.wavenumbers[idx] for idx in wavenumber_indices]
                                
                                # Calculate baseline intensities as the minimum value in each region
                                # For now, use a simple approach: take the minimum of the data
                                baseline_intensities = []
                                for idx in wavenumber_indices:
                                    # Use the minimum value in a small window around this point
                                    window_start = max(0, idx - 5)
                                    window_end = min(len(self.data[0]), idx + 6)
                                    min_intensity = np.min(self.data[:, window_start:window_end])
                                    baseline_intensities.append(min_intensity)
                                
                                # Create baseline points
                                points_list = []
                                for i, (wavenumber, intensity) in enumerate(zip(actual_wavenumbers, baseline_intensities)):
                                    points_list.append([wavenumber, intensity])
                                
                                # Add start and end points if not already included
                                if len(points_list) == 0:
                                    # No custom points, use default
                                    points_list = [[self.wavenumbers[0], np.min(self.data)], [self.wavenumbers[-1], np.min(self.data)]]
                                else:
                                    # Add start point if not already there
                                    if points_list[0][0] > self.wavenumbers[0]:
                                        start_intensity = np.min(self.data[:, :10])  # Use first 10 points
                                        points_list.insert(0, [self.wavenumbers[0], start_intensity])
                                    
                                    # Add end point if not already there
                                    if points_list[-1][0] < self.wavenumbers[-1]:
                                        end_intensity = np.min(self.data[:, -10:])  # Use last 10 points
                                        points_list.append([self.wavenumbers[-1], end_intensity])
                                
                                params["baseline_points"] = points_list
                                print(f"Using custom flat regions at wavenumbers: {actual_wavenumbers}")
                                print(f"Baseline points: {points_list}")
                            except Exception as e:
                                print(f"Warning: Invalid wavenumbers format: {baseline_points}, error: {str(e)}")
                                # Fallback to default
                                params["baseline_points"] = [[self.wavenumbers[0], np.min(self.data)], [self.wavenumbers[-1], np.min(self.data)]]
                        else:
                            # Use predefined flat region presets
                            if flat_region_presets == "auto_minima":
                                # Auto-detect minima in the data
                                from scipy.signal import find_peaks
                                # Use the mean spectrum to find minima
                                mean_spectrum = np.mean(self.data, axis=0)
                                # Find minima (negative peaks)
                                minima_indices, _ = find_peaks(-mean_spectrum, distance=20, prominence=0.01)
                                
                                if len(minima_indices) > 0:
                                    # Take up to 5 minima, evenly spaced
                                    if len(minima_indices) > 5:
                                        step = len(minima_indices) // 5
                                        minima_indices = minima_indices[::step][:5]
                                    
                                    points_list = []
                                    for idx in minima_indices:
                                        wavenumber = self.wavenumbers[idx]
                                        intensity = np.min(self.data[:, max(0, idx-5):min(len(self.data[0]), idx+6)])
                                        points_list.append([wavenumber, intensity])
                                    
                                    # Add start and end points
                                    start_intensity = np.min(self.data[:, :10])
                                    end_intensity = np.min(self.data[:, -10:])
                                    points_list.insert(0, [self.wavenumbers[0], start_intensity])
                                    points_list.append([self.wavenumbers[-1], end_intensity])
                                    
                                    params["baseline_points"] = points_list
                                else:
                                    # Fallback to start and end
                                    params["baseline_points"] = [[self.wavenumbers[0], np.min(self.data)], [self.wavenumbers[-1], np.min(self.data)]]
                                    
                            elif flat_region_presets == "quarter_points":
                                # Use quarter points of the wavenumber range
                                quarter_indices = [0, len(self.wavenumbers)//4, len(self.wavenumbers)//2, 3*len(self.wavenumbers)//4, len(self.wavenumbers)-1]
                                points_list = []
                                for idx in quarter_indices:
                                    wavenumber = self.wavenumbers[idx]
                                    window_start = max(0, idx - 5)
                                    window_end = min(len(self.data[0]), idx + 6)
                                    intensity = np.min(self.data[:, window_start:window_end])
                                    points_list.append([wavenumber, intensity])
                                
                                params["baseline_points"] = points_list
                                
                            elif flat_region_presets == "third_points":
                                # Use third points of the wavenumber range
                                third_indices = [0, len(self.wavenumbers)//3, 2*len(self.wavenumbers)//3, len(self.wavenumbers)-1]
                                points_list = []
                                for idx in third_indices:
                                    wavenumber = self.wavenumbers[idx]
                                    window_start = max(0, idx - 5)
                                    window_end = min(len(self.data[0]), idx + 6)
                                    intensity = np.min(self.data[:, window_start:window_end])
                                    points_list.append([wavenumber, intensity])
                                
                                params["baseline_points"] = points_list
                                
                            elif flat_region_presets == "start_end":
                                # Just start and end points
                                start_intensity = np.min(self.data[:, :10])
                                end_intensity = np.min(self.data[:, -10:])
                                params["baseline_points"] = [[self.wavenumbers[0], start_intensity], [self.wavenumbers[-1], end_intensity]]
                                
                            else:
                                # Default: use minimum of data
                                params["baseline_points"] = [[self.wavenumbers[0], np.min(self.data)], [self.wavenumbers[-1], np.min(self.data)]]
                            
                            print(f"Using preset flat region: {flat_region_presets}")
                    
                    # Filter out None values and other problematic values
                    filtered_params = {}
                    for k, v in params.items():
                        if v is not None and v != "" and not isinstance(v, dict):
                            filtered_params[k] = v
                    
                    self.process = Process(
                        target=baseline_correction_runner,
                        args=(self.data, self.wavenumbers, filtered_params, os.getpid(), self.redis_host, self.redis_port),
                    )
                    self.process.start()
                    self.current_pid = self.process.pid
                    return (
                        f"Running baseline correction with method: {baseline_method}...",
                        False,  # Enable process check immediately
                        "Running...",
                        "warning",
                        True,
                    )
                except Exception as e:
                    return (f"Error: {str(e)}", True, "Run Baseline Correction", "danger", False)

            elif trigger_id == "process-check" and self.process:
                if not self.process.is_alive():
                    # Check Redis for results
                    status_key = f"temp:baseline:{os.getpid()}:status"
                    corrected_key = f"temp:baseline:{os.getpid()}:corrected"
                    baseline_key = f"temp:baseline:{os.getpid()}:baseline"

                    status = self.redis_client.get(status_key)

                    if status:
                        status = status.decode()
                        if status == "done":
                            # Load the corrected and baseline data into memory
                            try:
                                corrected_data = self.redis_client.get(f"temp:baseline:{os.getpid()}:corrected")
                                baseline_data = self.redis_client.get(f"temp:baseline:{os.getpid()}:baseline")
                                
                                if corrected_data and baseline_data:
                                    corrected_array = np.frombuffer(corrected_data)
                                    baseline_array = np.frombuffer(baseline_data)
                                    
                                    # Calculate the correct shape based on the array size
                                    if len(corrected_array) == self.data.shape[0] * self.data.shape[1]:
                                        corrected_array = corrected_array.reshape(self.data.shape)
                                        baseline_array = baseline_array.reshape(self.data.shape)
                                    else:
                                        corrected_array = corrected_array.reshape(self.data.shape[0], -1)
                                        baseline_array = baseline_array.reshape(self.data.shape[0], -1)
                                    
                                    # Store the data in memory for display
                                    self.corrected_data = corrected_array
                                    self.baseline_data = baseline_array
                                    
                            except Exception as e:
                                pass
                            
                            self._cleanup_redis_keys(os.getpid())
                            self.process = None
                            return (
                                "Baseline correction completed successfully!",
                                True,
                                "Run Baseline Correction",
                                "success",
                                False,
                            )
                        elif status.startswith("Error") or status.startswith("Baseline correction failed"):
                            self._cleanup_redis_keys(os.getpid())
                            self.process = None
                            return (
                                f"Error: {status}",
                                True,
                                "Retry Baseline Correction",
                                "danger",
                                False,
                            )
                        else:
                            pass

                    self.process = None
                    return ("Error in computation", True, "Run Baseline Correction", "danger", False)
                return (
                    f"Still processing... (PID: {self.current_pid})",
                    False,
                    "Running...",
                    "warning",
                    True,
                )

            return (
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
            )

        # Redis status and current project callback (matches configure_umap)
        @self.app.callback(
            [Output("redis-status", "children"), Output("redis-status", "className"), Output("current-project", "children")],
            [Input("status-check", "n_intervals"), Input("redis-refresh-button", "n_clicks")],
        )
        def update_redis_status_and_project(n_intervals, n_clicks):
            try:
                self.redis_client.ping()
                redis_status = html.Span("Connected", className="text-success")
                redis_class = "text-success"
            except Exception as e:
                redis_status = html.Span(f"Failed: {str(e)}", className="text-danger")
                redis_class = "text-danger"
            
            # Get current project
            project = self.redis_client.get("current_project")
            if project:
                project = project.decode() if isinstance(project, bytes) else str(project)
            else:
                project = "(none)"
            
            return redis_status, redis_class, project

        @self.app.callback(
            [
                Output("load-data-status", "children"),
                Output("load-data-status", "className"),
            ],
            [Input("load-data-button", "n_clicks")],
            prevent_initial_call=True,
        )
        def handle_load_data(n_clicks):
            try:
                self.load_data()
                return "Data loaded successfully", "alert alert-success"
            except Exception as e:
                return f"Error loading data: {str(e)}", "alert alert-danger"

        @self.app.callback(
            [
                Output("imodpoly-params", "style"),
                Output("quantile-params", "style"),
                Output("rubberband-params", "style"),
                Output("pspline-params", "style"),
                Output("interp-params", "style"),
            ],
            [Input("baseline-method", "value")],
        )
        def toggle_parameter_sections(baseline_method):
            imodpoly_style = {"display": "block"} if baseline_method == "imodpoly" else {"display": "none"}
            quantile_style = {"display": "block"} if baseline_method == "quantile" else {"display": "none"}
            rubberband_style = {"display": "block"} if baseline_method == "rubberband" else {"display": "none"}
            pspline_style = {"display": "block"} if baseline_method == "pspline_arpls" else {"display": "none"}
            interp_style = {"display": "block"} if baseline_method == "interp_pts" else {"display": "none"}
            
            return imodpoly_style, quantile_style, rubberband_style, pspline_style, interp_style



        @self.app.callback(
            Output("spatial-plot", "figure"),
            [Input("load-data-status", "children")],
            prevent_initial_call=True,
        )
        def update_spatial_plot(status):
            if self.data is None or not hasattr(self, "spatial_coords"):
                return {
                    "data": [],
                    "layout": {
                        "title": "Load data to see spatial distribution",
                        "xaxis": {"title": "X Position"},
                        "yaxis": {"title": "Y Position"},
                    },
                }

            # Create spatial scatter plot
            spatial_fig = {
                "data": [
                    {
                        "type": "scatter",
                        "x": self.spatial_coords["X"],
                        "y": self.spatial_coords["Y"],
                        "mode": "markers",
                        "marker": {
                            "size": 5,
                            "color": "rgb(220,20,60)",
                            "opacity": 0.6,
                        },
                        "name": "Measurement Points",
                    }
                ],
                "layout": {
                    "title": "Spatial Distribution of Measurements",
                    "xaxis": {
                        "title": "X Position",
                        "scaleanchor": "y",
                        "scaleratio": 1,
                    },
                    "yaxis": {"title": "Y Position"},
                    "dragmode": "lasso",
                    "hovermode": "closest",
                },
            }

            return spatial_fig

        @self.app.callback(
            Output("spectrum-plot", "figure"),
            [Input("spatial-plot", "clickData")],
            prevent_initial_call=True,
        )
        def update_spectrum_plot(click_data):
            if self.data is None or click_data is None:
                return {
                    "data": [],
                    "layout": {
                        "title": "Click on a point to see its spectrum",
                        "xaxis": {"title": "Wavenumber (cm⁻¹)"},
                        "yaxis": {"title": "Intensity"},
                        "showlegend": True,
                    },
                }

            # Get the clicked point index
            point_index = click_data["points"][0]["pointIndex"]
            
            if point_index >= len(self.data):
                return {
                    "data": [],
                    "layout": {
                        "title": "Invalid point index",
                        "xaxis": {"title": "Wavenumber (cm⁻¹)"},
                        "yaxis": {"title": "Intensity"},
                        "showlegend": True,
                    },
                }

            # Get the original spectrum
            original_spectrum = self.data[point_index]
            
            # Get baseline data if available
            baseline_spectrum = None
            
            if hasattr(self, 'baseline_data') and self.baseline_data is not None:
                try:
                    baseline_spectrum = self.baseline_data[point_index]
                    print(f"Point {point_index}: Original shape {original_spectrum.shape}, Baseline shape {baseline_spectrum.shape}")
                except Exception as e:
                    print(f"Error getting baseline data for point {point_index}: {str(e)}")

            # Create traces
            traces = []
            
            # Original spectrum - ALWAYS show this
            traces.append({
                "type": "scatter",
                "x": self.wavenumbers,
                "y": original_spectrum,
                "mode": "lines",
                "line": {"color": "rgb(220,20,60)", "width": 2},
                "name": "Original",
                "showlegend": True,
            })
            
            # If baseline correction has been run, show baseline and corrected
            if baseline_spectrum is not None:
                # Baseline spectrum
                traces.append({
                    "type": "scatter",
                    "x": self.wavenumbers[:len(baseline_spectrum)],
                    "y": baseline_spectrum,
                    "mode": "lines",
                    "line": {"color": "rgb(255,165,0)", "width": 2, "dash": "dash"},
                    "name": "Baseline",
                    "showlegend": True,
                })
                
                # Corrected spectrum (original - baseline)
                corrected_spectrum = original_spectrum[:len(baseline_spectrum)] - baseline_spectrum
                traces.append({
                    "type": "scatter",
                    "x": self.wavenumbers[:len(corrected_spectrum)],
                    "y": corrected_spectrum,
                    "mode": "lines",
                    "line": {"color": "rgb(30,144,255)", "width": 2},
                    "name": "Corrected (Original - Baseline)",
                    "showlegend": True,
                })
            else:
                # If no baseline correction has been run, show message
                traces.append({
                    "type": "scatter",
                    "x": [self.wavenumbers[0], self.wavenumbers[-1]],
                    "y": [0, 0],
                    "mode": "lines",
                    "line": {"color": "rgba(0,0,0,0)"},
                    "name": "Run baseline correction to see baseline and corrected spectra",
                    "showlegend": True,
                })

            # Create the plot
            fig = {
                "data": traces,
                "layout": {
                    "title": f"Spectrum at Point {point_index} (X: {self.spatial_coords.iloc[point_index]['X']:.2f}, Y: {self.spatial_coords.iloc[point_index]['Y']:.2f})",
                    "xaxis": {
                        "title": "Wavenumber (cm⁻¹)",
                        "autorange": "reversed",  # Reverse x-axis for wavenumbers
                    },
                    "yaxis": {"title": "Intensity"},
                    "showlegend": True,
                    "hovermode": "closest",
                },
            }

            return fig

        @self.app.callback(
            Output("export-status", "children"),
            [Input("export-button", "n_clicks")],
            prevent_initial_call=True,
        )
        def export_corrected_data(n_clicks):
            try:
                # Check if we have corrected and baseline data in memory
                if hasattr(self, 'corrected_data') and hasattr(self, 'baseline_data'):
                    corrected_array = self.corrected_data
                    baseline_array = self.baseline_data
                    
                    # Get database path from Redis
                    db_path = self.redis_client.get("current_project")
                    if not db_path:
                        return "No current project found in Redis"
                    
                    db_path = db_path.decode() if isinstance(db_path, bytes) else str(db_path)
                    
                    # Connect to database and store both tables
                    conn = duckdb.connect(db_path)
                    
                    # Create DataFrame with corrected data
                    corrected_df = pd.DataFrame(corrected_array, columns=self.wavenumbers)
                    corrected_df.index.name = "spectrum_id"
                    
                    # Create DataFrame with baseline data
                    baseline_df = pd.DataFrame(baseline_array, columns=self.wavenumbers)
                    baseline_df.index.name = "spectrum_id"
                    
                    # Store both tables
                    store_df_in_db(conn, corrected_df, "baseline_corrected_data")
                    store_df_in_db(conn, baseline_df, "baseline_data")
                    
                    conn.close()
                    
                    return f"Exported to tables: baseline_corrected_data, baseline_data"
                else:
                    return "No corrected/baseline data available for export. Please run baseline correction first."
                    
            except Exception as e:
                return f"Error exporting data: {str(e)}"

    def run(self, debug=False, port=8057):
        """Run the Dash app"""
        # Suppress Werkzeug logging
        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        
        self.app.run_server(debug=debug, port=port, host="0.0.0.0")

    def _cleanup_redis_keys(self, pid):
        """Clean up Redis keys for a given process ID"""
        try:
            keys_to_delete = [
                f"temp:baseline:{pid}:corrected",
                f"temp:baseline:{pid}:baseline",
                f"temp:baseline:{pid}:status",
            ]
            for key in keys_to_delete:
                self.redis_client.delete(key)
        except Exception as e:
            print(f"Error cleaning up Redis keys: {str(e)}")

    def _connect_redis(self):
        """Test Redis connection"""
        try:
            r = redis.Redis(host=self.redis_host, port=self.redis_port)
            r.ping()
            return True
        except Exception as e:
            print(f"Redis connection failed: {str(e)}")
            return False

    def load_data(self):
        """Load spectral data from database"""
        try:
            # Get database path from Redis
            db_path = self.redis_client.get("current_project")
            if not db_path:
                raise Exception("No current project found in Redis. Please set a current project first.")
            
            db_path = db_path.decode() if isinstance(db_path, bytes) else str(db_path)
            print(f"\n=== Loading data from {db_path} ===")
            
            # Connect to database
            conn = duckdb.connect(db_path)
            
            # Check for required tables
            required_tables = ["measured_data", "wavenumbers", "HCD"]
            for table in required_tables:
                if not check_table_exists(conn, table):
                    conn.close()
                    raise Exception(f"Required table '{table}' not found in database")
            
            # Load spectral data
            measured_data_df = read_df_from_db(conn, "measured_data")
            wavenumbers_df = read_df_from_db(conn, "wavenumbers")
            hcd_df = read_df_from_db(conn, "HCD")
            
            if measured_data_df is None or measured_data_df.empty:
                raise Exception("No data found in measured_data table")
            
            if wavenumbers_df is None or wavenumbers_df.empty:
                raise Exception("No data found in wavenumbers table")
            
            if hcd_df is None or hcd_df.empty:
                raise Exception("No data found in HCD table")
            
            # Get wavenumbers
            self.wavenumbers = wavenumbers_df["wavenumber"].values
            
            # Extract spectral data (excluding hcd_indx column)
            self.data = measured_data_df.iloc[:, 1:].values  # Skip hcd_indx column
            
            # Get HCD indices from measured data
            measured_indices = measured_data_df["hcd_indx"].values
            
            # Get corresponding X, Y coordinates from HCD for our indices
            self.spatial_coords = hcd_df.loc[measured_indices, ["X", "Y"]]
            
            print(f"Loaded data: {self.data.shape}")
            print(f"Wavenumbers: {len(self.wavenumbers)} points")
            print(f"Spatial coordinates: {self.spatial_coords.shape}")
            print(f"X range: {self.spatial_coords['X'].min():.2f} to {self.spatial_coords['X'].max():.2f}")
            print(f"Y range: {self.spatial_coords['Y'].min():.2f} to {self.spatial_coords['Y'].max():.2f}")
            
            conn.close()
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise


def main():
    parser = argparse.ArgumentParser(description="Baseline Correction Tool")
    parser.add_argument("--redis-host", default="localhost", help="Redis host")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis port")
    parser.add_argument("--port", type=int, default=8057, help="Port for the web interface")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    
    args = parser.parse_args()
    
    app = BaselineCorrectionApp(redis_host=args.redis_host, redis_port=args.redis_port)
    app.run(debug=args.debug, port=args.port)


if __name__ == "__main__":
    main() 