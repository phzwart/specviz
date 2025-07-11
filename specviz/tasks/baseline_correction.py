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
            "polynomial": {
                "func": baseline_fitter.poly,
                "params": ["poly_order"]
            },
            "modpoly": {
                "func": baseline_fitter.modpoly,
                "params": ["poly_order", "tol", "max_iter", "use_original", "mask_initial_peaks"]
            },
            "imodpoly": {
                "func": baseline_fitter.imodpoly,
                "params": ["poly_order", "tol", "max_iter", "use_original", "mask_initial_peaks", "num_std"]
            },
            "penalized_poly": {
                "func": baseline_fitter.penalized_poly,
                "params": ["poly_order", "tol", "max_iter", "cost_function", "threshold", "alpha_factor"]
            },
            "loess": {
                "func": baseline_fitter.loess,
                "params": ["fraction", "total_points", "poly_order", "scale", "tol", "max_iter", "symmetric_weights", "use_threshold", "num_std", "use_original", "conserve_memory", "delta"]
            },
            "quantile": {
                "func": baseline_fitter.quant_reg,
                "params": ["poly_order", "quantile", "tol", "max_iter", "eps"]
            },
            "rolling_ball": {
                "func": baseline_fitter.rolling_ball,
                "params": ["half_window", "smooth_half_window"]  # Avoid pad_kwargs, window_kwargs, kwargs
            },
            "morphological": {
                "func": baseline_fitter.mor,
                "params": ["half_window"]  # Only pass half_window, avoid window_kwargs and kwargs
            },
            "rubberband": {
                "func": baseline_fitter.rubberband,
                "params": ["segments", "lam", "diff_order", "smooth_half_window"]
            },
            "mpls": {
                "func": baseline_fitter.mpls,
                "params": ["half_window", "lam", "p", "diff_order", "tol", "max_iter"]  # Avoid window_kwargs, kwargs
            },
            "arpls": {
                "func": baseline_fitter.arpls,
                "params": ["lam", "diff_order", "max_iter", "tol"]
            },
            "asls": {
                "func": baseline_fitter.asls,
                "params": ["lam", "p", "diff_order", "max_iter", "tol"]
            },
            "iasls": {
                "func": baseline_fitter.iasls,
                "params": ["lam", "p", "lam_1", "max_iter", "tol", "diff_order"]
            },
            "psalsa": {
                "func": baseline_fitter.psalsa,
                "params": ["lam", "p", "k", "diff_order", "max_iter", "tol"]
            },
            "drpls": {
                "func": baseline_fitter.drpls,
                "params": ["lam", "eta", "max_iter", "tol", "diff_order"]
            },
            "iarpls": {
                "func": baseline_fitter.iarpls,
                "params": ["lam", "diff_order", "max_iter", "tol"]
            },
            "aspls": {
                "func": baseline_fitter.aspls,
                "params": ["lam", "diff_order", "max_iter", "tol", "alpha", "asymmetric_coef"]
            },
            "pspline_asls": {
                "func": baseline_fitter.pspline_asls,
                "params": ["lam", "p", "num_knots", "spline_degree", "diff_order", "max_iter", "tol"]
            },
            "pspline_iasls": {
                "func": baseline_fitter.pspline_iasls,
                "params": ["lam", "p", "lam_1", "num_knots", "spline_degree", "max_iter", "tol", "diff_order"]
            },
            "pspline_arpls": {
                "func": baseline_fitter.pspline_arpls,
                "params": ["lam", "num_knots", "spline_degree", "diff_order", "max_iter", "tol"]
            },
            "pspline_iarpls": {
                "func": baseline_fitter.pspline_iarpls,
                "params": ["lam", "num_knots", "spline_degree", "diff_order", "max_iter", "tol"]
            },
            "pspline_psalsa": {
                "func": baseline_fitter.pspline_psalsa,
                "params": ["lam", "p", "k", "num_knots", "spline_degree", "diff_order", "max_iter", "tol"]
            },
            "pspline_drpls": {
                "func": baseline_fitter.pspline_drpls,
                "params": ["lam", "eta", "num_knots", "spline_degree", "diff_order", "max_iter", "tol"]
            },
            "pspline_aspls": {
                "func": baseline_fitter.pspline_aspls,
                "params": ["lam", "num_knots", "spline_degree", "diff_order", "max_iter", "tol", "alpha", "asymmetric_coef"]
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
                final_kwargs = {}
                for k, v in method_kwargs.items():
                    if v is not None and not isinstance(v, dict):
                        final_kwargs[k] = v
                
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
            "polynomial": ["poly_order"],
            "modpoly": ["poly_order", "tol", "max_iter", "use_original", "mask_initial_peaks"],
            "imodpoly": ["poly_order", "tol", "max_iter", "use_original", "mask_initial_peaks", "num_std"],
            "penalized_poly": ["poly_order", "tol", "max_iter", "cost_function", "threshold", "alpha_factor"],
            "loess": ["fraction", "total_points", "poly_order", "scale", "tol", "max_iter", "symmetric_weights", "use_threshold", "num_std", "use_original", "conserve_memory", "delta"],
            "quantile": ["poly_order", "quantile", "tol", "max_iter", "eps"],
            "rolling_ball": ["half_window", "smooth_half_window"],
            "morphological": ["half_window"],
            "rubberband": ["segments", "lam", "diff_order", "smooth_half_window"],
            "mpls": ["half_window", "lam", "p", "diff_order", "tol", "max_iter"],
            "arpls": ["lam", "diff_order", "max_iter", "tol"],
            "asls": ["lam", "p", "diff_order", "max_iter", "tol"],
            "iasls": ["lam", "p", "lam_1", "max_iter", "tol", "diff_order"],
            "psalsa": ["lam", "p", "k", "diff_order", "max_iter", "tol"],
            "drpls": ["lam", "eta", "max_iter", "tol", "diff_order"],
            "iarpls": ["lam", "diff_order", "max_iter", "tol"],
            "aspls": ["lam", "diff_order", "max_iter", "tol", "alpha", "asymmetric_coef"],
            "pspline_asls": ["lam", "p", "num_knots", "spline_degree", "diff_order", "max_iter", "tol"],
            "pspline_iasls": ["lam", "p", "lam_1", "num_knots", "spline_degree", "max_iter", "tol", "diff_order"],
            "pspline_arpls": ["lam", "num_knots", "spline_degree", "diff_order", "max_iter", "tol"],
            "pspline_iarpls": ["lam", "num_knots", "spline_degree", "diff_order", "max_iter", "tol"],
            "pspline_psalsa": ["lam", "p", "k", "num_knots", "spline_degree", "diff_order", "max_iter", "tol"],
            "pspline_drpls": ["lam", "eta", "num_knots", "spline_degree", "diff_order", "max_iter", "tol"],
            "pspline_aspls": ["lam", "num_knots", "spline_degree", "diff_order", "max_iter", "tol", "alpha", "asymmetric_coef"],
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
                if k in ["poly_order", "max_iter", "num_knots", "spline_degree", "half_window"]:
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
                                                                        {"label": "Polynomial", "value": "polynomial"},
                                                                        {"label": "Modified Polynomial", "value": "modpoly"},
                                                                        {"label": "Iterative Modified Polynomial", "value": "imodpoly"},
                                                                        {"label": "Penalized Polynomial", "value": "penalized_poly"},
                                                                        {"label": "LOESS", "value": "loess"},
                                                                        {"label": "Quantile", "value": "quantile"},
                                                                        {"label": "Rolling Ball", "value": "rolling_ball"},
                                                                        {"label": "Morphological", "value": "morphological"},
                                                                        {"label": "Rubberband", "value": "rubberband"},
                                                                        {"label": "MPLS", "value": "mpls"},
                                                                        {"label": "ARPLS", "value": "arpls"},
                                                                        {"label": "ASLS", "value": "asls"},
                                                                        {"label": "IASLS", "value": "iasls"},
                                                                        {"label": "PSALSA", "value": "psalsa"},
                                                                        {"label": "DRPLS", "value": "drpls"},
                                                                        {"label": "IARPLS", "value": "iarpls"},
                                                                        {"label": "ASPLS", "value": "aspls"},
                                                                        {"label": "PSpline ASLS", "value": "pspline_asls"},
                                                                        {"label": "PSpline IASLS", "value": "pspline_iasls"},
                                                                        {"label": "PSpline ARPLS", "value": "pspline_arpls"},
                                                                        {"label": "PSpline IARPLS", "value": "pspline_iarpls"},
                                                                        {"label": "PSpline PSALSA", "value": "pspline_psalsa"},
                                                                        {"label": "PSpline DRPLS", "value": "pspline_drpls"},
                                                                        {"label": "PSpline ASPLS", "value": "pspline_aspls"},
                                                                    ],
                                                                    value="polynomial",
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
                                                
                                                # Polynomial parameters
                                                html.Div(
                                                    id="polynomial-params",
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
                                                                    width=6,
                                                                ),
                                                                dbc.Col(
                                                                    [
                                                                        html.Label("Lambda Parameter:"),
                                                                        dcc.Input(
                                                                            id="lambda-param",
                                                                            type="number",
                                                                            value=1000.0,
                                                                            step=100.0,
                                                                            className="form-control",
                                                                        ),
                                                                    ],
                                                                    width=6,
                                                                ),
                                                            ]
                                                        ),
                                                    ]
                                                ),
                                                
                                                # LOESS parameters
                                                html.Div(
                                                    id="loess-params",
                                                    children=[
                                                        dbc.Row(
                                                            [
                                                                dbc.Col(
                                                                    [
                                                                        html.Label("Fraction:"),
                                                                        dcc.Input(
                                                                            id="frac",
                                                                            type="number",
                                                                            value=0.1,
                                                                            min=0.01,
                                                                            max=1.0,
                                                                            step=0.01,
                                                                            className="form-control",
                                                                        ),
                                                                    ],
                                                                    width=6,
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
                                                                    width=6,
                                                                ),
                                                            ]
                                                        ),
                                                    ],
                                                    style={"display": "none"},
                                                ),
                                                
                                                # Rolling ball parameters
                                                html.Div(
                                                    id="rolling-ball-params",
                                                    children=[
                                                        dbc.Row(
                                                            [
                                                                dbc.Col(
                                                                    [
                                                                        html.Label("Half Window:"),
                                                                        dcc.Input(
                                                                            id="half-window",
                                                                            type="number",
                                                                            value=50,
                                                                            min=1,
                                                                            max=1000,
                                                                            className="form-control",
                                                                        ),
                                                                    ],
                                                                    width=6,
                                                                ),
                                                            ]
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
                                                                    width=6,
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
                                                                    width=6,
                                                                ),
                                                            ]
                                                        ),
                                                    ],
                                                    style={"display": "none"},
                                                ),
                                                
                                                # PSpline parameters
                                                html.Div(
                                                    id="pspline-params",
                                                    children=[
                                                        dbc.Row(
                                                            [
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
                                                                    width=6,
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
                                                                    width=6,
                                                                ),
                                                            ]
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
                State("poly-order", "value"),
                State("lambda-param", "value"),
                State("frac", "value"),
                State("num-std", "value"),
                State("half-window", "value"),
                State("segments", "value"),
                State("rubberband-lam", "value"),
                State("num-knots", "value"),
                State("spline-degree", "value"),
            ],
            prevent_initial_call=True,
        )
        def update_status(
            n_clicks,
            n_intervals,
            median_filter_window,
            baseline_method,
            normalization_method,
            poly_order,
            lambda_param,
            frac,
            num_std,
            half_window,
            segments,
            rubberband_lam,
            num_knots,
            spline_degree,
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
                        "poly_order": poly_order,
                        "lambda_param": lambda_param,
                        "frac": frac,
                        "num_std": num_std,
                        "half_window": half_window,
                        "segments": segments,
                        "rubberband_lam": rubberband_lam,
                        "num_knots": num_knots,
                        "spline_degree": spline_degree,
                    }
                    
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
                Output("polynomial-params", "style"),
                Output("loess-params", "style"),
                Output("rolling-ball-params", "style"),
                Output("rubberband-params", "style"),
                Output("pspline-params", "style"),
            ],
            [Input("baseline-method", "value")],
        )
        def toggle_parameter_sections(baseline_method):
            polynomial_style = {"display": "block"} if baseline_method in ["polynomial", "modpoly", "imodpoly", "penalized_poly"] else {"display": "none"}
            loess_style = {"display": "block"} if baseline_method == "loess" else {"display": "none"}
            rolling_ball_style = {"display": "block"} if baseline_method in ["rolling_ball", "morphological"] else {"display": "none"}
            rubberband_style = {"display": "block"} if baseline_method == "rubberband" else {"display": "none"}
            pspline_style = {"display": "block"} if baseline_method.startswith("pspline_") else {"display": "none"}
            
            return polynomial_style, loess_style, rolling_ball_style, rubberband_style, pspline_style



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