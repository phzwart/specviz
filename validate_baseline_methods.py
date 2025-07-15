#!/usr/bin/env python3
"""
Script to validate baseline correction method parameters against pybaselines API.
This script extracts all available methods and their parameters from pybaselines
and compares them with our GUI configuration to ensure accuracy.
"""

import inspect
import numpy as np
from pybaselines import Baseline

def extract_method_info():
    """Extract all method names and their parameters from pybaselines Baseline class."""
    
    # Create a baseline instance with dummy data
    x = np.linspace(0, 100, 10)
    baseline_fitter = Baseline(x)
    
    # Get all methods from the Baseline class
    methods = {}
    
    # List of known baseline correction methods
    known_methods = [
        'poly', 'modpoly', 'imodpoly', 'penalized_poly', 'loess', 'quant_reg',
        'rolling_ball', 'mor', 'mpls', 'arpls', 'asls', 'iasls', 'psalsa',
        'drpls', 'iarpls', 'aspls', 'pspline_asls', 'pspline_iasls',
        'pspline_arpls', 'pspline_iarpls', 'pspline_psalsa', 'pspline_drpls',
        'pspline_aspls'
    ]
    
    print("=== EXTRACTING METHOD INFORMATION FROM PYBASELINES ===\n")
    
    for method_name in known_methods:
        try:
            # Get the method
            method = getattr(baseline_fitter, method_name, None)
            if method is not None:
                # Get the signature
                sig = inspect.signature(method)
                
                # Extract parameters (skip 'data' which is always first)
                params = []
                for param_name, param in sig.parameters.items():
                    if param_name != 'data':
                        params.append(param_name)
                
                methods[method_name] = params
                print(f"{method_name}: {params}")
            else:
                print(f"{method_name}: NOT FOUND")
        except Exception as e:
            print(f"{method_name}: ERROR - {str(e)}")
    
    return methods

def get_gui_config():
    """Get the current GUI configuration from baseline_correction.py."""
    
    # This is the configuration from our baseline_correction.py file
    gui_config = {
        "polynomial": ["poly_order", "weights"],
        "modpoly": ["poly_order", "tol", "max_iter", "weights", "use_original", "mask_initial_peaks", "return_coef"],
        "imodpoly": ["poly_order", "tol", "max_iter", "weights", "use_original", "mask_initial_peaks", "return_coef", "num_std"],
        "penalized_poly": ["poly_order", "tol", "max_iter", "weights", "cost_function", "threshold", "alpha_factor", "return_coef"],
        "loess": ["fraction", "total_points", "poly_order", "scale", "tol", "max_iter", "symmetric_weights", "use_threshold", "num_std", "use_original", "weights", "return_coef", "conserve_memory", "delta"],
        "quantile": ["quantile", "num_std", "max_iter", "tol", "use_original", "mask_initial_peaks"],
        "rolling_ball": ["half_window", "smooth_half_window", "use_original"],
        "morphological": ["half_window", "smooth_half_window", "use_original"],
        "mpls": ["lambda_param", "num_std", "max_iter", "tol", "use_original", "mask_initial_peaks"],
        "arpls": ["lambda_param", "num_std", "max_iter", "tol", "use_original", "mask_initial_peaks"],
        "asls": ["lambda_param", "num_std", "max_iter", "tol", "use_original", "mask_initial_peaks"],
        "iasls": ["lambda_param", "num_std", "max_iter", "tol", "use_original", "mask_initial_peaks"],
        "psalsa": ["lambda_param", "num_std", "max_iter", "tol", "use_original", "mask_initial_peaks"],
        "drpls": ["lambda_param", "num_std", "max_iter", "tol", "use_original", "mask_initial_peaks"],
        "iarpls": ["lambda_param", "num_std", "max_iter", "tol", "use_original", "mask_initial_peaks"],
        "aspls": ["lambda_param", "num_std", "max_iter", "tol", "use_original", "mask_initial_peaks"],
        "pspline_asls": ["lambda_param", "num_knots", "spline_degree", "num_std", "max_iter", "tol", "use_original", "mask_initial_peaks"],
        "pspline_iasls": ["lambda_param", "num_knots", "spline_degree", "num_std", "max_iter", "tol", "use_original", "mask_initial_peaks"],
        "pspline_arpls": ["lambda_param", "num_knots", "spline_degree", "num_std", "max_iter", "tol", "use_original", "mask_initial_peaks"],
        "pspline_iarpls": ["lambda_param", "num_knots", "spline_degree", "num_std", "max_iter", "tol", "use_original", "mask_initial_peaks"],
        "pspline_psalsa": ["lambda_param", "num_knots", "spline_degree", "num_std", "max_iter", "tol", "use_original", "mask_initial_peaks"],
        "pspline_drpls": ["lambda_param", "num_knots", "spline_degree", "num_std", "max_iter", "tol", "use_original", "mask_initial_peaks"],
        "pspline_aspls": ["lambda_param", "num_knots", "spline_degree", "num_std", "max_iter", "tol", "use_original", "mask_initial_peaks"],
    }
    
    return gui_config

def validate_configurations():
    """Compare the actual pybaselines methods with our GUI configuration."""
    
    print("=== VALIDATING METHOD CONFIGURATIONS ===\n")
    
    # Extract actual method info
    actual_methods = extract_method_info()
    
    # Get GUI configuration
    gui_config = get_gui_config()
    
    # Mapping from GUI names to actual method names
    name_mapping = {
        "polynomial": "poly",
        "modpoly": "modpoly", 
        "imodpoly": "imodpoly",
        "penalized_poly": "penalized_poly",
        "loess": "loess",
        "quantile": "quant_reg",
        "rolling_ball": "rolling_ball",
        "morphological": "mor",
        "mpls": "mpls",
        "arpls": "arpls",
        "asls": "asls",
        "iasls": "iasls",
        "psalsa": "psalsa",
        "drpls": "drpls",
        "iarpls": "iarpls",
        "aspls": "aspls",
        "pspline_asls": "pspline_asls",
        "pspline_iasls": "pspline_iasls",
        "pspline_arpls": "pspline_arpls",
        "pspline_iarpls": "pspline_iarpls",
        "pspline_psalsa": "pspline_psalsa",
        "pspline_drpls": "pspline_drpls",
        "pspline_aspls": "pspline_aspls",
    }
    
    print("=== VALIDATION RESULTS ===\n")
    
    issues_found = []
    
    for gui_name, actual_name in name_mapping.items():
        print(f"Checking {gui_name} -> {actual_name}")
        
        if actual_name not in actual_methods:
            print(f"  ❌ ERROR: Method '{actual_name}' not found in pybaselines")
            issues_found.append(f"Method '{actual_name}' not found")
            continue
        
        gui_params = set(gui_config.get(gui_name, []))
        actual_params = set(actual_methods[actual_name])
        
        # Check for missing parameters in GUI
        missing_in_gui = actual_params - gui_params
        if missing_in_gui:
            print(f"  ⚠️  WARNING: GUI missing parameters: {missing_in_gui}")
            issues_found.append(f"GUI missing parameters for {gui_name}: {missing_in_gui}")
        
        # Check for extra parameters in GUI
        extra_in_gui = gui_params - actual_params
        if extra_in_gui:
            print(f"  ⚠️  WARNING: GUI has extra parameters: {extra_in_gui}")
            issues_found.append(f"GUI has extra parameters for {gui_name}: {extra_in_gui}")
        
        if not missing_in_gui and not extra_in_gui:
            print(f"  ✅ OK: Parameters match")
    
    print(f"\n=== SUMMARY ===")
    print(f"Total issues found: {len(issues_found)}")
    
    if issues_found:
        print("\nIssues to fix:")
        for issue in issues_found:
            print(f"  - {issue}")
    else:
        print("✅ All configurations are correct!")
    
    return issues_found

def generate_corrected_config():
    """Generate the corrected configuration based on actual pybaselines methods."""
    
    print("\n=== GENERATING CORRECTED CONFIGURATION ===\n")
    
    actual_methods = extract_method_info()
    
    # Generate the corrected method_configs dictionary
    print("Corrected method_configs = {")
    
    name_mapping = {
        "polynomial": "poly",
        "modpoly": "modpoly", 
        "imodpoly": "imodpoly",
        "penalized_poly": "penalized_poly",
        "loess": "loess",
        "quantile": "quant_reg",
        "rolling_ball": "rolling_ball",
        "morphological": "mor",
        "mpls": "mpls",
        "arpls": "arpls",
        "asls": "asls",
        "iasls": "iasls",
        "psalsa": "psalsa",
        "drpls": "drpls",
        "iarpls": "iarpls",
        "aspls": "aspls",
        "pspline_asls": "pspline_asls",
        "pspline_iasls": "pspline_iasls",
        "pspline_arpls": "pspline_arpls",
        "pspline_iarpls": "pspline_iarpls",
        "pspline_psalsa": "pspline_psalsa",
        "pspline_drpls": "pspline_drpls",
        "pspline_aspls": "pspline_aspls",
    }
    
    for gui_name, actual_name in name_mapping.items():
        if actual_name in actual_methods:
            params = actual_methods[actual_name]
            print(f'    "{gui_name}": {{')
            print(f'        "func": baseline_fitter.{actual_name},')
            print(f'        "params": {params}')
            print(f'    }},')
    
    print("}")

if __name__ == "__main__":
    print("Baseline Correction Method Validator")
    print("=" * 50)
    
    # Run validation
    issues = validate_configurations()
    
    # Generate corrected configuration
    generate_corrected_config()
    
    print(f"\nValidation complete. Found {len(issues)} issues to address.") 