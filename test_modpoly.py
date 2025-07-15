#!/usr/bin/env python3

from pybaselines import Baseline
import numpy as np

# Create a simple test
print("Testing modpoly function...")

# Create baseline fitter
x = np.linspace(0, 100, 100)
b = Baseline(x)

# Create test data
data = np.random.random(100) + np.linspace(0, 1, 100)

print(f"Data shape: {data.shape}")
print(f"Data type: {type(data)}")
print(f"Data sample: {data[:5]}")

try:
    # Test modpoly with minimal parameters
    result = b.modpoly(data, poly_order=3)
    print("Success with minimal parameters!")
    print(f"Result type: {type(result)}")
    if isinstance(result, tuple):
        print(f"Result length: {len(result)}")
        print(f"First element shape: {result[0].shape}")
except Exception as e:
    print(f"Error with minimal parameters: {str(e)}")
    print(f"Error type: {type(e)}")

try:
    # Test modpoly with all parameters
    result = b.modpoly(data, poly_order=3, tol=0.001, max_iter=100)
    print("Success with all parameters!")
    print(f"Result type: {type(result)}")
    if isinstance(result, tuple):
        print(f"Result length: {len(result)}")
        print(f"First element shape: {result[0].shape}")
except Exception as e:
    print(f"Error with all parameters: {str(e)}")
    print(f"Error type: {type(e)}") 