#!/usr/bin/env python3
"""
Launcher script for the Baseline Correction Tool
"""

import sys
import os

# Add the specviz directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from specviz.tasks.baseline_correction import main

if __name__ == "__main__":
    main() 