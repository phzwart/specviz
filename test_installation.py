#!/usr/bin/env python3
"""
Test script to verify that all required dependencies can be imported.
Run this after installation to ensure everything is working correctly.
"""

import sys

def test_imports():
    """Test importing all required dependencies."""
    required_packages = [
        # Core dependencies
        ('typer', 'typer'),
        ('rich', 'rich'),
        
        # Scientific computing
        ('numpy', 'numpy'),
        ('scipy', 'scipy'),
        ('pandas', 'pandas'),
        
        # Machine learning
        ('sklearn', 'scikit-learn'),
        ('torch', 'torch'),
        ('pytorch_lightning', 'pytorch-lightning'),
        ('umap', 'umap-learn'),
        
        # Visualization
        ('matplotlib', 'matplotlib'),
        ('plotly', 'plotly'),
        ('dash', 'dash'),
        ('dash_bootstrap_components', 'dash-bootstrap-components'),
        
        # Database
        ('redis', 'redis'),
        ('duckdb', 'duckdb'),
        
        # Parallel computing
        ('dask', 'dask'),
        ('psutil', 'psutil'),
        
        # Utilities
        ('einops', 'einops'),
    ]
    
    failed_imports = []
    
    print("Testing imports...")
    print("=" * 50)
    
    for module_name, package_name in required_packages:
        try:
            __import__(module_name)
            print(f"✓ {package_name}")
        except ImportError as e:
            print(f"✗ {package_name}: {e}")
            failed_imports.append(package_name)
    
    print("=" * 50)
    
    if failed_imports:
        print(f"\n❌ Failed to import {len(failed_imports)} packages:")
        for package in failed_imports:
            print(f"  - {package}")
        print(f"\nPlease install missing packages:")
        print(f"pip install {' '.join(failed_imports)}")
        return False
    else:
        print(f"\n✅ All {len(required_packages)} packages imported successfully!")
        return True

def test_specviz_import():
    """Test importing the specviz package itself."""
    try:
        import specviz
        print(f"✅ SpecViz package imported successfully (version: {getattr(specviz, '__version__', 'unknown')})")
        return True
    except ImportError as e:
        print(f"❌ Failed to import specviz package: {e}")
        print("Make sure you've installed the package in development mode:")
        print("pip install -e .")
        return False

if __name__ == "__main__":
    print("SpecViz Installation Test")
    print("=" * 50)
    
    # Test Python version
    print(f"Python version: {sys.version}")
    if sys.version_info < (3, 9):
        print("❌ Python 3.9 or higher is required")
        sys.exit(1)
    else:
        print("✅ Python version is compatible")
    
    print()
    
    # Test all imports
    imports_ok = test_imports()
    print()
    
    # Test specviz package
    specviz_ok = test_specviz_import()
    
    if imports_ok and specviz_ok:
        print("\n🎉 All tests passed! SpecViz is ready to use.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please fix the issues above.")
        sys.exit(1) 