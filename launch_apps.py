import argparse
import os
import subprocess
import threading
import time
import webbrowser


def run_app(app_name, port):
    """Run a Dash app in a separate process"""
    # Get the absolute path to the specviz package
    current_dir = os.path.dirname(os.path.abspath(__file__))
    specviz_dir = os.path.join(current_dir, "specviz")
    cmd = ["python", "-m", f"specviz.tasks.{app_name}", "--port", str(port)]
    subprocess.Popen(cmd, cwd=current_dir)


def open_browser_tabs(ports):
    """Open browser tabs for each port after a short delay"""
    time.sleep(2)  # Give apps time to start
    for port in ports:
        webbrowser.open_new_tab(f"http://localhost:{port}")


def main():
    # Available apps and their default ports
    available_apps = {
        "setup_session": 8071,
        "measurement_planner": 8072,
        "acquire_data": 8073,
        "baseline_correction": 8074,
        "configure_umap": 8075,
        "explore_embedding": 8076,
        "nearest_neighbor_classifier": 8077,
        "interpolate_data": 8078,
    }

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Launch SpecViz applications")
    parser.add_argument(
        "apps",
        nargs="*",
        choices=list(available_apps.keys()) + ["all"],
        default=["all"],
        help="Apps to launch (default: all). Use 'all' to launch all apps.",
    )
    parser.add_argument(
        "--setup-port",
        type=int,
        default=available_apps["setup_session"],
        help="Port for Setup Session app",
    )
    parser.add_argument(
        "--planner-port",
        type=int,
        default=available_apps["measurement_planner"],
        help="Port for Measurement Planner app",
    )
    parser.add_argument(
        "--acquire-port",
        type=int,
        default=available_apps["acquire_data"],
        help="Port for Data Acquisition app",
    )
    parser.add_argument(
        "--baseline-port",
        type=int,
        default=available_apps["baseline_correction"],
        help="Port for Baseline Correction app",
    )
    parser.add_argument(
        "--umap-port",
        type=int,
        default=available_apps["configure_umap"],
        help="Port for UMAP Configuration app",
    )
    parser.add_argument(
        "--explorer-port",
        type=int,
        default=available_apps["explore_embedding"],
        help="Port for Embedding Explorer app",
    )
    parser.add_argument(
        "--classifier-port",
        type=int,
        default=available_apps["nearest_neighbor_classifier"],
        help="Port for Nearest Neighbor Classifier app",
    )
    parser.add_argument(
        "--interpolator-port",
        type=int,
        default=available_apps["interpolate_data"],
        help="Port for Interpolate Data app",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open browser tabs automatically",
    )

    args = parser.parse_args()

    # Determine which apps to launch
    if "all" in args.apps:
        apps_to_launch = list(available_apps.keys())
    else:
        apps_to_launch = args.apps

    # Create ports dictionary for selected apps
    ports = {}
    for app_name in apps_to_launch:
        if app_name == "setup_session":
            ports[app_name] = args.setup_port
        elif app_name == "measurement_planner":
            ports[app_name] = args.planner_port
        elif app_name == "acquire_data":
            ports[app_name] = args.acquire_port
        elif app_name == "baseline_correction":
            ports[app_name] = args.baseline_port
        elif app_name == "configure_umap":
            ports[app_name] = args.umap_port
        elif app_name == "explore_embedding":
            ports[app_name] = args.explorer_port
        elif app_name == "nearest_neighbor_classifier":
            ports[app_name] = args.classifier_port
        elif app_name == "interpolate_data":
            ports[app_name] = args.interpolator_port

    # Launch selected apps in separate processes
    for app_name, port in ports.items():
        print(f"Starting {app_name} on port {port}...")
        run_app(app_name, port)

    # Open browser tabs if not disabled
    if not args.no_browser:
        print("Opening browser tabs...")
        browser_thread = threading.Thread(
            target=open_browser_tabs, args=(list(ports.values()),)
        )
        browser_thread.start()

    print("\nAll applications started. Press Ctrl+C to stop all applications.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down applications...")


if __name__ == "__main__":
    main()
