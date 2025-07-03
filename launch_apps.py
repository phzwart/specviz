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
    # Default ports for each app
    ports = {
        "setup_session": 8051,
        "measurement_planner": 8053,
        "acquire_data": 8055,
        "configure_umap": 8056,
        "explore_embedding": 8057,
        "ensemble_classifier": 8058,
    }

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Launch SpecViz applications")
    parser.add_argument(
        "--setup-port",
        type=int,
        default=ports["setup_session"],
        help="Port for Setup Session app",
    )
    parser.add_argument(
        "--planner-port",
        type=int,
        default=ports["measurement_planner"],
        help="Port for Measurement Planner app",
    )
    parser.add_argument(
        "--acquire-port",
        type=int,
        default=ports["acquire_data"],
        help="Port for Data Acquisition app",
    )
    parser.add_argument(
        "--umap-port",
        type=int,
        default=ports["configure_umap"],
        help="Port for UMAP Configuration app",
    )
    parser.add_argument(
        "--explorer-port",
        type=int,
        default=ports["explore_embedding"],
        help="Port for Embedding Explorer app",
    )
    parser.add_argument(
        "--classifier-port",
        type=int,
        default=ports["ensemble_classifier"],
        help="Port for Ensemble Classifier app",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open browser tabs automatically",
    )

    args = parser.parse_args()

    # Update ports from arguments
    ports["setup_session"] = args.setup_port
    ports["measurement_planner"] = args.planner_port
    ports["acquire_data"] = args.acquire_port
    ports["configure_umap"] = args.umap_port
    ports["explore_embedding"] = args.explorer_port
    ports["ensemble_classifier"] = args.classifier_port

    # Launch each app in a separate process
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
