# --- Conceptual Flask App (pwr_flask_app.py) ---
# Requires: pip install Flask Flask-Cors numpy scipy
# Run with: python pwr_flask_app.py

from flask import Flask, jsonify, request
from flask_cors import CORS # To allow requests from the browser frontend
import numpy as np
import threading
import time

# --- Import your existing Simulation code ---
# Assume pwr_simulation_backend.py is in the same directory
# Make sure you are using the version with debug prints if needed
from pwr_simulation_backend import Simulation, scenario_control # Or your custom control logic

app = Flask(__name__)
CORS(app) # Allow requests from any origin (for development)

# --- Global variables to hold simulation state ---
sim_instance = None
sim_thread = None
simulation_running = False
latest_data_point = {} # Store the most recent data
sim_lock = threading.Lock() # To safely access shared simulation data

# --- Simulation Runner Function (to run in a separate thread) ---
def run_simulation_thread(duration, control_logic):
    """
    Runs the simulation loop in a separate thread.
    Continuously steps the simulation and updates the latest_data_point.
    """
    global sim_instance, simulation_running, latest_data_point
    print("Simulation thread started.")
    start_time = sim_instance.time
    end_time = start_time + duration

    while simulation_running and sim_instance.time < end_time:
        # Determine step duration, ensuring it doesn't overshoot end_time
        step_duration = min(0.1, end_time - sim_instance.time) # Simulate in chunks (e.g., 0.1s)
        if step_duration <= 0: # Avoid zero or negative step if near end_time
             break

        # Get control inputs for this step
        # NOTE: A more robust implementation for dynamic control would use a
        # shared queue or state object updated by the /control endpoint.
        # This example uses the provided control_logic function.
        current_state = sim_instance.state # Get current state for control logic
        # Ensure control_logic is callable
        if callable(control_logic):
            current_controls = control_logic(sim_instance.time, current_state)
        else:
            # Fallback if control_logic is not callable (e.g., use defaults)
            print("Warning: control_logic is not callable. Using default controls.")
            current_controls = {
                'valve_position': sim_instance.turbine.valve_position,
                'external_reactivity': sim_instance.reactor.rho_ext,
                'electrical_load_pu': sim_instance.grid.electrical_load_pu
            }


        # Acquire lock to safely interact with sim_instance and latest_data_point
        with sim_lock:
            if not simulation_running: # Check again inside lock in case stop was called
                 break
            # --- Run one simulation step ---
            success = sim_instance.step(step_duration, current_controls)
            if not success:
                print("Simulation thread: Solver failed.")
                simulation_running = False # Stop the loop
                break

            # --- Update latest data point ---
            # Retrieve all results generated so far by the Simulation class
            results = sim_instance.get_results()
            # Ensure results are not empty before accessing the last element
            if all(results.values()): # Check if all lists in results have at least one element
                 latest_data_point = {key: val[-1] for key, val in results.items()}
            else:
                 print("Warning: Simulation results dictionary is empty or incomplete.")
                 latest_data_point = {} # Reset or handle as appropriate

        # Small sleep to yield control and prevent busy-waiting
        # Adjust sleep time based on desired update rate vs CPU usage
        time.sleep(0.05) # e.g., sleep for 50ms

    print("Simulation thread finished.")
    # Ensure the global flag reflects the state accurately after the loop finishes
    simulation_running = False


# --- API Endpoints ---

@app.route('/start', methods=['POST'])
def start_sim():
    """
    API endpoint to initialize and start the simulation.
    Expects JSON payload with optional 'duration', 'initial_power', 'initial_load'.
    Starts the simulation in a background thread.
    """
    global sim_instance, sim_thread, simulation_running, latest_data_point
    if simulation_running:
        return jsonify({"status": "error", "message": "Simulation already running"}), 400

    try:
        # Get parameters from request JSON, providing defaults
        data = request.get_json() or {}
        sim_duration = data.get('duration', 60) # Default 60s simulation
        initial_power = data.get('initial_power', 0.9) # Default 90% power
        initial_load = data.get('initial_load', 0.9) # Default 90% load

        print(f"Received start request: duration={sim_duration}, power={initial_power}, load={initial_load}")

        # --- Initialize simulation ---
        sim_instance = Simulation(initial_power_fraction=initial_power, initial_load_pu=initial_load)
        # Store the initial state immediately after initialization
        initial_results = sim_instance.get_results()
        if all(initial_results.values()):
             latest_data_point = {key: val[0] for key, val in initial_results.items()} # Store initial state
        else:
             latest_data_point = {}
             print("Warning: Could not retrieve initial state.")


        simulation_running = True
        # --- Start simulation in a background thread ---
        # Pass the desired control logic function (e.g., scenario_control)
        sim_thread = threading.Thread(target=run_simulation_thread, args=(sim_duration, scenario_control))
        sim_thread.daemon = True # Allows app to exit even if thread is running (optional)
        sim_thread.start()

        print("Simulation thread initiated.")
        return jsonify({"status": "success", "message": "Simulation started"})
    except Exception as e:
        print(f"Error starting simulation: {e}")
        # Provide more detailed error logging in production if necessary
        return jsonify({"status": "error", "message": f"Internal server error: {e}"}), 500


@app.route('/stop', methods=['POST'])
def stop_sim():
    """
    API endpoint to stop the currently running simulation.
    Sets the global flag and waits briefly for the thread to join.
    """
    global simulation_running, sim_thread
    if not simulation_running:
        return jsonify({"status": "error", "message": "Simulation not running"}), 400

    print("Received stop request...")
    simulation_running = False # Signal the simulation thread to stop

    if sim_thread and sim_thread.is_alive():
        print("Waiting for simulation thread to join...")
        sim_thread.join(timeout=2.0) # Wait up to 2 seconds for the thread to finish cleanly
        if sim_thread.is_alive():
             print("Warning: Simulation thread did not stop within timeout.")
             # Consider more forceful termination if necessary, though generally avoided

    # Reset thread variable
    sim_thread = None
    # Optionally reset sim_instance = None if you want to clear state completely

    print("Simulation stopped.")
    return jsonify({"status": "success", "message": "Simulation stopped"})


@app.route('/data', methods=['GET'])
def get_data():
    """
    API endpoint for the frontend to poll for the latest simulation data point.
    Includes the current simulation status ('running', 'stopped', 'idle').
    """
    global latest_data_point, simulation_running, sim_instance

    # Determine current status
    if simulation_running:
        status = 'running'
    elif sim_instance is not None: # If instance exists but not running, it's stopped/finished
        status = 'stopped'
    else: # No instance exists
        status = 'idle'

    # Safely access the latest data point using the lock
    with sim_lock:
        data_to_send = latest_data_point.copy()

    # Add simulation status to the response
    data_to_send['status'] = status

    if not data_to_send and status == 'idle':
         # Handle case where simulation hasn't started yet
         return jsonify({"status": "idle", "message": "Simulation not started."})
    elif not data_to_send and status != 'idle':
         # Handle case where data might be temporarily unavailable but sim is running/stopped
         print("Warning: /data endpoint called but latest_data_point is empty.")
         return jsonify({"status": status, "message": "Data temporarily unavailable."})


    return jsonify(data_to_send)


@app.route('/control', methods=['POST'])
def set_control():
    """
    API endpoint to receive control commands from the frontend.
    Example: Update electrical load demand.
    NOTE: Needs refinement for dynamically influencing the control_logic
          used by the running simulation thread.
    """
    global sim_instance
    if not simulation_running or not sim_instance:
         return jsonify({"status": "error", "message": "Simulation not running"}), 400

    try:
        control_data = request.get_json()
        if not control_data:
             return jsonify({"status": "error", "message": "No control data received"}), 400

        print(f"Received control data: {control_data}")

        # Acquire lock to safely update simulation control parameters
        # This example directly modifies the sim_instance attributes.
        # The effectiveness depends on whether the 'control_logic' function
        # passed to the thread reads these attributes dynamically.
        with sim_lock:
            if 'electrical_load_pu' in control_data:
                 try:
                     load_value = float(control_data['electrical_load_pu'])
                     # Add validation if needed (e.g., check range 0.0 to 1.0+)
                     sim_instance.grid.set_electrical_load_pu(load_value)
                     print(f"Updated electrical load to: {load_value}")
                 except (ValueError, TypeError) as e:
                     print(f"Invalid value for electrical_load_pu: {control_data['electrical_load_pu']}, Error: {e}")
                     # Optionally return an error to the client

            if 'valve_position' in control_data:
                 try:
                     valve_value = float(control_data['valve_position'])
                     # Add validation (e.g., clamp between 0.0 and 1.0)
                     valve_value = max(0.0, min(1.0, valve_value))
                     sim_instance.turbine.set_valve_position(valve_value)
                     print(f"Updated valve position to: {valve_value}")
                 except (ValueError, TypeError) as e:
                     print(f"Invalid value for valve_position: {control_data['valve_position']}, Error: {e}")
                     # Optionally return an error to the client

            # Add handling for other controllable parameters (e.g., external_reactivity)
            if 'external_reactivity' in control_data:
                 try:
                     reactivity_value = float(control_data['external_reactivity'])
                     sim_instance.reactor.set_external_reactivity(reactivity_value)
                     print(f"Updated external reactivity to: {reactivity_value}")
                 except (ValueError, TypeError) as e:
                     print(f"Invalid value for external_reactivity: {control_data['external_reactivity']}, Error: {e}")


        return jsonify({"status": "success", "message": "Control parameters received"})
    except Exception as e:
        print(f"Error processing control data: {e}")
        return jsonify({"status": "error", "message": f"Internal server error: {e}"}), 500


if __name__ == '__main__':
    print("Starting Flask server for PWR Simulation...")
    # Run the Flask development server
    # host='0.0.0.0' makes it accessible from other devices on the network
    # debug=True enables auto-reloading and provides more detailed error pages (disable in production)
    # use_reloader=False is often recommended when using background threads to avoid issues
    app.run(debug=True, port=5000, host='127.0.0.1', use_reloader=False)

