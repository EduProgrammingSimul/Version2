# pwr_simulation_backend.py

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time

# --- ALL CONSTANTS ---
# (Reactor Physics, Thermal-Hydraulic, Turbine, Grid, etc.)
BETA_I = np.array([0.00025, 0.00164, 0.00147, 0.00296, 0.00086, 0.00032])
BETA = np.sum(BETA_I)
# ... (all other constants) ...
MAX_FUEL_TEMP = 2800.0 # °C
DT = 0.01 # Default time step for Simulation class

# --- Constants and Parameters (Representative PWR values, adjust as needed) ---

# Reactor Physics Parameters (Based on typical PWR data, e.g., NUREG/CR-6928, Kerlin & Upadhyaya)
BETA_I = np.array([0.00025, 0.00164, 0.00147, 0.00296, 0.00086, 0.00032]) # Delayed neutron fractions
BETA = np.sum(BETA_I) # Total delayed neutron fraction
LAMBDA_I = np.array([0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01]) # Decay constants (1/s)
PROMPT_NEUTRON_LIFETIME = 2.0e-5 # Prompt neutron lifetime (s)

# Thermal-Hydraulic Parameters
ALPHA_F = -3.0e-5 # Fuel temperature reactivity coefficient (dk/k/°C)
ALPHA_C = -1.5e-4 # Coolant temperature reactivity coefficient (dk/k/°C)
HEAT_CAP_F = 150.0 # Fuel heat capacity (MJ/°C)
HEAT_CAP_C = 350.0 # Coolant heat capacity (MJ/°C)
HEAT_TRANSFER_FC = 55.0 # Heat transfer coefficient Fuel-to-Coolant (MW/°C)
HEAT_TRANSFER_CS = 65.0 # Heat transfer coefficient Coolant-to-Secondary (Steam Gen.) (MW/°C)
NOMINAL_POWER = 3000.0 # Nominal thermal power (MWth) - Example for a ~1000 MWe plant
NOMINAL_FUEL_TEMP = 800.0 # Nominal average fuel temperature (°C)
NOMINAL_COOLANT_TEMP = 300.0 # Nominal average coolant temperature (°C)
NOMINAL_SEC_TEMP = 280.0 # Nominal secondary side (steam) temperature (°C)

# Turbine Parameters (Simplified lumped model)
TURBINE_TIME_CONST = 5.0 # Turbine time constant (s) - Needs careful definition/tuning
TURBINE_GAIN = 1.0 # Turbine gain (linking valve pos to power) - adjust based on valve characteristics
NOMINAL_TURBINE_SPEED_RPM = 1800.0 # Nominal speed for 60Hz grid (RPM)
NOMINAL_TURBINE_SPEED_RAD_S = NOMINAL_TURBINE_SPEED_RPM * (2 * np.pi / 60) # rad/s

# Generator and Grid Parameters (Swing Equation)
GENERATOR_INERTIA = 5.0 # Generator inertia constant (H) (s) - Typical range 3-7 for large turbines
DAMPING_COEFF = 0.01 # Damping coefficient (D) (pu power / pu freq deviation)
NOMINAL_GRID_FREQ = 60.0 # Hz
NOMINAL_GRID_FREQ_RAD_S = NOMINAL_GRID_FREQ * 2 * np.pi # rad/s

# Safety Limits (from proposal)
MAX_FUEL_TEMP = 2800.0 # °C
MAX_TURBINE_SPEED_RPM = 3600.0 # RPM (proposal mentions 3600, likely a typo, using 120% nominal below)
MAX_TURBINE_SPEED_RAD_S = (NOMINAL_TURBINE_SPEED_RPM * 1.2) * (2 * np.pi / 60) # Using 120% of nominal as a more typical limit
MAX_FREQ_DEVIATION_HZ = 0.5 # Hz
MAX_FREQ_DEVIATION_RAD_S = MAX_FREQ_DEVIATION_HZ * 2 * np.pi # rad/s

# Simulation Settings
SIM_TIME_STEP = 0.01 # s - as mentioned in proposal
DEFAULT_SIM_DURATION = 100 # s

# --- CLASS DEFINITIONS ---

class Reactor:
    """
    Models the PWR core using point kinetics with 6 delayed neutron groups
    and thermal feedback from fuel and coolant temperatures.
    Includes DEBUG prints in differential_equations.
    """
    def __init__(self, initial_power_fraction=1.0):
        # Initial conditions at steady state (nominal power)
        self.n_rel = initial_power_fraction # Relative neutron density (n/n0)
        # Ensure LAMBDA_I and PROMPT_NEUTRON_LIFETIME are non-zero before division
        if np.any(LAMBDA_I <= 0) or PROMPT_NEUTRON_LIFETIME <= 0:
             raise ValueError("LAMBDA_I and PROMPT_NEUTRON_LIFETIME must be positive.")
        self.C_i = (BETA_I / (LAMBDA_I * PROMPT_NEUTRON_LIFETIME)) * self.n_rel # Relative precursor concentrations (Ci/n0)
        self.T_f = NOMINAL_FUEL_TEMP # Fuel temperature (°C)
        self.T_c = NOMINAL_COOLANT_TEMP # Coolant temperature (°C)
        self.rho_ext = 0.0 # External reactivity (e.g., from control rods)

        # State vector: [n_rel, C1, C2, C3, C4, C5, C6, T_f, T_c]
        self.state = np.concatenate(([self.n_rel], self.C_i, [self.T_f, self.T_c]))
        self.thermal_power = self.n_rel * NOMINAL_POWER # MWth
        print(f"Reactor Initialized: n_rel={self.n_rel}, C_i shape={self.C_i.shape}, T_f={self.T_f}, T_c={self.T_c}") # Debug Init

    def set_external_reactivity(self, rho):
        """Sets the external reactivity input."""
        self.rho_ext = rho

    def _calculate_reactivity(self):
        """Calculates total reactivity including feedback."""
        # Ensure temperatures are valid numbers before calculation
        if not isinstance(self.T_f, (int, float, np.number)) or not isinstance(self.T_c, (int, float, np.number)):
             print(f"ERROR: Invalid temperature types! T_f type={type(self.T_f)}, T_c type={type(self.T_c)}")
             # Handle error appropriately, e.g., return zero reactivity or raise exception
             return self.rho_ext # Return only external reactivity as fallback
        rho_feedback = ALPHA_F * (self.T_f - NOMINAL_FUEL_TEMP) + \
                       ALPHA_C * (self.T_c - NOMINAL_COOLANT_TEMP)
        return self.rho_ext + rho_feedback

    def differential_equations(self, t, y):
        """
        Defines the system of ODEs for the reactor.
        y = [n_rel, C1, C2, C3, C4, C5, C6, T_f, T_c]
        Includes DEBUG prints.
        """
        # --- DEBUG: Check input state vector y ---
        if not isinstance(y, np.ndarray) or y.ndim != 1 or len(y) != 9:
            print(f"ERROR: Incorrect state vector type/shape at t={t:.4f}! Expected numpy array(9,), got {type(y)} with shape {getattr(y, 'shape', 'N/A')}. y = {y}")
            # Return zero derivatives to attempt graceful stop
            return np.zeros(9)

        n_rel, C1, C2, C3, C4, C5, C6, T_f, T_c = y
        C_i = np.array([C1, C2, C3, C4, C5, C6]) # Construct C_i array from state vector components

        # Update internal state variables for feedback calculation
        self.n_rel = n_rel
        self.T_f = T_f
        self.T_c = T_c
        # Ensure n_rel is valid before calculating power
        if not isinstance(n_rel, (int, float, np.number)):
             print(f"ERROR: Invalid n_rel type at t={t:.4f}! type={type(n_rel)}. Setting thermal power to 0.")
             self.thermal_power = 0.0
        else:
             self.thermal_power = n_rel * NOMINAL_POWER

        # Calculate total reactivity
        rho_total = self._calculate_reactivity()

        # Check for safety limits
        if isinstance(T_f, (int, float, np.number)) and T_f >= MAX_FUEL_TEMP:
            print(f"Warning: Max fuel temperature {MAX_FUEL_TEMP}°C exceeded at t={t:.2f}s! T_f = {T_f}")
            # Consider adding logic to handle this, e.g., trip simulation

        # --- Point Kinetics Equations ---

        # +++ DEBUG PRINTS +++
        # print(f"\n--- Debugging Reactor ODEs at t={t:.4f} ---") # Reduce frequency if too verbose
        # print(f"Input y: {y}")
        # print(f"Extracted n_rel: type={type(n_rel)}, value={n_rel}")
        # print(f"Extracted C_i: type={type(C_i)}, shape={C_i.shape}, value={C_i}")
        # print(f"Internal T_f: {self.T_f}, T_c: {self.T_c}")
        # print(f"Calculated rho_total: type={type(rho_total)}, value={rho_total}")
        # print(f"Global BETA: type={type(BETA)}, value={BETA}")
        # print(f"Global PNL: type={type(PROMPT_NEUTRON_LIFETIME)}, value={PROMPT_NEUTRON_LIFETIME}")
        # print(f"Global LAMBDA_I: type={type(LAMBDA_I)}, shape={LAMBDA_I.shape}, value={LAMBDA_I}")
        # Check for potential division by zero explicitly
        if PROMPT_NEUTRON_LIFETIME == 0:
             print("ERROR: PROMPT_NEUTRON_LIFETIME is zero!")
             return np.zeros(9) # Return zero derivatives
        # Check shapes before multiplication
        if LAMBDA_I.shape != C_i.shape:
            print(f"ERROR: Shape mismatch! LAMBDA_I shape={LAMBDA_I.shape}, C_i shape={C_i.shape}")
            return np.zeros(9) # Return zero derivatives
        # Check types are numeric before calculation
        numeric_types = (int, float, np.number)
        if not isinstance(n_rel, numeric_types): print(f"ERROR: n_rel is not a number! type={type(n_rel)}"); return np.zeros(9)
        if not isinstance(rho_total, numeric_types): print(f"ERROR: rho_total is not a number! type={type(rho_total)}"); return np.zeros(9)
        if not isinstance(C_i, np.ndarray): print(f"ERROR: C_i is not a numpy array! type={type(C_i)}"); return np.zeros(9)
        if not np.issubdtype(C_i.dtype, np.number): print(f"ERROR: C_i does not contain numbers! dtype={C_i.dtype}"); return np.zeros(9)
        # +++ END DEBUG PRINTS +++

        # The potentially failing line, wrapped in try-except:
        try:
            # Calculate terms separately for clarity
            term1 = (rho_total - BETA) / PROMPT_NEUTRON_LIFETIME * n_rel
            term2 = np.sum(LAMBDA_I * C_i)
            dn_rel_dt = term1 + term2

            # Also check the second equation's terms
            term3 = (BETA_I / PROMPT_NEUTRON_LIFETIME) * n_rel
            term4 = LAMBDA_I * C_i
            # Ensure shapes match for subtraction
            if term3.shape != term4.shape:
                 print(f"ERROR: Shape mismatch for dCi_dt! term3 shape={term3.shape}, term4 shape={term4.shape}")
                 return np.zeros(9)
            dCi_dt = term3 - term4

        except Exception as e:
            print(f"!!!!!!!! EXCEPTION CAUGHT in kinetics calculation at t={t:.4f} !!!!!!!!")
            print(f"Error message: {e}")
            print(f"Variables at time of error:")
            print(f"  rho_total={rho_total}, BETA={BETA}, PNL={PROMPT_NEUTRON_LIFETIME}, n_rel={n_rel}")
            print(f"  LAMBDA_I={LAMBDA_I} (shape {LAMBDA_I.shape})")
            print(f"  C_i={C_i} (shape {C_i.shape})")
            print(f"  BETA_I={BETA_I} (shape {BETA_I.shape})")
            # Re-raise the exception to stop execution and see the full traceback
            raise e

        # --- Thermal-Hydraulic Equations ---
        # Ensure temperatures are valid before using in calculations
        if not isinstance(T_f, numeric_types) or not isinstance(T_c, numeric_types):
             print(f"ERROR: Invalid temperature types for thermal calculation at t={t:.4f}! T_f type={type(T_f)}, T_c type={type(T_c)}")
             # Set derivatives to zero as fallback
             dTf_dt = 0.0
             dTc_dt = 0.0
        else:
             P_fuel = self.thermal_power # Use already calculated thermal power
             Q_fc = HEAT_TRANSFER_FC * (T_f - T_c)
             Q_cs = HEAT_TRANSFER_CS * (T_c - NOMINAL_SEC_TEMP)

             # Avoid division by zero in heat capacity
             if HEAT_CAP_F == 0 or HEAT_CAP_C == 0:
                  print("ERROR: Heat capacity cannot be zero!")
                  dTf_dt = 0.0
                  dTc_dt = 0.0
             else:
                  dTf_dt = (P_fuel - Q_fc) / HEAT_CAP_F
                  dTc_dt = (Q_fc - Q_cs) / HEAT_CAP_C

        # Combine derivatives into a single vector
        dydt = np.concatenate(([dn_rel_dt], dCi_dt, [dTf_dt, dTc_dt]))

        # Final check: ensure dydt has the correct shape
        if dydt.shape != (9,):
             print(f"ERROR: Final derivative vector dydt has wrong shape at t={t:.4f}! Shape is {dydt.shape}. dydt={dydt}")
             # Attempt to return zeros of the correct shape
             return np.zeros(9)

        return dydt

    def get_state(self):
        """Returns the current state vector."""
        return self.state

    def update_state(self, new_state):
        """Updates the state vector from the solver result."""
        if new_state is not None and len(new_state) == 9:
            self.state = new_state
            # Update individual variables for clarity and feedback calculation
            self.n_rel = new_state[0]
            self.C_i = new_state[1:7]
            self.T_f = new_state[7]
            self.T_c = new_state[8]
            # Recalculate thermal power based on updated n_rel
            if isinstance(self.n_rel, (int, float, np.number)):
                 self.thermal_power = self.n_rel * NOMINAL_POWER
            else:
                 self.thermal_power = 0.0 # Fallback if n_rel is invalid
        else:
            print(f"Warning: Attempted to update reactor state with invalid data: {new_state}")


class Turbine:
    """
    Models the steam turbine using a simplified lumped-parameter model.
    Focuses on rotational speed dynamics based on steam valve position.
    """
    def __init__(self, initial_speed_rad_s=NOMINAL_TURBINE_SPEED_RAD_S):
        # State vector: [omega_t] (turbine speed in rad/s)
        self.omega_t = initial_speed_rad_s
        self.state = np.array([self.omega_t])
        self.valve_position = 1.0 # Initial valve position (0.0 to 1.0)
        self.mechanical_power = 0.0 # MW (calculated during ODE solve)

    def set_valve_position(self, pos):
        """Sets the governor valve position (0 to 1)."""
        self.valve_position = np.clip(pos, 0.0, 1.0)

    def differential_equations(self, t, y, reactor_thermal_power, electrical_load_pu):
        """
        Defines the ODE for the turbine speed.
        y = [omega_t]
        Requires thermal power from reactor (influences steam conditions indirectly)
        and electrical load from grid (converted to MW).
        """
        if not isinstance(y, np.ndarray) or len(y) != 1:
             print(f"ERROR: Invalid turbine state vector y at t={t:.4f}. y={y}")
             return np.zeros(1)

        omega_t = y[0]
        self.omega_t = omega_t # Update internal variable

        # Ensure inputs are valid numbers
        numeric_types = (int, float, np.number)
        if not isinstance(reactor_thermal_power, numeric_types) or \
           not isinstance(electrical_load_pu, numeric_types) or \
           not isinstance(omega_t, numeric_types):
            print(f"ERROR: Invalid input types for turbine ODE at t={t:.4f}!")
            print(f"  reactor_power type: {type(reactor_thermal_power)}")
            print(f"  load_pu type: {type(electrical_load_pu)}")
            print(f"  omega_t type: {type(omega_t)}")
            return np.zeros(1)

        # Simplified relation: Mechanical power proportional to valve position and reactor power
        # Assume nominal thermal power produces nominal mechanical power at full valve opening and nominal speed.
        # Scale by relative speed to account for speed effects on power output.
        # Avoid division by zero if nominal speed is zero
        if NOMINAL_TURBINE_SPEED_RAD_S == 0:
             print("ERROR: NOMINAL_TURBINE_SPEED_RAD_S cannot be zero!")
             P_mech_potential = 0.0
        else:
             P_mech_potential = reactor_thermal_power * TURBINE_GAIN * (omega_t / NOMINAL_TURBINE_SPEED_RAD_S)

        P_mech_actual = P_mech_potential * self.valve_position
        self.mechanical_power = P_mech_actual # Store for grid use and output

        # Electrical power demand (converted from pu based on nominal thermal power)
        P_elec_mw = electrical_load_pu * NOMINAL_POWER

        # Check for safety limits
        current_speed_rpm = omega_t * 60 / (2 * np.pi)
        if current_speed_rpm >= MAX_TURBINE_SPEED_RPM * 1.01: # Add small buffer
             print(f"Warning: Max turbine speed {MAX_TURBINE_SPEED_RPM} RPM potentially exceeded at t={t:.2f}s! Speed = {current_speed_rpm:.1f} RPM")
             # Consider implementing trip logic or capping acceleration

        # Simplified turbine dynamics based on power imbalance (similar to swing equation concept)
        # d(omega_t)/dt = (P_mech - P_elec) / (Effective_Inertia * omega_t) -- problem near omega=0
        # Using a linear approximation or relating to TURBINE_TIME_CONST:
        # Assume TC represents time for power imbalance to cause significant speed change.
        # Let's use a simple power balance affecting acceleration, scaled by nominal conditions.
        # Avoid division by zero if nominal speed is zero
        if NOMINAL_TURBINE_SPEED_RAD_S == 0 or TURBINE_TIME_CONST == 0:
             print("ERROR: Cannot calculate turbine speed derivative due to zero nominal speed or time constant!")
             domega_t_dt = 0.0
        else:
             # This model assumes speed deviation is proportional to power mismatch / (TC * Nominal Speed)
             # Units: (MW - MW) / (s * rad/s) -> MW*s / (s*rad) -> MW/rad ?? Needs review/refinement based on TC definition.
             # Let's assume TC relates torque imbalance to angular acceleration: d(omega)/dt = Torque_imbalance / J
             # Torque ~ Power / omega. Let J = TC * Nominal_Torque / Nominal_Omega ?
             # Using a simpler approach: imbalance drives speed change over TC.
             power_imbalance_mw = P_mech_actual - P_elec_mw
             # Scale factor needed. Let's assume TC relates speed change to relative power imbalance.
             domega_t_dt = (power_imbalance_mw / NOMINAL_POWER) * (NOMINAL_TURBINE_SPEED_RAD_S / TURBINE_TIME_CONST)

        return [domega_t_dt]


    def get_state(self):
        return self.state

    def update_state(self, new_state):
        if new_state is not None and len(new_state) == 1:
            self.state = new_state
            self.omega_t = new_state[0]
        else:
            print(f"Warning: Attempted to update turbine state with invalid data: {new_state}")


    def get_mechanical_power_output(self):
        """Returns the current mechanical power output (MW) calculated in the last ODE step."""
        # This value is updated during the differential_equations call
        return self.mechanical_power



class GridInterface:
    """
    Models the grid interface using the swing equation for a single machine
    connected to an infinite bus. Calculates frequency deviation.
    """
    def __init__(self, initial_delta=0.0, initial_freq_dev_rad_s=0.0):
        # State vector: [delta, omega_deviation]
        # delta: Rotor angle difference (rad)
        # omega_deviation: Frequency deviation from nominal (rad/s)
        self.delta = initial_delta
        self.omega_dev = initial_freq_dev_rad_s
        self.state = np.array([self.delta, self.omega_dev])
        self.electrical_load_pu = 1.0 # Per unit electrical load demand (relative to NOMINAL_POWER)

    def set_electrical_load_pu(self, load_pu):
        """Sets the electrical load demand (per unit of nominal power)."""
        # Add validation if needed
        self.electrical_load_pu = max(0.0, load_pu) # Ensure non-negative load

    def differential_equations(self, t, y, P_mech_mw):
        """
        Defines the swing equations.
        y = [delta, omega_deviation]
        Requires mechanical power input from the turbine (MW).
        """
        if not isinstance(y, np.ndarray) or len(y) != 2:
             print(f"ERROR: Invalid grid state vector y at t={t:.4f}. y={y}")
             return np.zeros(2)

        delta, omega_dev = y
        self.delta = delta # Update internal variable
        self.omega_dev = omega_dev # Update internal variable

        # Ensure inputs are valid numbers
        numeric_types = (int, float, np.number)
        if not isinstance(P_mech_mw, numeric_types) or \
           not isinstance(delta, numeric_types) or \
           not isinstance(omega_dev, numeric_types):
            print(f"ERROR: Invalid input types for grid ODE at t={t:.4f}!")
            print(f"  P_mech_mw type: {type(P_mech_mw)}")
            print(f"  delta type: {type(delta)}")
            print(f"  omega_dev type: {type(omega_dev)}")
            return np.zeros(2)


        # Convert mechanical power to per unit
        if NOMINAL_POWER == 0:
             print("ERROR: NOMINAL_POWER cannot be zero!")
             P_mech_pu = 0.0
        else:
             P_mech_pu = P_mech_mw / NOMINAL_POWER

        # Electrical power output (simplified model: assumes P_elec = P_load instantly)
        P_elec_pu = self.electrical_load_pu

        # Check safety limits for frequency deviation
        current_freq_hz = NOMINAL_GRID_FREQ + omega_dev / (2 * np.pi)
        freq_dev_hz = current_freq_hz - NOMINAL_GRID_FREQ
        if abs(freq_dev_hz) >= MAX_FREQ_DEVIATION_HZ * 1.01: # Add small buffer
            print(f"Warning: Max frequency deviation {MAX_FREQ_DEVIATION_HZ} Hz potentially exceeded at t={t:.2f}s! Deviation = {freq_dev_hz:.3f} Hz")
            # Consider implementing grid protection logic (e.g., tripping)

        # --- Swing Equations ---
        # d(delta)/dt = omega_dev (Definition of frequency deviation)
        ddelta_dt = omega_dev

        # d(omega_dev)/dt = (omega_nom / (2*H)) * (P_mech_pu - P_elec_pu - D * (omega_dev / omega_nom))
        # Avoid division by zero for inertia H
        if GENERATOR_INERTIA == 0:
             print("ERROR: GENERATOR_INERTIA cannot be zero!")
             domega_dev_dt = 0.0
        else:
             # Calculate damping term separately
             damping_effect = DAMPING_COEFF * (omega_dev / NOMINAL_GRID_FREQ_RAD_S)
             # Calculate acceleration term
             power_mismatch_pu = P_mech_pu - P_elec_pu - damping_effect
             domega_dev_dt = (NOMINAL_GRID_FREQ_RAD_S / (2 * GENERATOR_INERTIA)) * power_mismatch_pu

        return [ddelta_dt, domega_dev_dt]

    def get_state(self):
        return self.state

    def update_state(self, new_state):
        if new_state is not None and len(new_state) == 2:
            self.state = new_state
            self.delta = new_state[0]
            self.omega_dev = new_state[1]
        else:
            print(f"Warning: Attempted to update grid state with invalid data: {new_state}")


    def get_frequency_hz(self):
        """Returns the current grid frequency in Hz."""
        return NOMINAL_GRID_FREQ + self.omega_dev / (2 * np.pi)




class Simulation:
    """
    Integrates the Reactor, Turbine, and Grid models... (From previous complete code)
    """
    def __init__(self, initial_power_fraction=1.0, initial_load_pu=0.9, dt=DT):
        # --- Initialize Models ---
        self.reactor = Reactor(initial_power_fraction=initial_power_fraction)
        self.turbine = Turbine(initial_valve_pos=initial_power_fraction) # Match initial valve to power
        self.grid = GridInterface(initial_load_pu=initial_load_pu)
        self.dt = dt
        self.time = 0.0
        # --- Store results ---
        self.results = {
            'time': [self.time],
            'reactor_power': [self.reactor.thermal_power],
            'fuel_temp': [self.reactor.T_f],
            'coolant_temp': [self.reactor.T_c],
            'turbine_speed': [self.turbine.speed_rpm],
            'valve_position': [self.turbine.valve_position_actual],
            'grid_frequency': [self.grid.frequency],
            'electrical_load_pu': [self.grid.electrical_load_pu],
            'external_reactivity': [self.reactor.rho_ext]
             # Add other states as needed
        }
        self.state = self._get_combined_state() # Get initial combined state
        print("Simulation Class Initialized.")


    def _get_combined_state(self):
        # Example: Combine key states for control logic or external access
         return {
             'time': self.time,
             'reactor_power': self.reactor.thermal_power,
             'fuel_temp': self.reactor.T_f,
             'coolant_temp': self.reactor.T_c,
             'turbine_speed': self.turbine.speed_rpm,
             'valve_position': self.turbine.valve_position_actual,
             'grid_frequency': self.grid.frequency,
             'electrical_load_pu': self.grid.electrical_load_pu,
             'external_reactivity': self.reactor.rho_ext
         }


    def get_results(self):
         # Return a copy to avoid external modification
         return {k: list(v) for k, v in self.results.items()}


    def step(self, duration, controls):
        """
        Steps the simulation forward by a given duration, applying controls.
        'controls' is a dict: {'valve_position': V, 'external_reactivity': R, 'electrical_load_pu': L}
        Returns True if successful, False otherwise.
        """
        try:
            # --- Apply controls ---
            if 'valve_position' in controls:
                self.turbine.set_valve_position(controls['valve_position']) # Assuming Turbine has this method
            if 'external_reactivity' in controls:
                self.reactor.set_external_reactivity(controls['external_reactivity'])
            if 'electrical_load_pu' in controls:
                self.grid.set_electrical_load_pu(controls['electrical_load_pu']) # Assuming Grid has this method

            # --- Prepare initial state for solver ---
            y0 = np.concatenate((
                self.reactor.get_state(),
                # Add turbine and grid states similarly if they use ODEs internally
                # If not, handle their stepping logic separately before/after reactor ODE
            ))

            # --- Solve Reactor ODE ---
            sol = solve_ivp(
                self.reactor.differential_equations,
                [self.time, self.time + duration],
                y0[0:9], # Pass only the reactor part of the state vector
                method='RK45', # Or 'LSODA' for stiff systems
                t_eval=[self.time + duration] # Evaluate only at the end point
                # args=(...) # Pass any extra arguments needed by differential_equations
            )

            if not sol.success:
                 print(f"ODE Solver failed at time {self.time:.2f}: {sol.message}")
                 return False

            # --- Update states ---
            self.reactor.update_state(sol.y[:, -1])
            # --- Step Turbine and Grid (if they have their own step logic) ---
            # Example: self.turbine.step(duration, self.reactor.T_c, self.grid.electrical_load_pu)
            # Example: self.grid.step(duration, self.turbine.get_mechanical_power_pu())
            # NOTE: The exact coupling depends heavily on your Turbine/Grid implementations.
            #       If they were ODE-based like in the Gym env, you'd solve them together.
            #       This simplified example assumes only Reactor uses solve_ivp directly here.


            self.time += duration
            self.state = self._get_combined_state() # Update combined state


            # --- Append results ---
            self.results['time'].append(self.time)
            self.results['reactor_power'].append(self.reactor.thermal_power)
            self.results['fuel_temp'].append(self.reactor.T_f)
            self.results['coolant_temp'].append(self.reactor.T_c)
            self.results['turbine_speed'].append(self.turbine.speed_rpm) # Assuming turbine updates its speed
            self.results['valve_position'].append(self.turbine.valve_position_actual) # Assuming turbine updates valve
            self.results['grid_frequency'].append(self.grid.frequency) # Assuming grid updates freq
            self.results['electrical_load_pu'].append(self.grid.electrical_load_pu)
            self.results['external_reactivity'].append(self.reactor.rho_ext)


            return True

        except Exception as e:
             print(f"Error during Simulation step at time {self.time:.2f}: {e}")
             import traceback
             traceback.print_exc() # Print full stack trace
             return False


# --- CONTROL LOGIC FUNCTION (Example) ---
def scenario_control(time, state):
    """
    Example control logic function.
    Determines control inputs based on time and current state.
    """
    # Default values (maintain current state)
    controls = {
        'valve_position': state.get('valve_position', 0.9), # Use current valve pos if available
        'external_reactivity': state.get('external_reactivity', 0.0), # Use current rho_ext
        'electrical_load_pu': state.get('electrical_load_pu', 0.9) # Use current load
    }

    # Example: Implement a simple load change scenario
    if 10 <= time < 40:
        controls['electrical_load_pu'] = 0.95 # Increase load between t=10 and t=40
    elif time >= 40:
         controls['electrical_load_pu'] = 0.9 # Return to base load


    # Example: Simple proportional control for reactivity based on temp deviation (basic)
    # temp_deviation = state.get('coolant_temp', 300.0) - 300.0
    # controls['external_reactivity'] = - temp_deviation * 1e-4


    # NOTE: You would replace this with more sophisticated logic or
    # allow the /control endpoint to modify target values used here.

    return controls

# --- (End of pwr_simulation_backend.py) ---