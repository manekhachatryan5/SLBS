# slbs_model.py
"""
Mathematical core of the Susceptible-Latent-Breaking-out-Susceptible (SLBS) model.
This module defines:
1. The deterministic SLBS equations (both delay and non-delay versions)
2. Node-level transition rates for stochastic Gillespie simulation
3. Parameter validation and management
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import IntEnum


class NodeState(IntEnum):
    """Enumeration of possible node states in the SLBS model."""
    SUSCEPTIBLE = 0
    LATENT = 1
    BREAKING_OUT = 2


@dataclass
class SLBSParameters:
    """
    Container for all SLBS model parameters with validation.
    
    Parameters:
    -----------
    beta1 : float
        Infection rate from latent nodes to susceptible nodes (per neighbor per unit time)
    beta2 : float
        Infection rate from breaking-out nodes to susceptible nodes (per neighbor per unit time)
    gamma1 : float
        Cleaning rate for latent nodes (per node per unit time)
    gamma2 : float
        Cleaning rate for breaking-out nodes (per node per unit time)
    alpha : float
        Activation rate from latent to breaking-out state (per node per unit time)
    theta : float
        Infection rate via removable media (non-network infection, per node per unit time)
    delta : float
        Disconnection/removal rate for any node (per node per unit time)
    tau : float
        Delay in cleaning/antivirus response (time units). Set to 0 for no delay.
    mu1 : float
        Inflow rate of new susceptible nodes (nodes per unit time)
    mu2 : float
        Inflow rate of new latent nodes (nodes per unit time)
    """
    # Infection parameters
    beta1: float = 0.01    # Latent infection rate
    beta2: float = 0.02    # Breaking-out infection rate
    alpha: float = 0.2     # Activation rate (L -> B)
    theta: float = 0.02    # Removable media infection rate

    # Cleaning/recovery parameters
    gamma1: float = 0.1    # Latent cleaning rate
    gamma2: float = 0.3    # Breaking-out cleaning rate

    # Network dynamics parameters
    delta: float = 0.1     # Disconnection rate
    tau: float = 0.0       # Cleaning delay (0 for no delay)

    # Inflow parameters
    mu1: float = 4.0       # Inflow of susceptible nodes
    mu2: float = 2.0       # Inflow of latent nodes

    def __post_init__(self):
        """Validate all parameters after initialization."""
        self.validate()

    def validate(self) -> None:
        """
        Validate that all parameters are within reasonable bounds.
        Raises ValueError if any parameter is invalid.
        """
        # All rates should be non-negative
        rate_params = ['beta1', 'beta2', 'gamma1', 'gamma2', 'alpha',
                       'theta', 'delta', 'mu1', 'mu2']

        for param in rate_params:
            value = getattr(self, param)
            if value < 0:
                raise ValueError(f"Parameter {param} must be non-negative, got {value}")

        # Delay should be non-negative
        if self.tau < 0:
            raise ValueError(f"Delay tau must be non-negative, got {self.tau}")

        # Check for obviously unreasonable values (optional)
        if self.beta1 > 10 or self.beta2 > 10:
            print(f"Warning: High infection rates (beta1={self.beta1}, beta2={self.beta2})")

        if self.gamma1 > 10 or self.gamma2 > 10:
            print(f"Warning: High cleaning rates (gamma1={self.gamma1}, gamma2={self.gamma2})")

    def copy(self) -> 'SLBSParameters':
        """Create a copy of the parameters."""
        return SLBSParameters(
            beta1=self.beta1, beta2=self.beta2,
            gamma1=self.gamma1, gamma2=self.gamma2,
            alpha=self.alpha, theta=self.theta,
            delta=self.delta, tau=self.tau,
            mu1=self.mu1, mu2=self.mu2
        )

    def to_dict(self) -> Dict[str, float]:
        """Convert parameters to dictionary for easy serialization."""
        return {
            'beta1': self.beta1, 'beta2': self.beta2,
            'gamma1': self.gamma1, 'gamma2': self.gamma2,
            'alpha': self.alpha, 'theta': self.theta,
            'delta': self.delta, 'tau': self.tau,
            'mu1': self.mu1, 'mu2': self.mu2
        }


class SLBSModel:
    """
    Mathematical implementation of the SLBS model.
    
    This class provides:
    1. Deterministic ODE/DDE equations for population-level dynamics
    2. Node-level transition rate calculations for stochastic simulation
    3. Helper methods for analysis and validation
    """

    def __init__(self, params: Optional[SLBSParameters] = None):
        """
        Initialize the SLBS model with parameters.
        
        Args:
            params: SLBS model parameters. If None, uses default parameters.
        """
        self.params = params if params is not None else SLBSParameters()

    # =========================================================================
    # DETERMINISTIC EQUATIONS (POPULATION LEVEL)
    # =========================================================================

    def deterministic_ode(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Compute the right-hand side of the ODE system (no delay).
        
        Args:
            t: Current time (unused for autonomous system, but required by ODE solvers)
            y: State vector [S, L, B]
            
        Returns:
            dy/dt: Derivative vector [dS/dt, dL/dt, dB/dt]
        """
        S, L, B = y
        p = self.params

        # Susceptible equation
        dS_dt = (p.mu1
                 - (p.beta1 * L + p.beta2 * B) * S
                 + p.gamma1 * L  # No delay
                 + p.gamma2 * B  # No delay
                 - (p.delta + p.theta) * S)

        # Latent equation
        dL_dt = (p.mu2
                 + (p.beta1 * L + p.beta2 * B) * S
                 - p.gamma1 * L  # No delay
                 - (p.alpha + p.delta) * L
                 + p.theta * S)

        # Breaking-out equation
        dB_dt = (p.alpha * L
                 - p.gamma2 * B  # No delay
                 - p.delta * B)

        return np.array([dS_dt, dL_dt, dB_dt])

    def deterministic_dde(self, t: float, y: np.ndarray,
                          y_past: np.ndarray) -> np.ndarray:
        """
        Compute the right-hand side of the DDE system (with delay).
        
        Args:
            t: Current time
            y: Current state vector [S, L, B]
            y_past: State vector at time t - tau [S(t-tau), L(t-tau), B(t-tau)]
            
        Returns:
            dy/dt: Derivative vector [dS/dt, dL/dt, dB/dt]
        """
        S, L, B = y
        S_past, L_past, B_past = y_past
        p = self.params

        # Susceptible equation with delay
        dS_dt = (p.mu1
                 - (p.beta1 * L + p.beta2 * B) * S
                 + p.gamma1 * L_past  # Delayed cleaning
                 + p.gamma2 * B_past  # Delayed cleaning
                 - (p.delta + p.theta) * S)

        # Latent equation with delay
        dL_dt = (p.mu2
                 + (p.beta1 * L + p.beta2 * B) * S
                 - p.gamma1 * L_past  # Delayed cleaning
                 - (p.alpha + p.delta) * L
                 + p.theta * S)

        # Breaking-out equation with delay
        dB_dt = (p.alpha * L
                 - p.gamma2 * B_past  # Delayed cleaning
                 - p.delta * B)

        return np.array([dS_dt, dL_dt, dB_dt])

    # =========================================================================
    # NODE-LEVEL TRANSITION RATES (FOR STOCHASTIC SIMULATION)
    # =========================================================================

    def compute_node_rates(self, node_state: NodeState,
                           neighbor_counts: Dict[NodeState, int],
                           params: Optional[SLBSParameters] = None) -> Dict[str, float]:
        """
        Compute all possible transition rates for a node given its state and neighbors.
        
        Args:
            node_state: Current state of the node
            neighbor_counts: Dictionary mapping neighbor states to counts
            params: Optional parameters (uses model params if None)
            
        Returns:
            Dictionary mapping event names to their rates for this node
        """
        p = params if params is not None else self.params

        rates = {}

        if node_state == NodeState.SUSCEPTIBLE:
            # Infection from neighbors
            L_nbr = neighbor_counts.get(NodeState.LATENT, 0)
            B_nbr = neighbor_counts.get(NodeState.BREAKING_OUT, 0)

            infection_rate = p.beta1 * L_nbr + p.beta2 * B_nbr
            if infection_rate > 0:
                rates['infection'] = infection_rate

            # Non-network infection (removable media)
            if p.theta > 0:
                rates['removable_media_infection'] = p.theta

            # Disconnection
            if p.delta > 0:
                rates['disconnect'] = p.delta

        elif node_state == NodeState.LATENT:
            # Activation to breaking-out
            if p.alpha > 0:
                rates['activate'] = p.alpha

            # Cleaning (delayed or not)
            # Note: In Gillespie, delayed events are handled differently
            # For now, we treat cleaning as a regular event
            if p.gamma1 > 0:
                rates['clean'] = p.gamma1

            # Disconnection
            if p.delta > 0:
                rates['disconnect'] = p.delta

        elif node_state == NodeState.BREAKING_OUT:
            # Cleaning (delayed or not)
            if p.gamma2 > 0:
                rates['clean'] = p.gamma2

            # Disconnection
            if p.delta > 0:
                rates['disconnect'] = p.delta

        return rates

    def get_all_event_rates(self, network_state: List[NodeState],
                            adjacency_list: List[List[int]],
                            params: Optional[SLBSParameters] = None) -> Tuple[List[float], List[Dict]]:
        """
        Compute event rates for all nodes in the network.
        
        Args:
            network_state: List of node states (length = number of nodes)
            adjacency_list: List of lists, where adjacency_list[i] contains neighbors of node i
            params: Optional parameters (uses model params if None)
            
        Returns:
            Tuple of (rates_list, event_info_list) where:
                rates_list: List of event rates for all possible events
                event_info_list: List of dictionaries with event details
        """
        p = params if params is not None else self.params
        n_nodes = len(network_state)

        rates_list = []
        event_info_list = []

        for node_id in range(n_nodes):
            state = network_state[node_id]

            # Count neighbors by state
            neighbor_counts = {NodeState.SUSCEPTIBLE: 0,
                               NodeState.LATENT: 0,
                               NodeState.BREAKING_OUT: 0}

            for neighbor_id in adjacency_list[node_id]:
                neighbor_state = network_state[neighbor_id]
                neighbor_counts[neighbor_state] += 1

            # Get rates for this node
            node_rates = self.compute_node_rates(state, neighbor_counts, p)

            # Add each event to the lists
            for event_type, rate in node_rates.items():
                if rate > 0:
                    rates_list.append(rate)
                    event_info_list.append({
                        'node_id': node_id,
                        'event_type': event_type,
                        'current_state': state
                    })

        return rates_list, event_info_list

    # =========================================================================
    # ANALYTICAL METHODS
    # =========================================================================

    def compute_basic_reproduction_number(self) -> float:
        """
        Compute the basic reproduction number R0 for the mean-field model.
        
        Returns:
            R0 value (approximate for the simplified model)
            
        Note: This is a simplified calculation. For the full network model,
        R0 would depend on network structure.
        """
        p = self.params

        # For the mean-field SLBS model without delay, R0 can be approximated as:
        # R0 ≈ (β1/γ1 + β2α/(γ1γ2)) * S0
        # where S0 is the disease-free equilibrium susceptible fraction

        # Disease-free equilibrium susceptible proportion
        S0 = p.mu1 / (p.delta + p.theta) if (p.delta + p.theta) > 0 else 1.0

        # Contribution from latent infections
        R0_latent = p.beta1 * S0 / (p.gamma1 + p.delta) if (p.gamma1 + p.delta) > 0 else float('inf')

        # Contribution from breaking-out infections
        R0_breaking = (p.beta2 * p.alpha * S0 /
                       ((p.gamma1 + p.delta) * (p.gamma2 + p.delta))) if (p.gamma1 + p.delta) * (p.gamma2 + p.delta) > 0 else float('inf')

        R0 = R0_latent + R0_breaking

        return R0

    def compute_equilibria(self) -> Dict[str, np.ndarray]:
        """
        Compute disease-free and endemic equilibria for the mean-field model.
        
        Returns:
            Dictionary with 'disease_free' and 'endemic' equilibrium points
        """
        p = self.params

        # Disease-free equilibrium
        S_dfe = p.mu1 / (p.delta + p.theta) if (p.delta + p.theta) > 0 else 1.0
        L_dfe = 0.0
        B_dfe = 0.0
        dfe = np.array([S_dfe, L_dfe, B_dfe])

        # Endemic equilibrium (simplified calculation)
        # This solves the equations numerically - for exact solution we'd need
        # to solve the cubic equation from the paper
        R0 = self.compute_basic_reproduction_number()

        if R0 <= 1:
            # No endemic equilibrium if R0 <= 1
            endemic = np.array([np.nan, np.nan, np.nan])
        else:
            # Approximate endemic equilibrium
            # We could implement the exact solution from the paper here
            # For now, return placeholder
            endemic = np.array([np.nan, np.nan, np.nan])

        return {
            'disease_free': dfe,
            'endemic': endemic,
            'R0': R0
        }

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def get_state_names(self) -> List[str]:
        """Get names of the compartments."""
        return ['S', 'L', 'B']

    def get_state_colors(self) -> Dict[NodeState, str]:
        """Get color mapping for visualization."""
        return {
            NodeState.SUSCEPTIBLE: 'green',
            NodeState.LATENT: 'yellow',
            NodeState.BREAKING_OUT: 'red'
        }

    def get_state_labels(self) -> Dict[NodeState, str]:
        """Get label mapping for visualization."""
        return {
            NodeState.SUSCEPTIBLE: 'Susceptible',
            NodeState.LATENT: 'Latent',
            NodeState.BREAKING_OUT: 'Breaking-out'
        }

    def create_initial_state(self, n_nodes: int,
                             initial_infected: float = 0.01) -> List[NodeState]:
        """
        Create an initial network state.
        
        Args:
            n_nodes: Total number of nodes
            initial_infected: Fraction of nodes initially in breaking-out state
            
        Returns:
            List of initial node states
        """
        n_infected = int(n_nodes * initial_infected)
        n_latent = int(n_nodes * initial_infected * 0.5)  # Half as many latent

        states = [NodeState.SUSCEPTIBLE] * n_nodes

        # Set breaking-out nodes
        breaking_indices = np.random.choice(n_nodes, n_infected, replace=False)
        for idx in breaking_indices:
            states[idx] = NodeState.BREAKING_OUT

        # Set latent nodes (from remaining nodes)
        remaining_indices = [i for i in range(n_nodes) if i not in breaking_indices]
        latent_indices = np.random.choice(remaining_indices,
                                          min(n_latent, len(remaining_indices)),
                                          replace=False)
        for idx in latent_indices:
            states[idx] = NodeState.LATENT

        return states


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_deterministic_ode():
    """Test the deterministic ODE implementation."""
    print("Testing deterministic ODE...")

    params = SLBSParameters()
    model = SLBSModel(params)

    # Test point near equilibrium
    y0 = np.array([0.8, 0.1, 0.1])
    dydt = model.deterministic_ode(0, y0)

    print(f"Initial state: S={y0[0]:.3f}, L={y0[1]:.3f}, B={y0[2]:.3f}")
    print(f"Derivatives: dS/dt={dydt[0]:.3f}, dL/dt={dydt[1]:.3f}, dB/dt={dydt[2]:.3f}")

    # Test mass conservation (without inflow/outflow)
    params_no_flow = SLBSParameters(mu1=0, mu2=0, delta=0, theta=0)
    model_no_flow = SLBSModel(params_no_flow)
    total = np.sum(y0)
    dydt_no_flow = model_no_flow.deterministic_ode(0, y0)

    print(f"\nMass conservation test (no inflow/outflow):")
    print(f"Sum of derivatives: {np.sum(dydt_no_flow):.6f} (should be 0)")

    return dydt


def test_node_rates():
    """Test node-level rate calculations."""
    print("\nTesting node-level rates...")

    params = SLBSParameters()
    model = SLBSModel(params)

    # Test susceptible node with infected neighbors
    neighbor_counts = {
        NodeState.LATENT: 2,
        NodeState.BREAKING_OUT: 3,
        NodeState.SUSCEPTIBLE: 5
    }

    rates = model.compute_node_rates(NodeState.SUSCEPTIBLE, neighbor_counts)
    print(f"Susceptible node rates: {rates}")

    # Test latent node
    rates = model.compute_node_rates(NodeState.LATENT, neighbor_counts)
    print(f"Latent node rates: {rates}")

    # Test breaking-out node
    rates = model.compute_node_rates(NodeState.BREAKING_OUT, neighbor_counts)
    print(f"Breaking-out node rates: {rates}")

    return rates


def test_reproduction_number():
    """Test R0 calculation."""
    print("\nTesting basic reproduction number...")

    params = SLBSParameters()
    model = SLBSModel(params)

    R0 = model.compute_basic_reproduction_number()
    print(f"R0 = {R0:.3f}")

    # Test with different parameters
    high_infection = SLBSParameters(beta1=0.1, beta2=0.2)
    model_high = SLBSModel(high_infection)
    R0_high = model_high.compute_basic_reproduction_number()
    print(f"R0 (high infection) = {R0_high:.3f}")

    high_cleaning = SLBSParameters(gamma1=0.5, gamma2=0.8)
    model_clean = SLBSModel(high_cleaning)
    R0_clean = model_clean.compute_basic_reproduction_number()
    print(f"R0 (high cleaning) = {R0_clean:.3f}")

    return R0


if __name__ == "__main__":
    """Run tests if this file is executed directly."""
    print("=" * 60)
    print("SLBS Model Implementation Tests")
    print("=" * 60)

    # Run all tests
    test_deterministic_ode()
    test_node_rates()
    test_reproduction_number()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)