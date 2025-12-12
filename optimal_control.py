import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from slbs_model import SLBSModel, SLBSParameters, NodeState


class ControlType(Enum):
    PREVENTIVE = "preventive"
    REACTIVE_LATENT = "reactive_latent"
    REACTIVE_ACTIVE = "reactive_active"


@dataclass
class ControlParameters:

    A1: float = 1.0
    A2: float = 5.0
    A3: float = 10.0


    B1: float = 0.1
    B2: float = 0.2
    B3: float = 0.5


    u_max: float = 1.0


    T: float = 100.0
    n_time_steps: int = 1000


    tolerance: float = 1e-4
    max_iterations: int = 100
    alpha: float = 0.1
    def __post_init__(self):
        if any(w < 0 for w in [self.A1, self.A2, self.A3, self.B1, self.B2, self.B3]):
            raise ValueError("All weights must be non-negative")
        if self.u_max <= 0:
            raise ValueError(f"u_max must be positive, got {self.u_max}")
        if self.T <= 0:
            raise ValueError(f"T must be positive, got {self.T}")
        if self.n_time_steps <= 0:
            raise ValueError(f"n_time_steps must be positive, got {self.n_time_steps}")
        if self.tolerance <= 0:
            raise ValueError(f"tolerance must be positive, got {self.tolerance}")
        if self.max_iterations <= 0:
            raise ValueError(f"max_iterations must be positive, got {self.max_iterations}")


@dataclass
class ControlSolution:

    time_points: np.ndarray
    state_trajectory: np.ndarray
    control_trajectory: np.ndarray
    adjoint_trajectory: np.ndarray
    objective_value: float
    convergence_info: Dict[str, Any]

    def get_state_at_time(self, t: float) -> np.ndarray:
        idx = np.searchsorted(self.time_points, t)
        if idx >= len(self.time_points):
            return self.state_trajectory[:, -1]
        elif idx == 0:
            return self.state_trajectory[:, 0]
        else:
            t0, t1 = self.time_points[idx-1], self.time_points[idx]
            alpha = (t - t0) / (t1 - t0)
            return (1-alpha) * self.state_trajectory[:, idx-1] + alpha * self.state_trajectory[:, idx]

    def get_control_at_time(self, t: float) -> np.ndarray:
        idx = np.searchsorted(self.time_points, t)
        if idx >= len(self.time_points):
            return self.control_trajectory[:, -1]
        elif idx == 0:
            return self.control_trajectory[:, 0]
        else:
            t0, t1 = self.time_points[idx-1], self.time_points[idx]
            alpha = (t - t0) / (t1 - t0)
            return (1-alpha) * self.control_trajectory[:, idx-1] + alpha * self.control_trajectory[:, idx]

    def compute_total_cost_breakdown(self) -> Dict[str, float]:
        dt = self.time_points[1] - self.time_points[0]

        # State cost
        S_cost = np.sum(self.state_trajectory[0, :]) * dt
        L_cost = np.sum(self.state_trajectory[1, :]) * dt
        B_cost = np.sum(self.state_trajectory[2, :]) * dt

        # Control cost
        u1_cost = np.sum(self.control_trajectory[0, :]**2) * dt
        u2_cost = np.sum(self.control_trajectory[1, :]**2) * dt
        u3_cost = np.sum(self.control_trajectory[2, :]**2) * dt

        return {
            'total_state_cost': S_cost + L_cost + B_cost,
            'total_control_cost': u1_cost + u2_cost + u3_cost,
            'S_cost': S_cost,
            'L_cost': L_cost,
            'B_cost': B_cost,
            'u1_cost': u1_cost,
            'u2_cost': u2_cost,
            'u3_cost': u3_cost
        }


class OptimalControlSolver:
    def __init__(self,
                 slbs_params: SLBSParameters,
                 control_params: Optional[ControlParameters] = None):
        self.slbs_params = slbs_params
        self.control_params = control_params if control_params is not None else ControlParameters()
        self.time_points = np.linspace(0, self.control_params.T, self.control_params.n_time_steps)
        self.dt = self.time_points[1] - self.time_points[0]


        self.state_traj = None
        self.control_traj = None
        self.adjoint_traj = None

    def state_equations(self, t: float, y: np.ndarray, u: np.ndarray) -> np.ndarray:
        S, L, B = y
        u1, u2, u3 = u
        p = self.slbs_params

        u1 = np.clip(u1, 0, self.control_params.u_max)
        u2 = np.clip(u2, 0, self.control_params.u_max)
        u3 = np.clip(u3, 0, self.control_params.u_max)

        dS_dt = (p.mu1
                 - (p.beta1 * L + p.beta2 * B) * S
                 + p.gamma1 * L
                 + p.gamma2 * B
                 - (p.delta + p.theta) * S
                 - u1 * S)

        dL_dt = (p.mu2
                 + (p.beta1 * L + p.beta2 * B) * S
                 - p.gamma1 * L
                 - (p.alpha + p.delta) * L
                 + p.theta * S
                 - u2 * L)

        dB_dt = (p.alpha * L
                 - p.gamma2 * B
                 - p.delta * B
                 - u3 * B)

        return np.array([dS_dt, dL_dt, dB_dt])

    def adjoint_equations(self, t: float, lambda_vec: np.ndarray,
                          y: np.ndarray, u: np.ndarray) -> np.ndarray:
        S, L, B = y
        u1, u2, u3 = u
        p = self.slbs_params
        cp = self.control_params

        u1 = np.clip(u1, 0, cp.u_max)
        u2 = np.clip(u2, 0, cp.u_max)
        u3 = np.clip(u3, 0, cp.u_max)

        lambda_S, lambda_L, lambda_B = lambda_vec

        dH_dS = (cp.A1
                 + lambda_S * (-(p.beta1*L + p.beta2*B) - (p.delta + p.theta) - u1)
                 + lambda_L * ((p.beta1*L + p.beta2*B) + p.theta))

        dH_dL = (cp.A2
                 + lambda_S * (-p.beta1*S + p.gamma1)
                 + lambda_L * (p.beta1*S - p.gamma1 - (p.alpha + p.delta) - u2)
                 + lambda_B * p.alpha)

        dH_dB = (cp.A3
                 + lambda_S * (-p.beta2*S + p.gamma2)
                 + lambda_L * (p.beta2*S)
                 + lambda_B * (-p.gamma2 - p.delta - u3))

        dlambda_S_dt = -dH_dS
        dlambda_L_dt = -dH_dL
        dlambda_B_dt = -dH_dB

        return np.array([dlambda_S_dt, dlambda_L_dt, dlambda_B_dt])

    def optimal_control(self, y: np.ndarray, lambda_vec: np.ndarray) -> np.ndarray:
        S, L, B = y
        lambda_S, lambda_L, lambda_B = lambda_vec
        cp = self.control_params
        u1_star = -lambda_S * S / (2 * cp.B1)
        u2_star = -lambda_L * L / (2 * cp.B2)
        u3_star = -lambda_B * B / (2 * cp.B3)

        u1_star = np.clip(u1_star, 0, cp.u_max)
        u2_star = np.clip(u2_star, 0, cp.u_max)
        u3_star = np.clip(u3_star, 0, cp.u_max)

        return np.array([u1_star, u2_star, u3_star])

    def forward_sweep(self, control_traj: np.ndarray,
                      initial_state: np.ndarray) -> np.ndarray:
        n_steps = len(self.time_points)
        state_traj = np.zeros((3, n_steps))
        state_traj[:, 0] = initial_state

        for i in range(n_steps - 1):
            t = self.time_points[i]
            y = state_traj[:, i]
            u = control_traj[:, i]


            dy_dt = self.state_equations(t, y, u)


            state_traj[:, i+1] = y + dy_dt * self.dt


            state_traj[:, i+1] = np.maximum(state_traj[:, i+1], 0)

        return state_traj

    def backward_sweep(self, state_traj: np.ndarray,
                       control_traj: np.ndarray) -> np.ndarray:

        n_steps = len(self.time_points)
        adjoint_traj = np.zeros((3, n_steps))


        adjoint_traj[:, -1] = np.zeros(3)
        for i in range(n_steps - 2, -1, -1):
            t = self.time_points[i]
            y = state_traj[:, i]
            u = control_traj[:, i]
            lambda_next = adjoint_traj[:, i+1]

            dlambda_dt = self.adjoint_equations(t, lambda_next, y, u)

            adjoint_traj[:, i] = lambda_next - dlambda_dt * self.dt

        return adjoint_traj

    def compute_objective(self, state_traj: np.ndarray,
                          control_traj: np.ndarray) -> float:
        cp = self.control_params
        state_cost = (cp.A1 * np.sum(state_traj[0, :]) +
                      cp.A2 * np.sum(state_traj[1, :]) +
                      cp.A3 * np.sum(state_traj[2, :]))

        control_cost = (cp.B1 * np.sum(control_traj[0, :]**2) +
                        cp.B2 * np.sum(control_traj[1, :]**2) +
                        cp.B3 * np.sum(control_traj[2, :]**2))

        return (state_cost + control_cost) * self.dt

    def solve(self, initial_state: Optional[np.ndarray] = None,
              initial_control: Optional[np.ndarray] = None) -> ControlSolution:
        cp = self.control_params
        n_steps = len(self.time_points)

        if initial_state is None:
            initial_state = np.array([0.9, 0.05, 0.05])

        if initial_control is None:

            control_traj = np.zeros((3, n_steps))
        else:
            control_traj = initial_control.copy()

        if np.sum(initial_state) > 0:
            initial_state = initial_state / np.sum(initial_state)

        prev_control = control_traj.copy()
        iteration = 0
        control_diff = float('inf')

        convergence_info = {
            'iterations': 0,
            'converged': False,
            'objective_history': [],
            'control_diff_history': []
        }

        print(f"\nStarting forward-backward sweep...")
        print(f"  Tolerance: {cp.tolerance}")
        print(f"  Max iterations: {cp.max_iterations}")

        while iteration < cp.max_iterations and control_diff > cp.tolerance:
            state_traj = self.forward_sweep(control_traj, initial_state)
            adjoint_traj = self.backward_sweep(state_traj, control_traj)
            new_control = np.zeros_like(control_traj)
            for i in range(n_steps):
                y = state_traj[:, i]
                lambda_vec = adjoint_traj[:, i]
                new_control[:, i] = self.optimal_control(y, lambda_vec)

            control_traj = cp.alpha * new_control + (1 - cp.alpha) * prev_control
            control_diff = np.max(np.abs(control_traj - prev_control))
            prev_control = control_traj.copy()

            J = self.compute_objective(state_traj, control_traj)

            convergence_info['objective_history'].append(J)
            convergence_info['control_diff_history'].append(control_diff)

            if iteration % 10 == 0:
                print(f"  Iteration {iteration}: J = {J:.6f}, Δu = {control_diff:.6f}")

            iteration += 1
        state_traj = self.forward_sweep(control_traj, initial_state)
        J_final = self.compute_objective(state_traj, control_traj)

        convergence_info['iterations'] = iteration
        convergence_info['converged'] = control_diff <= cp.tolerance
        convergence_info['final_objective'] = J_final
        convergence_info['final_control_diff'] = control_diff

        print(f"\nOptimization {'converged' if convergence_info['converged'] else 'did not converge'}")
        print(f"  Final objective: {J_final:.6f}")
        print(f"  Iterations: {iteration}")
        print(f"  Final control difference: {control_diff:.6f}")
        self.state_traj = state_traj
        self.control_traj = control_traj
        self.adjoint_traj = adjoint_traj

        return ControlSolution(
            time_points=self.time_points,
            state_trajectory=state_traj,
            control_trajectory=control_traj,
            adjoint_trajectory=adjoint_traj,
            objective_value=J_final,
            convergence_info=convergence_info
        )

    def plot_solution(self, solution: ControlSolution,
                      figsize: Tuple[int, int] = (15, 10)):
        fig, axes = plt.subplots(3, 2, figsize=figsize)

        # State trajectories
        axes[0, 0].plot(solution.time_points, solution.state_trajectory[0], 'b-', linewidth=2, label='S')
        axes[0, 0].plot(solution.time_points, solution.state_trajectory[1], 'g-', linewidth=2, label='L')
        axes[0, 0].plot(solution.time_points, solution.state_trajectory[2], 'r-', linewidth=2, label='B')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Proportion')
        axes[0, 0].set_title('State Trajectories (Optimal Control)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Control trajectories
        axes[0, 1].plot(solution.time_points, solution.control_trajectory[0], 'b-', linewidth=2, label='u1 (Preventive)')
        axes[0, 1].plot(solution.time_points, solution.control_trajectory[1], 'g-', linewidth=2, label='u2 (Reactive Latent)')
        axes[0, 1].plot(solution.time_points, solution.control_trajectory[2], 'r-', linewidth=2, label='u3 (Reactive Active)')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Control Effort')
        axes[0, 1].set_title('Optimal Control Trajectories')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([0, self.control_params.u_max * 1.1])

        # Adjoint trajectories
        axes[1, 0].plot(solution.time_points, solution.adjoint_trajectory[0], 'b-', linewidth=2, label='λ_S')
        axes[1, 0].plot(solution.time_points, solution.adjoint_trajectory[1], 'g-', linewidth=2, label='λ_L')
        axes[1, 0].plot(solution.time_points, solution.adjoint_trajectory[2], 'r-', linewidth=2, label='λ_B')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Adjoint Value')
        axes[1, 0].set_title('Adjoint (Costate) Trajectories')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(solution.state_trajectory[0], solution.control_trajectory[0], 'bo-', alpha=0.5)
        axes[1, 1].set_xlabel('S (Susceptible Proportion)')
        axes[1, 1].set_ylabel('u1 (Preventive Control)')
        axes[1, 1].set_title('Control Policy: u1 vs S')
        axes[1, 1].grid(True, alpha=0.3)

        if solution.convergence_info['objective_history']:
            axes[2, 0].plot(solution.convergence_info['objective_history'], 'b-', linewidth=2)
            axes[2, 0].set_xlabel('Iteration')
            axes[2, 0].set_ylabel('Objective Value')
            axes[2, 0].set_title('Convergence History (Objective)')
            axes[2, 0].grid(True, alpha=0.3)

            control_diff_values = np.array(solution.convergence_info['control_diff_history'])
            if len(control_diff_values) > 0:
                control_diff_values = np.maximum(control_diff_values, 1e-10)

                if np.any(control_diff_values > 0):
                    axes[2, 1].plot(control_diff_values, 'r-', linewidth=2)
                    axes[2, 1].set_xlabel('Iteration')
                    axes[2, 1].set_ylabel('Max Control Difference')
                    axes[2, 1].set_title('Convergence History (Control)')
                    axes[2, 1].set_yscale('log')
                    axes[2, 1].grid(True, alpha=0.3)
                else:
                    axes[2, 1].plot(control_diff_values, 'r-', linewidth=2)
                    axes[2, 1].set_xlabel('Iteration')
                    axes[2, 1].set_ylabel('Max Control Difference')
                    axes[2, 1].set_title('Convergence History (Control) - Linear Scale')
                    axes[2, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig, axes

    def compare_with_no_control(self, solution: ControlSolution,
                                initial_state: Optional[np.ndarray] = None):
        if initial_state is None:
            initial_state = solution.state_trajectory[:, 0]

        no_control = np.zeros((3, len(self.time_points)))
        state_no_control = self.forward_sweep(no_control, initial_state)
        J_no_control = self.compute_objective(state_no_control, no_control)

        improvement = 0.0
        if J_no_control > 0:
            improvement = ((J_no_control - solution.objective_value) / J_no_control) * 100
        final_state_control = solution.state_trajectory[:, -1]
        final_state_no_control = state_no_control[:, -1]

        return {
            'J_optimal': solution.objective_value,
            'J_no_control': J_no_control,
            'improvement_percent': improvement,
            'final_state_control': final_state_control,
            'final_state_no_control': final_state_no_control,
            'state_no_control': state_no_control,
            'control_no_control': no_control
        }

    def sensitivity_analysis(self, parameter: str, values: List[float],
                             initial_state: Optional[np.ndarray] = None) -> Dict[str, Any]:

        results = []

        print(f"\nSensitivity analysis for {parameter}:")

        for val in values:

            cp_dict = self.control_params.__dict__.copy()
            cp_dict[parameter] = val
            modified_cp = ControlParameters(**cp_dict)


            solver = OptimalControlSolver(self.slbs_params, modified_cp)


            solution = solver.solve(initial_state)

            cost_breakdown = solution.compute_total_cost_breakdown()
            results.append({
                'parameter_value': val,
                'objective': solution.objective_value,
                'final_S': solution.state_trajectory[0, -1],
                'final_L': solution.state_trajectory[1, -1],
                'final_B': solution.state_trajectory[2, -1],
                'avg_u1': np.mean(solution.control_trajectory[0]),
                'avg_u2': np.mean(solution.control_trajectory[1]),
                'avg_u3': np.mean(solution.control_trajectory[2]),
                'cost_breakdown': cost_breakdown
            })

            print(f"  {parameter}={val}: J={solution.objective_value:.4f}, "
                  f"Final B={solution.state_trajectory[2, -1]:.4f}")

        return {
            'parameter': parameter,
            'values': values,
            'results': results
        }


def test_optimal_control_simple():
    """Simplified test for optimal control solver."""
    print("=" * 60)
    print("Testing Optimal Control Solver (Simplified)")
    print("=" * 60)

    try:

        slbs_params = SLBSParameters(
            beta1=0.01, beta2=0.02,
            gamma1=0.1, gamma2=0.3,
            alpha=0.2, theta=0.02,
            delta=0.1, tau=0.0,
            mu1=4.0, mu2=2.0
        )

        control_params = ControlParameters(
            A1=1.0, A2=5.0, A3=10.0,
            B1=0.1, B2=0.2, B3=0.5,
            u_max=0.5,
            T=20.0,
            n_time_steps=100,
            max_iterations=20,
            tolerance=1e-3
        )


        solver = OptimalControlSolver(slbs_params, control_params)


        initial_state = np.array([0.9, 0.05, 0.05])

        print(f"\nSolving optimal control problem...")
        print(f"  Horizon: {control_params.T}")
        print(f"  Time steps: {control_params.n_time_steps}")


        solution = solver.solve(initial_state)


        print(f"\nResults:")
        print(f"  Objective value: {solution.objective_value:.4f}")
        print(f"  Converged: {solution.convergence_info['converged']}")
        print(f"  Iterations: {solution.convergence_info['iterations']}")

        try:
            fig, axes = solver.plot_solution(solution, figsize=(12, 8))
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"  Plotting error (may be due to non-interactive environment): {e}")

        return solver, solution

    except Exception as e:
        print(f"Error in test: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    solver, solution = test_optimal_control_simple()

    if solver is not None and solution is not None:
        print("\n" + "=" * 60)
        print("Optimal control test completed successfully!")
        print("=" * 60)