import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation, PillowWriter
import networkx as nx
from typing import List, Tuple, Dict, Any, Optional, Union
from dataclasses import dataclass
import seaborn as sns
from scipy import stats
import pandas as pd

from slbs_model import SLBSModel, NodeState, SLBSParameters
from network_generator import NetworkGenerator, NetworkParameters
from simulation_engine import SimulationEngine, SimulationState, SimulationParameters
from optimal_control import OptimalControlSolver, ControlSolution


@dataclass
class VisualizationParameters:
    figsize: Tuple[int, int] = (12, 8)
    dpi: int = 100
    color_map: Optional[Dict[NodeState, str]] = None
    style: str = 'seaborn-v0_8-darkgrid'
    save_format: str = 'png'
    animation_fps: int = 10

    def __post_init__(self):
        """Set default color map if not provided."""
        if self.color_map is None:
            self.color_map = {
                NodeState.SUSCEPTIBLE: 'green',
                NodeState.LATENT: 'yellow',
                NodeState.BREAKING_OUT: 'red'
            }


class SLBSVisualizer:
    def __init__(self, params: Optional[VisualizationParameters] = None):

        self.params = params if params is not None else VisualizationParameters()

        plt.style.use(self.params.style)

        self.color_map = self.params.color_map

        self.state_labels = {
            NodeState.SUSCEPTIBLE: 'Susceptible',
            NodeState.LATENT: 'Latent',
            NodeState.BREAKING_OUT: 'Breaking-out'
        }

    def plot_epidemic_curves(self,
                             time_points: np.ndarray,
                             state_data: Dict[str, np.ndarray],
                             title: str = "Epidemic Curves",
                             ax: Optional[plt.Axes] = None,
                             show_legend: bool = True,
                             alpha: float = 1.0) -> plt.Axes:

        if ax is None:
            fig, ax = plt.subplots(figsize=self.params.figsize, dpi=self.params.dpi)

        if 'S' in state_data:
            ax.plot(time_points, state_data['S'],
                    color=self.color_map[NodeState.SUSCEPTIBLE],
                    label='Susceptible (S)', linewidth=2, alpha=alpha)

        if 'L' in state_data:
            ax.plot(time_points, state_data['L'],
                    color=self.color_map[NodeState.LATENT],
                    label='Latent (L)', linewidth=2, alpha=alpha)

        if 'B' in state_data:
            ax.plot(time_points, state_data['B'],
                    color=self.color_map[NodeState.BREAKING_OUT],
                    label='Breaking-out (B)', linewidth=2, alpha=alpha)

        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Proportion / Count', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')

        if show_legend:
            ax.legend(loc='best', fontsize=10)

        ax.grid(True, alpha=0.3)
        ax.set_xlim([time_points[0], time_points[-1]])

        return ax

    def plot_multiple_runs(self,
                           all_histories: List[List[SimulationState]],
                           interval: float = 1.0,
                           title: str = "Multiple Simulation Runs",
                           show_confidence: bool = True,
                           confidence_alpha: float = 0.3) -> plt.Figure:

        if not all_histories:
            raise ValueError("No simulation data provided")

        max_time = max(max(state.time for state in history) for history in all_histories)
        time_points = np.arange(0, max_time + interval, interval)

        S_series, L_series, B_series = [], [], []

        for history in all_histories:
            times = [state.time for state in history]
            S_counts = [state.counts[NodeState.SUSCEPTIBLE] for state in history]
            L_counts = [state.counts[NodeState.LATENT] for state in history]
            B_counts = [state.counts[NodeState.BREAKING_OUT] for state in history]

            S_interp = np.interp(time_points, times, S_counts,
                                 left=S_counts[0], right=S_counts[-1])
            L_interp = np.interp(time_points, times, L_counts,
                                 left=L_counts[0], right=L_counts[-1])
            B_interp = np.interp(time_points, times, B_counts,
                                 left=B_counts[0], right=B_counts[-1])

            S_series.append(S_interp)
            L_series.append(L_interp)
            B_series.append(B_interp)

        S_series = np.array(S_series)
        L_series = np.array(L_series)
        B_series = np.array(B_series)

        S_mean, S_std = np.mean(S_series, axis=0), np.std(S_series, axis=0)
        L_mean, L_std = np.mean(L_series, axis=0), np.std(L_series, axis=0)
        B_mean, B_std = np.mean(B_series, axis=0), np.std(B_series, axis=0)

        fig, axes = plt.subplots(3, 1, figsize=(self.params.figsize[0],
                                                self.params.figsize[1] * 1.5),
                                 sharex=True)

        compartments = [
            ('S', S_mean, S_std, NodeState.SUSCEPTIBLE),
            ('L', L_mean, L_std, NodeState.LATENT),
            ('B', B_mean, B_std, NodeState.BREAKING_OUT)
        ]

        for idx, (comp_name, mean, std, state) in enumerate(compartments):
            ax = axes[idx]

            ax.plot(time_points, mean, color=self.color_map[state],
                    linewidth=2, label=f'{comp_name} (mean)')

            if show_confidence and len(all_histories) > 1:
                ax.fill_between(time_points, mean - std, mean + std,
                                color=self.color_map[state], alpha=confidence_alpha,
                                label=f'{comp_name} (±1 std)')

            n_individual_runs = min(3, len(all_histories))
            for run_idx in range(n_individual_runs):
                alpha_ind = 0.2 if run_idx > 0 else 0.4
                if comp_name == 'S':
                    ax.plot(time_points, S_series[run_idx],
                            color=self.color_map[state], alpha=alpha_ind,
                            linewidth=0.5)

            ax.set_ylabel(f'{comp_name} Count', fontsize=10)
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)

            ax.set_ylim(bottom=0)

        axes[-1].set_xlabel('Time', fontsize=12)
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        return fig

    def plot_degree_distribution(self,
                                 network_generator: NetworkGenerator,
                                 log_log: bool = True,
                                 title: str = "Degree Distribution",
                                 ax: Optional[plt.Axes] = None,
                                 fit_power_law: bool = True) -> plt.Axes:
        if ax is None:
            fig, ax = plt.subplots(figsize=self.params.figsize, dpi=self.params.dpi)

        degrees, counts = network_generator.get_degree_distribution()

        if len(degrees) == 0:
            ax.text(0.5, 0.5, "No degree data available",
                    ha='center', va='center', transform=ax.transAxes)
            return ax

        if log_log:
            mask = (degrees > 0) & (counts > 0)
            degrees_plot = degrees[mask]
            counts_plot = counts[mask]
        else:
            degrees_plot = degrees
            counts_plot = counts

        ax.scatter(degrees_plot, counts_plot, alpha=0.7, s=30,
                   color='blue', edgecolor='black', linewidth=0.5)

        if fit_power_law and len(degrees_plot) > 5 and log_log:
            try:

                log_degrees = np.log(degrees_plot)
                log_counts = np.log(counts_plot)

                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    log_degrees, log_counts
                )

                x_fit = np.linspace(log_degrees.min(), log_degrees.max(), 100)
                y_fit = slope * x_fit + intercept

                ax.plot(np.exp(x_fit), np.exp(y_fit), 'r--', linewidth=2,
                        label=f'Power law fit: γ = {-slope:.2f}\nR² = {r_value ** 2:.3f}')

                ax.legend(loc='best')

                title += f" (γ ≈ {-slope:.2f})"

            except Exception as e:
                print(f"Power law fitting failed: {e}")

        if log_log:
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('Degree k (log scale)', fontsize=12)
            ax.set_ylabel('P(k) (log scale)', fontsize=12)
        else:
            ax.set_xlabel('Degree k', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both' if log_log else 'major')

        stats_text = (
            f"N = {network_generator.get_network_size()}\n"
            f"Avg k = {network_generator.get_average_degree():.2f}\n"  # Changed from ⟨k⟩ to Avg k
            f"k_max = {network_generator.get_max_degree()}"
        )

        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        return ax

    def plot_network_state(self,
                           network_generator: NetworkGenerator,
                           node_states: List[NodeState],
                           title: str = "Network State",
                           highlight_hubs: bool = True,
                           max_nodes: int = 300,
                           node_size_base: int = 50,
                           ax: Optional[plt.Axes] = None) -> plt.Axes:

        if ax is None:
            fig, ax = plt.subplots(figsize=self.params.figsize, dpi=self.params.dpi)

        G = network_generator.to_networkx()

        if len(G.nodes()) > max_nodes:
            all_nodes = list(G.nodes())
            hubs = network_generator.get_hub_nodes(top_n=min(20, max_nodes // 3))
            other_nodes = [n for n in all_nodes if n not in hubs]

            sample_size = min(len(other_nodes), max_nodes - len(hubs))
            if sample_size > 0:
                sampled_other = np.random.choice(other_nodes, sample_size, replace=False)
                nodes_to_keep = list(hubs) + list(sampled_other)
            else:
                nodes_to_keep = list(hubs)[:max_nodes]

            G = G.subgraph(nodes_to_keep)

        pos = nx.spring_layout(G, seed=42, iterations=50)

        node_colors = []
        node_sizes = []
        node_labels = {}

        hub_nodes = set()
        if highlight_hubs:
            all_hubs = network_generator.get_hub_nodes(top_n=10)
            hub_nodes = set(all_hubs).intersection(set(G.nodes()))

        for node in G.nodes():
            node_idx = list(network_generator.node_ids).index(node) \
                if node in network_generator.node_ids else 0
            state = node_states[node_idx] if node_idx < len(node_states) \
                else NodeState.SUSCEPTIBLE

            node_colors.append(self.color_map[state])

            if node in hub_nodes:
                node_sizes.append(node_size_base * 3)
                node_labels[node] = str(node)
            else:
                node_sizes.append(node_size_base)

        nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                               node_size=node_sizes, alpha=0.8, ax=ax,
                               edgecolors='black', linewidths=0.5)

        nx.draw_networkx_edges(G, pos, alpha=0.15, width=0.3, ax=ax)

        if highlight_hubs and node_labels:
            nx.draw_networkx_labels(G, pos, labels=node_labels,
                                    font_size=8, ax=ax)

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=self.color_map[NodeState.SUSCEPTIBLE],
                  label='Susceptible (S)'),
            Patch(facecolor=self.color_map[NodeState.LATENT],
                  label='Latent (L)'),
            Patch(facecolor=self.color_map[NodeState.BREAKING_OUT],
                  label='Breaking-out (B)')
        ]

        if highlight_hubs:
            legend_elements.append(
                Patch(facecolor='white', edgecolor='black',
                      linewidth=2, label='Hub Node')
            )

        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')

        stats_text = f"Nodes: {len(G.nodes())}\nEdges: {G.number_of_edges()}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        return ax

    def plot_comparison_scenarios(self,
                                  scenarios: Dict[str, Dict[str, Any]],
                                  time_points: np.ndarray,
                                  compare_metric: str = 'B',  # 'S', 'L', or 'B'
                                  title: str = "Scenario Comparison",
                                  ax: Optional[plt.Axes] = None) -> plt.Axes:

        if ax is None:
            fig, ax = plt.subplots(figsize=self.params.figsize, dpi=self.params.dpi)

        colors = plt.cm.Set3(np.linspace(0, 1, len(scenarios)))

        for idx, (scenario_name, scenario_data) in enumerate(scenarios.items()):
            if compare_metric in scenario_data:
                ax.plot(time_points, scenario_data[compare_metric],
                        color=colors[idx], linewidth=2,
                        label=f"{scenario_name}", alpha=0.8)

        ax.set_xlabel('Time', fontsize=12)

        metric_labels = {
            'S': 'Susceptible Proportion',
            'L': 'Latent Proportion',
            'B': 'Breaking-out Proportion'
        }
        ax.set_ylabel(metric_labels.get(compare_metric, compare_metric), fontsize=12)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        return ax

    def plot_delay_effects(self,
                           delay_values: List[float],
                           final_infections: List[float],
                           title: str = "Effect of Cleaning Delay",
                           ax: Optional[plt.Axes] = None) -> plt.Axes:

        if ax is None:
            fig, ax = plt.subplots(figsize=self.params.figsize, dpi=self.params.dpi)

        ax.plot(delay_values, final_infections, 'bo-', linewidth=2,
                markersize=8, markerfacecolor='red')

        ax.set_xlabel('Cleaning Delay τ', fontsize=12)
        ax.set_ylabel('Final Breaking-out Proportion', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')

        if len(delay_values) > 1:
            diffs = np.diff(final_infections)
            if len(diffs) > 0:
                max_diff_idx = np.argmax(np.abs(diffs))
                if max_diff_idx < len(delay_values) - 1:
                    critical_delay = delay_values[max_diff_idx + 1]
                    ax.axvline(x=critical_delay, color='red', linestyle='--',
                               alpha=0.7, label=f'Critical delay ≈ {critical_delay:.2f}')
                    ax.legend(loc='best')

        ax.grid(True, alpha=0.3)
        ax.set_xlim([min(delay_values), max(delay_values)])

        return ax

    def plot_optimal_control_results(self,
                                     control_solution: ControlSolution,
                                     no_control_state: Optional[np.ndarray] = None,
                                     title: str = "Optimal Control Results",
                                     figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:

        if figsize is None:
            figsize = (self.params.figsize[0], self.params.figsize[1] * 1.2)

        fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=self.params.dpi)
        axes = axes.flatten()

        time_points = control_solution.time_points

        ax = axes[0]
        ax.plot(time_points, control_solution.state_trajectory[0],
                'b-', linewidth=2, label='S (with control)')
        ax.plot(time_points, control_solution.state_trajectory[1],
                'g-', linewidth=2, label='L (with control)')
        ax.plot(time_points, control_solution.state_trajectory[2],
                'r-', linewidth=2, label='B (with control)')

        if no_control_state is not None:
            ax.plot(time_points, no_control_state[0], 'b--', alpha=0.5,
                    linewidth=1, label='S (no control)')
            ax.plot(time_points, no_control_state[1], 'g--', alpha=0.5,
                    linewidth=1, label='L (no control)')
            ax.plot(time_points, no_control_state[2], 'r--', alpha=0.5,
                    linewidth=1, label='B (no control)')

        ax.set_xlabel('Time')
        ax.set_ylabel('Proportion')
        ax.set_title('State Trajectories')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(time_points, control_solution.control_trajectory[0],
                'b-', linewidth=2, label='u1 (Preventive)')
        ax.plot(time_points, control_solution.control_trajectory[1],
                'g-', linewidth=2, label='u2 (Reactive Latent)')
        ax.plot(time_points, control_solution.control_trajectory[2],
                'r-', linewidth=2, label='u3 (Reactive Active)')

        ax.set_xlabel('Time')
        ax.set_ylabel('Control Effort')
        ax.set_title('Optimal Control Trajectories')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[2]
        scatter = ax.scatter(control_solution.state_trajectory[0],
                             control_solution.control_trajectory[0],
                             c=time_points, cmap='viridis', alpha=0.6,
                             s=20)
        ax.set_xlabel('Susceptible Proportion (S)')
        ax.set_ylabel('Preventive Control (u1)')
        ax.set_title('Control Policy: u1 vs S')
        ax.grid(True, alpha=0.3)

        plt.colorbar(scatter, ax=ax, label='Time')

        ax = axes[3]
        cost_breakdown = control_solution.compute_total_cost_breakdown()

        cost_labels = ['State Cost', 'Control Cost']
        cost_values = [
            cost_breakdown['total_state_cost'],
            cost_breakdown['total_control_cost']
        ]

        wedges, texts, autotexts = ax.pie(cost_values, labels=cost_labels,
                                          autopct='%1.1f%%', startangle=90,
                                          colors=['lightblue', 'lightcoral'])

        ax.set_title('Cost Breakdown')

        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        return fig

    def create_network_animation(self,
                                 network_generator: NetworkGenerator,
                                 state_history: List[List[NodeState]],
                                 time_points: List[float],
                                 output_file: str = 'network_animation.gif',
                                 interval: int = 200,
                                 max_frames: int = 50) -> None:
        n_frames = min(len(state_history), max_frames)
        step = max(1, len(state_history) // n_frames)

        sampled_indices = range(0, len(state_history), step)
        sampled_states = [state_history[i] for i in sampled_indices]
        sampled_times = [time_points[i] for i in sampled_indices]

        G_full = network_generator.to_networkx()

        if len(G_full.nodes()) > 200:
            all_nodes = list(G_full.nodes())
            hubs = network_generator.get_hub_nodes(top_n=30)
            other_nodes = [n for n in all_nodes if n not in hubs]
            sampled_other = np.random.choice(other_nodes,
                                             min(170, len(other_nodes)),
                                             replace=False)
            nodes_to_keep = list(hubs) + list(sampled_other)
            G = G_full.subgraph(nodes_to_keep)
        else:
            G = G_full

        pos = nx.spring_layout(G, seed=42)

        fig, ax = plt.subplots(figsize=(10, 8), dpi=100)

        def update_frame(frame_idx):
            ax.clear()

            current_states = sampled_states[frame_idx]
            current_time = sampled_times[frame_idx]

            node_colors = []
            for node in G.nodes():
                node_idx = list(network_generator.node_ids).index(node) \
                    if node in network_generator.node_ids else 0
                state = current_states[node_idx] if node_idx < len(current_states) \
                    else NodeState.SUSCEPTIBLE
                node_colors.append(self.color_map[state])

            nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                                   node_size=100, alpha=0.8, ax=ax)
            nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.5, ax=ax)

            ax.set_title(f'Time: {current_time:.1f}', fontsize=14, fontweight='bold')
            ax.text(0.02, 0.98, f'Frame: {frame_idx + 1}/{len(sampled_states)}',
                    transform=ax.transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax.axis('off')

            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=self.color_map[NodeState.SUSCEPTIBLE],
                      label='S'),
                Patch(facecolor=self.color_map[NodeState.LATENT],
                      label='L'),
                Patch(facecolor=self.color_map[NodeState.BREAKING_OUT],
                      label='B')
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

        anim = FuncAnimation(fig, update_frame, frames=len(sampled_states),
                             interval=interval, repeat=True)

        print(f"Saving animation to {output_file}...")
        anim.save(output_file, writer=PillowWriter(fps=self.params.animation_fps))
        print("Animation saved!")

        plt.close(fig)

    def plot_summary_dashboard(self,
                               simulation_engine: SimulationEngine,
                               network_generator: NetworkGenerator,
                               title: str = "SLBS Model Dashboard") -> plt.Figure:

        fig = plt.figure(figsize=(16, 12), dpi=self.params.dpi)

        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, :2])
        if simulation_engine.state_counts_history:
            df = simulation_engine.get_results_dataframe()
            time_points = df['time'].values
            state_data = {
                'S': df['S'].values,
                'L': df['L'].values,
                'B': df['B'].values
            }
            self.plot_epidemic_curves(time_points, state_data,
                                      title="Epidemic Evolution",
                                      ax=ax1)

        ax2 = fig.add_subplot(gs[0, 2])
        self.plot_degree_distribution(network_generator,
                                      title="Network Degree Distribution",
                                      ax=ax2)

        ax3 = fig.add_subplot(gs[1, 0])
        if simulation_engine.node_states:
            self.plot_network_state(network_generator,
                                    simulation_engine.node_states,
                                    title="Current Network State",
                                    ax=ax3,
                                    max_nodes=150)

        ax4 = fig.add_subplot(gs[1, 1])
        if simulation_engine.event_counts:
            event_stats = simulation_engine.get_event_statistics()
            event_types = list(event_stats['event_distribution'].keys())
            event_counts = list(event_stats['event_distribution'].values())

            if event_types:
                bars = ax4.bar(range(len(event_types)), event_counts,
                               color=plt.cm.Set3(np.linspace(0, 1, len(event_types))))
                ax4.set_xlabel('Event Type')
                ax4.set_ylabel('Count')
                ax4.set_title('Event Distribution')
                ax4.set_xticks(range(len(event_types)))
                ax4.set_xticklabels(event_types, rotation=45, ha='right')

                for bar, count in zip(bars, event_counts):
                    height = bar.get_height()
                    ax4.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                             f'{count}', ha='center', va='bottom', fontsize=8)

        ax5 = fig.add_subplot(gs[1, 2])
        ax5.axis('off')

        stats_text = (
            "NETWORK STATISTICS\n"
            f"Active Nodes: {network_generator.get_network_size()}\n"
            f"Total Edges: {network_generator.get_total_edges()}\n"
            f"Average Degree: {network_generator.get_average_degree():.2f}\n"
            f"Maximum Degree: {network_generator.get_max_degree()}\n"
            f"Hubs (top 5%): {len(network_generator.get_hub_nodes())}\n\n"
            "SIMULATION STATISTICS\n"
        )

        if simulation_engine.state_counts_history:
            final_counts = simulation_engine.state_counts_history[-1]
            stats_text += (
                f"Final S: {final_counts['S']}\n"
                f"Final L: {final_counts['L']}\n"
                f"Final B: {final_counts['B']}\n"
                f"Total Events: {simulation_engine.total_events}\n"
                f"Final Time: {simulation_engine.current_time:.1f}"
            )

        ax5.text(0.1, 0.5, stats_text, fontsize=10,
                 verticalalignment='center',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        ax6 = fig.add_subplot(gs[2, 0])
        try:
            model = SLBSModel()
            R0 = model.compute_basic_reproduction_number()

            ax6.text(0.5, 0.7, f"Basic Reproduction Number",
                     ha='center', va='center', fontsize=12, fontweight='bold')
            ax6.text(0.5, 0.5, f"R₀ = {R0:.3f}",
                     ha='center', va='center', fontsize=24, color='red')

            if R0 < 1:
                interpretation = "Epidemic will die out"
                color = 'green'
            elif R0 < 2:
                interpretation = "Moderate spread"
                color = 'orange'
            else:
                interpretation = "Severe epidemic"
                color = 'red'

            ax6.text(0.5, 0.3, interpretation,
                     ha='center', va='center', fontsize=10, color=color)

        except Exception as e:
            ax6.text(0.5, 0.5, f"R0 calculation error:\n{e}",
                     ha='center', va='center', fontsize=10)

        ax6.set_xlim([0, 1])
        ax6.set_ylim([0, 1])
        ax6.axis('off')

        ax7 = fig.add_subplot(gs[2, 1:])
        ax7.axis('off')

        param_text = "MODEL PARAMETERS\n"

        try:
            if hasattr(simulation_engine, 'slbs_model'):
                params = simulation_engine.slbs_model.params
                param_text += (
                    f"β₁: {params.beta1:.3f}  β₂: {params.beta2:.3f}\n"
                    f"γ₁: {params.gamma1:.3f}  γ₂: {params.gamma2:.3f}\n"
                    f"α: {params.alpha:.3f}  θ: {params.theta:.3f}\n"
                    f"δ: {params.delta:.3f}  τ: {params.tau:.1f}\n"
                    f"μ₁: {params.mu1:.1f}  μ₂: {params.mu2:.1f}\n"
                )
        except:
            param_text += "SLBS parameters not available\n"

        if hasattr(simulation_engine, 'params'):
            sim_params = simulation_engine.params
            param_text += (
                f"\nSIMULATION PARAMETERS\n"
                f"Final time: {sim_params.final_time:.1f}\n"
                f"Record interval: {sim_params.record_interval:.1f}\n"
                f"Random seed: {sim_params.random_seed or 'None'}"
            )

        ax7.text(0.1, 0.5, param_text, fontsize=9,
                 verticalalignment='center',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()

        return fig


def test_visualization():
    viz = SLBSVisualizer()

    print("\n1. Testing epidemic curves...")
    time_points = np.linspace(0, 100, 100)
    state_data = {
        'S': 0.8 + 0.2 * np.exp(-0.05 * time_points),
        'L': 0.1 * (1 - np.exp(-0.1 * time_points)),
        'B': 0.1 * np.exp(-0.02 * time_points)
    }

    fig, ax = plt.subplots(figsize=viz.params.figsize)
    viz.plot_epidemic_curves(time_points, state_data,
                             title="Test Epidemic Curves", ax=ax)
    plt.tight_layout()
    plt.show()

    print("\n2. Testing degree distribution plot...")

    np.random.seed(42)
    degrees = np.random.pareto(2.5, 1000) + 1
    degrees = degrees.astype(int)
    unique_degrees, counts = np.unique(degrees, return_counts=True)

    class MockNetwork:
        def get_degree_distribution(self):
            return unique_degrees, counts

        def get_network_size(self):
            return 1000

        def get_average_degree(self):
            return np.mean(degrees)

        def get_max_degree(self):
            return np.max(degrees)

        def get_hub_nodes(self, top_n=10):
            return list(range(top_n))

    mock_network = MockNetwork()

    fig, ax = plt.subplots(figsize=viz.params.figsize)
    viz.plot_degree_distribution(mock_network,
                                 title="Test Degree Distribution",
                                 ax=ax)
    plt.tight_layout()
    plt.show()

    print("\n3. Testing scenario comparison...")

    scenarios = {
        'No Delay (τ=0)': {
            'S': 0.8 + 0.2 * np.exp(-0.05 * time_points),
            'L': 0.1 * (1 - np.exp(-0.1 * time_points)),
            'B': 0.1 * np.exp(-0.02 * time_points)
        },
        'Small Delay (τ=5)': {
            'S': 0.7 + 0.3 * np.exp(-0.04 * time_points),
            'L': 0.15 * (1 - np.exp(-0.08 * time_points)),
            'B': 0.15 * np.exp(-0.015 * time_points)
        },
        'Large Delay (τ=20)': {
            'S': 0.6 + 0.4 * np.exp(-0.03 * time_points),
            'L': 0.2 * (1 - np.exp(-0.06 * time_points)),
            'B': 0.2 * np.exp(-0.01 * time_points)
        }
    }

    fig, ax = plt.subplots(figsize=viz.params.figsize)
    viz.plot_comparison_scenarios(scenarios, time_points,
                                  compare_metric='B',
                                  title="Effect of Cleaning Delay on Breaking-out Nodes",
                                  ax=ax)
    plt.tight_layout()
    plt.show()

    print("\n4. Testing delay effects plot...")

    delay_values = [0, 5, 10, 15, 20, 25, 30]
    final_infections = [0.05, 0.08, 0.12, 0.18, 0.25, 0.30, 0.35]

    fig, ax = plt.subplots(figsize=viz.params.figsize)
    viz.plot_delay_effects(delay_values, final_infections,
                           title="Effect of Cleaning Delay on Final Infection Level",
                           ax=ax)
    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 60)
    print("Visualization tests completed!")
    print("=" * 60)

    return viz


if __name__ == "__main__":
    viz = test_visualization()