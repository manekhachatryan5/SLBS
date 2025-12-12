import numpy as np
import random
from typing import List, Tuple, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import IntEnum
import time
from collections import defaultdict
import pandas as pd

from slbs_model import SLBSModel, NodeState, SLBSParameters
from network_generator import NetworkGenerator, NetworkParameters


@dataclass
class SimulationParameters:
    final_time: float = 100.0
    random_seed: Optional[int] = None
    record_interval: float = 1.0
    max_events: int = 1000000
    track_network_history: bool = False
    track_node_history: bool = False
    verbose: bool = True

    def __post_init__(self):
        if self.final_time <= 0:
            raise ValueError(f"final_time must be positive, got {self.final_time}")
        if self.record_interval <= 0:
            raise ValueError(f"record_interval must be positive, got {self.record_interval}")
        if self.max_events <= 0:
            raise ValueError(f"max_events must be positive, got {self.max_events}")


@dataclass
class SimulationState:
    time: float
    node_states: List[NodeState]
    counts: Dict[NodeState, int]  # Count of nodes in each state
    fractions: Dict[NodeState, float]  # Fraction of nodes in each state
    network_stats: Optional[Dict[str, Any]] = None  # Network statistics
    event_counts: Optional[Dict[str, int]] = None  # Counts of different event types

    def to_dict(self) -> Dict[str, Any]:
        result = {
            'time': self.time,
            'S_count': self.counts[NodeState.SUSCEPTIBLE],
            'L_count': self.counts[NodeState.LATENT],
            'B_count': self.counts[NodeState.BREAKING_OUT],
            'S_frac': self.fractions[NodeState.SUSCEPTIBLE],
            'L_frac': self.fractions[NodeState.LATENT],
            'B_frac': self.fractions[NodeState.BREAKING_OUT],
        }

        if self.network_stats:
            result.update(self.network_stats)

        if self.event_counts:
            result.update({f'event_{k}': v for k, v in self.event_counts.items()})

        return result


class SimulationEngine:
    def __init__(self,
                 slbs_model: SLBSModel,
                 network_generator: NetworkGenerator,
                 sim_params: Optional[SimulationParameters] = None):

        self.slbs_model = slbs_model
        self.network = network_generator
        self.params = sim_params if sim_params is not None else SimulationParameters()

        if self.params.random_seed is not None:
            np.random.seed(self.params.random_seed)
            random.seed(self.params.random_seed)

        self.current_time = 0.0
        self.node_states = []
        self.total_events = 0
        self.event_history = []
        self.state_history = []
        self.network_history = []


        self.event_counts = defaultdict(int)
        self.state_counts_history = []
        self.state_fractions_history = []
        self.start_time = None
        self.end_time = None
        self._initialize_node_states()

    def _initialize_node_states(self):

        initial_infected = 0.01
        self.node_states = self.slbs_model.create_initial_state(
            self.network.get_network_size(),
            initial_infected
        )

        if len(self.node_states) != self.network.get_network_size():
            current_size = len(self.node_states)
            target_size = self.network.get_network_size()

            if current_size < target_size:
                self.node_states.extend([NodeState.SUSCEPTIBLE] *
                                        (target_size - current_size))
            else:
                self.node_states = self.node_states[:target_size]

        self._record_state()

    def _record_state(self):
        counts = self._count_states()
        total_nodes = len(self.node_states)

        fractions = {
            state: count / total_nodes if total_nodes > 0 else 0.0
            for state, count in counts.items()
        }

        network_stats = None
        if self.params.track_network_history:
            network_stats = {
                'network_size': self.network.get_network_size(),
                'total_edges': self.network.get_total_edges(),
                'avg_degree': self.network.get_average_degree(),
                'max_degree': self.network.get_max_degree(),
            }

        state = SimulationState(
            time=self.current_time,
            node_states=self.node_states.copy(),
            counts=counts,
            fractions=fractions,
            network_stats=network_stats,
            event_counts=dict(self.event_counts) if self.event_counts else None
        )

        self.state_history.append(state)

        self.state_counts_history.append({
            'time': self.current_time,
            'S': counts[NodeState.SUSCEPTIBLE],
            'L': counts[NodeState.LATENT],
            'B': counts[NodeState.BREAKING_OUT]
        })

        self.state_fractions_history.append({
            'time': self.current_time,
            'S': fractions[NodeState.SUSCEPTIBLE],
            'L': fractions[NodeState.LATENT],
            'B': fractions[NodeState.BREAKING_OUT]
        })

    def _count_states(self) -> Dict[NodeState, int]:
        counts = {
            NodeState.SUSCEPTIBLE: 0,
            NodeState.LATENT: 0,
            NodeState.BREAKING_OUT: 0
        }

        for state in self.node_states:
            counts[state] += 1

        return counts

    def _get_all_event_rates(self) -> Tuple[List[float], List[Dict[str, Any]]]:

        adjacency_list = []
        node_id_map = {}

        active_nodes = self.network.node_ids
        for idx, network_node_id in enumerate(active_nodes):
            neighbors = self.network.get_neighbors(network_node_id)
            neighbor_indices = []
            for neighbor_id in neighbors:
                if neighbor_id in active_nodes:
                    neighbor_idx = active_nodes.index(neighbor_id)
                    neighbor_indices.append(neighbor_idx)
            adjacency_list.append(neighbor_indices)
            node_id_map[idx] = network_node_id


        rates_list, event_info_list = self.slbs_model.get_all_event_rates(
            self.node_states,
            adjacency_list,
            self.slbs_model.params
        )

        for event_info in event_info_list:
            sim_idx = event_info['node_id']
            event_info['network_node_id'] = node_id_map.get(sim_idx, sim_idx)

        return rates_list, event_info_list

    def _select_event(self, rates_list: List[float],
                      event_info_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not rates_list:
            raise ValueError("No events available (rates_list is empty)")

        total_rate = sum(rates_list)

        if total_rate <= 0:
            return None

        r = random.random() * total_rate
        cumulative = 0.0

        for i, rate in enumerate(rates_list):
            cumulative += rate
            if cumulative >= r:
                return event_info_list[i]

        return event_info_list[-1]

    def _apply_event(self, event_info: Dict[str, Any]):

        event_type = event_info['event_type']
        node_idx = event_info['node_id']
        current_state = event_info['current_state']

        if event_type == 'infection':
            if current_state == NodeState.SUSCEPTIBLE:
                self.node_states[node_idx] = NodeState.LATENT
                self.event_counts['infection'] += 1
            else:
                print(f"Warning: Infection event for non-susceptible node {node_idx}")

        elif event_type == 'removable_media_infection':
            if current_state == NodeState.SUSCEPTIBLE:
                self.node_states[node_idx] = NodeState.LATENT
                self.event_counts['removable_media_infection'] += 1
            else:
                print(f"Warning: Removable media infection for non-susceptible node {node_idx}")

        elif event_type == 'activate':
            if current_state == NodeState.LATENT:
                self.node_states[node_idx] = NodeState.BREAKING_OUT
                self.event_counts['activation'] += 1
            else:
                print(f"Warning: Activation event for non-latent node {node_idx}")

        elif event_type == 'clean':
            if current_state == NodeState.LATENT:
                self.node_states[node_idx] = NodeState.SUSCEPTIBLE
                self.event_counts['clean_latent'] += 1
            elif current_state == NodeState.BREAKING_OUT:
                self.node_states[node_idx] = NodeState.SUSCEPTIBLE
                self.event_counts['clean_breaking'] += 1
            else:
                print(f"Warning: Clean event for non-infected node {node_idx}")

        elif event_type == 'disconnect':
            self.event_counts['disconnect'] += 1

        else:
            print(f"Warning: Unknown event type '{event_type}'")

    def _update_network_for_event(self, event_info: Dict[str, Any], delta_t: float):
        self.network.advance_time(delta_t)
        current_network_size = self.network.get_network_size()
        current_state_size = len(self.node_states)

        if current_network_size > current_state_size:
            new_nodes = current_network_size - current_state_size
            self.node_states.extend([NodeState.SUSCEPTIBLE] * new_nodes)

        elif current_network_size < current_state_size:
            self.node_states = self.node_states[:current_network_size]

    def run_single_simulation(self) -> List[SimulationState]:

        if self.params.verbose:
            print(f"\nStarting Gillespie simulation...")
            print(f"  Final time: {self.params.final_time}")
            print(f"  Initial network size: {len(self.node_states)}")
            print(f"  Initial infected: {self._count_states()[NodeState.BREAKING_OUT]}")

        self.start_time = time.time()
        self.current_time = 0.0
        self.total_events = 0
        self.event_counts.clear()

        self._record_state()
        last_record_time = 0.0

        while (self.current_time < self.params.final_time and
               self.total_events < self.params.max_events):

            rates_list, event_info_list = self._get_all_event_rates()

            total_rate = sum(rates_list)

            if total_rate <= 0:
                if self.params.verbose:
                    print("  No events possible, ending simulation early")
                break

            delta_t = np.random.exponential(1.0 / total_rate)

            if (self.current_time + delta_t >= last_record_time + self.params.record_interval or
                    self.current_time + delta_t >= self.params.final_time):


                record_time = min(
                    last_record_time + self.params.record_interval,
                    self.params.final_time
                )
                time_to_record = record_time - self.current_time
                if time_to_record > 0:
                    self.network.advance_time(time_to_record)
                    self.current_time = record_time
                    self._synchronize_network_state()
                    self._record_state()
                    last_record_time = record_time


            self.current_time += delta_t

            if self.current_time > self.params.final_time:
                self.current_time = self.params.final_time
                break

            event_info = self._select_event(rates_list, event_info_list)

            if event_info is None:
                continue
            self._apply_event(event_info)
            self._update_network_for_event(event_info, delta_t)
            self.event_history.append({
                'time': self.current_time,
                'event_type': event_info['event_type'],
                'node_id': event_info['node_id'],
                'network_node_id': event_info.get('network_node_id', event_info['node_id'])
            })

            self.total_events += 1
            if self.params.verbose and self.total_events % 1000 == 0:
                print(f"  Events: {self.total_events}, Time: {self.current_time:.1f}/{self.params.final_time}")

        if self.state_history[-1].time < self.current_time:
            self._record_state()

        self.end_time = time.time()

        if self.params.verbose:
            elapsed = self.end_time - self.start_time
            print(f"\nSimulation completed in {elapsed:.2f} seconds")
            print(f"  Total events: {self.total_events}")
            print(f"  Final time: {self.current_time:.2f}")
            print(f"  Final counts: S={self.state_counts_history[-1]['S']}, "
                  f"L={self.state_counts_history[-1]['L']}, "
                  f"B={self.state_counts_history[-1]['B']}")
            print(f"  Event distribution: {dict(self.event_counts)}")

        return self.state_history

    def _synchronize_network_state(self):
        current_network_size = self.network.get_network_size()
        current_state_size = len(self.node_states)

        if current_network_size > current_state_size:

            new_nodes = current_network_size - current_state_size
            self.node_states.extend([NodeState.SUSCEPTIBLE] * new_nodes)

        elif current_network_size < current_state_size:
            # Remove nodes from end (simplistic approach)
            # In a more advanced implementation, we'd track which nodes were removed
            self.node_states = self.node_states[:current_network_size]

    def run_multiple_simulations(self, n_runs: int = 5) -> List[List[SimulationState]]:
        if self.params.verbose:
            print(f"\nRunning {n_runs} independent simulations...")

        all_histories = []

        for run_idx in range(n_runs):
            if self.params.verbose:
                print(f"\nRun {run_idx + 1}/{n_runs}")
            self.current_time = 0.0
            self.total_events = 0
            self.event_history = []
            self.state_history = []
            self.state_counts_history = []
            self.state_fractions_history = []
            self.event_counts.clear()
            random_seed = (self.params.random_seed + run_idx if self.params.random_seed
                           else None)
            if random_seed is not None:
                np.random.seed(random_seed)
                random.seed(random_seed)

            self._initialize_node_states()
            history = self.run_single_simulation()
            all_histories.append(history)

        return all_histories

    def get_results_dataframe(self) -> pd.DataFrame:
        if not self.state_counts_history:
            raise ValueError("No simulation data available. Run simulation first.")

        return pd.DataFrame(self.state_counts_history)

    def get_fractions_dataframe(self) -> pd.DataFrame:
        if not self.state_fractions_history:
            raise ValueError("No simulation data available. Run simulation first.")

        return pd.DataFrame(self.state_fractions_history)

    def get_event_statistics(self) -> Dict[str, Any]:
        total_events = sum(self.event_counts.values())

        if total_events == 0:
            return {
                'total_events': 0,
                'event_distribution': {},
                'event_rates': {}
            }
        if self.current_time > 0:
            event_rates = {
                event_type: count / self.current_time
                for event_type, count in self.event_counts.items()
            }
        else:
            event_rates = {event_type: 0.0 for event_type in self.event_counts.keys()}

        return {
            'total_events': total_events,
            'event_distribution': dict(self.event_counts),
            'event_rates': event_rates,
            'simulation_time': self.current_time,
            'events_per_time': total_events / self.current_time if self.current_time > 0 else 0.0
        }

    def get_network_statistics(self) -> Dict[str, Any]:
        return {
            'network_size': self.network.get_network_size(),
            'total_edges': self.network.get_total_edges(),
            'average_degree': self.network.get_average_degree(),
            'maximum_degree': self.network.get_max_degree(),
            'hub_nodes': len(self.network.get_hub_nodes()),
            'top_hubs': self.network.get_hub_nodes(top_n=5)
        }

    def reset(self):
        self.current_time = 0.0
        self.total_events = 0
        self.event_history = []
        self.state_history = []
        self.state_counts_history = []
        self.state_fractions_history = []
        self.event_counts.clear()
        self._initialize_node_states()

def aggregate_simulation_results(all_histories: List[List[SimulationState]],
                                 interval: float = 1.0) -> Dict[str, Any]:
    if not all_histories:
        return {}

    max_time = max(max(state.time for state in history) for history in all_histories)
    time_points = np.arange(0, max_time + interval, interval)

    S_series = []
    L_series = []
    B_series = []
    S_frac_series = []
    L_frac_series = []
    B_frac_series = []

    for history in all_histories:
        times = [state.time for state in history]
        S_counts = [state.counts[NodeState.SUSCEPTIBLE] for state in history]
        L_counts = [state.counts[NodeState.LATENT] for state in history]
        B_counts = [state.counts[NodeState.BREAKING_OUT] for state in history]
        S_fracs = [state.fractions[NodeState.SUSCEPTIBLE] for state in history]
        L_fracs = [state.fractions[NodeState.LATENT] for state in history]
        B_fracs = [state.fractions[NodeState.BREAKING_OUT] for state in history]

        # Interpolate to common time grid
        S_interp = np.interp(time_points, times, S_counts, left=S_counts[0], right=S_counts[-1])
        L_interp = np.interp(time_points, times, L_counts, left=L_counts[0], right=L_counts[-1])
        B_interp = np.interp(time_points, times, B_counts, left=B_counts[0], right=B_counts[-1])
        S_frac_interp = np.interp(time_points, times, S_fracs, left=S_fracs[0], right=S_fracs[-1])
        L_frac_interp = np.interp(time_points, times, L_fracs, left=L_fracs[0], right=L_fracs[-1])
        B_frac_interp = np.interp(time_points, times, B_fracs, left=B_fracs[0], right=B_fracs[-1])

        S_series.append(S_interp)
        L_series.append(L_interp)
        B_series.append(B_interp)
        S_frac_series.append(S_frac_interp)
        L_frac_series.append(L_frac_interp)
        B_frac_series.append(B_frac_interp)

    # Calculate statistics across runs
    S_mean = np.mean(S_series, axis=0)
    L_mean = np.mean(L_series, axis=0)
    B_mean = np.mean(B_series, axis=0)
    S_std = np.std(S_series, axis=0)
    L_std = np.std(L_series, axis=0)
    B_std = np.std(B_series, axis=0)

    S_frac_mean = np.mean(S_frac_series, axis=0)
    L_frac_mean = np.mean(L_frac_series, axis=0)
    B_frac_mean = np.mean(B_frac_series, axis=0)
    S_frac_std = np.std(S_frac_series, axis=0)
    L_frac_std = np.std(L_frac_series, axis=0)
    B_frac_std = np.std(B_frac_series, axis=0)

    return {
        'time_points': time_points,
        'S_mean': S_mean, 'S_std': S_std,
        'L_mean': L_mean, 'L_std': L_std,
        'B_mean': B_mean, 'B_std': B_std,
        'S_frac_mean': S_frac_mean, 'S_frac_std': S_frac_std,
        'L_frac_mean': L_frac_mean, 'L_frac_std': L_frac_std,
        'B_frac_mean': B_frac_mean, 'B_frac_std': B_frac_std,
        'n_runs': len(all_histories)
    }

def test_simulation_engine():
    print("=" * 60)
    print("Testing Simulation Engine")
    print("=" * 60)

    slbs_params = SLBSParameters(
        beta1=0.01, beta2=0.02,
        gamma1=0.1, gamma2=0.3,
        alpha=0.2, theta=0.02,
        delta=0.1, tau=0.0,
        mu1=4.0, mu2=2.0
    )
    model = SLBSModel(slbs_params)

    network_params = NetworkParameters(
        n_nodes=200,
        m0=5,
        m=2,
        turnover_rate=0.01,
        seed=42
    )
    network = NetworkGenerator(network_params)

    sim_params = SimulationParameters(
        final_time=50.0,
        record_interval=2.0,
        random_seed=123,
        verbose=True
    )

    engine = SimulationEngine(model, network, sim_params)

    print("\nRunning single simulation...")
    history = engine.run_single_simulation()
    df = engine.get_results_dataframe()
    print(f"\nSimulation results shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())

    stats = engine.get_event_statistics()
    print(f"\nEvent statistics:")
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")

    print("\n" + "-" * 40)
    print("Testing multiple simulations...")

    engine.reset()

    n_runs = 3
    all_histories = engine.run_multiple_simulations(n_runs=n_runs)

    aggregated = aggregate_simulation_results(all_histories, interval=5.0)
    print(f"\nAggregated {n_runs} runs:")
    print(f"  Time points: {len(aggregated['time_points'])}")
    print(f"  Final S mean: {aggregated['S_mean'][-1]:.1f} ± {aggregated['S_std'][-1]:.1f}")
    print(f"  Final L mean: {aggregated['L_mean'][-1]:.1f} ± {aggregated['L_std'][-1]:.1f}")
    print(f"  Final B mean: {aggregated['B_mean'][-1]:.1f} ± {aggregated['B_std'][-1]:.1f}")

    return engine, aggregated


def test_performance():
    print("\n" + "=" * 60)
    print("Performance Testing")
    print("=" * 60)

    slbs_params = SLBSParameters()
    model = SLBSModel(slbs_params)

    network_sizes = [100, 500, 1000]

    for n_nodes in network_sizes:
        print(f"\nTesting with network size: {n_nodes}")

        network_params = NetworkParameters(
            n_nodes=n_nodes,
            m0=5,
            m=2,
            turnover_rate=0.0,
            seed=42
        )
        network = NetworkGenerator(network_params)

        sim_params = SimulationParameters(
            final_time=30.0,
            record_interval=5.0,
            verbose=False
        )

        engine = SimulationEngine(model, network, sim_params)

        start_time = time.time()
        engine.run_single_simulation()
        elapsed = time.time() - start_time

        print(f"  Time: {elapsed:.2f} seconds")
        print(f"  Events: {engine.total_events}")
        print(f"  Events/sec: {engine.total_events/elapsed:.0f}")


if __name__ == "__main__":
    test_simulation_engine()
    test_performance()

    print("\n" + "=" * 60)
    print("Simulation engine tests completed!")
    print("=" * 60)