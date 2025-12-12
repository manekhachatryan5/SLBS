import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import random


@dataclass
class NetworkParameters:
    n_nodes: int = 1000          # Target number of nodes
    m0: int = 5                  # Initial seed graph size
    m: int = 3                   # Edges per new node
    turnover_rate: float = 0.01  # Node turnover rate
    seed: Optional[int] = None   # Random seed

    def __post_init__(self):
        self.validate()

    def validate(self):
        if self.n_nodes <= 0:
            raise ValueError(f"n_nodes must be positive, got {self.n_nodes}")
        if self.m0 <= 0:
            raise ValueError(f"m0 must be positive, got {self.m0}")
        if self.m <= 0:
            raise ValueError(f"m must be positive, got {self.m}")
        if self.m > self.m0:
            raise ValueError(f"m must be <= m0, got m={self.m}, m0={self.m0}")
        if self.turnover_rate < 0:
            raise ValueError(f"turnover_rate must be non-negative, got {self.turnover_rate}")

    def copy(self):
        return NetworkParameters(
            n_nodes=self.n_nodes,
            m0=self.m0,
            m=self.m,
            turnover_rate=self.turnover_rate,
            seed=self.seed
        )


class NetworkGenerator:

    def __init__(self, params: Optional[NetworkParameters] = None):
        self.params = params if params is not None else NetworkParameters()

        if self.params.seed is not None:
            np.random.seed(self.params.seed)
            random.seed(self.params.seed)

        self.adjacency_list = []
        self.node_ids = []
        self.next_node_id = 0

        self.creation_times = {}
        self.removal_times = {}
        self.network_history = []


        self._current_time = 0.0
        self._degree_distribution = None
        self._hub_nodes = None

        self._generate_initial_network()

    def _generate_initial_network(self):
        params = self.params

        print(f"Generating initial Barabási-Albert network with {params.n_nodes} nodes...")

        self.adjacency_list = [[] for _ in range(params.m0)]
        self.node_ids = list(range(params.m0))
        self.next_node_id = params.m0

        for i in range(params.m0):
            for j in range(i + 1, params.m0):
                self.adjacency_list[i].append(j)
                self.adjacency_list[j].append(i)

        for node_id in self.node_ids:
            self.creation_times[node_id] = 0.0

        for new_node_id in range(params.m0, params.n_nodes):
            self._add_node_with_preferential_attachment(new_node_id)
            self.node_ids.append(new_node_id)
            self.creation_times[new_node_id] = 0.0

        print(f"Initial network generated with {len(self.node_ids)} nodes and {self.get_total_edges()} edges")
        print(f"Average degree: {self.get_average_degree():.2f}")
        print(f"Maximum degree: {self.get_max_degree()}")

        self._update_statistics()

    def _add_node_with_preferential_attachment(self, node_id: int):

        params = self.params


        if node_id >= len(self.adjacency_list):
            self.adjacency_list.extend([[] for _ in range(node_id - len(self.adjacency_list) + 1)])

        active_nodes = [nid for nid in self.node_ids if nid != node_id]

        if not active_nodes:
            return


        total_degree = sum(len(self.adjacency_list[nid]) for nid in active_nodes)

        if total_degree == 0:

            targets = random.sample(active_nodes, min(params.m, len(active_nodes)))
        else:

            targets = []
            for _ in range(min(params.m, len(active_nodes))):

                r = random.random() * total_degree
                cumulative = 0
                for target_id in active_nodes:
                    if target_id in targets:
                        continue
                    cumulative += len(self.adjacency_list[target_id])
                    if cumulative >= r:
                        targets.append(target_id)
                        break

        for target_id in targets:
            self.adjacency_list[node_id].append(target_id)
            self.adjacency_list[target_id].append(node_id)

    def _update_statistics(self):
        degrees = [self.get_node_degree(node_id) for node_id in self.node_ids]
        self._degree_distribution = degrees

        if degrees:
            sorted_indices = np.argsort(degrees)[::-1]
            hub_count = max(1, int(len(self.node_ids) * 0.05))
            self._hub_nodes = [self.node_ids[i] for i in sorted_indices[:hub_count]]
        else:
            self._hub_nodes = []

    def get_network_size(self) -> int:
        return len(self.node_ids)

    def get_total_edges(self) -> int:
        total = 0
        for neighbors in self.adjacency_list:
            total += len(neighbors)
        return total // 2

    def get_node_degree(self, node_id: int) -> int:
        if node_id < 0 or node_id >= len(self.adjacency_list):
            return 0
        return len(self.adjacency_list[node_id])

    def get_average_degree(self) -> float:
        if not self.node_ids:
            return 0.0
        total_degree = sum(self.get_node_degree(node_id) for node_id in self.node_ids)
        return total_degree / len(self.node_ids)

    def get_max_degree(self) -> int:
        if not self.node_ids:
            return 0
        return max(self.get_node_degree(node_id) for node_id in self.node_ids)

    def get_degree_distribution(self) -> Tuple[np.ndarray, np.ndarray]:
        degrees = self._degree_distribution
        if not degrees:
            return np.array([]), np.array([])

        unique_degrees, counts = np.unique(degrees, return_counts=True)
        return unique_degrees, counts

    def get_hub_nodes(self, top_n: Optional[int] = None) -> List[int]:
        if not self._hub_nodes:
            self._update_statistics()

        if top_n is None:
            return self._hub_nodes.copy()
        else:
            nodes_with_degrees = [(node_id, self.get_node_degree(node_id))
                                  for node_id in self.node_ids]
            nodes_with_degrees.sort(key=lambda x: x[1], reverse=True)
            return [node_id for node_id, _ in nodes_with_degrees[:top_n]]

    def apply_node_turnover(self, time_step: float = 1.0):
        params = self.params
        n_active = len(self.node_ids)

        if n_active == 0:
            return

        expected_removals = params.turnover_rate * time_step * n_active
        n_removals = np.random.poisson(expected_removals)


        for _ in range(min(n_removals, n_active)):
            if n_active == 0:
                break


            remove_index = random.randint(0, n_active - 1)
            node_to_remove = self.node_ids[remove_index]


            self._remove_node(node_to_remove)

            self.node_ids.pop(remove_index)
            n_active -= 1


            self.removal_times[node_to_remove] = self._current_time


        n_additions = np.random.poisson(expected_removals)

        for _ in range(n_additions):
            new_node_id = self.next_node_id
            self.next_node_id += 1

            self._add_node_with_preferential_attachment(new_node_id)
            self.node_ids.append(new_node_id)
            self.creation_times[new_node_id] = self._current_time

        self._update_statistics()
        if hasattr(self, 'record_history') and self.record_history:
            snapshot = {
                'time': self._current_time,
                'n_nodes': len(self.node_ids),
                'n_edges': self.get_total_edges(),
                'avg_degree': self.get_average_degree(),
                'max_degree': self.get_max_degree()
            }
            self.network_history.append(snapshot)

    def _remove_node(self, node_id: int):
        for neighbor in self.adjacency_list[node_id]:
            if neighbor in self.node_ids:
                try:
                    self.adjacency_list[neighbor].remove(node_id)
                except ValueError:
                    pass

        self.adjacency_list[node_id] = []

    def advance_time(self, delta_t: float = 1.0):
        self._current_time += delta_t
        self.apply_node_turnover(delta_t)

    def get_neighbors(self, node_id: int) -> List[int]:
        if node_id < 0 or node_id >= len(self.adjacency_list):
            return []
        return self.adjacency_list[node_id].copy()

    def get_adjacency_matrix(self, sparse: bool = True):
        n_nodes = len(self.adjacency_list)

        if sparse:
            from scipy.sparse import lil_matrix
            adj = lil_matrix((n_nodes, n_nodes), dtype=int)
            for i in range(n_nodes):
                for j in self.adjacency_list[i]:
                    adj[i, j] = 1
            return adj.tocsr()
        else:
            adj = np.zeros((n_nodes, n_nodes), dtype=int)
            for i in range(n_nodes):
                for j in self.adjacency_list[i]:
                    adj[i, j] = 1
            return adj

    def to_networkx(self) -> nx.Graph:
        G = nx.Graph()
        for node_id in self.node_ids:
            G.add_node(node_id)
        for node_id in self.node_ids:
            for neighbor in self.adjacency_list[node_id]:
                if neighbor > node_id:  # Add edge only once
                    G.add_edge(node_id, neighbor)

        return G

    def plot_degree_distribution(self, log_log: bool = True, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        degrees, counts = self.get_degree_distribution()

        if len(degrees) == 0:
            ax.text(0.5, 0.5, "No data to plot",
                    ha='center', va='center', transform=ax.transAxes)
            return ax

        if log_log:
            mask = (degrees > 0) & (counts > 0)
            if np.any(mask):
                ax.loglog(degrees[mask], counts[mask], 'bo-', alpha=0.7, linewidth=2)
                ax.set_xlabel('Degree k (log scale)')
                ax.set_ylabel('P(k) (log scale)')
            else:
                ax.text(0.5, 0.5, "No non-zero data for log-log plot",
                        ha='center', va='center', transform=ax.transAxes)
        else:
            ax.plot(degrees, counts, 'bo-', alpha=0.7, linewidth=2)
            ax.set_xlabel('Degree k')
            ax.set_ylabel('Count')

        ax.set_title(f'Degree Distribution (n={len(self.node_ids)})')
        ax.grid(True, alpha=0.3)

        return ax

    def plot_network(self, highlight_hubs: bool = True, max_nodes: int = 200,
                     node_size: int = 50, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))

        G = self.to_networkx()

        if len(G.nodes()) > max_nodes:
            nodes_to_keep = list(G.nodes())
            if len(nodes_to_keep) > max_nodes:
                hubs = self.get_hub_nodes(top_n=min(20, max_nodes // 2))
                other_nodes = [n for n in nodes_to_keep if n not in hubs]
                sample_size = min(len(other_nodes), max_nodes - len(hubs))
                sampled_other = random.sample(other_nodes, sample_size)
                nodes_to_keep = hubs + sampled_other


            G = G.subgraph(nodes_to_keep)

        pos = nx.spring_layout(G, seed=42)

        node_colors = []
        node_sizes = []

        hub_nodes = set(self.get_hub_nodes())

        for node in G.nodes():
            if highlight_hubs and node in hub_nodes:
                node_colors.append('red')

                node_sizes.append(node_size * 3)
            else:
                node_colors.append('skyblue')
                node_sizes.append(node_size)


        nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                               node_size=node_sizes, alpha=0.8, ax=ax)
        nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.5, ax=ax)

        if highlight_hubs:
            hub_labels = {node: str(node) for node in G.nodes() if node in hub_nodes}
            nx.draw_networkx_labels(G, pos, labels=hub_labels, font_size=8, ax=ax)

        ax.set_title(f'Network Structure (showing {len(G.nodes())} of {len(self.node_ids)} nodes)')
        ax.axis('off')

        return ax

    def print_statistics(self):
        print("\n" + "="*60)
        print("NETWORK STATISTICS")
        print("="*60)
        print(f"Active nodes: {len(self.node_ids)}")
        print(f"Total edges: {self.get_total_edges()}")
        print(f"Average degree: {self.get_average_degree():.2f}")
        print(f"Maximum degree: {self.get_max_degree()}")

        if self._hub_nodes:
            top_hubs = self.get_hub_nodes(top_n=5)
            print(f"\nTop 5 hub nodes (highest degree):")
            for hub in top_hubs:
                degree = self.get_node_degree(hub)
                print(f"  Node {hub}: degree = {degree}")

        degrees = self._degree_distribution
        if degrees:
            print(f"\nDegree distribution moments:")
            print(f"  Variance: {np.var(degrees):.2f}")
            print(f"  Skewness: {float(np.mean((degrees - np.mean(degrees))**3) / np.std(degrees)**3):.2f}")

        print("="*60)



def test_network_generation():
    print("Testing NetworkGenerator...")

    params = NetworkParameters(
        n_nodes=100,
        m0=5,
        m=2,
        turnover_rate=0.05,
        seed=42
    )

    generator = NetworkGenerator(params)

    generator.print_statistics()

    degrees, counts = generator.get_degree_distribution()
    print(f"\nDegree distribution shape: {degrees.shape}")

    hubs = generator.get_hub_nodes(top_n=3)
    print(f"\nTop 3 hubs: {hubs}")
    for hub in hubs:
        print(f"  Node {hub}: degree = {generator.get_node_degree(hub)}")

    test_node = hubs[0] if hubs else 0
    neighbors = generator.get_neighbors(test_node)
    print(f"\nNeighbors of node {test_node}: {neighbors[:5]}... (total: {len(neighbors)})")

    print("\nApplying node turnover...")
    initial_size = generator.get_network_size()
    generator.apply_node_turnover()
    new_size = generator.get_network_size()
    print(f"Network size: {initial_size} -> {new_size}")

    return generator


def test_visualization():
    print("\nTesting visualization...")

    params = NetworkParameters(
        n_nodes=50,
        m0=3,
        m=2,
        turnover_rate=0.0,
        seed=123
    )

    generator = NetworkGenerator(params)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    generator.plot_degree_distribution(log_log=True, ax=axes[0])
    generator.plot_network(highlight_hubs=True, max_nodes=50, ax=axes[1])
    plt.tight_layout()
    plt.show()
    return generator


def test_dynamic_network():
    print("\nTesting dynamic network evolution...")

    params = NetworkParameters(
        n_nodes=200,
        m0=5,
        m=2,
        turnover_rate=0.1,
        seed=456
    )

    generator = NetworkGenerator(params)

    time_points = []
    sizes = []
    avg_degrees = []

    print("Simulating network evolution over 10 time steps...")
    for t in range(10):
        generator.advance_time(delta_t=1.0)

        time_points.append(t)
        sizes.append(generator.get_network_size())
        avg_degrees.append(generator.get_average_degree())

        if t % 2 == 0:
            print(f"  Time {t}: {sizes[-1]} nodes, avg degree {avg_degrees[-1]:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(time_points, sizes, 'bo-', linewidth=2)
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Number of Nodes')
    axes[0].set_title('Network Size Over Time')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_points, avg_degrees, 'ro-', linewidth=2)
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Average Degree')
    axes[1].set_title('Average Degree Over Time')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return generator


if __name__ == "__main__":
    """Run tests if this file is executed directly."""
    print("=" * 60)
    print("Network Generator Tests")
    print("=" * 60)

    test_network_generation()
    test_visualization()
    test_dynamic_network()

    print("\n" + "=" * 60)
    print("Network generator tests completed!")
    print("=" * 60)