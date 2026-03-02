"""
Prim’s algorithm is a classic "greedy" algorithm used to find the Minimum Spanning Tree (MST) of a weighted, undirected graph. 
In an interview, the goal is to show you understand how to build a network that connects all points with the lowest possible total "cost" without any loops.
"""
import heapq

def prims_algorithm(graph, start_node):
    # graph is an adjacency list: {node: [(weight, neighbor), ...]}
    mst_edges = []
    visited = set()
    # Min-heap stores: (weight, current_node, parent_node)
    min_heap = [(0, start_node, None)]
    total_cost = 0

    while min_heap:
        weight, u, parent = heapq.heappop(min_heap)

        if u in visited:
            continue

        # Add to MST
        visited.add(u)
        total_cost += weight
        if parent is not None:
            mst_edges.append((parent, u, weight))

        # Check neighbors
        for edge_weight, v in graph[u]:
            if v not in visited:
                heapq.heappush(min_heap, (edge_weight, v, u))

    return mst_edges, total_cost]

# --- Test Case ---
if __name__ == "__main__":
    # Representing a simple triangle graph with 3 nodes
    # A -5- B, B -1- C, A -10- C
    example_graph = {
        'A': [(5, 'B'), (10, 'C')],
        'B': [(5, 'A'), (1, 'C')],
        'C': [(10, 'A'), (1, 'B')]
    }

    edges, cost = prims_algorithm(example_graph, 'A')
    
    print(f"MST Edges: {edges}")
    print(f"Total Minimum Cost: {cost}")
    # Expected: ('A', 'B', 5) and ('B', 'C', 1), Total: 6