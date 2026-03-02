
def bellman_ford(graph, start, n):
    distances = { id:float('inf') for id in range(n)}
    distances[start] = 0

    # iterate over all nodes
    for _ in range(n):
        # edge interation loop
        for src in graph.keys():
            edges = graph[src]
            for dst, w in edges:
                if distances[src] != float('inf') and (distances[src] + w) < distances[dst]:
                    distances[dst] = distances[src] + w

    # neg edge check
    for src in graph.keys():
        edges = graph[src]
        for dst, w in edges:
            if distances[src] != float('inf'):
                if (distances[src]+w) < distances[dst]:
                    raise ValueError("negative cycle detected")
                # return distances #negative loop
    
    return distances








def run_tests(bellman_ford_func):
    # Test Case 1: Simple Linear
    # graph format: { node: [(neighbor, weight), ...] }
    g1 = {0: [(1, 5)], 1: [(2, 10)], 2: []}
    print(f"Test 1 (Linear): {bellman_ford_func(g1, 0, 3)}")

    # Test Case 2: The Short-Cut
    g2 = {0: [(2, 10), (1, 2)], 1: [(2, 3)], 2: []}
    print(f"Test 2 (Short-cut): {bellman_ford_func(g2, 0, 3)}")

    # Test Case 3: Disconnected
    g3 = {0: [(1, 5)], 1: [], 2: []}
    print(f"Test 3 (Disconnected): {bellman_ford_func(g3, 0, 3)}")

    # Test Case 4: Cycles
    g4 = {0: [(1, 1)], 1: [(2, 2), (3, 5)], 2: [(0, 1)], 3: []}
    print(f"Test 4 (Cycle): {bellman_ford_func(g4, 0, 4)}")

# To run it, just call:
run_tests(bellman_ford)