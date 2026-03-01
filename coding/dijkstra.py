
import heapq

def dijkstra(graph, start, n):
    """
    graph: {0: [(1, 5)], 1: [(2, 10)], 2: []}
    start: source node
    n: number of nodes
    """
    # float('inf')

    distances = {node: float('inf') for node in range(n)}
    distances[start] = 0

    pq = [(0, start)]
    while pq:
        dist_u, u = heapq.heappop(pq)

        for v, weight in graph[u]:
            distance = dist_u + weight
            if distance < distances[v]:
                distances[v] = distance
                heapq.heappush(pq, (distance,v))
    return distances

















# def dijkstra(graph, start, n):
#     """
#     graph: {0: [(1, 5)], 1: [(2, 10)], 2: []}
#     start: source node
#     n: number of nodes
#     """
#     # distances[i] will hold the shortest distance from start to i
#     distances = {node: float('inf') for node in range(n)}
#     distances[start] = 0
    
#     # Priority Queue stores (distance, node)
#     pq = [(0, start)]
    
#     while pq:
#         current_dist, u = heapq.heappop(pq)
        
#         # Nodes can be added to PQ multiple times; only process the best one
#         if current_dist > distances[u]:
#             continue
            
#         for v, weight in graph[u]:
#             distance = current_dist + weight
            
#             # If a shorter path to v is found
#             if distance < distances[v]:
#                 distances[v] = distance
#                 heapq.heappush(pq, (distance, v))
                
#     return distances




def run_tests(dijkstra_func):
    # Test Case 1: Simple Linear
    # graph format: { node: [(neighbor, weight), ...] }
    g1 = {0: [(1, 5)], 1: [(2, 10)], 2: []}
    print(f"Test 1 (Linear): {dijkstra_func(g1, 0, 3)}")

    # Test Case 2: The Short-Cut
    g2 = {0: [(2, 10), (1, 2)], 1: [(2, 3)], 2: []}
    print(f"Test 2 (Short-cut): {dijkstra_func(g2, 0, 3)}")

    # Test Case 3: Disconnected
    g3 = {0: [(1, 5)], 1: [], 2: []}
    print(f"Test 3 (Disconnected): {dijkstra_func(g3, 0, 3)}")

    # Test Case 4: Cycles
    g4 = {0: [(1, 1)], 1: [(2, 2), (3, 5)], 2: [(0, 1)], 3: []}
    print(f"Test 4 (Cycle): {dijkstra_func(g4, 0, 4)}")

# To run it, just call:
run_tests(dijkstra)
