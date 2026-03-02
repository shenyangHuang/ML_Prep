"""
only works on DAG, no cycles
"""
from collections import deque, defaultdict


def find_topological_sort(num_courses, prerequisites):
    """
    :param num_courses: int (Total number of nodes/courses)
    :param prerequisites: List[List[int]] (Edges as [destination, source])
    :return: List[int] (A valid topological ordering or empty list if cycle exists)
    """
    in_degrees = {}
    edges = {}
    for pr in prerequisites:
        src, dst = pr[0], pr[1]
        if src not in edges:
            edges[src] = [dst]
        else:
            edges[src].append(dst)

        if dst not in in_degrees:
            in_degrees[dst] = 1
        else:
            in_degrees[dst] += 1
        
        if src not in in_degrees:
            in_degrees[src] = 0
    no_req = [k for k, v in in_degrees.items() if v == 0]
    dq = deque(no_req)
    
    schedule = []
    while dq:
        src = dq.popleft()
        if src in edges:
            for dst in edges[src]:
                in_degrees[dst] -= 1
                if in_degrees[dst] == 0:
                    dq.append(dst)
        schedule.append(src)
    
    if len(schedule) != len(in_degrees):
        return []
    else:
        return schedule



    




# def find_topological_sort(num_courses, prerequisites):
#     """
#     :param num_courses: int (Total number of nodes/courses)
#     :param prerequisites: List[List[int]] (Edges as [destination, source])
#     :return: List[int] (A valid topological ordering or empty list if cycle exists)
#     """
#     # 1. Initialize the graph and in-degree counter
#     adj = defaultdict(list)
#     in_degree = {i: 0 for i in range(num_courses)}
    
#     # 2. Build the graph
#     # Usually, LeetCode gives prerequisites as [dest, src] (src -> dest)
#     for dest, src in prerequisites:
#         adj[src].append(dest)
#         in_degree[dest] += 1
        
#     # 3. Find all nodes with 0 in-degree (no dependencies)
#     queue = deque([node for node in in_degree if in_degree[node] == 0])
    
#     topo_order = []
    
#     # 4. Process the queue
#     while queue:
#         current = queue.popleft()
#         topo_order.append(current)
        
#         # Check neighbors
#         for neighbor in adj[current]:
#             in_degree[neighbor] -= 1
#             # If in-degree becomes 0, it's ready to be processed
#             if in_degree[neighbor] == 0:
#                 queue.append(neighbor)
                
#     # 5. Cycle Detection
#     # If the topo_order length doesn't match num_courses, there's a cycle
#     if len(topo_order) == num_courses:
#         return topo_order
#     else:
#         return [] # No valid order possible




def run_tests():
    test_cases = [
        {
            "name": "Linear Path",
            "n": 4,
            "edges": [[1, 0], [2, 1], [3, 2]],
        },
        {
            "name": "Diamond (Multiple Valid)",
            "n": 4,
            "edges": [[1, 0], [2, 0], [3, 1], [3, 2]],
        },
        {
            "name": "Cyclic (Impossible)",
            "n": 2,
            "edges": [[1, 0], [0, 1]],
        }
    ]

    for case in test_cases:
        result = find_topological_sort(case["n"], case["edges"])
        print(f"Test: {case['name']}")
        print(f"Result: {result}\n")

run_tests()