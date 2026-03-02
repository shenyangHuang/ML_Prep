import torch

def compute_hits_at_k(y_pred, y_true, k=10):
    pred_topk, idx_topk = torch.topk(y_pred, k=k)
    true_idx = y_true.view(-1,1)
    matches = (idx_topk == true_idx) # (-1, k) == (-1,1) expand to (-1, k) == (-1,k) matches = (-1)
    return matches.sum() / y_pred.shape[0]





# def compute_hits_at_k(y_pred, y_true, k=10):
#     """
#     Args:
#         y_pred: (batch_size, num_classes) - The model's scores/logits
#         y_true: (batch_size,) - The index of the single correct class
#         k: The threshold for a 'hit'
#     """
#     # 1. Get the indices of only the top K predictions
#     # This is O(n log k) instead of O(n log n) for a full sort
#     _, topk_indices = torch.topk(y_pred, k, dim=1)
    
#     # 2. Reshape y_true from (batch_size) to (batch_size, 1) 
#     # so we can compare it against all k columns at once (broadcasting)
#     y_true_reshaped = y_true.view(-1, 1)
    
#     # 3. Check for matches: (batch_size, k) boolean tensor
#     # True if the ground truth index is in the top k for that row
#     matches = (topk_indices == y_true_reshaped)
    
#     # 4. Check if ANY of the k positions in each row is a hit
#     # matches.any(dim=1) results in (batch_size,) boolean tensor
#     hits_per_sample = matches.any(dim=1)
    
#     # 5. Convert booleans to floats and take the average
#     # A result of 0.7 means 70% of targets were in the top K
#     return hits_per_sample.float().mean().item()

# --- Quick Test ---
preds = torch.tensor([
    [0.1, 0.7, 0.2], # Target 1 is at index 1 (Rank 1) -> Hit@1
    [0.4, 0.5, 0.1], # Target 0 is at index 0 (Rank 2) -> Hit@2
])
targets = torch.tensor([1, 0])

print(f"Hits@1: {compute_hits_at_k(preds, targets, k=1)}") # 0.5
print(f"Hits@2: {compute_hits_at_k(preds, targets, k=2)}") # 1.0