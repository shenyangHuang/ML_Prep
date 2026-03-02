import torch

def mean_reciprocal_rank(y_pred, y_true):

    #! sort the prob by rank
    sort_idx = torch.argsort(y_pred)

    # find the index of the y_true
    true_idx = y_true.reshape(-1, 1)

    ranks = torch.where(sort_idx == true_idx, sort_idx, float('inf'))
    ranks = ranks + 1  # index off by 1
    
    rr = 1 / ranks
    mrr = torch.sum(rr) / y_pred.shape[0]
    return mrr









# def mean_reciprocal_rank(y_pred, y_true):
#     """
#     Args:
#         y_pred: (batch_size, num_classes) - Model logits
#         y_true: (batch_size,) - Ground truth indices
#     """
#     # 1. Sort predictions in descending order
#     # indices shape: (batch_size, num_classes)
#     indices = torch.argsort(y_pred, dim=1, descending=True)
    
#     # 2. Find where the true label matches the sorted indices
#     # result is a boolean mask: (batch_size, num_classes)
#     matches = (indices == y_true.view(-1, 1))
    
#     # 3. Find the index (position) of the True value in each row
#     # torch.nonzero returns (row_idx, col_idx). We want the col_idx (the rank-1).
#     # We add 1 because ranks are 1-based (1st place, 2nd place, etc.)
#     _, ranks = matches.nonzero(as_tuple=True)
#     ranks = ranks + 1 
    
#     # 4. Calculate Reciprocal Rank: 1 / rank
#     reciprocal_ranks = 1.0 / ranks.float()
    
#     # 5. Return the Mean
#     return reciprocal_ranks.mean().item()

# # Example:
# # Row 0: Target is class 2. Preds say [class 1, class 2...]. Rank is 2. RR = 0.5
# # Row 1: Target is class 0. Preds say [class 0, class 1...]. Rank is 1. RR = 1.0
# # MRR = (0.5 + 1.0) / 2 = 0.75


# --- Quick Test ---
preds = torch.tensor([
    [0.1, 0.7, 0.2], # Target 1 is at index 1 (Rank 1) -> Hit@1
    [0.4, 0.5, 0.1], # Target 0 is at index 0 (Rank 2) -> Hit@2
])
targets = torch.tensor([1, 0])

print(f"MRR: {mean_reciprocal_rank(preds, targets)}")
print(f"MRR: {mean_reciprocal_rank(preds, targets)}")