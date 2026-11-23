"""
Group-Aware Deterministic Dataset Splitting
"""
import hashlib


def assign_split(group_id: str, train_ratio: float = 0.90, val_ratio: float = 0.05) -> str:
    """
    Deterministically assigns a split ('train', 'val', 'test') based on MD5 hash of group_id.
    Ensures:
    - All turns in a conversation stay in the same split
    - All rows sharing the same group/order/cluster stay in the same split
    - 100% reproducible across machines and runs
    """
    if not group_id:
        group_id = "default_group"
        
    hash_val = int(hashlib.md5(group_id.encode('utf-8')).hexdigest(), 16) % 10000
    train_thresh = int(train_ratio * 10000)
    val_thresh = int((train_ratio + val_ratio) * 10000)
    
    if hash_val < train_thresh:
        return "train"
    elif hash_val < val_thresh:
        return "val"
    else:
        return "test"
