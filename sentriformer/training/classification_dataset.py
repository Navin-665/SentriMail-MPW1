"""
PyTorch Dataset for SentriFormer Customer Support Classification Fine-Tuning
"""
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional
import pyarrow.parquet as pq


class ClassificationParquetDataset(Dataset):
    """
    Random-access Parquet Dataset for Supervised Classification.
    Reads text narratives and maps product categories to class IDs.
    """
    def __init__(
        self,
        parquet_path: str,
        tokenizer,
        label2id: Dict[str, int],
        max_seq_length: int = 512,
        limit: Optional[int] = None
    ):
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_seq_length = max_seq_length
        
        table = pq.read_table(parquet_path, columns=["text", "product"])
        if limit:
            table = table.slice(0, limit)
            
        self.texts = table["text"].to_pylist()
        self.products = table["product"].to_pylist()
        
    def __len__(self) -> int:
        return len(self.texts)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text = self.texts[idx]
        product = self.products[idx]
        
        label_id = self.label2id.get(product, 0)
        
        # Tokenize text
        token_ids = self.tokenizer.encode(
            text,
            truncation=True,
            max_length=self.max_seq_length,
            add_special_tokens=True
        )
        return {
            "token_ids": token_ids,
            "label_id": label_id
        }


class ClassificationDataCollator:
    """
    Dynamic Batch Collator for Sequence Classification.
    """
    def __init__(self, pad_token_id: int = 0, max_seq_length: int = 512):
        self.pad_token_id = pad_token_id
        self.max_seq_length = max_seq_length
        
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_size = len(batch)
        max_len = min(self.max_seq_length, max(len(x["token_ids"]) for x in batch))
        
        input_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        labels = torch.zeros(batch_size, dtype=torch.long)
        
        for i, item in enumerate(batch):
            ids = item["token_ids"][:max_len]
            input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, :len(ids)] = 1
            labels[i] = item["label_id"]
            
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
