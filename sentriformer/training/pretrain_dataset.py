"""
PyTorch Dataset & DataCollator for SentriFormer Pretraining with On-the-Fly Span Corruption
"""
import torch
from torch.utils.data import Dataset, IterableDataset
from typing import List, Dict, Any, Optional
import pyarrow.parquet as pq
import json
from .span_corruption import SpanCorruptor


class PretrainParquetDataset(Dataset):
    """
    Random-access Parquet Dataset for PyTorch DataLoader.
    Reads pre-tokenized or text records from train.parquet.
    """
    def __init__(self, parquet_path: str, tokenizer, max_seq_length: int = 512, limit: Optional[int] = None):
        self.parquet_path = parquet_path
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        
        # Read table
        table = pq.read_table(parquet_path, columns=["text", "response"])
        if limit:
            table = table.slice(0, limit)
            
        self.texts = table["text"].to_pylist()
        self.responses = table["response"].to_pylist()
        
    def __len__(self) -> int:
        return len(self.texts)
        
    def __getitem__(self, idx: int) -> List[int]:
        text = self.texts[idx]
        resp = self.responses[idx]
        
        # Combine text and response if present
        full_text = text
        if resp and len(resp.strip()) > 0:
            full_text = f"{text} </s> {resp}"
            
        # Tokenize
        enc = self.tokenizer.encode(
            full_text,
            truncation=True,
            max_length=self.max_seq_length,
            add_special_tokens=False
        )
        return enc


class PretrainDataCollator:
    """
    Data Collator that performs on-the-fly T5 span corruption and dynamic batch padding.
    """
    def __init__(
        self,
        tokenizer,
        noise_density: float = 0.15,
        mean_noise_span_length: float = 3.0,
        pad_token_id: int = 0,
        eos_token_id: int = 3,
        max_seq_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.max_seq_length = max_seq_length
        
        # Determine extra_ids_start token ID
        extra_ids_start = tokenizer.convert_tokens_to_ids("<extra_id_0>")
        if extra_ids_start is None or extra_ids_start < 0:
            extra_ids_start = 6
            
        self.corruptor = SpanCorruptor(
            noise_density=noise_density,
            mean_noise_span_length=mean_noise_span_length,
            extra_ids_start=extra_ids_start,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id
        )

    def __call__(self, batch: List[List[int]]) -> Dict[str, torch.Tensor]:
        corrupted_batch = []
        target_batch = []
        
        for token_ids in batch:
            if not token_ids:
                token_ids = [self.eos_token_id]
            corr_ids, tgt_ids = self.corruptor.corrupt(token_ids)
            corrupted_batch.append(corr_ids)
            target_batch.append(tgt_ids)
            
        # Dynamic padding lengths
        max_input_len = min(self.max_seq_length, max(len(x) for x in corrupted_batch))
        max_target_len = min(self.max_seq_length, max(len(x) for x in target_batch))
        
        batch_size = len(batch)
        
        input_ids = torch.full((batch_size, max_input_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_input_len), dtype=torch.long)
        labels = torch.full((batch_size, max_target_len), -100, dtype=torch.long)
        
        for i in range(batch_size):
            corr = corrupted_batch[i][:max_input_len]
            tgt = target_batch[i][:max_target_len]
            
            input_ids[i, :len(corr)] = torch.tensor(corr, dtype=torch.long)
            attention_mask[i, :len(corr)] = 1
            labels[i, :len(tgt)] = torch.tensor(tgt, dtype=torch.long)
            
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


# Alias for backward compatibility
PretrainDataset = PretrainParquetDataset

