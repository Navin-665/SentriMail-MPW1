"""
SentriFormer Pretraining Engine
Implements mixed-precision T5 span-corruption pretraining with AdamW, Warmup+Cosine Scheduler,
Gradient Clipping, and Checkpointing.
"""
import os
import sys
import math
import time
import json
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

ROOT_DIR = r"c:\Users\nan33\OneDrive\Desktop\6th sem projects\SENTRIMAIL-MPW1"
sys.path.insert(0, ROOT_DIR)

from sentriformer.model.config import SentriFormerConfig
from sentriformer.model.sentriformer import SentriFormerForConditionalGeneration
from sentriformer.training.pretrain_dataset import PretrainParquetDataset, PretrainDataCollator


def create_pretrain_dataloader(
    parquet_path: str,
    tokenizer,
    batch_size: int = 32,
    max_seq_length: int = 512,
    limit: Optional[int] = None,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """Creates a PyTorch DataLoader with dynamic span corruption collation."""
    dataset = PretrainParquetDataset(
        parquet_path=parquet_path,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        limit=limit
    )
    collator = PretrainDataCollator(
        tokenizer=tokenizer,
        noise_density=0.15,
        mean_noise_span_length=3.0,
        max_seq_length=max_seq_length
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )


def run_training_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: torch.device,
    use_amp: bool = True
) -> torch.Tensor:
    """Executes a single forward + backward step with mixed precision."""
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    
    if use_amp and device.type == "cuda":
        with torch.cuda.amp.autocast(dtype=torch.float16):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs["loss"]
    else:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs["loss"]
        
    return loss
