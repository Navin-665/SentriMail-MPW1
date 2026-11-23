"""
PyTorch Dataset and collate function for the SentriMail multi-task model.

Not every row has every label - Twitter rows have no category, CFPB rows
have no response text. has_category and has_response flags travel
alongside every batch so the loss function (in losses.py) knows exactly
which rows to skip for which task, rather than training on a placeholder
value as if it were real supervision.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset

import config


def _label_to_index(label, label_vocabulary):
    """Turns a string label like "billing" into its position in the fixed
    vocabulary list, e.g. 0. Returns None if the label is missing."""
    if label is None or (isinstance(label, float) and pd.isna(label)):
        return None
    return label_vocabulary.index(label)


class SentriDataset(Dataset):
    def __init__(self, parquet_path):
        self.dataframe = pd.read_parquet(parquet_path)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, row_index):
        row = self.dataframe.iloc[row_index]

        sentiment_val = row["sentiment"] if "sentiment" in self.dataframe.columns else None
        emotion_val = row["emotion"] if "emotion" in self.dataframe.columns else None
        category_val = row["category"] if "category" in self.dataframe.columns else None
        response_val = row["response"] if "response" in self.dataframe.columns else None

        sentiment_index = _label_to_index(sentiment_val, config.SENTIMENT_LABELS)
        emotion_index = _label_to_index(emotion_val, config.EMOTION_LABELS)
        category_index = _label_to_index(category_val, config.CATEGORY_LABELS)

        has_sentiment = sentiment_index is not None
        has_emotion = emotion_index is not None
        has_category = category_index is not None
        has_response = response_val is not None and not (
            isinstance(response_val, float) and pd.isna(response_val)
        )

        return {
            "complaint_text": row["text"],
            "sentiment_index": sentiment_index if has_sentiment else 0,
            "has_sentiment": has_sentiment,
            "emotion_index": emotion_index if has_emotion else 0,
            "has_emotion": has_emotion,
            # a missing label still needs a placeholder int so the
            # batch tensor can be built - has_* flags tell the loss and metric functions
            # to ignore them
            "category_index": category_index if has_category else 0,
            "has_category": has_category,
            "response_text": response_val if has_response else "",
            "has_response": has_response,
        }


def build_collate_fn(tokenizer):
    """Returns a collate_fn closed over the tokenizer, so batching does the
    actual tokenization and padding for this specific batch - padding to
    the longest sequence in the batch, not one fixed length for the whole
    dataset, which would waste compute on every short Bitext/Twitter row
    sitting next to a long CFPB narrative."""

    def collate_fn(batch_items):
        complaint_texts = [item["complaint_text"] for item in batch_items]
        encoded_inputs = tokenizer(
            complaint_texts,
            padding=True,
            truncation=True,
            max_length=config.MAX_INPUT_TOKENS,
            return_tensors="pt",
        )

        response_texts = [item["response_text"] for item in batch_items]
        encoded_responses = tokenizer(
            response_texts,
            padding=True,
            truncation=True,
            max_length=config.MAX_RESPONSE_TOKENS,
            return_tensors="pt",
        )
        # T5 expects -100 at padded decoder-target positions so the loss
        # ignores them automatically - done once here rather than repeated
        # in the training loop
        response_labels = encoded_responses["input_ids"].clone()
        response_labels[encoded_responses["attention_mask"] == 0] = -100

        has_response_tensor = torch.tensor([item["has_response"] for item in batch_items])
        # an empty string still tokenizes to at least one real token (e.g.
        # an EOS token), which would otherwise leak a sliver of generation
        # loss from rows that have no real response at all (CFPB) - force
        # the whole row to -100 regardless of what the tokenizer produced
        response_labels[~has_response_tensor] = -100

        return {
            "input_ids": encoded_inputs["input_ids"],
            "attention_mask": encoded_inputs["attention_mask"],
            "sentiment_labels": torch.tensor([item["sentiment_index"] for item in batch_items]),
            "has_sentiment": torch.tensor([item["has_sentiment"] for item in batch_items]),
            "emotion_labels": torch.tensor([item["emotion_index"] for item in batch_items]),
            "has_emotion": torch.tensor([item["has_emotion"] for item in batch_items]),
            "category_labels": torch.tensor([item["category_index"] for item in batch_items]),
            "has_category": torch.tensor([item["has_category"] for item in batch_items]),
            "response_labels": response_labels,
            "has_response": has_response_tensor,
        }

    return collate_fn


if __name__ == "__main__":
    # quick sanity check - run "python dataset.py" to see a real batch's
    # shapes before any model code exists, exactly the point of building
    # this file first
    from transformers import T5Tokenizer
    from torch.utils.data import DataLoader

    tokenizer = T5Tokenizer.from_pretrained(config.BACKBONE_MODEL_NAME)
    train_dataset = SentriDataset(config.TRAIN_PARQUET_PATH)
    print(f"loaded {len(train_dataset):,} training rows")

    data_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True,
        collate_fn=build_collate_fn(tokenizer),
    )
    sample_batch = next(iter(data_loader))

    print("\nbatch tensor shapes:")
    for key, value in sample_batch.items():
        print(f"  {key}: shape={tuple(value.shape)} dtype={value.dtype}")

    print("\nhas_category:", sample_batch["has_category"].tolist())
    print("has_response:", sample_batch["has_response"].tolist())
