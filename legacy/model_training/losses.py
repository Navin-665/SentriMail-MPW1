"""
Masked, class-weighted multi-task loss for SentriModel.

Sentiment and emotion are present on every row, so their loss is a
straightforward weighted cross-entropy. Category is missing on Twitter
rows, so it needs an explicit mask - unlike generation, where T5's own
loss already handles missing targets correctly via the -100 ignore_index
already baked into response_labels by dataset.py's collate_fn.
"""

import torch
import torch.nn.functional as F

import config


def compute_class_weights(label_series, label_vocabulary, method="inverse_frequency", beta=0.999):
    """Two weighting methods:

    "inverse_frequency" (the original) - weight = total / (num_classes * count).
    This overcorrects for classes like "other" that are broad and heterogeneous
    but not genuinely rare in the way small classes are - it pushed the model
    to over-predict "other" at billing's expense.

    "class_balanced" - uses the effective number of samples instead of raw count:
    effective_num = (1 - beta^n) / (1 - beta). As beta -> 1, this approaches raw
    inverse frequency; smaller beta (e.g. 0.999) tempers it substantially. This is
    the standard fix for class imbalance (Cui et al., "Class-Balanced Loss Based
    on Effective Number of Samples")."""
    label_counts = label_series.value_counts()
    total_count = label_counts.sum()
    num_classes = len(label_vocabulary)

    weights = torch.ones(num_classes)
    for class_index, class_name in enumerate(label_vocabulary):
        count_for_class = label_counts.get(class_name, 0)
        if count_for_class == 0:
            continue
        if method == "class_balanced":
            effective_num = (1.0 - (beta ** count_for_class)) / (1.0 - beta)
            weights[class_index] = 1.0 / max(effective_num, 1e-6)
        else:
            weights[class_index] = total_count / (num_classes * count_for_class)

    # normalize so weights average to 1 - keeps overall loss scale stable
    # across methods, so LOSS_WEIGHT_CATEGORY in config.py doesn't need
    # retuning just because the weighting method changed
    weights = weights * (num_classes / weights.sum())
    return weights


def compute_masked_classification_loss(logits, labels, valid_mask, class_weights=None):
    """Cross-entropy averaged only over rows where valid_mask is True -
    rows without a real label (e.g. Twitter's missing category) contribute
    nothing, rather than being scored against a meaningless placeholder."""
    if not valid_mask.any():
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    per_example_loss = F.cross_entropy(
        logits, labels, weight=class_weights, reduction="none"
    )
    masked_loss = per_example_loss * valid_mask.float()
    num_valid_examples = valid_mask.float().sum().clamp(min=1.0)
    return masked_loss.sum() / num_valid_examples


def compute_multitask_loss(model_outputs, batch, class_weights):
    """Combines all four task losses using the weights in config.py.
    Returns the total loss plus a dict of the individual components, so
    train.py can log each one separately - essential for noticing if one
    task is being neglected relative to the others during training."""
    batch_size = batch["sentiment_labels"].shape[0]
    always_valid_mask = torch.ones(batch_size, dtype=torch.bool, device=batch["sentiment_labels"].device)

    sentiment_mask = batch.get("has_sentiment", always_valid_mask)
    emotion_mask = batch.get("has_emotion", always_valid_mask)
    category_mask = batch.get("has_category", always_valid_mask)

    sentiment_loss = compute_masked_classification_loss(
        model_outputs["sentiment_logits"], batch["sentiment_labels"],
        sentiment_mask, class_weights["sentiment"],
    )
    emotion_loss = compute_masked_classification_loss(
        model_outputs["emotion_logits"], batch["emotion_labels"],
        emotion_mask, class_weights["emotion"],
    )
    category_loss = compute_masked_classification_loss(
        model_outputs["category_logits"], batch["category_labels"],
        category_mask, class_weights["category"],
    )
    # generation_loss can be None or NaN if response_labels wasn't passed or
    # all items in the batch are masked to -100 (e.g. all-CFPB batches)
    generation_loss = model_outputs["generation_loss"]
    if generation_loss is None or torch.isnan(generation_loss):
        generation_loss = torch.tensor(0.0, device=sentiment_loss.device)

    total_loss = (
        config.LOSS_WEIGHT_SENTIMENT * sentiment_loss
        + config.LOSS_WEIGHT_EMOTION * emotion_loss
        + config.LOSS_WEIGHT_CATEGORY * category_loss
        + config.LOSS_WEIGHT_GENERATION * generation_loss
    )

    loss_breakdown = {
        "sentiment_loss": sentiment_loss.item(),
        "emotion_loss": emotion_loss.item(),
        "category_loss": category_loss.item(),
        "generation_loss": generation_loss.item(),
        "total_loss": total_loss.item(),
    }
    return total_loss, loss_breakdown
