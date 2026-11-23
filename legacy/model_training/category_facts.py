"""
Category facts, grounding knowledge, and canonical safe fallbacks for SentriMail.

Provides:
1. Verified grounding facts and SLA policies per category.
2. Verified canonical response templates (used as safe fallbacks when generative text fails safety checks).
3. Confidence gating utility to prevent low-confidence predictions (< 65-70%) from blindly auto-routing.
"""

from typing import Dict, Any, Tuple, Optional
import torch

CATEGORY_FACTS = {
    "billing": {
        "department": "Billing & Accounts",
        "sla_hours": 24,
        "verified_portal": "https://sentrimail.internal/portal/billing",
        "policy_summary": "Invoices and transaction charges can be reviewed and adjusted within 30 days.",
        "fallback_template": (
            "Thank you for contacting SentriMail Billing Support. We have received your inquiry regarding "
            "your billing details and transaction history. Our team is actively reviewing your account and "
            "will provide an itemized breakdown and resolution within 24 business hours."
        ),
    },
    "technical": {
        "department": "Technical Engineering Support",
        "sla_hours": 12,
        "verified_portal": "https://sentrimail.internal/portal/support",
        "policy_summary": "Service availability issues and system errors are logged and investigated by engineering.",
        "fallback_template": (
            "Thank you for reporting this technical issue. Our engineering team has logged your complaint "
            "and started system diagnostics. We will follow up with troubleshooting steps or resolution updates shortly."
        ),
    },
    "delivery": {
        "department": "Logistics & Fulfillment",
        "sla_hours": 24,
        "verified_portal": "https://sentrimail.internal/portal/tracking",
        "policy_summary": "Shipment status and tracking updates are synchronized with logistics carrier feeds.",
        "fallback_template": (
            "Thank you for reaching out regarding your shipment status. We are tracking your delivery details "
            "with our logistics carrier and will update you with current tracking information and estimated arrival."
        ),
    },
    "customer_service": {
        "department": "Customer Experience Team",
        "sla_hours": 24,
        "verified_portal": "https://sentrimail.internal/portal/help",
        "policy_summary": "Support quality feedback is reviewed by team leads for immediate resolution and coaching.",
        "fallback_template": (
            "Thank you for reaching out to us. We take customer satisfaction seriously and have assigned "
            "your case to a senior support representative who will assist you directly."
        ),
    },
    "product": {
        "department": "Product Quality & Inventory",
        "sla_hours": 24,
        "verified_portal": "https://sentrimail.internal/portal/products",
        "policy_summary": "Product quality, defect, and availability concerns are reviewed for replacement or repair.",
        "fallback_template": (
            "We are sorry to hear that you experienced an issue with your product. Our product quality team "
            "is reviewing your report and will reach out with replacement or resolution options."
        ),
    },
    "refund": {
        "department": "Refunds & Settlements",
        "sla_hours": 48,
        "verified_portal": "https://sentrimail.internal/portal/refunds",
        "policy_summary": "Approved refunds are processed back to the original payment method within 3-5 business days.",
        "fallback_template": (
            "Thank you for contacting us regarding your refund request. We have initiated a review of your "
            "order and return status, and our refund specialist will provide a status update within 48 business hours."
        ),
    },
    "other": {
        "department": "General Inquiries",
        "sla_hours": 48,
        "verified_portal": "https://sentrimail.internal/portal/help",
        "policy_summary": "General inquiries are triaged and routed to the appropriate specialized team.",
        "fallback_template": (
            "Thank you for contacting SentriMail. We have logged your inquiry and our support team is "
            "reviewing the details to ensure it reaches the right department for a prompt response."
        ),
    },
}


def get_category_fallback(category: str, username: str = "Customer") -> str:
    """Returns the verified, safe canonical response template for a given category."""
    cat_key = (category or "other").lower().strip()
    entry = CATEGORY_FACTS.get(cat_key, CATEGORY_FACTS["other"])
    return entry["fallback_template"]


def get_category_facts(category: str) -> Dict[str, Any]:
    """Returns grounding facts and verified links for a category."""
    cat_key = (category or "other").lower().strip()
    return CATEGORY_FACTS.get(cat_key, CATEGORY_FACTS["other"])


def route_with_confidence_gate(
    category_logits_or_probs: Any,
    category_labels: list,
    threshold: float = 0.65,
) -> Tuple[str, float, bool]:
    """
    Evaluates classification confidence against a threshold.
    
    Returns:
        (predicted_category, confidence, needs_human_review)
    """
    if isinstance(category_logits_or_probs, torch.Tensor):
        probs = torch.softmax(category_logits_or_probs, dim=-1).squeeze()
        conf, idx = torch.max(probs, dim=-1)
        confidence = conf.item()
        cat_idx = idx.item()
    elif isinstance(category_logits_or_probs, list):
        import numpy as np
        probs = np.array(category_logits_or_probs)
        cat_idx = int(np.argmax(probs))
        confidence = float(probs[cat_idx])
    else:
        return "other", 0.0, True

    predicted_cat = category_labels[cat_idx]
    needs_review = confidence < threshold
    return predicted_cat, confidence, needs_review
