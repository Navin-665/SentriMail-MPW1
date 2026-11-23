"""
Category mapping for SentriMail dataset preparation.

Maps raw category strings from various datasets (Bitext, CFPB, E-commerce)
into the 7 canonical SentriMail categories:
- billing
- technical
- delivery
- customer_service
- product
- refund
- other
"""

from typing import Dict, Optional

# Bitext category taxonomy -> SentriMail canonical categories
# Handles both uppercase category column and lowercase intent column
BITEXT_MAP = {
    # Raw Bitext Category column (UPPERCASE)
    "ACCOUNT": "technical",
    "ORDER": "product",
    "REFUND": "refund",
    "INVOICE": "billing",
    "PAYMENT": "billing",
    "CONTACT": "customer_service",
    "FEEDBACK": "customer_service",
    "DELIVERY": "delivery",
    "SHIPPING": "delivery",
    "SUBSCRIPTION": "billing",
    "CANCEL": "product",

    # Lowercase variants / Bitext intent column
    "check_invoice": "billing",
    "get_invoice": "billing",
    "check_payment_methods": "billing",
    "payment_issue": "billing",
    "accounting": "billing",
    "check_cancellation_fee": "billing",
    "get_refund": "refund",
    "track_refund": "refund",
    "check_refund_policy": "refund",
    "create_account": "technical",
    "delete_account": "technical",
    "edit_account": "technical",
    "recover_password": "technical",
    "registration_problems": "technical",
    "switch_account": "technical",
    "change_shipping_address": "delivery",
    "set_up_shipping_address": "delivery",
    "delivery_options": "delivery",
    "delivery_period": "delivery",
    "track_order": "delivery",
    "contact_customer_service": "customer_service",
    "contact_human_agent": "customer_service",
    "complaint": "customer_service",
    "review": "customer_service",
    "newsletter_subscription": "customer_service",
    "cancel_order": "product",
    "change_order": "product",
    "order": "product",
    "place_order": "product",
}

# CFPB Product taxonomy -> SentriMail canonical categories
CFPB_MAP = {
    "Credit reporting, credit repair services, or other personal consumer reports": "billing",
    "Credit reporting": "billing",
    "Debt collection": "billing",
    "Mortgage": "billing",
    "Credit card or prepaid card": "billing",
    "Credit card": "billing",
    "Prepaid card": "billing",
    "Checking or savings account": "billing",
    "Bank account or service": "billing",
    "Money transfer, virtual currency, or money service": "billing",
    "Money transfers": "billing",
    "Vehicle loan or lease": "billing",
    "Consumer Loan": "billing",
    "Student loan": "billing",
    "Payday loan, title loan, or personal loan": "billing",
    "Payday loan": "billing",
    "Other financial service": "other",
}

# E-commerce customer support dataset -> SentriMail canonical categories
ECOMMERCE_MAP = {
    # Exact category values in Ataur77/ecommerce-customer-support
    "Payments": "billing",
    "Checkout": "billing",
    "Subscriptions": "billing",
    "Billing and Payments": "billing",
    "Billing and Invoicing": "billing",
    "Payment Issues": "billing",
    "Refunds": "refund",
    "Returns": "refund",
    "Refunds and Exchanges": "refund",
    "Returns and Refunds": "refund",
    "Refund": "refund",
    "Account": "technical",
    "Technical Issues": "technical",
    "Technical Support": "technical",
    "Website and App Issues": "technical",
    "Account Management": "technical",
    "Shipping": "delivery",
    "Order Tracking": "delivery",
    "Account & Shipping": "delivery",
    "Shipping and Delivery": "delivery",
    "Order Tracking and Delivery": "delivery",
    "Delivery": "delivery",
    "Product Quality": "product",
    "Product Issues": "product",
    "Order Issues": "product",
    "Inventory": "product",
    "Product Inquiries": "product",
    "Product Availability": "product",
    "Order Inquiries": "product",
    "Special Requests": "customer_service",
    "Customer Service Experience": "customer_service",
    "Customer Service": "customer_service",
    "Feedback and Complaints": "customer_service",
    "Promotions": "other",
    "General Inquiries": "other",
}


def map_category(raw_category: Optional[str], mapping: Dict[str, str]) -> str:
    """
    Map a raw category string to a canonical SentriMail category.
    Falls back to 'other' if the key is missing, null, or unmapped.
    """
    if not raw_category or not isinstance(raw_category, str):
        return "other"
    
    clean_cat = raw_category.strip()
    if clean_cat in mapping:
        return mapping[clean_cat]
    if clean_cat.upper() in mapping:
        return mapping[clean_cat.upper()]
    if clean_cat.lower() in mapping:
        return mapping[clean_cat.lower()]
    return "other"