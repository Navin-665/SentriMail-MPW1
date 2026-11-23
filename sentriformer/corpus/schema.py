"""
Canonical Schema for SentriCorpus
"""
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, Dict, Any


class CanonicalSource(str, Enum):
    CFPB = "cfpb"
    TWCS = "twcs"
    CUSTOMER_SUPPORT = "customer_support"
    BITEXT = "bitext"


class ExampleType(str, Enum):
    COMPLAINT = "complaint"
    CONVERSATION = "conversation"
    SUPPORT_RESPONSE = "support_response"


@dataclass
class SentriRecord:
    id: str
    source: str
    example_type: str
    group_id: str
    parent_id: Optional[str] = None
    text: str = ""
    response: Optional[str] = None
    split: str = "train"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def validate(self) -> bool:
        if not self.id or not isinstance(self.id, str):
            return False
        if not self.text or not isinstance(self.text, str):
            return False
        if self.split not in ("train", "val", "test"):
            return False
        return True
