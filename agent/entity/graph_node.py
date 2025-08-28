from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np


@dataclass
class GraphNode:
    key: str
    name: str
    type: str  # Activity, Gateway, Event, Role
    search_text: str
    embedding: Optional[np.ndarray] = None
    properties: Dict[str, Any] = None
