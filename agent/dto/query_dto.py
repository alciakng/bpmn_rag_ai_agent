from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from agent.entity.graph_node import GraphNode
from agent.enums.enums import QueryType

@dataclass
class QueryResult:
    query_type: QueryType
    candidate_nodes: List[GraphNode]
    graph_context: Dict[str, Any]
    answer: str
    image_path: str
    requires_user_interaction: bool = False
    interaction_message: Optional[str] = None


@dataclass
class QueryContext:
    query_type: QueryType
    process_name: Optional[str]
    process_action: str  # "use_current", "search_new", "confirm_change", "select_from_candidates"
    image_requested: bool
    requires_user_interaction: bool
    interaction_message: Optional[str]