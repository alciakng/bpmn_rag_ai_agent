from enum import Enum

class ProcessState(Enum): 
    INITIAL = "initial" 
    SEARCHING = "searching" 
    MULTIPLE_FOUND = "multiple" 
    CONFIRMED = "confirmed" 
    CHANGING = "changing"

class QueryType(Enum):
    FLOW_OVERVIEW = "flow_overview"  # 전체적인 흐름과 지엽적 기능
    PATH_FINDING = "path_finding"    # 특정 노드 간 경로
    GENERAL = "general"              # 기타 질의