from enum import Enum

class ProcessState(Enum):
    INITIAL = "initial"
    SEARCHING = "searching"
    MULTIPLE_FOUND = "multiple"
    CONFIRMED = "confirmed"
    CHANGING = "changing"