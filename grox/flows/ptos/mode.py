from enum import Enum

from grox.flows.ptos import constants


class SafetyPtosMode(str, Enum):
    STANDARD = "standard"
    DELUXE = "deluxe"
    RECOVERY = "recovery"
    BACKFILL = "backfill"
    LIVE_CLUSTER_ANCHORS = "live_cluster_anchors"

    @classmethod
    def from_task_type(cls, task_type: str | None) -> "SafetyPtosMode":
        match task_type:
            case constants.SAFETY_PTOS_DELUXE:
                return cls.DELUXE
            case constants.SAFETY_PTOS_RECOVERY:
                return cls.RECOVERY
            case constants.SAFETY_PTOS_BACKFILL:
                return cls.BACKFILL
            case constants.SAFETY_PTOS_LIVE_CLUSTER_ANCHORS:
                return cls.LIVE_CLUSTER_ANCHORS
            case _:
                return cls.STANDARD

    @property
    def is_deluxe(self) -> bool:
        return self is SafetyPtosMode.DELUXE
