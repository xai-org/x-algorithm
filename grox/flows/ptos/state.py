from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from protos.abuse_cluster_anchor.abuse_cluster_anchor_pb2 import (
    LiveClusterAnchorInput as LiveClusterAnchorInputProto,
)
from pydantic import BaseModel


class SafetyPolicyCategory(str, Enum):
    HateOrAbuse = "HateOrAbuse"
    ViolentSpeech = "ViolentSpeech"
    ChildSafety = "ChildSafety"
    IllegalAndRegulatedBehaviors = "IllegalAndRegulatedBehaviors"
    Spam = "Spam"
    SuicideOrSelfHarm = "SuicideOrSelfHarm"
    AdultContent = "AdultContent"
    ViolentMedia = "ViolentMedia"
    TerrorismOrViolentExtremism = "TerrorismOrViolentExtremism"
    CivicIntegrity = "CivicIntegrity"


class SafetyPolicyType(str, Enum):
    NoViolation = "NoViolation"
    ViolentMediaGraphicMedia = "ViolentMediaGraphicMedia"
    ViolentMediaGore = "ViolentMediaGore"
    ViolentMediaWithException = "ViolentMediaWithException"
    ViolentMediaViolentSexualConduct = "ViolentMediaViolentSexualConduct"
    ViolentMediaBestialityAndNecrophilia = "ViolentMediaBestialityAndNecrophilia"
    AdultContentSexualHard = "AdultContentSexualHard"
    AdultContentSexualSoft = "AdultContentSexualSoft"
    SpamEngagementFarming = "SpamEngagementFarming"
    SpamEngagementTrading = "SpamEngagementTrading"
    SpamFinancialScam = "SpamFinancialScam"
    SpamHashTagAbuse = "SpamHashTagAbuse"
    SpamScamLinks = "SpamScamLinks"
    SpamBotProduced = "SpamBotProduced"
    SpamPlatformManipulation = "SpamPlatformManipulation"
    SpamEngagementBaiting = "SpamEngagementBaiting"
    SpamCardManipulation = "SpamCardManipulation"
    SpamMentionAbuse = "SpamMentionAbuse"
    IllegalAndRegulatedBehaviorsSexualServices = (
        "IllegalAndRegulatedBehaviorsSexualServices"
    )
    IllegalAndRegulatedBehaviorsIllegalDrugs = (
        "IllegalAndRegulatedBehaviorsIllegalDrugs"
    )
    HateOrAbuseSlursWithTarget = "HateOrAbuseSlursWithTarget"
    HateOrAbuseInsultsWithTarget = "HateOrAbuseInsultsWithTarget"
    ViolentSpeechMajorAndSpecificThreatWithHumanTarget = (
        "ViolentSpeechMajorAndSpecificThreatWithHumanTarget"
    )
    ViolentSpeechMajorAndNonSpecificThreatWithHumanTarget = (
        "ViolentSpeechMajorAndNonSpecificThreatWithHumanTarget"
    )
    ViolentSpeechMajorAndSpecificThreatWithoutHumanTarget = (
        "ViolentSpeechMajorAndSpecificThreatWithoutHumanTarget"
    )
    ViolentSpeechMajorAndNonSpecificThreatWithoutHumanTarget = (
        "ViolentSpeechMajorAndNonSpecificThreatWithoutHumanTarget"
    )
    ViolentSpeechMinorWithHumanTarget = "ViolentSpeechMinorWithHumanTarget"
    ViolentSpeechMinorWithoutHumanTarget = "ViolentSpeechMinorWithoutHumanTarget"
    SuicideOrSelfHarmEncouraging = "SuicideOrSelfHarmEncouraging"
    SuicideOrSelfHarmDepiction = "SuicideOrSelfHarmDepiction"
    SuicideOrSelfHarmEngaging = "SuicideOrSelfHarmEngaging"
    SuicideOrSelfHarmExpressingDesire = "SuicideOrSelfHarmExpressingDesire"
    ChildSafetySexualAbuseMaterial = "ChildSafetySexualAbuseMaterial"
    ChildSafetySexualExploitation = "ChildSafetySexualExploitation"


class SafetyPolicy(BaseModel):
    policyType: SafetyPolicyType
    confidenceScore: int | None = None
    reason: str | None = None


class SafetyPtosViolatedPolicy(BaseModel):
    category: SafetyPolicyCategory
    score: int | None = None
    reason: str | None = None
    safetyPolicy: SafetyPolicy | None = None


class SafetyPostAnnotations(BaseModel):
    violatedPolicies: list[SafetyPtosViolatedPolicy] | None = None


class SafemodelResult(BaseModel):
    scored: bool = False
    positive: bool = False
    confidence: float = 0.0


class LiveClusterAnchorInput(BaseModel):
    anchorPostId: int
    clusterId: int

    @classmethod
    def from_proto_kafka_message(cls, message_value: Any) -> "LiveClusterAnchorInput":
        if not message_value:
            raise ValueError("LiveClusterAnchorInput proto empty")
        proto_obj = LiveClusterAnchorInputProto()
        proto_obj.ParseFromString(message_value)
        if proto_obj.anchor_post_id == 0:
            raise ValueError("LiveClusterAnchorInput proto missing anchor_post_id")
        if proto_obj.cluster_id == 0:
            raise ValueError("LiveClusterAnchorInput proto missing cluster_id")
        return cls(
            anchorPostId=proto_obj.anchor_post_id,
            clusterId=proto_obj.cluster_id,
        )


class LiveClusterAnchorKafkaVerdict(BaseModel):
    anchorPostId: int
    clusterId: int
    enforceDecision: bool
    timestampMs: int


@dataclass
class SafetyPtosState:
    annotations: SafetyPostAnnotations | None = None
    safemodel_sex_nudity: SafemodelResult = field(default_factory=SafemodelResult)
