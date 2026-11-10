"""domain 包的公开接口。

下游只准许 import 这里导出的名字，不要直接 import 子模块——
后面把实现拆成独立包时，才不会牵连一堆 import。
"""

from mc_domain.catalog_loader import (
    LoadReport,
    build_tree,
    load_csv,
    sorted_for_insert,
    to_upsert_rows,
)
from mc_domain.classify import (
    R_COMBINATION,
    R_EXCLUSION,
    R_HIGHEST_CLASS,
    R_MOST_SPECIFIC,
    ClassificationPipeline,
    CodeRetriever,
    ProfileNormalizer,
    Rule,
)
from mc_domain.enums import (
    CertStatus,
    CertType,
    ClassificationOutcome,
    ClauseKind,
    ContactDuration,
    ContactType,
    EnergyType,
    ManagementClass,
    ProductStatus,
    RegulationKind,
    RequirementKey,
    VersionStatus,
)
from mc_domain.gate import DeviceGate, GateResult
from mc_domain.retrieval import KeywordCodeRetriever
from mc_domain.schemas import (
    Candidate,
    ClassificationResult,
    Evidence,
    GateDecision,
    MissingSlot,
    ProductProfile,
    RejectedCandidate,
    StepTrace,
)
from mc_domain.tree import (
    ClassificationTree,
    CodeNode,
    ExclusionHit,
    ExclusionMatcher,
    extract_exclusion_terms,
)

__all__ = [
    # tree
    "ClassificationTree",
    "CodeNode",
    "ExclusionHit",
    "ExclusionMatcher",
    "extract_exclusion_terms",
    # pipeline
    "ClassificationPipeline",
    "CodeRetriever",
    "DeviceGate",
    "GateResult",
    "ProfileNormalizer",
    "Rule",
    "R_HIGHEST_CLASS",
    "R_MOST_SPECIFIC",
    "R_COMBINATION",
    "R_EXCLUSION",
    # retrieval
    "KeywordCodeRetriever",
    # schemas
    "Candidate",
    "ClassificationResult",
    "Evidence",
    "MissingSlot",
    "ProductProfile",
    "RejectedCandidate",
    "StepTrace",
    # catalog
    "LoadReport",
    "build_tree",
    "load_csv",
    "sorted_for_insert",
    "to_upsert_rows",
    # enums
    "CertStatus",
    "CertType",
    "ClassificationOutcome",
    "ClauseKind",
    "GateDecision",
    "ContactDuration",
    "ContactType",
    "EnergyType",
    "ManagementClass",
    "ProductStatus",
    "RegulationKind",
    "RequirementKey",
    "VersionStatus",
]
