"""领域枚举。

约定：所有枚举继承 (str, Enum)，既能直接比较字符串，也能被 Pydantic / SQLAlchemy 双向使用。
数据库列一律存 value 字符串（非数据库原生 ENUM），保证 MySQL / PostgreSQL 可移植。
"""

from __future__ import annotations

from enum import Enum


class ManagementClass(str, Enum):
    """医疗器械管理类别。决定注册路径、审评机构与资料要求。"""

    CLASS_I = "I"      # 第一类：备案管理
    CLASS_II = "II"    # 第二类：省级注册
    CLASS_III = "III"  # 第三类：国家局注册

    @property
    def rank(self) -> int:
        """用于「多类别取最高」的比较。"""
        return {"I": 1, "II": 2, "III": 3}[self.value]


class ClassificationOutcome(str, Enum):
    """分类界定的**四种**结论，而不是三种。

    「不作为医疗器械」不是管理类别，而是对定义的否定——它必须排在类别判定之前。
    官方《分类界定结果汇总》里这类结论占比很高（大量日常用品、非医疗用途的
    电子产品都会被申请界定），只建 I/II/III 三个值会让系统无法表达真实结论，
    评测时也无从统计。

    判定顺序（与《医疗器械分类规则》一致）：
        1. 先判"是否属于医疗器械" → NOT_A_DEVICE 或继续
        2. 再判管理类别 → CLASS_I / CLASS_II / CLASS_III
    """

    CLASS_I = "I"
    CLASS_II = "II"
    CLASS_III = "III"
    NOT_A_DEVICE = "not_a_device"

    @classmethod
    def from_management_class(
        cls, mc: ManagementClass | None
    ) -> ClassificationOutcome | None:
        if mc is None:
            return None
        return cls(mc.value)

    @property
    def is_device(self) -> bool:
        return self is not ClassificationOutcome.NOT_A_DEVICE

    @property
    def management_class(self) -> ManagementClass | None:
        """能映射回 I/II/III 时返回，NOT_A_DEVICE 返回 None。"""
        if self is ClassificationOutcome.NOT_A_DEVICE:
            return None
        return ManagementClass(self.value)

    @property
    def rank(self) -> int:
        """NOT_A_DEVICE 排最低，便于「取最高」时被正常类别压过。"""
        return {"not_a_device": 0, "I": 1, "II": 2, "III": 3}[self.value]


class RegulationKind(str, Enum):
    """监管文件类型。

    注意：这里把「指导原则」和「标准」收进同一个 Regulation 表，用 kind 区分。
    它们是同一种东西的两面——都带文号、都有版本生命周期、都产出可引用的条款，
    差别只在标识符格式与强制力。分开建表会立刻遇到 Clause 的多态外键问题。
    """

    GUIDELINE = "guideline"        # 器审中心指导原则（技术审评口径）
    STANDARD = "standard"          # GB / YY 标准
    REGULATION = "regulation"      # 条例、部门规章、规范性文件
    NOTICE = "notice"              # 公告、通告、分类界定通知
    CATALOG = "catalog"            # 分类目录本身（可自引用做版本）


class VersionStatus(str, Enum):
    """法规版本生命周期。P1 的「版本时效」全靠这个字段。"""

    DRAFT = "draft"              # 征求意见稿，不可作为依据
    PROPOSED = "proposed"        # 已发布未实施，可提示「将于 YYYY-MM-DD 实施」
    CURRENT = "current"          # 现行有效
    SUPERSEDED = "superseded"    # 已被新版替代
    ABOLISHED = "abolished"      # 废止，且无替代

    @property
    def citable(self) -> bool:
        """能否作为「今天生效的依据」被引用。"""
        return self is VersionStatus.CURRENT


class ContactType(str, Enum):
    """与人体的接触类型——《医疗器械分类规则》定类的首要维度。"""

    NONE = "none"                        # 不接触人体
    SURFACE = "surface"                  # 表面接触
    INDIRECT = "indirect"                # 间接接触（经体液/管路）
    INVASIVE = "invasive"                # 侵入（经体表进入体内）
    IMPLANT = "implant"                  # 植入


class ContactDuration(str, Enum):
    """接触时间（暂时 / 短期 / 长期）。"""

    NONE = "none"
    TRANSIENT = "transient"   # 暂时，< 60 min
    SHORT = "short"           # 短期，< 30 d
    LONG = "long"             # 长期，>= 30 d


class EnergyType(str, Enum):
    """有源 / 无源。分类目录子目录 01–07 多为有源，08–22 多为无源。"""

    ACTIVE = "active"
    PASSIVE = "passive"


class ProductStatus(str, Enum):
    DRAFT = "draft"
    CLASSIFYING = "classifying"
    CLASSIFIED = "classified"      # P1 已给出分类结论
    REGISTERED = "registered"      # 已有注册证（P2 / P3 可用）


class CertType(str, Enum):
    """注册证类型。境内三/二类、进口、以及第一类的备案凭证。"""

    DOMESTIC_III = "domestic_iii"  # 国械注准
    DOMESTIC_II = "domestic_ii"    # 省械注准
    IMPORTED_III = "imported_iii"  # 国械注进
    IMPORTED_II = "imported_ii"    # 国械注进
    FILING = "filing"              # 第一类备案凭证


class CertStatus(str, Enum):
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"        # 撤销
    CANCELLED = "cancelled"    # 注销
    SUSPENDED = "suspended"    # 暂停


class RequirementKey(str, Enum):
    """机读的条款键——P2 的必标项就是靠它把「法规条款」映射到「检查表项」。

    这是 P1/P2/P3 之间少数的硬编码耦合点之一，故意做成闭合枚举：
    新增必标项 = 加一行枚举 + 一条 Clause，不需要改判定引擎。
    """

    LABEL_PRODUCT_NAME = "label.product_name"
    LABEL_MODEL_SPEC = "label.model_spec"
    LABEL_REGISTRANT_NAME = "label.registrant_name"
    LABEL_REGISTRANT_ADDRESS = "label.registrant_address"
    LABEL_REGISTRANT_CONTACT = "label.registrant_contact"
    LABEL_CERT_NO = "label.cert_no"
    LABEL_PRODUCTION_LICENSE = "label.production_license"
    LABEL_MANUFACTURE_DATE = "label.manufacture_date"
    LABEL_EXPIRY_DATE = "label.expiry_date"
    LABEL_STORAGE = "label.storage"
    LABEL_UDI = "label.udi"
    LABEL_STERILE_MARK = "label.sterile_mark"
    LABEL_SINGLE_USE = "label.single_use"
    LABEL_WARNING = "label.warning"
    LABEL_IFU_REF = "label.ifu_ref"


class ClauseKind(str, Enum):
    """条款颗粒度。审查报告里引用时要区分「要求」和「解释」。"""

    REQUIREMENT = "requirement"  # 构成强制性要求（可作为违规判定依据）
    GUIDANCE = "guidance"        # 技术审评口径、建议
    DEFINITION = "definition"    # 定义、术语
    EXCLUSION = "exclusion"      # 排除性说明（分类排除法用这个）
