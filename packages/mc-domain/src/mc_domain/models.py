"""domain/ 六个实体。

    ClassificationCode   分类目录树节点（子目录 → 一级类别 → 二级类别）
    Product              产品（P1 的输入，标注化后的产品画像）
    RegistrationCert     注册证 / 备案凭证（P2 校验、P3 核对的锚点）
    Regulation           监管文件身份（指导原则 / 标准 / 规章 / 公告）
    RegulationVersion    监管文件的具体版本（**版本时效就活在这里**）
    Clause               条款（可引用、可判定、带机读键）

另有一张关联表 RegulationApplicability（不算第六个实体，是「类别适用哪些法规」的关系），
它是 P1 回答「这个类别适用什么」的唯一入口。

设计取舍见 README.md。
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.ext.mutable import MutableDict, MutableList
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from mc_domain.enums import (
    CertStatus,
    CertType,
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


class Base(DeclarativeBase):
    pass


#: 主键类型。用 BigInteger 保证与 MySQL 生产一致；
#: 但在 SQLite 上退化为 Integer —— SQLite 的 AUTOINCREMENT 只对
#: `INTEGER PRIMARY KEY` 生效，BIGINT 主键插入 NULL 时不会自增。
#: 没有这个 variant，所有用内存 SQLite 跑的单测都会在第一次 insert 时炸掉。
PkType = BigInteger().with_variant(Integer, "sqlite")


def _enum(enum_cls, length: int = 24):
    """统一用字符串列存枚举，避免 MySQL ENUM 的迁移地狱。"""
    from sqlalchemy import Enum as SAEnum

    return SAEnum(
        enum_cls,
        native_enum=False,
        length=length,
        values_callable=lambda e: [m.value for m in e],
    )


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


# ---------------------------------------------------------------------------
# 1. ClassificationCode —— 分类目录树
# ---------------------------------------------------------------------------


class ClassificationCode(TimestampMixin, Base):
    """分类目录中的一个节点。

    树形结构用 parent_id + 冗余的 path / name_path：
      path      = "07/07-01/07-01-01"   代码路径
      name_path = "医用成像器械/超声诊断设备/眼科超声"   名称路径
    name_path 是刻意的反规范化——它让 BM25 与向量检索能直接命中整条路径的语义，
    不必在查询时递归拼名字。

    官方目录的层级深度以实际发布版本为准，这里用 self-reference + level 支持任意深度，
    不把「三级」写死。
    """

    __tablename__ = "classification_code"

    id: Mapped[int] = mapped_column(PkType, primary_key=True, autoincrement=True)

    catalog_version: Mapped[str] = mapped_column(String(32), nullable=False)
    """目录版本，如 "2017" / "2023-adjust"。目录会被公告调整，所以同一 code 可以有多版记录。"""

    code: Mapped[str] = mapped_column(String(32), nullable=False)
    """分类编码："07" / "07-01" / "07-01-01"。"""

    level: Mapped[int] = mapped_column(Integer, nullable=False)
    """0 = 子目录，1 = 一级产品类别，2 = 二级产品类别……"""

    parent_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("classification_code.id", ondelete="RESTRICT"), nullable=True
    )
    path: Mapped[str] = mapped_column(String(255), nullable=False)
    name_path: Mapped[str] = mapped_column(String(512), nullable=False)

    name: Mapped[str] = mapped_column(String(255), nullable=False)

    management_class: Mapped[ManagementClass | None] = mapped_column(
        _enum(ManagementClass, 8), nullable=True
    )
    """只有可定类的节点才有值。子目录级通常为 NULL。"""

    definition: Mapped[str | None] = mapped_column(Text, nullable=True)
    """类别描述 / 子目录说明。"""

    scope_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    """「本类包括……」——正向范围。"""

    exclusion_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    """「本类不包括……」——**排除法判定的唯一依据**，P1 第 2 阶段读它。"""

    examples: Mapped[list[str]] = mapped_column(MutableList.as_mutable(JSON), default=list)
    """品名举例。P1 的匹配语料就是它，也是最容易扩充的部分。"""

    effective_from: Mapped[date | None] = mapped_column(Date, nullable=True)
    effective_to: Mapped[date | None] = mapped_column(Date, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    sort_order: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    parent: Mapped[ClassificationCode | None] = relationship(
        remote_side="ClassificationCode.id", back_populates="children"
    )
    children: Mapped[list[ClassificationCode]] = relationship(
        back_populates="parent", order_by="ClassificationCode.sort_order"
    )

    __table_args__ = (
        UniqueConstraint("catalog_version", "code", name="uq_cc_version_code"),
        Index("ix_cc_version_parent", "catalog_version", "parent_id"),
        Index("ix_cc_path", "path"),
        CheckConstraint("level >= 0", name="ck_cc_level_nonneg"),
    )

    @property
    def is_leaf(self) -> bool:
        return not self.children

    @property
    def ancestor_codes(self) -> list[str]:
        return [c for c in self.path.split("/") if c]

    def is_within(self, other: ClassificationCode) -> bool:
        """self 是否在 other 的子树内（用于「取最具体类别」的规则）。"""
        return self.path == other.path or self.path.startswith(other.path + "/")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<ClassificationCode {self.catalog_version}:{self.code} {self.name}>"


# ---------------------------------------------------------------------------
# 2. Product —— 产品画像
# ---------------------------------------------------------------------------


class Product(TimestampMixin, Base):
    """一个待定类 / 已定类的产品。

    这里的字段不是「随便记点信息」，而是**分类判定树的输入向量**：
    contact_type / contact_duration / energy_type / is_sterile / is_reusable
    是《医疗器械分类规则》定类的核心维度，缺一项判定就无法收敛（→ needs_more_info）。
    """

    __tablename__ = "product"

    id: Mapped[int] = mapped_column(PkType, primary_key=True, autoincrement=True)
    tenant_id: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False, index=True)

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    models: Mapped[list[str]] = mapped_column(MutableList.as_mutable(JSON), default=list)
    """型号规格列表。P3 的跨文档一致性就以它为基准。"""

    structure_composition: Mapped[str | None] = mapped_column(Text, nullable=True)
    intended_use: Mapped[str | None] = mapped_column(Text, nullable=True)
    principle: Mapped[str | None] = mapped_column(Text, nullable=True)
    materials: Mapped[list[str]] = mapped_column(MutableList.as_mutable(JSON), default=list)

    contact_type: Mapped[ContactType | None] = mapped_column(_enum(ContactType), nullable=True)
    contact_duration: Mapped[ContactDuration | None] = mapped_column(
        _enum(ContactDuration), nullable=True
    )
    energy_type: Mapped[EnergyType | None] = mapped_column(_enum(EnergyType), nullable=True)
    is_sterile: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    is_reusable: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    """None 表示「尚未确认」，与 False 严格区分——这是 needs_more_info 的判定依据。"""

    # --- P1 写入的分类结论 ---
    classification_code_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("classification_code.id", ondelete="SET NULL"), nullable=True
    )
    management_class: Mapped[ManagementClass | None] = mapped_column(
        _enum(ManagementClass, 8), nullable=True
    )
    """冗余自 ClassificationCode，便于 P2/P3 直接读，不必 join 树。"""

    classified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    classification_method: Mapped[str | None] = mapped_column(String(64), nullable=True)
    """如 "tree_rule_v1"。—— 结论必须可追溯是哪个引擎给的。"""

    status: Mapped[ProductStatus] = mapped_column(
        _enum(ProductStatus), default=ProductStatus.DRAFT, nullable=False
    )

    classification_code: Mapped[ClassificationCode | None] = relationship()
    certificates: Mapped[list[RegistrationCert]] = relationship(
        back_populates="product", cascade="all, delete-orphan"
    )

    __table_args__ = (Index("ix_product_tenant_status", "tenant_id", "status"),)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Product {self.id} {self.name}>"


# ---------------------------------------------------------------------------
# 3. RegistrationCert —— 注册证 / 备案凭证
# ---------------------------------------------------------------------------


class RegistrationCert(TimestampMixin, Base):
    """注册证或第一类备案凭证。

    这张表是三个应用的公共锚点：
      P1  读它来回答「已有哪些同类产品注册」
      P2  拿 cert_no 校验标签上印的注册证号格式与真伪
      P3  拿 models / structure_composition 校验申报资料是否与证载一致
    """

    __tablename__ = "registration_cert"

    id: Mapped[int] = mapped_column(PkType, primary_key=True, autoincrement=True)
    tenant_id: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False, index=True)
    product_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("product.id", ondelete="CASCADE"), nullable=True
    )

    cert_no: Mapped[str] = mapped_column(String(64), nullable=False)
    """如 国械注准20243010123 / 苏械注准20242020001 / 国械备20240001。"""

    cert_type: Mapped[CertType] = mapped_column(_enum(CertType), nullable=False)

    registrant_name: Mapped[str] = mapped_column(String(255), nullable=False)
    registrant_address: Mapped[str | None] = mapped_column(String(512), nullable=True)
    registrant_contact: Mapped[str | None] = mapped_column(String(255), nullable=True)

    models: Mapped[list[str]] = mapped_column(MutableList.as_mutable(JSON), default=list)
    structure_composition: Mapped[str | None] = mapped_column(Text, nullable=True)
    intended_use: Mapped[str | None] = mapped_column(Text, nullable=True)

    issue_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    valid_from: Mapped[date | None] = mapped_column(Date, nullable=True)
    valid_to: Mapped[date | None] = mapped_column(Date, nullable=True)

    province_code: Mapped[str | None] = mapped_column(String(8), nullable=True)
    """二类境内证的省级行政区划码（cert_no 第 1 位汉字对应的省份）。"""
    status: Mapped[CertStatus] = mapped_column(
        _enum(CertStatus), default=CertStatus.ACTIVE, nullable=False
    )

    product: Mapped[Product | None] = relationship(back_populates="certificates")

    __table_args__ = (
        UniqueConstraint("cert_no", name="uq_cert_no"),
        Index("ix_cert_product", "product_id"),
        Index("ix_cert_valid_to", "status", "valid_to"),
    )

    def is_valid_at(self, as_of: date) -> bool:
        """P2/P3 用：证在某日是否有效。"""
        if self.status is not CertStatus.ACTIVE:
            return False
        if self.valid_from and as_of < self.valid_from:
            return False
        if self.valid_to and as_of > self.valid_to:
            return False
        return True

    def __repr__(self) -> str:  # pragma: no cover
        return f"<RegistrationCert {self.cert_no}>"


# ---------------------------------------------------------------------------
# 4. Regulation —— 监管文件身份
# ---------------------------------------------------------------------------


class Regulation(TimestampMixin, Base):
    """一份监管文件的**身份**，与「内容」分离。

    身份 = kind + doc_no + title + issuer，在多版之间不变。
    内容 = 各个 RegulationVersion（可能有多个版本并存、替代、废止）。

    这是把「指导原则」和「标准」合并成一张表的理由：
    两者的版本生命周期语义完全同构，只是 doc_no 格式不同。
    """

    __tablename__ = "regulation"

    id: Mapped[int] = mapped_column(PkType, primary_key=True, autoincrement=True)

    kind: Mapped[RegulationKind] = mapped_column(_enum(RegulationKind), nullable=False)
    doc_no: Mapped[str] = mapped_column(String(128), nullable=False)
    """指导原则用文号（如「2023年第7号」），标准用标准号（如「YY/T 0466.1-2023」）。"""

    title: Mapped[str] = mapped_column(String(512), nullable=False)
    issuer: Mapped[str | None] = mapped_column(String(255), nullable=True)
    scope_note: Mapped[str | None] = mapped_column(Text, nullable=True)

    meta: Mapped[dict] = mapped_column(MutableDict.as_mutable(JSON), default=dict)
    """类型特有条目：标准性质（强制/推荐）、采标关系（ISO 15223-1）、是否等同采用等。"""

    versions: Mapped[list[RegulationVersion]] = relationship(
        back_populates="regulation", order_by="RegulationVersion.effective_date"
    )

    __table_args__ = (UniqueConstraint("kind", "doc_no", name="uq_regulation_kind_docno"),)

    def version_at(self, as_of: date) -> RegulationVersion | None:
        """取 as_of 当天生效的版本。多个都生效时取最晚发布的那个。"""
        hits = [v for v in self.versions if v.is_effective_at(as_of)]
        return max(hits, key=lambda v: v.effective_date or date.min) if hits else None

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Regulation {self.kind}:{self.doc_no}>"


# ---------------------------------------------------------------------------
# 5. RegulationVersion —— 版本（时效的核心）
# ---------------------------------------------------------------------------


class RegulationVersion(TimestampMixin, Base):
    """一次发布的具体版本。

    「同一个问题在旧版和新版下答案相反」这件事，就靠
    effective_date / expiry_date / status / superseded_by 四个字段表达。
    P1 检索时按 as_of 过滤；答案里强制带上版本号与日期。
    """

    __tablename__ = "regulation_version"

    id: Mapped[int] = mapped_column(PkType, primary_key=True, autoincrement=True)
    regulation_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("regulation.id", ondelete="CASCADE"), nullable=False
    )

    version_label: Mapped[str] = mapped_column(String(64), nullable=False)
    """如 "2017版" / "2023修订" / "第2版"。"""

    publish_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    effective_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    expiry_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    """停止实施日。None = 目前仍有效（配合 status 判断）。"""

    status: Mapped[VersionStatus] = mapped_column(
        _enum(VersionStatus), default=VersionStatus.CURRENT, nullable=False
    )

    superseded_by_version_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("regulation_version.id", ondelete="SET NULL"), nullable=True
    )

    source_url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    checksum: Mapped[str | None] = mapped_column(String(64), nullable=True)
    """原文 hash，用于「法规是否被静默替换」的检测。"""

    regulation: Mapped[Regulation] = relationship(back_populates="versions")
    clauses: Mapped[list[Clause]] = relationship(
        back_populates="version", cascade="all, delete-orphan", order_by="Clause.seq"
    )
    superseded_by: Mapped[RegulationVersion | None] = relationship(
        remote_side="RegulationVersion.id", foreign_keys=[superseded_by_version_id]
    )

    __table_args__ = (
        UniqueConstraint("regulation_id", "version_label", name="uq_rv_regulation_label"),
        Index("ix_rv_status_effective", "status", "effective_date"),
        CheckConstraint("publish_date IS NULL OR effective_date IS NULL OR effective_date >= publish_date",
                        name="ck_rv_effective_after_publish"),
    )

    def is_effective_at(self, as_of: date) -> bool:
        """判断该版本在 as_of 当天是否生效。

        **关键陷阱：status 描述的是「现在」，不是「as_of 那时」。**
        一个现在已被替代的版本，在它自己的生效窗口内必须仍然算数——
        否则系统就无法回答「2020 年当时依据的是哪一版」，而这恰恰是 P1 的存在理由。
        所以这里只认日期窗口，status 仅用于排除从未生效的征求意见稿。

        推论：SUPERSEDED / ABOLISHED 的版本**必须**有 expiry_date，
        或有 superseded_by 且其后继带 effective_date，否则窗口是开区间。
        """
        if self.status is VersionStatus.DRAFT:
            return False
        start, end = self.effective_window()
        if start and as_of < start:
            return False
        if end and as_of > end:
            return False
        return True

    def effective_window(self) -> tuple[date | None, date | None]:
        """(生效日, 失效日)。失效日优先取 expiry_date。

        没填 expiry_date 时退回用后继版本的生效日前一天推导——
        这样「只维护后继关系、不补失效日」也不会留下开区间。
        """
        end = self.expiry_date
        if end is None and self.superseded_by is not None:
            successor_start = self.superseded_by.effective_date
            if successor_start is not None:
                end = successor_start - timedelta(days=1)
        return self.effective_date, end

    def __repr__(self) -> str:  # pragma: no cover
        return f"<RegulationVersion {self.version_label} {self.status.value}>"


# ---------------------------------------------------------------------------
# 6. Clause —— 条款
# ---------------------------------------------------------------------------


class Clause(TimestampMixin, Base):
    """可引用的最小单位。

    P1 的答案是「依据 XX 指导原则 5.2 条」；
    P2 的检查表项是「依据《说明书和标签管理规定》第 X 条」；
    P3 的冲突结论要定位到「文件 + 章节 + 原文」。
    三个应用引用的是同一种东西，所以它是一个实体而不是三段文本。
    """

    __tablename__ = "clause"

    id: Mapped[int] = mapped_column(PkType, primary_key=True, autoincrement=True)
    regulation_version_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("regulation_version.id", ondelete="CASCADE"), nullable=False
    )

    clause_no: Mapped[str] = mapped_column(String(64), nullable=False)
    """结构化条款号：「第十七条」/「5.2.1」/「附录A.3」。"""

    heading: Mapped[str | None] = mapped_column(String(255), nullable=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)

    seq: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    """版本内排序，用于「上一条 / 下一条」与检索上下文扩展。"""

    kind: Mapped[ClauseKind] = mapped_column(
        _enum(ClauseKind), default=ClauseKind.REQUIREMENT, nullable=False
    )

    parent_clause_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("clause.id", ondelete="CASCADE"), nullable=True
    )
    """条款自身的层级（章 → 节 → 条）。"""

    requirement_key: Mapped[RequirementKey | None] = mapped_column(
        _enum(RequirementKey, 48), nullable=True
    )
    """机读键。P2 的必标项引擎只认这个，不解析自然语言。
    新增一个标签审查项 = 加枚举值 + 一条带该键的条款。"""

    version: Mapped[RegulationVersion] = relationship(back_populates="clauses")
    parent: Mapped[Clause | None] = relationship(
        remote_side="Clause.id", back_populates="children"
    )
    children: Mapped[list[Clause]] = relationship(back_populates="parent")

    __table_args__ = (
        Index("ix_clause_version_seq", "regulation_version_id", "seq"),
        Index("ix_clause_requirement_key", "requirement_key"),
    )

    def cite(self) -> str:
        """统一引用格式，三个应用共用，保证报告里引用风格一致。"""
        return f"{self.clause_no}" if not self.heading else f"{self.clause_no} {self.heading}"

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Clause {self.clause_no}>"


# ---------------------------------------------------------------------------
# 关联：类别 ↔ 法规（P1「适用哪些法规」的唯一入口）
# ---------------------------------------------------------------------------


class RegulationApplicability(TimestampMixin, Base):
    """某个分类编码适用某个法规版本。

    做成带属性的关联表而不是纯 M2M，因为「适用」本身有语义：
    是强制适用还是参考？依据哪一条判定的？
    """

    __tablename__ = "regulation_applicability"

    id: Mapped[int] = mapped_column(PkType, primary_key=True, autoincrement=True)

    classification_code_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("classification_code.id", ondelete="CASCADE"), nullable=False
    )
    regulation_version_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("regulation_version.id", ondelete="CASCADE"), nullable=False
    )

    is_mandatory: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    basis_clause_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("clause.id", ondelete="SET NULL"), nullable=True
    )
    """「依据哪一条说它适用」——让适用关系本身也可追溯。"""

    note: Mapped[str | None] = mapped_column(Text, nullable=True)

    classification_code: Mapped[ClassificationCode] = relationship()
    regulation_version: Mapped[RegulationVersion] = relationship()
    basis_clause: Mapped[Clause | None] = relationship(foreign_keys=[basis_clause_id])

    __table_args__ = (
        UniqueConstraint(
            "classification_code_id", "regulation_version_id", name="uq_applicability_pair"
        ),
        Index("ix_applicability_code", "classification_code_id"),
    )
