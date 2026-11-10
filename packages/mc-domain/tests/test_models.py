"""ORM 层单测，跑在内存 SQLite 上。

不要小看这层：`is_effective_at` 的语义如果错了，
P1「已废止版本引用率 = 0」这条指标就是假的。所以它必须有回归测试。
"""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from mc_domain.enums import (
    CertType,
    ContactType,
    EnergyType,
    ManagementClass,
    RegulationKind,
    RequirementKey,
    VersionStatus,
)
from mc_domain.models import (
    Base,
    Clause,
    ClassificationCode,
    Product,
    RegistrationCert,
    Regulation,
    RegulationVersion,
)


@pytest.fixture
def session():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def test_all_tables_created(session):
    assert sorted(Base.metadata.tables) == [
        "classification_code",
        "clause",
        "product",
        "registration_cert",
        "regulation",
        "regulation_applicability",
        "regulation_version",
    ]


def _seed_tree(s: Session) -> ClassificationCode:
    root = ClassificationCode(
        catalog_version="v1", code="07", level=0, name="医用成像器械",
        path="07", name_path="医用成像器械", sort_order=1,
    )
    s.add(root)
    s.flush()
    child = ClassificationCode(
        catalog_version="v1", code="07-01", level=1, parent_id=root.id, name="超声诊断设备",
        path="07/07-01", name_path="医用成像器械/超声诊断设备", sort_order=1,
    )
    s.add(child)
    s.flush()
    leaf = ClassificationCode(
        catalog_version="v1", code="07-01-01", level=2, parent_id=child.id,
        name="眼科超声诊断设备", path="07/07-01/07-01-01",
        name_path="医用成像器械/超声诊断设备/眼科超声诊断设备",
        management_class=ManagementClass.CLASS_II, examples=["眼科A型超声"], sort_order=1,
    )
    s.add(leaf)
    s.flush()
    return leaf


def test_tree_relationship_and_autoincrement(session):
    leaf = _seed_tree(session)
    # PkType 在 SQLite 上必须退化为 INTEGER 才能自增
    assert leaf.id == 3
    assert leaf.is_leaf is True
    assert leaf.parent.code == "07-01"
    assert leaf.parent.parent.code == "07"
    assert leaf.ancestor_codes == ["07", "07-01", "07-01-01"]


def test_product_and_cert(session):
    leaf = _seed_tree(session)
    product = Product(
        name="眼科超声诊断设备", models=["A-100"], intended_use="观察眼底",
        contact_type=ContactType.SURFACE, energy_type=EnergyType.ACTIVE,
        classification_code_id=leaf.id, management_class=ManagementClass.CLASS_II,
    )
    session.add(product)
    session.flush()

    cert = RegistrationCert(
        cert_no="国械注准20243010123", cert_type=CertType.DOMESTIC_III,
        registrant_name="某医疗科技有限公司", models=["A-100"],
        valid_from=date(2024, 1, 1), valid_to=date(2029, 1, 1), product_id=product.id,
    )
    session.add(cert)
    session.commit()

    assert cert.is_valid_at(date(2026, 11, 10)) is True
    assert cert.is_valid_at(date(2030, 1, 1)) is False
    assert [c.cert_no for c in product.certificates] == ["国械注准20243010123"]


def _two_versions(s: Session) -> Regulation:
    reg = Regulation(
        kind=RegulationKind.GUIDELINE, doc_no="2023年第7号",
        title="某注册审查指导原则", issuer="器审中心",
    )
    s.add(reg)
    s.flush()
    old = RegulationVersion(
        regulation_id=reg.id, version_label="2019版",
        effective_date=date(2019, 1, 1), expiry_date=date(2023, 6, 30),
        status=VersionStatus.SUPERSEDED,
    )
    s.add(old)
    s.flush()
    new = RegulationVersion(
        regulation_id=reg.id, version_label="2023版",
        effective_date=date(2023, 7, 1), status=VersionStatus.CURRENT,
    )
    s.add(new)
    s.flush()
    old.superseded_by_version_id = new.id
    s.commit()
    return reg


def test_superseded_version_still_counts_historically(session):
    """这是本项目最容易搞错的一条语义。

    2019 版现在是 SUPERSEDED，但 2020 年当时它就是现行版。
    如果这里返回 None，「按提问时点检索法规」就整个失效了。
    """
    reg = _two_versions(session)
    assert reg.version_at(date(2020, 1, 1)).version_label == "2019版"
    assert reg.version_at(date(2024, 1, 1)).version_label == "2023版"
    assert reg.version_at(date(2015, 1, 1)) is None


def test_window_closed_by_successor_when_expiry_missing(session):
    """只填后继关系、不补 expiry_date 时，窗口也要能闭合。"""
    reg = Regulation(kind=RegulationKind.GUIDELINE, doc_no="2024年第1号", title="测试")
    session.add(reg)
    session.flush()
    v1 = RegulationVersion(
        regulation_id=reg.id, version_label="v1", effective_date=date(2020, 1, 1),
        status=VersionStatus.SUPERSEDED,
    )
    session.add(v1)
    session.flush()
    v2 = RegulationVersion(
        regulation_id=reg.id, version_label="v2", effective_date=date(2023, 1, 1),
        status=VersionStatus.CURRENT,
    )
    session.add(v2)
    session.flush()
    v1.superseded_by_version_id = v2.id
    session.commit()

    assert v1.effective_window() == (date(2020, 1, 1), date(2022, 12, 31))
    assert v1.is_effective_at(date(2024, 1, 1)) is False


def test_draft_version_never_effective(session):
    reg = Regulation(kind=RegulationKind.STANDARD, doc_no="YY/T 9999-2025", title="征求意见稿")
    session.add(reg)
    session.flush()
    draft = RegulationVersion(
        regulation_id=reg.id, version_label="征求意见稿",
        effective_date=date(2020, 1, 1), status=VersionStatus.DRAFT,
    )
    session.add(draft)
    session.commit()
    assert draft.is_effective_at(date(2021, 1, 1)) is False


def test_clause_requirement_key_is_queryable(session):
    """P2 的必标项引擎只认这个键，所以它必须能直接查。"""
    reg = _two_versions(session)
    clause = Clause(
        regulation_version_id=reg.version_at(date(2026, 1, 1)).id,
        clause_no="第十七条",
        text="医疗器械标签应当标注医疗器械注册证编号。",
        requirement_key=RequirementKey.LABEL_CERT_NO,
        seq=1,
    )
    session.add(clause)
    session.commit()

    found = session.scalars(
        select(Clause).where(Clause.requirement_key == RequirementKey.LABEL_CERT_NO)
    ).one()
    assert found.cite() == "第十七条"
    assert found.requirement_key.value == "label.cert_no"
