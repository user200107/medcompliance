"""P3 DossierReview 卷宗审查 —— 申报资料的跨文档一致性与完整性。

**它解决的不是检索问题**：RAG 能回答"技术要求里写了什么"，
但回答不了「技术要求写 Ø3.20、检验报告写 Ø3.50，谁对？」

**当前实现范围**
- ✅ 章节完整性（按 121 号通告的 RPS 简化目录）
- ✅ 6 类结构化属性的跨文档一致性（型号/尺寸/灭菌方式/有效期/生产地址/结构组成）
- ✅ 值归一化：写法差异不报、实质差异必报
- ✅ 三态结论：冲突 / 一致 / 无法核对
- ⬜ 不涵盖技术要求全文与临床评价的实质内容（已在 caveats 中声明）
"""

from __future__ import annotations

from mc_dossier.main import create_app
from mc_dossier.version import __version__

__all__ = ["create_app", "__version__"]
