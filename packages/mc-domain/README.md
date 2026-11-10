# MedCompliance Suite — `domain/` 领域底座 + P1 分类判定

这是套件的地基，也是简历上最容易被追问的部分。它要同时喂饱三个应用：

| 应用 | 用到底座的什么 |
|---|---|
| **P1 MediReg**（注册路径问答） | 分类目录树 + 法规版本 + 检索 |
| **P2 LabelGuard**（标签视觉审核） | `RegistrationCert` 校验注册证号、`Clause.requirement_key` 驱动必标项 |
| **P3 DossierReview**（申报资料审查） | `Product.models` 做跨文档一致性基准、`Clause` 做引用溯源 |

当前状态：**`domain/` 与 P1 的确定性判定流水线已实现，45 个单测全部通过**。
检索用的是零依赖基线实现（可替换为 Milvus+BM25），LLM 槽位抽取未接入。

---

## 1. 六个实体

```
        ┌──────────────────┐
        │ ClassificationCode│  分类目录树
        │  code / path      │  07 → 07-01 → 07-01-01
        │  management_class │
        │  exclusion_note ──┼──┐ 排除法唯一依据
        └────────┬─────────┘  │
                 │ 1:N        │
        ┌────────▼─────────┐  │
        │   Applicability   │  │  类别 ↔ 法规（带 is_mandatory / 依据条款）
        └────────┬─────────┘  │
                 │ N:1        │
        ┌────────▼──────────┐ │
        │ RegulationVersion │ │  版本时效的核心
        │  effective_date   │ │
        │  status           │ │
        │  superseded_by ───┼─┼─┐
        └────────┬──────────┘ │ │
                 │ 1:N        │ │
        ┌────────▼─────────┐  │ │
        │      Clause       │◄─┘ │  requirement_key（P2 必标项）
        │  clause_no / text │    │
        └───────────────────┘    │
        ┌────────▼──────────┐    │
        │     Regulation     │◄───┘  文件身份
        │  kind / doc_no     │
        └───────────────────┘

        ┌──────────┐  1:N   ┌──────────────────┐
        │ Product  │───────►│ RegistrationCert │
        └──────────┘        └──────────────────┘
              │ classification_code_id
              └──────────────────────► ClassificationCode
```

| 实体 | 一句话 | 关键设计 |
|---|---|---|
| `ClassificationCode` | 分类目录节点 | `path` / `name_path` 冗余存储，让检索能命中整条路径语义 |
| `Product` | 产品画像 | 槽位（接触类型 / 有无源 / 无菌…）就是判定树的输入向量；`None` ≠ `False` |
| `RegistrationCert` | 注册证 / 备案凭证 | 三个应用的公共锚点；`is_valid_at()` 做时点校验 |
| `Regulation` | 文件身份 | `kind` + `doc_no` 唯一，多版之间不变 |
| `RegulationVersion` | 具体版本 | **版本时效全在这里**：生效日 / 失效日 / 状态 / 被谁替代 |
| `Clause` | 条款 | `requirement_key` 是机读键，P2 的必标项只认它，不解析自然语言 |

### 1.1 为什么把「指导原则」和「标准」合并成 `Regulation`

原本的设计是 `Guideline` 和 `Standard` 两张表。落地时发现三个问题：

1. **`Clause` 立刻变成多态外键**。条款要么属于指导原则要么属于标准，
   要么加 `owner_type + owner_id`（无法外键约束），要么建两张 `Clause` 表（所有查询写两遍）。
2. **两者的生命周期完全同构**：都有发布日、实施日、废止、被替代。
   分开建表等于把同一套版本逻辑写两遍。
3. **它们的差别只有标识符格式和强制力**：文号 vs 标准号、强制 vs 推荐。
   这正好是 `kind` + `meta(JSONB)` 能表达的东西。

代价：想只查指导原则时要加 `WHERE kind='guideline'`。这个代价可以忽略。

### 1.2 为什么单独拆出 `RegulationVersion`

因为**「同一问题在旧版和新版下答案相反」是 P1 存在的理由**。
如果版本只是 `Regulation` 上的几个字段，你就无法同时保存"现行版"和"已废止版"的条款，
也就无法回答"这个结论依据的是哪一版、那一版现在还作数吗"。

`RegulationVersion.is_effective_at(as_of)` 是版本路由的唯一入口。

**这里埋着整个项目最容易搞错的一条语义：**

> `status` 描述的是「现在」，不是「as_of 那时」。

一个现在标为 `SUPERSEDED` 的版本，在 2019–2023 那个窗口里**就是当时的现行版**。
如果按 `status == CURRENT` 过滤，`version_at(2020-01-01)` 会返回 `None`，
「按提问时点检索法规」这个功能整个失效——而这恰恰是 P1 存在的理由。

所以正确的判定是**只认日期窗口**，`status` 仅用于排除从未生效的征求意见稿：

```python
def is_effective_at(self, as_of: date) -> bool:
    if self.status is VersionStatus.DRAFT:
        return False
    start, end = self.effective_window()   # 失效日可由后继版本的生效日推导
    return (not start or as_of >= start) and (not end or as_of <= end)
```

这个 bug 是我在写完 smoke test 之后才发现的（见 §5）。**它是那种"跑得通、单测不写就永远发现不了"的错误**——
功能表面上正常，只有问"2020 年当时依据哪一版"时才会暴露。

### 1.3 `Clause.requirement_key` —— 三个应用唯一的硬耦合点

P2 的必标项引擎**不解析自然语言，只匹配枚举键**：

```python
LABEL_CERT_NO = "label.cert_no"   # ← 标签必须标注册证编号
```

于是「新增一个标签审查项」= 加一个枚举值 + 一条带该键的 `Clause`，**不改判定引擎**。
这是把领域知识放进数据而不是代码的具体做法，也是 P2 能快速扩充品类的原因。

---

## 2. 分类判定树

### 2.1 结构

```
level 0   07 医用成像器械         exclusion_note: 本类不包括：超声治疗设备…
          ├─ scope_note: 本类包括超声、X射线、磁共振…
level 1   ├─ 07-01 超声诊断设备
level 2   │   ├─ 07-01-01 眼科超声诊断设备   management_class: II
          │   └─ 07-01-02 腹部超声诊断设备   management_class: II
          09 物理治疗器械
           └─ 09-01 超声治疗设备              management_class: III
```

层级不是写死的：`parent_id` 自引用 + `level`，官方目录几层都吃得下。
**只有 `management_class` 非空的节点才参与定类**，纯分类容器节点（如 07-01）不参与排名。

### 2.2 五个阶段（模型只站在第 0 阶段）

| 阶段 | 谁在做 | 做什么 | 是否确定 |
|---|---|---|---|
| `normalize` | **LLM** | 口语 → `ProductProfile` 槽位 | 否（唯一允许不确定的地方） |
| `coarse` | 检索 | 粗筛子目录，`coarse_limit=6` | 是 |
| `exclude` | 规则 | 排除说明命中即剔除，进 `rejected` 留痕 | 是 |
| `match` | 检索 | 在存活子树内匹配到可定类叶子 | 是 |
| `rule` | 规则 | 取最具体 → 取最高类别 | 是 |
| `rank` | 规则 | Top-3 + 歧义检测 | 是 |

**设计主张：模型只做翻译，判定交给确定性引擎。**
LLM 一旦被允许直接输出分类编码，你就失去了可复现性和可解释性，
而这两样正是注册专员愿意为工具付费的全部理由。

### 2.3 排除法的实现（为什么用子串而不是分词）

```python
extract_exclusion_terms("本类不包括：超声治疗设备、超声理疗仪")
# → ["超声治疗设备", "超声理疗仪"]
```

剥引导词 → 按标点与连词切 → 丢掉虚指词（`其他` / `等`）和过短片段 →
对产品文本做**子串匹配**。

刻意不用分词：分词会引入不确定性，而这一步必须完全确定、可复现、可单测。
代价是召回偏低（"超声理疗仪"无法匹配"超声理疗设备"）——这是**保守**选择：
宁可少排除（候选变多，进 `rejected` 可见，用户能推翻），也不错排除。

### 2.4 歧义不猜

Top-1 与 Top-2 分数差小于 15% 时，状态改成 `needs_more_info`，
并生成一个**指向性追问**（"「眼科超声诊断设备」与「腹部超声诊断设备」都可能适用，
能否补充预期用途"）。每轮最多问 2 个槽位——问多了用户会放弃。

---

## 3. 目录数据契约

`app/domain/data/catalog_sample.csv`：

| 列 | 必填 | 说明 |
|---|---|---|
| `code` | ✅ | 层级编码，`07` / `07-01` / `07-01-01` |
| `name` | ✅ | 类别名称 |
| `parent_code` | 根节点留空 | 父级编码 |
| `management_class` | 叶子必填 | `I` / `II` / `III`，也接受「III类」「第三类」 |
| `definition` | | 类别定义 |
| `scope_note` | | 「本类包括……」 |
| `exclusion_note` | | 「本类不包括……」← 排除法依据 |
| `examples` | | 品名举例，`|` 分隔 |
| `sort_order` | | 排序 |

> ⚠️ **`catalog_sample.csv` 是结构演示用的合成目录，不是真实的《医疗器械分类目录》。**
> 真实目录请自行导入，字段结构一致。导入前会跑树完整性校验，
> 层级断裂、编码前缀不符、叶子缺管理类别、成环——任一问题都会中止导入并列出明细。

`level` 由父链推导，**不接受 CSV 输入**——防止层级写错还蒙混过关。

---

## 4. 跑起来

```bash
# 在仓库根目录执行
python -m pytest packages/mc-domain -q    # 45 passed
python scripts/demo_classify.py           # 看完整依据链
```

无需数据库、无需网络、无需 API Key。这是刻意的：分类规则会反复调整，
如果每改一次都要起全套中间件，最后就没人跑了。

### 实际输出（`demo_classify.py`）

```
状态：classified    引擎：tree_rule_v1    耗时：0.0 ms

── 判定路径 ──
1. normalize  识别到 0 个必需槽位缺失
2. coarse     命中 2 个子目录：医用成像器械、物理治疗器械
3. exclude    排除说明剔除 1 个子目录  [exclusion]
4. match      在 1 个子目录内匹配到 1 个可定类类别
5. rank       输出 Top-1

── 结论 ──
分类编码：09-01   管理类别：III 类
类别路径：物理治疗器械/超声治疗设备

── 已排除 ──
  × 07 医用成像器械：命中排除说明：超声治疗设备
     依据：「超声治疗设备」命中 07 的排除说明：本类不包括：超声治疗设备、超声理疗仪
```

注意 `已排除` 这一段——**「为什么不是它」和「为什么是它」一样是交付物**。
注册专员推翻结论时，需要看到系统原本考虑过哪些类别、因为什么放弃。

---

## 5. 诚实清单

### 已实现且已验证

- 六个实体 + 关联表的完整 ORM 模型（索引、唯一约束、自引用关系、状态机），
  在内存 SQLite 上建表、插入、关系加载、`requirement_key` 查询全部跑通
- 分类目录树：任意深度、路径推导、完整性校验（断链 / 前缀 / 叶子缺类别 / 成环）
- 排除法：引导词剥离、虚指词过滤、确定性子串匹配
- 判定流水线：五阶段 + 依据链 + 被排除候选留痕 + 歧义转追问 + 缺槽位转追问
- 规则：取最具体、取最高类别（纯函数，独立单测）
- 版本时效：`is_effective_at` / `effective_window`，含历史查询与后继推导闭环
- 零依赖检索基线（IDF 加权字符 bigram）
- **45 个单测，写完即跑**（含评测层的标签归一化与指标）

### 写完之后才发现的四个 bug（都是测试逼出来的）

| # | 症状 | 根因 | 为什么危险 |
|---|---|---|---|
| 1 | 目录树所有非根节点 `level=0` | `level = _depth_of(parent_code)` 少加 1 | 树校验必挂，但只有导入真目录才暴露 |
| 2 | 「本类不包括其他」把**所有**类别都排除掉 | 虚指词 `其他` 被当成可匹配类别词 | 一条措辞含糊的目录说明会废掉整个子目录 |
| 3 | 所有内存 SQLite 单测在第一次 insert 就炸 | SQLite 只对 `INTEGER PRIMARY KEY` 自增，`BIGINT` 主键插入 NULL 不自增 | 不修的话这层永远没法测 |
| 4 | `version_at("2020-01-01")` 返回 `None` | 用 `status == CURRENT` 过滤，而 `status` 描述的是"现在" | **版本时效这个核心卖点直接失效**（详见 §1.2） |

> 第 4 个是这轮最有价值的产出：它不是拼写错误，是**语义错误**。
> 面试时讲这一条，比讲十个功能有说服力。

### 未实现

| 项 | 说明 | 优先级 |
|---|---|---|
| LLM 槽位抽取 | `ProfileNormalizer` 只有协议，没有真实实现。真实版要**解析失败抛异常**，不许返回空画像 | P0，接 P1 时第一个做 |
| 真实目录导入 | 目前只有 8 行合成数据 | P0 |
| 混合检索 | 现在是字符 bigram；生产应接 Milvus + BM25 + RRF + rerank（复用 fintech-brain） | P1 |
| `applicability_loader` | 协议已留，`RegulationApplicability` 的查询未写 | P1 |
| Alembic 迁移 | 只有 model，没有迁移脚本 | P1 |
| 分类规则条款号 | `Rule.basis` 目前是描述性文字，正式版要填准确条款号 | P2 |
| 目录多版本并存 | 表结构支持（`catalog_version` 唯一键含它），但"同编码两版同时存在"的查询未实现 | P2 |

### 已知局限

1. **排除匹配召回偏低**。子串匹配无法处理"超声理疗仪 → 超声理疗设备"这类变体。
   改进方向：在确定性匹配之外加一路同义词扩展，但**扩展结果要标注来源**，
   否则会把"可解释"变成"半可解释"。
2. **`_SPLIT_RE` 里的「及」「或」「和」是字符级切分**，会误切内含这些字的词（如"涉及"）。
   真实目录导入后需要抽查排除说明的切分结果。
3. **`coarse_limit=6` 是拍脑袋的经验值**，应该在 200 条评测集上扫一遍再定。

---

## 6. 与 P2 / P3 的接口

| 下游需要 | 底座提供 | 形态 |
|---|---|---|
| P2 校验标签上的注册证号 | `RegistrationCert.is_valid_at()` + `cert_no` 格式 | 直接查 ORM |
| P2 生成必标项清单 | `Clause WHERE requirement_key IS NOT NULL`，按 `as_of` 过滤版本 | 一次查询 |
| P3 做跨文档型号一致性 | `Product.models` / `RegistrationCert.models` | 基准值 |
| P3 生成引用 | `Clause.cite()` 统一格式 | 字符串 |

P1 → P2/P3 的交接物是 `ClassificationResult.applicable_regulations`
（`ApplicableRegulation`：法规 + 版本 + 生效日 + 是否强制 + 依据条款号）。
**下游拿到的是精确到版本的法规清单，不是一堆文本片段**——这是三个应用能真正串起来的原因。

---

## 7. 目录结构

```
packages/mc-domain/
├── src/mc_domain/
│   ├── enums.py            枚举（RequirementKey 机读键、ClassificationOutcome）
│   ├── models.py           六个实体 + 关联表
│   ├── schemas.py          P1 输入/输出契约 + 实体读模型
│   ├── tree.py             目录树 + 排除法（纯数据，不依赖 ORM）
│   ├── retrieval.py        零依赖检索基线
│   ├── gate.py             前置闸门：是否属于医疗器械（高精度规则，其余转 uncertain）
│   ├── classify.py         判定流水线（normalize → gate → coarse → exclude → match → rule → rank）
│   ├── catalog_loader.py   目录导入与完整性校验
│   ├── evaluation/         评测层：标签归一化 / 分层抽样 / 指标 / 报告
│   └── data/catalog_sample.csv   合成目录（8 行）
└── tests/
    ├── test_classification.py   树 / 排除法 / 流水线 / 规则（12 个）
    ├── test_models.py           ORM / 版本时效 / 证有效期（7 个）
    └── test_evaluation.py       标签归一化 / 指标 / 闸门顺序（26 个）
```

仓库根还有：`conftest.py`（把 `packages/*/src` 加进 `sys.path`）与
`scripts/{demo_classify,serve,crawl_adjudications,eval_adjudications}.py`。

`tree.py` 刻意不依赖 SQLAlchemy——树与排除法是最需要反复回归的部分，
如果它们绑在 session 上，每次改规则都要起数据库，最后就没人测了。
