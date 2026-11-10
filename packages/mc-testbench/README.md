# mc-testbench 测试中台

**它解决的不是"跑评测"，是"知道有没有变差"。**

三个真实的痛点驱动它：

| 痛点 | 现在怎么做 | 之前怎么做 |
|---|---|---|
| 改了配置后不知道哪项指标退化 | `compare` 自动按方向判定并标红 | 手工对比两份 markdown |
| 评测结果和数据版本的对应关系丢失 | 记录里存数据指纹 + git commit | 靠文件名和日期猜 |
| 每个 eval 脚本次重复实现抽样/指纹/报告 | 任务注册表统一收口 | 各写各的 |

## 用法

```bash
python scripts/testbench.py run labels --mode fields
python scripts/testbench.py run labels --mode text
python scripts/testbench.py list
python scripts/testbench.py compare <base_run_id> <head_run_id>
```

`compare` 退出码：**0** 无退化 / **1** 有退化 / **2** 不可比。

## 四个刻意的设计

1. **方向是声明的，不是猜的**。`undeterminable_rate` 越低越好；如果工具默认"变大就是变好"，
   它会把最常见的看板指标报成改进。未声明的指标判 `unknown` 并警告。
2. **容差**。0.0005 的抖动不算退化；没有容差的工具会在噪声上报警，然后被人关掉。
3. **不可比就拒绝**。任务不同、或数据指纹不同 → 退出码 2，不出结论。
   把两次不同数据上的评测并排比，是这类工具最危险的失败方式：它不报错，只给错结论。
4. **退出码即门禁**。有退化返回 1，因此可以直接挂进 CI 挡住合并。

## 刻意不做

不做调度器（cron 足够）、不做看板（终端表格足够）、不做分布式执行（评测是分钟级）。
把它做成一个"平台"很容易，做成一个**有人天天用**的工具很难——所以只做后者。
