# CPU 准备阶段总结

2026-09-09完成。本阶段交付环境、工程基础和CPU验证；没有执行正式A/B训练、完整7B前向、GPU性能测试或质量benchmark。挂卡后按 [resume.md](resume.md) 继续，同一本账本和同一份清单保持不变。

## 完成结果

- 任务书、主账本、论文解读、源码/环境审计、参数清单、重启交接文档已建立。
- uv环境锁定Python3.10、torch2.7.1+cu128、torchvision0.22.1+cu128、Transformers4.51.3、PEFT0.15.2、Accelerate1.6.0及指定FLA commit。普通包用阿里源，PyTorch用官方cu128；未修改全局Python环境。
- 最终统一测试 **145通过、4项GPU跳过**，见 [测试日志](../../../artifacts/experiments/prefix_ttt/evidence/pytest.log) 和 [JUnit结果](../../../artifacts/experiments/prefix_ttt/evidence/pytest-results.xml)。覆盖算子输出/全部输入梯度/因果/分段、真实块批量Local、LoRA、tiny图文训练、optimizer和RNG恢复、混合缓存greedy以及SQLite恢复。
- tiny图文生成确认视觉编码只执行一次，之后query长度为1；测试了首层TTT、不同anchor布局、左右padding、异长batch、finished停止写和重排。这不是完整7B或FLA GPU结论。
- 生产规格meta参数审计：新增27,131,904、LoRA79,953,920，共107,085,824个可训练参数。基座冻结，不将S加入optimizer。

## 数据和完整性

665,298条全部完成原LLaVA模板/tokenizer标签审计，预处理错误0。仅2条原始assistant内容为空，索引245723、247644；没有因tokenizer兼容错误删除样本，也没有修改原JSON或图片。

349,034张图片全部完成RGB解码，缺图0、解码失败0。147条展开前总长超过2048，但右截断后仍有有效回答监督，按统一协议保留。

固定清单为train **663,248**、dev **2,048**、A **50,000**。图片、同来源原ID和重复对话形成传递分组，A不与dev重叠。输入SHA和全部顺序哈希已核对，见 [清单摘要](data_manifest_summary.json)；完整清单保存在项目 `artifacts/cpu/fixed_manifest.json`，后续不得静默重建。

按有效batch128，A为391步，完整B为5182步，B warmup为156步，Pilot在391步/50048条处保存。这里是计算出的训练规划，不是已执行步数。

三个模型分片完整SHA256和686个tensor的严格shape/coverage验证已完成。GQA/POPE/MME的9个Arrow文件通过字节/SHA/可读性检查，路径映射另存，来源manifest不变；未进行质量推理。

## 发现的问题与修复

1. 原v1固定round偏移与本地legacy=false不兼容，会将监督清空。改为同一原模板渲染前缀确定assistant边界，保留输入token与tokenizer语义；局部fixture及全量审计通过。
2. bridge整体转BF16会量化原FP32 RoPE频率。改为仅转换参数，保留原始buffer；HF BF16加载对照中的频率和高位置图文logits在tiny CPU上逐值一致。加载后只能转device，不再次整体转dtype。
3. reference在外层autocast下可能隐式降低einsum精度；现计算区显式关闭autocast。新增FP32参数配BF16基座的推理投影使用明确AMP，参数storage保持FP32。
4. 调度恢复时空进程identity可能被误认为仍存活，已修复并覆盖。若进程身份无法安全判断仍拒绝重复派发。
5. 扩展缓存测试期间有旧fixture顺序错误及误放断言的NameError，修正测试后复验；不将这些失败隐藏为一次全通过。

关键失败/修复日志归档在evidence或artifacts/cpu，主账本保留时间顺序。

## 交接边界

FLA已安装，但本机无GPU，直接导入触发`0 active drivers`；这不是kernel通过。systemd包已安装但用户服务总线未就绪，正式持久托管尚待验收。CPU阶段实际配额4核，全量数据审计为主要耗时；下载未采用短间隔高频轮询。

正式A/B runner、统一清单消费和benchmark输入适配仍需接续实现；现有loss/optimizer/trajectory基础与tiny闭环不能当作已能启动665k训练。GPU挂载后先验证FLA和完整7B smoke，再按E0→A→整模型切换诊断→E1/E2同预算Pilot→连续完整B推进。

参考仓库、原始数据包及权重均保留；当前没有后台训练或质量评测任务。
