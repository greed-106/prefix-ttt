# Pilot 后诊断与仅 E2 续训

2026-09-10，用户要求简短dev/生成检查后先续训E2，不提交E1对照组续训。

E2正式Pilot checkpoint严格加载后，在与A切换诊断相同的首4条图文、首4条纯文本dev上，全部有效assistant预测位置加权CE为1.357264；A切换后对应值为10.111575，原E0为0.268383。E2已明显恢复但仍有差距；该8条诊断不能代替benchmark或完整dev结论。每模态短greedy及cache/full数值对照通过既定阈值，原始成功报告为 `artifacts/training/pilot-diagnostic/E2.json`。

E1诊断失败于cache/full第2个decode位置：最大绝对差0.265625，超过既定0.25；相对Frobenius误差0.016589低于0.02。未放宽阈值，未生成成功报告；根因尚未定位，不能断言只是舍入。日志保留在 `/var/tmp/prefix-ttt-1000/e1-pilot-diagnostic.log`，E1续训未提交。

初次E2续训被不必要的双报告门槛阻止，在加载模型前退出，步骤日志仍391条。随后将续训门槛改为核对该模型自身的通过报告和checkpoint SHA；5项定向测试通过。没有重跑E2成功诊断，也没有隐藏或重试E1失败。

活动清单 `configs/jobs-e2-resume.json` 仅含E2完整B续训，数据库为 `/var/tmp/prefix-ttt-1000/queue-e2-resume.sqlite3`。从保留的pilot.pt恢复模型、optimizer、scheduler及四rank RNG后，实际第392步已完成：累计50176样本，loss1.119555，学习率延续原cosine计划。后续累计训练到5182步/663248样本；不是重新开始一遍。原Pilot元数据归档在 `artifacts/training/E2/pilot-record/`。
