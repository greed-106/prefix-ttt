# 阶段 A 后整模型切换诊断

2026-09-10，在固定dev顺序中选择前4条图文和前4条纯文本，顺序运行原E0及加载正式A权重的A9-T23；无LoRA、不重训练，不属于benchmark结果。

全部有效shifted assistant预测位置的token加权指标：E0 CE为0.268383，切换模型CE为10.111575，teacher KL为9.770452。说明直接替换后出现明显质量退化，不能将A训练完成解释成质量保持。该8条子集不代表完整dev或benchmark。

每种模态的短greedy及cache/full logits对照通过既定绝对0.25、相对2%阈值，有限值检查通过。因此本次工程诊断成功，但质量没有宣称通过。原生报告及固定样本ID见 `artifacts/training/switch-A/result.json`，日志见 `/var/tmp/prefix-ttt-1000/a-whole-model-switch.log`。

按既定不设质量门槛的协议继续E2/E1同预算Pilot，不改结构、rank或loss。正式E2已进行四卡SFT并保存首步checkpoint；Pilot终点391步/50048样本，完整B规划5182步。E1随后使用相同样本顺序与预算。下一阶段需核查Pilot的训练与质量恢复情况，之后才继续完整轨迹。
