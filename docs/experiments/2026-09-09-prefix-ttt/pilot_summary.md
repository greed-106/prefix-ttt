# E1/E2 同预算 Pilot 完成总结

2026-09-10核查：两项正式Pilot均退出0，每项391步、50048样本。Supervisor消费者13:55结束，四A40空闲。当前未执行完整B续训，也未执行Pilot checkpoint的dev或benchmark质量评测。

| 指标 | E1 Full-MSA-LoRA | E2 A9-T23 + LoRA |
| --- | ---: | ---: |
| 调度器总耗时（含加载与保存） | 2小时36分 | 3小时26分 |
| 末步训练CE | 0.542736 | 1.105848 |
| 最后20步训练CE均值 | 0.573154 | 1.146149 |
| 最后20步平均步耗时 | 23.84秒 | 32.12秒 |

两模型每一步的样本累计量及有效监督token数完全对应，391条日志连续，loss和裁剪前grad_norm均有限；同预算数据协议通过此次核对。训练loss表明E2已从初始高loss下降，但仍高于E1，不能据此断言benchmark能力恢复。上述训练耗时也不是推理效率结论。

## 保存与恢复

`artifacts/training/E1/`及`artifacts/training/E2/`各自的pilot.pt/latest.pt均成功以CPU mmap方式读取。global_step为391、samples_seen为50048、scheduler.last_epoch为391，保存四rank RNG、optimizer和可训练张量。完整规划仍为5182步；complete=false是预期的Pilot结束状态，不表示失败。本次未重新构建7B模型验证恢复前向。

配置、原始步骤日志和结果分别为各目录run.json、steps.jsonl、result.json；调度证据为 `/var/tmp/prefix-ttt-1000/queue-pilot.sqlite3` 与对应任务日志。

## 结论与下一步

本轮两项训练任务均成功，没有触发失败重试。现有证据支持Pilot执行及恢复状态完整，不支持宣称E2质量达标。下一步先进行固定dev/生成诊断，再从各自checkpoint继续剩余4791步、613200样本，不重置optimizer/scheduler或重新开始epoch。本次状态核查未提交这些后续任务。
