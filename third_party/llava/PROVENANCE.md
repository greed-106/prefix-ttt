# LLaVA 本地维护副本

来源：https://github.com/haotian-liu/LLaVA

Commit: c121f0432da27facab705978f83c4ada465e46fd

Apache-2.0，许可证见 LICENSE。从只读 preference/LLaVA 复制 llava 包；运行时不依赖参考目录。Prefix-TTT 修改限定于新版 Transformers 兼容、内置视觉塔、同步展开 metadata、v1 assistant 监督边界修复和 HybridCache/greedy 入口。训练及评测入口保留，尚未声称全部入口已通过新版依赖回归。

v1 修复保留原模板和 tokenizer 输入；用原模板渲染的 assistant 前缀推导边界，替代在本地 legacy=false tokenizer 上不兼容的固定 round 偏移。不能对齐时报告错误，不删除样本。缓存和 greedy 的验证当前为 tiny CPU 模型；未据此声称 7B/GPU/FLA 验收通过。
