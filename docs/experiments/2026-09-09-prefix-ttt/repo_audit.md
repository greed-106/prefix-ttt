# 仓库、权重与数据来源审计

2026-09-09 CPU阶段；实际机器读数见 environment.lock.json。根目录起初无Git，现建立独立prefix-ttt分支；不提交数据/权重和参考仓库。

## 只读参考与复用边界

| 参考 | 本地commit | 用途/许可 |
|---|---|---|
| preference/LLaVA | c121f0432da27facab705978f83c4ada465e46fd | 主框架，Apache-2.0；包复制到third_party/llava |
| preference/LaCT | 1c2109fef1cf707c1025643d73684fe5e0920cc6 | KV Binding算子参考；根MIT，但所审阅NVIDIA文件含Apache-2.0头，不能混同 |
| preference/Spatial-TTT | e2e33a62b6f92c33b7e24ff042be9737d05c5bdb | QKV/anchor/门控/cache组织参考；根Apache-2.0，README MIT badge不一致 |
| FastWAM | c13e1534ece96093c95e0bff48d4ee6506ec40d9 | SQLite骨架参考，MIT，声明见configs/systemd/FASTWAM_LICENSE |
| flash-linear-attention | c51953382397da5c3b7b8a41e568915b703e2934 | 独立uv git依赖，MIT，未从参考目录导入 |

LaCT本地branch为main但含指定only_w1_no_wn_parallel.py；该文件cumsum(dw1)-dw1为chunk-exclusive，不能作为token-inclusive最终实现。Spatial的pending/tail只读、3D卷积、视频和固定比例未复制。

LLaVA副本改动集中于embedded CLIP、HF严格桥接、现代forward参数、多模态metadata、v1真实监督边界。原QKV/FFN/Norm结构、模板构造、图像预处理与dataset路径保留。旧v1固定偏移的legacy=false兼容错误已通过本地tokenizer fixture复现并修复。

## 数据与权重

直接使用data/llava-v1.5-assets-v1，源tar不删除，不重新下载模型。asset_manifest记载模型repo为llava-hf/llava-1.5-7b-hf，revision b234b804b114d9e37bb655e11cbbb5f5e971b7a9。

真实safetensors头部与meta模型严格检查：686 tensors，7,063,427,072参数，tensor payload 14,126,854,144 bytes，embedding/head 32064×4096，32 decoder层；无缺失/多余/shape错误。原始结果在artifacts/cpu/checkpoint-header-audit.json。随后三个完整分片的SHA256已计算至artifacts/cpu/checkpoint-sha256.txt；仍未执行7B前向。

BF16桥接实际回归曾发现整体dtype转换会量化FP32 RoPE buffers；现仅转换参数，HF BF16加载对照中的inv_freq/original_inv_freq和高位置图文logits精确一致（tiny CPU）。加载后仅转device，不再次整体转dtype。

原数据665298条、349034独立图片，benchmark包只包含GQA/POPE/MME。来源manifest含其他机器绝对路径，必须通过另存路径映射适配；不能据旧manifest的7任务总表声称其他4项benchmark已在本机。

完整数据解码/标签审计进展及产物以ledger和artifacts/cpu/data*为准；清点数不等于最终train有效数。正式train/dev划分需在标签实现和全量审计通过后冻结。

## 环境限制

本轮用户指定阿里PyPI+官方PyTorch cu128，优先于AGENTS默认清华源。项目uv.lock锁定实际分发与哈希，环境不安装在全局。当前无可用GPU；FLA已安装不等于kernel通过。systemd包安装成功不等于user manager可用，用户总线仍不可连接，正式服务未启动。
