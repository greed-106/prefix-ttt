"""Generate a Chinese report from existing mamba_bench JSON evidence; no GPU work."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
from urllib.parse import quote


SHAPE = re.compile(r'(core|layer|stack|decode)-b(\d+)-t(\d+)$')
ARMS = ('baseline', 'production', 'optimized', 'mha')


def number(value):
    return f'{value:.3f}' if isinstance(value, (int, float)) and math.isfinite(value) else '—'


def median(entry, clock, per_step=False):
    return entry.get(clock + '_ms_per_step') if per_step else entry.get(clock + '_ms', {}).get('median')


def latency(entry, per_step=False):
    return ' / '.join(number(median(entry, clock, per_step)) for clock in ('cuda', 'wall'))


def stable(entry):
    if entry.get('memory', {}).get('output_finite') is False:
        return '不可判定：非有限输出'
    if len(entry.get('round_cuda_medians', [])) < 5:
        return '轮次不足'
    value = entry.get('stable_lead_vs_mha')
    return '是' if value is True else '否' if value is False else '未提供'


def table(headers, rows):
    def row(values):
        return '| ' + ' | '.join(str(value).replace('|', '\\|').replace('\n', ' ') for value in values) + ' |'
    return '\n'.join([row(headers), row(['---'] * len(headers)), *(row(values) for values in rows)])


def link(path, directory):
    return quote(Path(os.path.relpath(path, directory)).as_posix(), safe='/._-')


def records(sources, stage):
    found = []
    for source in sources:
        for key, measurement in source['data'].get('measurements', {}).items():
            match = SHAPE.fullmatch(key)
            if match and match[1] == stage and 'summary' in measurement:
                found.append((source, int(match[2]), int(match[3]), measurement['summary']))
    return sorted(found, key=lambda item: (item[0]['id'], item[1], item[2]))


def comparison_rows(items, per_step=False):
    rows = []
    for source, batch, length, summary in items:
        candidate = next((name for name in ('production', 'optimized') if name in summary), None)
        entry = summary.get(candidate, {})
        rows.append([source['id'], batch, length, latency(summary.get('baseline', {}), per_step),
                     candidate or '未测', latency(entry, per_step),
                     latency(summary.get('mha', {}), per_step), number(entry.get('speedup_vs_mha')),
                     stable(entry) if candidate else '未测'])
    return rows


def plot_layers(items, output):
    if not items:
        return []
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    image_dir = output.parent / 'images'
    image_dir.mkdir(parents=True, exist_ok=True)
    references = []
    groups = sorted({(source['id'], batch) for source, batch, _, _ in items})
    for source_id, batch in groups:
        selected = [(length, summary) for source, b, length, summary in items
                    if source['id'] == source_id and b == batch]
        figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        ratio_count = 0
        for arm in ARMS:
            points = [(length, summary[arm]) for length, summary in selected if arm in summary]
            times = [(length, median(entry, 'cuda')) for length, entry in points
                     if median(entry, 'cuda') is not None]
            if times:
                axes[0].plot(*zip(*times), marker='o', label=arm)
            ratios = [(length, entry['speedup_vs_mha']) for length, entry in points
                      if 'speedup_vs_mha' in entry and entry.get('memory', {}).get('output_finite') is not False]
            if ratios:
                line, = axes[1].plot(*zip(*ratios), marker='o', label=arm)
                stable_points = [(length, entry['speedup_vs_mha']) for length, entry in points
                                 if stable(entry) == '是']
                if stable_points:
                    axes[1].scatter(*zip(*stable_points), marker='*', s=120, color=line.get_color(), zorder=3)
                ratio_count += 1
        axes[0].set(xlabel='Sequence length', ylabel='CUDA event median (ms)', yscale='log')
        axes[0].legend()
        axes[1].axhline(1, color='black', linestyle='--', linewidth=1)
        axes[1].axhline(1.05, color='gray', linestyle=':', linewidth=1)
        axes[1].set(xlabel='Sequence length', ylabel='Flash SDPA / TTT latency')
        if ratio_count:
            axes[1].legend()
        else:
            axes[1].text(.5, .5, 'No measured speedup', transform=axes[1].transAxes, ha='center')
        for axis in axes:
            axis.set_xscale('log', base=2)
            axis.grid(True, alpha=.25)
        figure.suptitle(f'Complete attention layer: {source_id}, batch={batch}')
        image_path = image_dir / f'{output.stem}-{source_id}-b{batch}.png'
        figure.savefig(image_path, dpi=180)
        plt.close(figure)
        references.append(f'![{source_id} batch={batch} 完整层 CUDA 延迟与加速比]({link(image_path, output.parent)})')
    return references


def make_report(sources, output):
    lines = ['# Mamba 启发的 Prefix-TTT prefill：测量汇总', '',
             '本文件由已有 JSON 生成。只汇总实际完成的测量；不同来源分别标记，不合并采样、不推算未测长度。'
             '最终采用方案、真实 checkpoint 质量回归与失败原因分析由阶段总结补充。', '',
             '## 口径与边界', '',
             '- Prefix-TTT 与 softmax MHA 计算不同函数；性能领先不代表质量等价。'
             '长度超过 2048 的结果仅用于硬件性能外推，不证明现有 checkpoint 的长上下文质量。',
             '- 完整层计入 QKV/RoPE、TTT 特征和 Local-32、门控、输出投影与缓存生成；不包含共同 MLP。'
             '核心算子使用已准备 Q/K/V，不能代替完整层或完整模型速度。',
             '- `production` 包含真实生产分派；`optimized` 是已知全有效输入的实验路径，'
             '可能省略生产有效性判定成本。表中优先展示 production，缺失时展示 optimized，并保留名称。',
             '- 表中延迟以“CUDA Events / 同步墙钟”表示。CUDA Events 是调用的设备时间跨度，'
             '可能包含发射间隙，不能称为纯 kernel 执行时间；profiler 的 kernel 时间不混入计时样本。',
             '- 加速比为 MHA CUDA 中位数除以对应 TTT CUDA 中位数。稳定领先要求至少五轮、'
             '总中位数加速比至少 1.05 且各轮均领先；少于五轮只记为轮次不足。',
             '- 请求的初始/最终持久状态和递推累加器为 FP32；锁定 FLA 前向块边界中间状态为 BF16。'
             '块边界临时内存仍随序列增长，不能将持久缓存容量当作峰值运行显存。', '',
             '## 数据来源', '']
    source_rows = []
    for source in sources:
        data, path = source['data'], source['path']
        config, environment = data.get('config', {}), data.get('environment', {})
        source_rows.append([source['id'], f'[{path.name}]({link(path, output.parent)})',
                            environment.get('gpu', '未记录'), environment.get('torch', '未记录'),
                            config.get('tile_size', '未记录'), config.get('rounds', '未记录'),
                            config.get('repetitions', '未记录'),
                            'checkpoint 层权重' if config.get('weights') else '合成层权重'])
    lines += [table(['来源', 'JSON', 'GPU', 'PyTorch', '完整层 tile', '轮数', '每轮次数', '权重'], source_rows), '']
    for source in sources:
        config = source['data'].get('config', {})
        lines += [f"- {source['id']} SHA256：`{source['sha256']}`；seed={config.get('seed', '未记录')}；"
                  f"warmup={config.get('warmup', '未记录')}；baseline commit=`{config.get('baseline_commit', '未记录')}`。"]
    header = ['来源', 'Batch', '长度', 'Baseline ms', '候选', '候选 ms', 'MHA ms', '加速比', '稳定领先']
    layers = records(sources, 'layer')
    lines += ['', '## 完整层 prefill', '']
    lines += [table(header, comparison_rows(layers)) if layers else '输入中没有已完成的完整层计时。', '']
    for source in sources:
        for batch in sorted({b for s, b, _, _ in layers if s['id'] == source['id']}):
            points = []
            for s, b, length, summary in layers:
                candidate = next((name for name in ('production', 'optimized') if name in summary), None)
                if s['id'] == source['id'] and b == batch and candidate and stable(summary[candidate]) == '是':
                    points.append(str(length))
            lines.append(f"- {source['id']}，batch={batch}：稳定领先的已测长度为"
                         + ('、'.join(points) if points else '无') + '；未据此推断未测交叉点。')
    pictures = plot_layers(layers, output)
    if pictures:
        lines += ['', '图中星号表示达到稳定领先判据的观测点；线段仅连接观测值。墙钟中位数见表。', '']
        for picture in pictures:
            lines += [picture, '']
    memory_rows = []
    for source, batch, length, summary in layers:
        candidate = next((name for name in ('production', 'optimized') if name in summary), None)
        for arm in (candidate, 'mha'):
            if arm not in summary:
                continue
            memory = summary[arm].get('memory', {})
            sizes = [memory.get(key) for key in ('peak_increment_bytes', 'temporary_over_retained_bytes',
                                               'retained_cache_bytes')]
            memory_rows.append([source['id'], batch, length, arm,
                                *(number(value / 2**20) if value is not None else '—' for value in sizes)])
    lines += ['## 完整层显存', '',
              '下表单位为 MiB。峰值增量是调用峰值 allocated 减调用前 allocated；临时超额是调用峰值减返回时 allocated；'
              '持久缓存为返回 cache 的张量容量。峰值增量包含返回输出和缓存，临时超额不等于所有临时张量分配量之和。', '',
              '同进程中 TTT 参数及实验包装由比较臂共用并常驻，因此不将总峰值 allocated 解释为两个独立模型的绝对显存占用。'
              '新增 A/B 拼接 BF16 prepared buffer 在本轮 32 头、头维 128 下为每个 TTT 层 2 MiB；'
              '23 层合计 46 MiB，FP32 参数 master 不变。这是权重副本容量，不包含在请求缓存中。', '',
              table(['来源', 'Batch', '长度', '变体', '峰值增量 MiB', '临时超额 MiB', '持久缓存 MiB'], memory_rows)
              if memory_rows else '输入中没有完整层显存记录。', '']
    core_rows = []
    for source, batch, length, summary in records(sources, 'core'):
        for arm, entry in summary.items():
            core_rows.append([source['id'], batch, length, arm, latency(entry),
                              number(entry.get('speedup_vs_mha')), stable(entry) if arm != 'mha' else '基准'])
    lines += ['', '## 核心算子与 tile 消融', '',
              table(['来源', 'Batch', '长度', '变体 / tile', 'CUDA / wall ms', '加速比', '稳定领先'], core_rows)
              if core_rows else '输入中没有已完成的核心算子计时。', '']
    for stage, title in [('stack', '多层串联'), ('decode', '固定输入 decode')]:
        items = records(sources, stage)
        lines += [f'## {title}', '']
        if stage == 'stack':
            lines += ['各来源的层数：' + '；'.join(
                f"{source['id']}={source['data'].get('config', {}).get('depth', '未记录')}"
                for source in sources) + '。层权重是否独立或重复，以原始 JSON 的 `protocol.stack_weights` 为准。', '']
        else:
            lines += ['延迟为整段 decode 总时间除以步数，属于平均每步耗时，不是逐 token 延迟中位数。'
                      '长度列表示初始前缀；加速比与稳定判据来自整段计时。各来源步数：' + '；'.join(
                          f"{source['id']}={source['data'].get('config', {}).get('decode_steps', '未记录')}"
                          for source in sources) + '。', '']
        lines += [table(header, comparison_rows(items, per_step=stage == 'decode'))
                  if items else '输入中没有该阶段已完成的计时。', '']
    correctness_rows = []
    for source in sources:
        checks = [entry for key, entry in source['data'].get('measurements', {}).items()
                  if key.startswith('correctness-')]
        variants = sorted({name for entry in checks for name in entry.get('variants', {})})
        for name in variants:
            entries = [entry['variants'][name] for entry in checks if name in entry.get('variants', {})]
            errors = []
            for field in ('output', 'state'):
                values = [entry.get(field, {}).get('relative_l2') for entry in entries]
                values = [value for value in values if value is not None]
                errors.append(f'{max(values):.6g}' if values and all(math.isfinite(value) for value in values) else '—')
            correctness_rows.append([source['id'], name,
                                     f"{sum(entry.get('passed') is True for entry in checks)}/{len(checks)}",
                                     f"{sum(entry.get('passed') is True for entry in entries)}/{len(entries)}", *errors])
    lines += ['## 已执行的数值正确性', '',
              '下表仅汇总 correctness stage 对冻结 TTT 基线的结果。误差列取该来源所有已测配置的最大相对 L2 误差；'
              'stage 与变体的 passed 分别按 JSON 原值计数，不把缺失记录算作通过，也不代替任务质量评估。', '',
              table(['来源', '变体', 'Stage 通过/配置数', '变体通过/配置数', '输出最大相对 L2', '状态最大相对 L2'], correctness_rows)
              if correctness_rows else '输入中没有已完成的 correctness stage 变体记录。', '',
              '## 失败与证据完整性', '']
    failures = [(source, source['data'].get('failure')) for source in sources if source['data'].get('failure')]
    if failures:
        lines += [table(['来源', '失败配置', '类型', '信息'],
                        [[source['id'], failure.get('shape', '未记录'), failure.get('type', '未记录'),
                          failure.get('message', '未记录')] for source, failure in failures]), '',
                  '失败配置没有被补成零耗时；上表前的成功测量只是该文件中已完成的部分。', '']
    else:
        lines += ['输入 JSON 未记录 `failure`；这不表示所有计划任务已经执行或通过。', '']
    profiles = [(source['id'], key, value)
                for source in sources for key, value in source['data'].get('measurements', {}).items()
                if key.startswith('profile-')]
    profile_rows = []
    for source_id, key, entry in profiles:
        candidate = next((name for name in ('production', 'optimized') if name in entry), None)
        flash = entry.get('mha', {}).get('flash_operator_verified')
        profile_rows.append([source_id, key, entry.get('baseline', {}).get('kernel_count', '—'), candidate or '未测',
                             entry.get(candidate, {}).get('kernel_count', '—'), entry.get('mha', {}).get('kernel_count', '—'),
                             '是' if flash is True else '否' if flash is False else '未提供'])
    lines += ['Flash 算子验证仅列出实际 profile 配置，不推广到未 profile 的配置。kernel 数取 JSON 的 `kernel_count`，'
              '不额外加总 CPU/GPU 标记、Memcpy 或 Memset；内核数量减少本身不代表延迟降低。', '',
              table(['来源', '配置', 'Baseline kernel 数', '候选', '候选 kernel 数', 'MHA kernel 数', '观察到 Flash SDPA'],
                    profile_rows) if profile_rows else '输入中没有 Flash 后端 profile 验证记录。', '',
              '本生成器不替代数值正确性、MME/POPE 或调度器完成状态验收。原始样本、配置、源码指纹及其他阶段记录保留在来源 JSON 中。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite an existing report: {args.output}')
    sources = []
    for index, path in enumerate(args.inputs, 1):
        raw = path.read_bytes()
        sources.append({'id': f'S{index:02d}', 'path': path.resolve(), 'data': json.loads(raw),
                        'sha256': hashlib.sha256(raw).hexdigest()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = make_report(sources, args.output.resolve())
    args.output.write_text(report, encoding='utf-8')
    print(args.output)


if __name__ == '__main__':
    main()
