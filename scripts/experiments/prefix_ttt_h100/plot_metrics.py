"""Figures and a machine-readable summary for the E0 vs E2 inference-cost study.

Quality comes from the official lmms-eval result files; cost comes from the
per-request logs that `prefix_ttt.instrument` appends during those same benchmark
runs, so cost is always paired with the run that produced the score and nothing is
recomputed here.

Every benchmark used here feeds 576 image tokens plus a short question, so the
input length only spans ~634-663 tokens and is reported as context rather than
plotted as an axis. Length is read from `prefill_positions` when the log has it;
older logs predate that field, so it is then reconstructed from the full-attention
KV bytes (only the anchor layers keep KV in the hybrid layout, so bytes per token
differ per model and must be passed with `--kv-bytes-per-token`). The
reconstruction counts the few decoded tokens too, hence the ~1-3 token offset.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

COLORS = {'e0': 'tab:blue', 'e2': 'tab:red'}
LABELS = {'e0': 'E0 (base LLaVA)', 'e2': 'E2 (LoRA + TTT)'}
SCORES = {
    'mme': {'perception': 'mme_perception_score', 'cognition': 'mme_cognition_score'},
    'pope': {'accuracy': 'pope_accuracy', 'f1': 'pope_f1_score'},
    'gqa': {'exact_match': 'exact_match'},
}
CACHE_BUCKETS = ('full_kv_bytes', 'ttt_state_bytes', 'local_window_bytes')
PANELS = (('prefill_ms', 'prefill latency (ms)', 1.0),
          ('tpot_ms', 'TPOT (ms per token)', 1.0),
          ('peak_allocated_bytes', 'peak allocated memory (GiB)', 2 ** 30))


def read_cost(root, label, task, kv_per_token):
    """One row per request: (length, prefill ms, TPOT ms, peak bytes, cache buckets)."""
    path = Path(root) / f'{label}-{task}.jsonl'
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        positions = record.get('prefill_positions')
        if positions is None:
            positions = record['cache']['full_kv_bytes'] / kv_per_token
        cache = record['cache']
        rows.append({
            'task': task,
            'length': positions,
            'prefill_ms': record['prefill_ms'],
            'tpot_ms': record['tpot_ms'],
            'peak_allocated_bytes': record['peak_allocated_bytes'],
            'peak_reserved_bytes': record['peak_reserved_bytes'],
            **{name: cache[name] for name in CACHE_BUCKETS},
        })
    return rows


def load_scores(root, tasks):
    scores = {}
    for label in ('e0', 'e2'):
        for task in tasks:
            files = sorted((Path(root) / label / task).rglob('*_results.json'))
            if not files:
                continue
            report = json.loads(files[-1].read_text())
            block = report['results'][task]
            scores.setdefault(label, {})[task] = {
                name: block.get(f'{key},none') for name, key in SCORES[task].items()}
            scores[label][task]['_samples'] = report['n-samples'][task]['effective']
            scores[label][task]['_seconds'] = report.get('total_evaluation_time_seconds')
    return scores


def groups(costs, tasks):
    """(task, label, rows) in a stable order, skipping tasks a model never ran."""
    for task in tasks:
        for label in sorted(costs):
            rows = [row for row in costs[label] if row['task'] == task]
            if rows:
                yield task, label, rows


def save(fig, images, name):
    fig.tight_layout()
    fig.savefig(Path(images) / name, dpi=150)
    plt.close(fig)
    print(f'  wrote {name}')


def plot_distributions(costs, tasks, images):
    """Box plots per task and model: spread over requests, not over length."""
    fig, axes = plt.subplots(1, len(PANELS), figsize=(4.2 * len(PANELS), 4.2))
    entries = list(groups(costs, tasks))
    for axis, (field, ylabel, scale) in zip(axes, PANELS):
        values = [[row[field] / scale for row in rows] for _, _, rows in entries]
        boxes = axis.boxplot(values, widths=.55, patch_artist=True, showfliers=False)
        for patch, (_, label, _) in zip(boxes['boxes'], entries):
            patch.set_facecolor(COLORS[label])
            patch.set_alpha(.55)
        for median in boxes['medians']:
            median.set_color('black')
        for position, value in enumerate(values, start=1):
            axis.text(position, np.median(value), f'{np.median(value):.3g}',
                      ha='center', va='bottom', fontsize=7)
        axis.set_xticks(range(1, len(entries) + 1))
        axis.set_xticklabels([f'{task}\n{label.upper()}' for task, label, _ in entries], fontsize=8)
        axis.set_ylabel(ylabel)
        axis.grid(axis='y', alpha=.3)
    axes[0].legend(handles=[plt.Line2D([], [], color=COLORS[label], lw=6, alpha=.55,
                                       label=LABELS[label]) for label in sorted(costs)],
                   fontsize=8)
    save(fig, images, 'cost_by_task.png')


def plot_cache(costs, tasks, images):
    """Median cache split: E0 keeps KV everywhere, E2 only in the anchor layers."""
    entries = list(groups(costs, tasks))
    fig, ax = plt.subplots(figsize=(1.6 * len(entries) + 3, 4.5))
    base = np.zeros(len(entries))
    for name in CACHE_BUCKETS:
        values = np.array([np.median([row[name] for row in rows]) / 2 ** 20 for _, _, rows in entries])
        if values.max() == 0:
            continue
        ax.bar(range(len(entries)), values, bottom=base, label=name.replace('_bytes', ''))
        for x, (value, bottom) in enumerate(zip(values, base)):
            if value / max(base.sum() / len(entries), 1e-9) > .12:
                ax.text(x, bottom + value / 2, f'{value:.0f}', ha='center', va='center', fontsize=7)
        base += values
    for x, total in enumerate(base):
        ax.text(x, total, f'{total:.0f} MiB', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(range(len(entries)))
    ax.set_xticklabels([f'{task}\n{label.upper()}' for task, label, _ in entries], fontsize=8)
    ax.set_ylabel('cache after decoding (MiB)')
    ax.set_ylim(0, base.max() * 1.15)
    ax.grid(axis='y', alpha=.3)
    ax.legend(fontsize=8)
    save(fig, images, 'cache_by_task.png')


def plot_score_vs_cost(scores, costs, images, reference='e0', candidate='e2'):
    """What the extra cost buys: E2 relative to E0 on both axes.

    Absolute scores are not comparable across tasks (MME is a 0-2000 point sum,
    POPE is an accuracy), so both axes are ratios against the reference model:
    x is the cost ratio, y the score ratio in percent. Absolute values live in the
    summary JSON and in the report tables.
    """
    metrics = [(field, xlabel, scale) for field, xlabel, scale in PANELS]
    metrics.append(('cache_total', 'cache after decoding', 2 ** 20))
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.4 * len(metrics), 4.2))
    for axis, (field, xlabel, scale) in zip(axes, metrics):
        for task, values in scores.get(candidate, {}).items():
            base = scores.get(reference, {}).get(task)
            if base is None:
                continue
            for name, value in values.items():
                if name.startswith('_') or value is None or not base.get(name):
                    continue
                ratio = []
                for label in (reference, candidate):
                    rows = [row for row in costs.get(label, []) if row['task'] == task]
                    if field == 'cache_total':
                        ratio.append(float(np.median(
                            [sum(row[bucket] for bucket in CACHE_BUCKETS) for row in rows])) / scale)
                    else:
                        ratio.append(float(np.median([row[field] for row in rows])) / scale)
                axis.scatter(ratio[1] / ratio[0], 100 * value / base[name],
                             color=COLORS[candidate], marker='o')
                axis.annotate(f'{task}:{name}', (ratio[1] / ratio[0], 100 * value / base[name]),
                              fontsize=6, textcoords='offset points', xytext=(4, 2))
        axis.axvline(1.0, color='grey', lw=.8, ls='--')
        axis.axhline(100.0, color='grey', lw=.8, ls='--')
        axis.set_xlabel(f'{xlabel}: {candidate.upper()} / {reference.upper()}')
        axis.grid(alpha=.3)
    axes[0].set_ylabel(f'official score: {candidate.upper()} / {reference.upper()} (%)')
    save(fig, images, 'score_vs_cost.png')


def summarise(costs, scores, tasks):
    summary = {'scores': scores, 'tasks': {}}
    for task, label, rows in groups(costs, tasks):
        block = {'requests': len(rows)}
        for field, _, scale in PANELS:
            values = [row[field] / scale for row in rows]
            block[field] = {'median': float(np.median(values)),
                            'p25': float(np.percentile(values, 25)),
                            'p75': float(np.percentile(values, 75))}
        block['length'] = {'median': float(np.median([row['length'] for row in rows])),
                           'min': float(min(row['length'] for row in rows)),
                           'max': float(max(row['length'] for row in rows))}
        block['cache_median_bytes'] = {name: float(np.median([row[name] for row in rows]))
                                       for name in CACHE_BUCKETS}
        block['cache_median_total_bytes'] = float(np.median(
            [sum(row[name] for name in CACHE_BUCKETS) for row in rows]))
        summary['tasks'][f'{label}-{task}'] = block
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cost-root', required=True, help='directory with <label>-<task>.jsonl')
    parser.add_argument('--eval-root', required=True, help='official lmms-eval results')
    parser.add_argument('--kv-bytes-per-token', nargs='+', required=True,
                        help='label=bytes, used only for logs without prefill_positions')
    parser.add_argument('--tasks', nargs='+', default=['mme', 'pope'])
    parser.add_argument('--images', required=True)
    parser.add_argument('--summary', required=True)
    args = parser.parse_args()

    kv = {key: float(value) for key, value in
          (pair.split('=', 1) for pair in args.kv_bytes_per_token)}
    costs = {label: [row for task in args.tasks
                     for row in read_cost(args.cost_root, label, task, kv[label])]
             for label in kv}
    scores = load_scores(args.eval_root, args.tasks)
    Path(args.images).mkdir(parents=True, exist_ok=True)
    plot_distributions(costs, args.tasks, args.images)
    plot_cache(costs, args.tasks, args.images)
    plot_score_vs_cost(scores, costs, args.images)
    summary = summarise(costs, scores, args.tasks)
    Path(args.summary).write_text(json.dumps(summary, indent=2) + '\n')
    print(f'  wrote {args.summary}')
    print(json.dumps({label: len(rows) for label, rows in costs.items()}))


if __name__ == '__main__':
    main()
