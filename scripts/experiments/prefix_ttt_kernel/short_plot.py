"""Render the completed short-sequence evidence into the document image archive."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root = Path(__file__).resolve().parents[3]
    evidence = root / 'artifacts/experiments/prefix_ttt_kernel/evidence'
    data = json.loads((evidence / 'mamba-short-graph/results.json').read_text())
    if data.get('status') != 'succeeded':
        raise ValueError('Only completed evidence may be plotted')
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    styles = [('eager-optimized', 'TTT eager', '#b76039', '--'),
              ('eager-mha', 'Flash eager', '#5276a7', '--'),
              ('graph-optimized', 'TTT Graph', '#9c583a', ':'),
              ('graph-warp1', 'TTT Graph + 1 warp', '#df6c35', '-'),
              ('graph-mha', 'Flash Graph', '#245b9b', '-')]
    lengths = data['config']['lengths']
    for column, batch in enumerate(data['config']['batches']):
        summaries = [data['measurements'][f'b{batch}-t{length}']['summary'] for length in lengths]
        ax = axes[0, column]
        for key, label, color, line in styles:
            ax.plot(lengths, [row[key]['cuda_ms']['median'] for row in summaries],
                    marker='o', markersize=3, label=label, color=color, linestyle=line)
        ax.set_title(f'Batch {batch}: complete attention layer')
        ax.set_ylabel('Latency (ms)')
        ax.grid(alpha=.2)
        ax = axes[1, column]
        for mode, variant, label, color in [('eager', 'optimized', 'Eager / Eager', '#9c583a'),
                                           ('graph', 'warp1', 'Graph / Graph (+ 1 warp)', '#df6c35')]:
            ratios = [row[f'{mode}-mha']['cuda_ms']['median'] /
                      row[f'{mode}-{variant}']['cuda_ms']['median'] for row in summaries]
            ax.plot(lengths, ratios, marker='o', markersize=3, color=color, label=label)
        ax.axhline(1, color='#333333', linewidth=1, linestyle='--')
        ax.text(150, 1.02, 'TTT leads above 1', fontsize=9)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel('Flash latency / TTT latency')
        ax.set_xlabel('Tokens')
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8, loc='lower right')
    fig.suptitle('H100, BF16, d_model=4096, 32 heads x 128; median of 5 x 20 samples')
    fig.text(.5, .01, 'Fixed-shape all-valid experiment. Graph replay excludes input copies and capture; no production Graph integration.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .025, 1, .96))
    destination = root / 'docs/experiments/2026-09-11-prefix-ttt-kernel/images/short_graph.png'
    destination.parent.mkdir(exist_ok=True)
    fig.savefig(destination, dpi=160)
    plt.close(fig)
    print(destination)


if __name__ == '__main__':
    main()
