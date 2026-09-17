"""Plot completed ordinary-eager decode evidence next to its Chinese report."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = ROOT / 'artifacts/experiments/prefix_ttt_kernel/evidence'
DOCUMENT = ROOT / 'docs/experiments/2026-09-11-prefix-ttt-kernel'
LABELS = {'production': 'Current TTT', 'local_prepare': 'TTT + Local preparation fusion',
          'mha_static': 'Flash, preallocated KV', 'mha_dynamic': 'Flash, growing KV',
          'cache_upper_bound': 'TTT cache diagnostic (not deployable)'}
COLORS = {'production': '#333333', 'local_prepare': '#0072b2', 'mha_static': '#d55e00',
          'mha_dynamic': '#e69f00', 'cache_upper_bound': '#009e73'}


def main():
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={'width_ratios': [1.4, 1]})
    for axis, filename, batch in zip(axes, ('b1-decode-curve', 'b4-decode-check'), (1, 4)):
        result = json.loads((EVIDENCE / 'mamba-decode-curve' / f'{filename}.json').read_text())
        assert not result.get('failure') and not result['protocol']['cuda_graph']
        entries = result['measurements']
        lengths = sorted(int(key.split('-t')[1]) for key in entries)
        for name, label in LABELS.items():
            rows = [entries[f'b{batch}-t{length}'] for length in lengths]
            assert all(row['status'] == 'succeeded' for row in rows)
            values = [row['summary'][name]['cuda_ms_per_step'] for row in rows]
            axis.plot(lengths, values, marker='o', markersize=4, label=label,
                      color=COLORS[name], linestyle='--' if name == 'cache_upper_bound' else '-')
        axis.set_xscale('log', base=2)
        axis.set_xticks(lengths, [str(length) for length in lengths])
        axis.set_xlabel('Initial prefix tokens')
        axis.set_ylabel('Complete attention layer: ms / decode step')
        axis.set_title(f'Batch {batch}; 128 consecutive decode steps')
        axis.grid(alpha=.2)
        axis.set_ylim(bottom=0)
    axes[0].legend(fontsize=8, loc='best')
    figure.suptitle('H100 / BF16 / FP32 TTT state / ordinary eager / no CUDA Graph', fontsize=11)
    figure.tight_layout()
    output = DOCUMENT / 'images/decode_layer.png'
    output.parent.mkdir(exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(output)


if __name__ == '__main__':
    main()
