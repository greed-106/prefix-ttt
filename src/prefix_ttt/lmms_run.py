"""Invoke pinned lmms-eval APIs; no task, inference loop or metric reimplementation."""
import argparse
from importlib.metadata import version
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pretrained', required=True)
    parser.add_argument('--checkpoint')
    parser.add_argument('--tasks', default='mme,pope,gqa')
    parser.add_argument('--output-path', required=True)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    tasks = args.tasks.split(',')
    if not tasks or not set(tasks) <= {'mme', 'pope', 'gqa'} or len(set(tasks)) != len(tasks):
        parser.error('Select only the fixed official mme,pope,gqa tasks')
    if args.limit is not None and args.limit <= 0:
        parser.error('limit must be positive (debug only)')
    if version('lmms-eval') != '0.7.2':
        raise RuntimeError('Evaluation protocol requires lmms-eval==0.7.2')
    output = Path(args.output_path)
    if list(output.rglob('*_results.json')):
        raise FileExistsError('Existing evaluation results: choose an explicit new run directory')
    import lmms_eval.tasks
    from lmms_eval import evaluator
    from lmms_eval.loggers import EvaluationTracker
    from lmms_eval.tasks import TaskManager
    from lmms_eval.utils import get_datetime_str, make_table
    import prefix_ttt.lmms_model  # Register only the checkpoint-loading adapter.

    # The wheel has unrelated incomplete video tasks. Use the public task-manager
    # selection API to index only original installed YAML/utils, without copying
    # or editing tasks. The evaluator and logger below are entirely official.
    task_root = Path(lmms_eval.tasks.__file__).parent
    manager = TaskManager(include_defaults=False,
                          include_path=[str(task_root / name) for name in tasks],
                          model_name='prefix_ttt_llava')
    if any(',' in value for value in (args.pretrained, args.checkpoint or '')):
        parser.error('Model paths must not contain commas')
    model_args = f'pretrained={args.pretrained},conv_template=vicuna_v1'
    if args.checkpoint:
        model_args += f',checkpoint={args.checkpoint}'
    tracker = EvaluationTracker(output_path=str(output))
    stamp = get_datetime_str(timezone='Asia/Shanghai')
    results = evaluator.simple_evaluate(model='prefix_ttt_llava', model_args=model_args,
        tasks=tasks, task_manager=manager, batch_size=1, limit=args.limit,
        log_samples=True, evaluation_tracker=tracker, datetime_str=stamp,
        random_seed=42, numpy_random_seed=42, torch_random_seed=42,
        fewshot_random_seed=42, verbosity='INFO')
    if results is not None:  # Official evaluator returns results only on rank 0.
        samples = results.pop('samples')
        tracker.save_results_aggregated(results=results, samples=samples, datetime_str=stamp)
        for task in results['configs']:
            tracker.save_results_samples(task_name=task, samples=samples[task])
        print(make_table(results), flush=True)


if __name__ == '__main__':
    main()
