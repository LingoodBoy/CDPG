import argparse
import sys
import os

from ensemble_clustering_temporal import (
    EnsembleClusteringBuilder,
    run_experiment
)
import numpy as np


def check_experiment_done(output_dir, exp_id, target_name):
    main_file = os.path.join(output_dir, f'temporal_emb_exp{exp_id}_target_{target_name}.npy')
    exp_dir = os.path.join(output_dir, f'exp{exp_id}')

    if os.path.exists(main_file) and os.path.exists(exp_dir):
        print(f"✓ Exp{exp_id}has completed: {main_file}")
        return True
    return False

# Run only Experiment 1
# python ensemble_clustering_temporal_v2.py --exp 1
# Run only Experiment 2
# python ensemble_clustering_temporal_v2.py --exp 2
# Run only Experiment 3
# python ensemble_clustering_temporal_v2.py --exp 3
# Run all experiments (default)
# python ensemble_clustering_temporal_v2.py --exp all
# Skip completed experiments and continue with unfinished ones
# python ensemble_clustering_temporal_v2.py --exp all --skip-completed

# Experiment 1: source = [sh, nc], target = nj
# Experiment 2: source = [sh, nj], target = nc
# Experiment 3: source = [nc, nj], target = sh

def main():
    parser = argparse.ArgumentParser(
        description='Ensemble Clustering Temporal Embedding',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        temporal
        """
    )

    parser.add_argument(
        '--exp',
        type=str,
        default='all',
        choices=['1', '2', '3', 'all'],
        help='runing exp (1, 2, 3, or all)'
    )

    parser.add_argument(
        '--skip-completed',
        action='store_true',
        help=''
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default='/Users/xiaolinzi/mac-syncthing/project/my_seq/data/handled',
        help=''
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='/Users/xiaolinzi/mac-syncthing/project/my_seq/temporal_v3/results',
        help=''
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print(" " * 15 + "Ensemble Clustering Temporal Embedding")
    print(" " * 20 + "")
    print("=" * 70)
    print()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("loading data")
    print("=" * 70)

    all_data = {
        'sh': np.load(f'{args.data_dir}/sh/sh_4505_1488.npy')[:, :, 0].T,
        'nc': np.load(f'{args.data_dir}/nc/nc_7702_336.npy')[:, :, 0].T,
        'nj': np.load(f'{args.data_dir}/nj/nj_8000_2784.npy')[:, :, 0].T
    }

    for city, data in all_data.items():
        print(f"✓ {city.upper()}: {data.shape} ({data.shape[1]/48:.1f}day)")
    print()

    all_experiments = {
        1: {'exp_id': 1, 'source': ['sh', 'nc'], 'target': 'nj'},
        2: {'exp_id': 2, 'source': ['sh', 'nj'], 'target': 'nc'},
        3: {'exp_id': 3, 'source': ['nc', 'nj'], 'target': 'sh'}
    }

    if args.exp == 'all':
        experiments_to_run = [1, 2, 3]
    else:
        experiments_to_run = [int(args.exp)]

    print("=" * 70)
    print("Configuration")
    print("=" * 70)
    print(f"runing exp: {experiments_to_run}")
    if args.skip_completed:
        print("skip exp")
    print()

    results = {}
    skipped = []
    completed = []

    for exp_id in experiments_to_run:
        exp = all_experiments[exp_id]

        if args.skip_completed and check_experiment_done(
            args.output_dir, exp['exp_id'], exp['target']
        ):
            skipped.append(exp_id)
            print(f"→ skip exp {exp_id}\n")
            continue

        try:
            result = run_experiment(
                exp_id=exp['exp_id'],
                source_names=exp['source'],
                target_name=exp['target'],
                all_data=all_data,
                output_dir=args.output_dir
            )
            results[exp['exp_id']] = result
            completed.append(exp_id)
        except Exception as e:
            print(f"\nexp{exp_id}failed: {e}\n")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print()

    if completed:
        print(f"compelete exp: {completed}")
        for exp_id in completed:
            exp = all_experiments[exp_id]
            print(f"  exp{exp_id}: {'+'.join([c.upper() for c in exp['source']])} → {exp['target'].upper()}")
            print(f"    file: temporal_emb_exp{exp_id}_target_{exp['target']}.npy")

    if skipped:
        print(f"\n→ skip exp: {skipped}")

    if not completed and not skipped:
        print("non completed")
        sys.exit(1)

    print()
    print("output dir:", args.output_dir)
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
