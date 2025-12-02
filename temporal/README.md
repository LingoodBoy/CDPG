Run only Experiment 1
python ensemble_clustering_temporal_v2.py --exp 1
Run only Experiment 2
python ensemble_clustering_temporal_v2.py --exp 2
Run only Experiment 3
python ensemble_clustering_temporal_v2.py --exp 3
Run all experiments (default)
python ensemble_clustering_temporal_v2.py --exp all
Skip completed experiments and continue with unfinished ones
python ensemble_clustering_temporal_v2.py --exp all --skip-completed

Experiment 1: source = [sh, nc], target = nj
Experiment 2: source = [sh, nj], target = nc
Experiment 3: source = [nc, nj], target = sh