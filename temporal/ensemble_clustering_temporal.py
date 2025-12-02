import numpy as np
import os
import pickle
from datetime import datetime
from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import dtw
from sklearn.preprocessing import StandardScaler
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


class EnsembleClusteringBuilder:

    def __init__(self, segment_length=128, n_clusters=8,
                 n_samples_per_station=10, random_state=42):
        self.segment_length = segment_length
        self.n_clusters = n_clusters
        self.n_samples_per_station = n_samples_per_station
        self.random_state = random_state

        self.source_centroids = None
        self.kmeans_model = None

        np.random.seed(random_state)

    def _preprocess_data(self, data):

        data_log = np.log10(data + 1)
        scaler = StandardScaler()
        data_normalized = scaler.fit_transform(data_log.T).T
        return data_normalized

    def _preprocess_single(self, data):

        data_log = np.log10(data + 1)
        data_normalized = (data_log - data_log.mean()) / (data_log.std() + 1e-8)
        return data_normalized

    def random_sample_segments(self, city_data, city_name):

        n_stations, time_steps = city_data.shape

        if time_steps < self.segment_length:
            raise ValueError(
                f"{city_name}: time step({time_steps}) smaller than ({self.segment_length})"
            )

        all_segments = []
        station_ids = []

        max_start = time_steps - self.segment_length

        for station_id in range(n_stations):
            station_ts = city_data[station_id]

            actual_n_samples = min(self.n_samples_per_station, max_start + 1)
            start_positions = np.random.choice(
                max_start + 1,
                size=actual_n_samples,
                replace=False
            )

            for start in start_positions:
                segment = station_ts[start:start + self.segment_length]
                all_segments.append(segment)
                station_ids.append(station_id)

        segments = np.array(all_segments)
        station_ids = np.array(station_ids)

        print(f"  {city_name}: {n_stations}base stations × {actual_n_samples}sample times = {len(segments)} segments")

        return segments, station_ids

    def build_source_embeddings(self, source_cities_data, source_names):

        print("\n" + "=" * 70)
        print("step1: Source city - Ensemble Clustering")
        print("=" * 70)
        print(f"Source cities: {source_names}")
        print(f"K: {self.n_clusters}")
        print(f"sample segments: {self.n_samples_per_station}")
        print()

        all_segments = []
        all_station_ids = []
        city_n_stations = []

        for city_data, city_name in zip(source_cities_data, source_names):
            segments, station_ids = self.random_sample_segments(city_data, city_name)
            all_segments.append(segments)
            all_station_ids.append(station_ids)
            city_n_stations.append(city_data.shape[0])

        all_segments = np.concatenate(all_segments, axis=0)

        station_id_offset = 0
        for i in range(len(all_station_ids)):
            all_station_ids[i] = all_station_ids[i] + station_id_offset
            station_id_offset += city_n_stations[i]

        all_station_ids = np.concatenate(all_station_ids, axis=0)
        total_n_stations = sum(city_n_stations)

        print(f"\nTotal: {len(all_segments)}segments, {total_n_stations}base stations")
        print()

        all_segments_normalized = self._preprocess_data(all_segments)
        print()

        print(f"DTW K-means Clustering（K={self.n_clusters}）:")
        print()

        self.kmeans_model = TimeSeriesKMeans(
            n_clusters=self.n_clusters,
            metric="dtw",
            max_iter=10,
            n_jobs=-1,
            random_state=self.random_state,
            verbose=1
        )

        segment_labels = self.kmeans_model.fit_predict(all_segments_normalized)
        self.source_centroids = self.kmeans_model.cluster_centers_

        print()
        print("✓ Complete!")
        unique, counts = np.unique(segment_labels, return_counts=True)
        for cluster_id, count in zip(unique, counts):
            print(f"    Cluster {cluster_id}: {count:5d} segments ({count/len(segment_labels)*100:5.1f}%)")
        print()
        source_embeddings_dict = {}
        source_labels_dict = {}

        station_id_start = 0
        for city_name, n_stations in zip(source_names, city_n_stations):
            city_embeddings = []
            city_labels = []

            for local_station_id in range(n_stations):
                global_station_id = station_id_start + local_station_id

                mask = (all_station_ids == global_station_id)
                station_segment_labels = segment_labels[mask]

                if len(station_segment_labels) > 0:
                    final_label = stats.mode(station_segment_labels, keepdims=True)[0][0]
                else:
                    final_label = 0

                city_labels.append(final_label)
                city_embeddings.append(self.source_centroids[final_label].squeeze())

            station_id_start += n_stations

            city_embeddings = np.array(city_embeddings)
            city_labels = np.array(city_labels)

            source_embeddings_dict[city_name] = city_embeddings
            source_labels_dict[city_name] = city_labels

            unique, counts = np.unique(city_labels, return_counts=True)
            for cluster_id, count in zip(unique, counts):
                print(f"    Cluster {cluster_id}: {count:5d} base station ({count/n_stations*100:5.1f}%)")

        print()
        print("=" * 70)
        print()

        return source_embeddings_dict, source_labels_dict, self.source_centroids

    def build_target_embeddings(self, target_city_data, target_name):
        if self.source_centroids is None:
            raise ValueError("use build_source_embeddings()")

        print("=" * 70)
        print("DTW Matching")
        print("=" * 70)
        print(f"Target city: {target_name}")
        print(f"use {self.segment_length} step for matching")
        print()

        n_stations, time_steps = target_city_data.shape

        target_embeddings = []
        target_labels = []
        target_distances = []

        for station_id in range(n_stations):
            if (station_id + 1) % 1000 == 0:
                print(f"  progress: {station_id + 1}/{n_stations}")

            pattern = target_city_data[station_id, :self.segment_length]

            pattern_normalized = self._preprocess_single(pattern)

            dtw_distances = []
            for centroid in self.source_centroids:
                dist = dtw(pattern_normalized, centroid.squeeze())
                dtw_distances.append(dist)

            nearest_idx = np.argmin(dtw_distances)
            target_labels.append(nearest_idx)
            target_embeddings.append(self.source_centroids[nearest_idx].squeeze())
            target_distances.append(dtw_distances[nearest_idx])

        target_embeddings = np.array(target_embeddings)
        target_labels = np.array(target_labels)
        target_distances = np.array(target_distances)

        print("Matching complete！")
        unique, counts = np.unique(target_labels, return_counts=True)
        for cluster_id, count in zip(unique, counts):
            print(f"    Cluster {cluster_id}: {count:5d} base station ({count/n_stations*100:5.1f}%)")
        print(f"  avg dtw: {np.mean(target_distances):.3f}")
        print(f"  dtw range: [{np.min(target_distances):.3f}, {np.max(target_distances):.3f}]")
        print()
        print("=" * 70)
        print()

        return target_embeddings, target_labels, target_distances

    def merge_embeddings(self, embeddings_dict, city_order=['sh', 'nc', 'nj']):
        merged = []
        for city_name in city_order:
            if city_name in embeddings_dict:
                merged.append(embeddings_dict[city_name])

        merged = np.concatenate(merged, axis=0)
        return merged

    def visualize_centroids(self, output_path):
        if self.source_centroids is None:
            return

        fig, axes = plt.subplots(3, 3, figsize=(20, 12))
        axes = axes.flatten()
        colors = plt.cm.tab10(np.linspace(0, 1, self.n_clusters))

        ax = axes[0]
        for i, (centroid, color) in enumerate(zip(self.source_centroids, colors)):
            ax.plot(centroid.squeeze(), color=color, linewidth=2, label=f'Cluster {i}')
        ax.set_title('compare', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        for i in range(min(8, self.n_clusters)):
            ax = axes[i + 1]
            centroid = self.source_centroids[i].squeeze()
            ax.plot(centroid, color=colors[i], linewidth=2)
            ax.fill_between(range(len(centroid)), centroid, alpha=0.3, color=colors[i])
            ax.set_title(f'Cluster {i}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()


def run_experiment(exp_id, source_names, target_name, all_data, output_dir):
    print("\n\n")
    print("=" * 70)
    print(f"exp{exp_id}: Source=[{', '.join(source_names)}], Target=[{target_name}]")
    print("=" * 70)
    print()

    builder = EnsembleClusteringBuilder(
        segment_length=128,
        n_clusters=8,
        n_samples_per_station=10,
        random_state=42
    )

    source_cities_data = [all_data[name] for name in source_names]
    source_embeddings_dict, source_labels_dict, centroids = builder.build_source_embeddings(
        source_cities_data, source_names
    )

    target_city_data = all_data[target_name]
    target_embeddings, target_labels, target_distances = builder.build_target_embeddings(
        target_city_data, target_name
    )

    all_embeddings_dict = {}
    all_embeddings_dict.update(source_embeddings_dict)
    all_embeddings_dict[target_name] = target_embeddings

    merged_embeddings = builder.merge_embeddings(
        all_embeddings_dict,
        city_order=['sh', 'nc', 'nj']
    )

    print("=" * 70)
    print(f"exp{exp_id}complete！")
    print("=" * 70)
    exp_output_dir = os.path.join(output_dir, f'exp{exp_id}')
    os.makedirs(exp_output_dir, exist_ok=True)

    np.save(
        os.path.join(output_dir, f'temporal_emb_exp{exp_id}_target_{target_name}.npy'),
        merged_embeddings
    )

    np.save(os.path.join(exp_output_dir, 'centroids.npy'), centroids)

    for city_name in ['sh', 'nc', 'nj']:
        if city_name in source_labels_dict:
            np.save(
                os.path.join(exp_output_dir, f'{city_name}_labels.npy'),
                source_labels_dict[city_name]
            )
        elif city_name == target_name:
            np.save(
                os.path.join(exp_output_dir, f'{city_name}_labels.npy'),
                target_labels
            )
            np.save(
                os.path.join(exp_output_dir, f'{city_name}_distances.npy'),
                target_distances
            )

    metadata = {
        'exp_id': exp_id,
        'source_cities': source_names,
        'target_city': target_name,
        'segment_length': 128,
        'n_clusters': 8,
        'n_samples': 10,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(os.path.join(exp_output_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)

    builder.visualize_centroids(
        os.path.join(exp_output_dir, 'centroids_visualization.png')
    )

    print(f"saving to: {output_dir}/")
    print(f"  - temporal_emb_exp{exp_id}_target_{target_name}.npy (main file)")
    print(f"  - exp{exp_id}/ (detail)")
    print()

    return merged_embeddings


def main():

    print("\n" + "=" * 70)
    print(" " * 15 + "Ensemble Clustering Temporal Embedding")
    print("=" * 70)
    print()

    DATA_DIR = '/Users/xiaolinzi/mac-syncthing/project/my_seq/data/handled'
    OUTPUT_DIR = '/Users/xiaolinzi/mac-syncthing/project/my_seq/temporal_v3/results'

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("loading data")
    print("=" * 70)

    all_data = {
        'sh': np.load(f'{DATA_DIR}/sh/sh_4505_1488.npy')[:, :, 0].T,  # (4505, 1488)
        'nc': np.load(f'{DATA_DIR}/nc/nc_7702_336.npy')[:, :, 0].T,   # (7702, 336)
        'nj': np.load(f'{DATA_DIR}/nj/nj_8000_2784.npy')[:, :, 0].T   # (8000, 2784)
    }

    for city, data in all_data.items():
        print(f"✓ {city.upper()}: {data.shape} ({data.shape[1]/48:.1f}day)")
    print()

    experiments = [
        {'exp_id': 1, 'source': ['sh', 'nc'], 'target': 'nj'},
        {'exp_id': 2, 'source': ['sh', 'nj'], 'target': 'nc'},
        {'exp_id': 3, 'source': ['nc', 'nj'], 'target': 'sh'}
    ]

    results = {}

    for exp in experiments:
        result = run_experiment(
            exp_id=exp['exp_id'],
            source_names=exp['source'],
            target_name=exp['target'],
            all_data=all_data,
            output_dir=OUTPUT_DIR
        )
        results[exp['exp_id']] = result

    print("\n" + "=" * 70)
    print("All exps has completed！")
    print("=" * 70)
    print()
    print("generated files:")
    print(f"  {OUTPUT_DIR}/")
    print(f"    ├── temporal_emb_exp1_target_nj.npy  (sh+nc → nj)")
    print(f"    ├── temporal_emb_exp2_target_nc.npy  (sh+nj → nc)")
    print(f"    ├── temporal_emb_exp3_target_sh.npy  (nc+nj → sh)")
    print()
    print("(20207, 128)")
    print("range: sh[0:4505] + nc[4505:12207] + nj[12207:20207]")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
