import json
import sys
import os
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

sys.path.append(os.getcwd() + '/src')
from dataloader import Dataloader
from cluster_tree_model import *  # required for unpickling ClusterTreeModel
from cdr_generator import CDRGenerator

COLORS = ['#00549f', '#407fb7', '#8ebae5', '#c7ddf2', '#e8f1fa', '#E30066', '#F19DAF']
# --- Setup ---
method = 'gmm'
experiment_name = 'AC_model_gmm'
basepath = os.getcwd()

dataloader = Dataloader(basepath, experiment_name=experiment_name)
model = dataloader.load_model('cluster_models', 'cluster_tree_model.pkl')
model.dataloader = dataloader
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
config_path = os.path.join(parent_dir, "config.json")


def load_config():
    with open(config_path, "r") as file:
        return json.load(file)

config = load_config()


def ecdf(x):
    x_sorted = np.sort(x)
    y = np.arange(1, len(x_sorted) + 1) / len(x_sorted)
    return x_sorted, y


def _convert_units(rv, gv, feat):
    """Convert arrays to display units (Wh→kWh, seconds→hours)."""
    if feat == 'quantity_in_wh':
        return rv / 1000, gv / 1000
    if feat == 'duration_sec':
        return rv / 3600, gv / 3600
    return rv, gv


def plot_qq_all_clusters(real_dict, gen_dict, features, clusters, basepath, experiment_name, method, config):
    """Create one QQ plot (with top/right marginals) per feature overlaying all clusters.

    Each feature gets its own figure matching the style of `plot_ecdf` (joint QQ with
    marginal histograms). Curves for clusters are colored consistently and a legend
    is added to the joint axis.
    """
    clusters_list = [c for c in clusters if c in real_dict and c in gen_dict]
    if len(clusters_list) == 0:
        return

    # quantile sampling for comparable-length QQ curves
    n_quantiles = 10000
    qs = np.linspace(0, 100, n_quantiles)

    # Use blue shades for A/C-clusters and red shades for B/D-clusters
    blue_shades = ['#00549f', '#407fb7', '#8ebae5', '#c7ddf2']
    red_shades = ['#E30066', '#F19DAF', '#FBEEF2']

    acdc = 'AC' if 'AC' in experiment_name else 'DC'
    reverse_cluster_map = {v: k for k, v in config["cluster_map"][acdc].items()}

    cluster_color_map = {}
    all_unique_clusters_translated = sorted([config["cluster_map"][acdc][c] for c in clusters_list])
    for cluster in all_unique_clusters_translated:
        if cluster.startswith('A'):
            idx = int(cluster.split('A')[1]) - 1 if cluster.split('A')[1].isdigit() else 0
            cluster_color_map[cluster] = blue_shades[idx % len(blue_shades)]
        elif cluster.startswith('B'):
            idx = int(cluster.split('B')[1]) - 1 if cluster.split('B')[1].isdigit() else 0
            cluster_color_map[cluster] = red_shades[idx % len(red_shades)]
        elif cluster.startswith('C'):
            idx = int(cluster.split('C')[1]) - 1 if cluster.split('C')[1].isdigit() else 0
            cluster_color_map[cluster] = blue_shades[idx % len(blue_shades)]
        elif cluster.startswith('D'):
            idx = int(cluster.split('D')[1]) - 1 if cluster.split('D')[1].isdigit() else 0
            cluster_color_map[cluster] = red_shades[idx % len(red_shades)]
        else:
            cluster_color_map[cluster] = COLORS[0]

    for feat in features:
        available = False
        mins = []
        maxs = []
        for cluster in clusters_list:
            arr_r = real_dict[cluster].get(feat, None) if isinstance(real_dict.get(cluster, {}), dict) else real_dict[cluster][feat]
            arr_g = gen_dict[cluster].get(feat, None) if isinstance(gen_dict.get(cluster, {}), dict) else gen_dict[cluster][feat]

            if arr_r is None or arr_g is None:
                continue

            rv = np.asarray(arr_r.dropna()) if hasattr(arr_r, 'dropna') else np.asarray(arr_r)
            gv = np.asarray(arr_g.dropna()) if hasattr(arr_g, 'dropna') else np.asarray(arr_g)
            rv = rv[~np.isnan(rv)] if rv.size else rv
            gv = gv[~np.isnan(gv)] if gv.size else gv
            if rv.size == 0 or gv.size == 0:
                continue

            rv, gv = _convert_units(rv, gv, feat)
            if feat == 'duration_sec':
                rv = np.clip(rv, 0, 70)
                gv = np.clip(gv, 0, 70)

            available = True
            mins.append(min(rv.min(), gv.min()))
            maxs.append(max(rv.max(), gv.max()))

        if not available:
            continue

        vmin = min(mins)
        vmax = max(maxs)
        if feat == 'duration_sec':
            vmin = 0
            vmax = 70

        fig = plt.figure(figsize=(7/2.54, 5/2.54))
        gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4), hspace=0.05, wspace=0.05)
        ax_marg_x = fig.add_subplot(gs[0, 0])
        ax_joint = fig.add_subplot(gs[1, 0])
        ax_marg_y = fig.add_subplot(gs[1, 1])

        handles = []
        labels = []
        real_qs = []
        gen_qs = []
        colors = []

        for translated_cluster in all_unique_clusters_translated:
            cluster = reverse_cluster_map.get(translated_cluster)
            if cluster is None or cluster not in real_dict or cluster not in gen_dict:
                continue

            arr_r = real_dict[cluster].get(feat, None) if isinstance(real_dict.get(cluster, {}), dict) else real_dict[cluster][feat]
            arr_g = gen_dict[cluster].get(feat, None) if isinstance(gen_dict.get(cluster, {}), dict) else gen_dict[cluster][feat]

            if arr_r is None or arr_g is None:
                continue

            rv = np.asarray(arr_r.dropna()) if hasattr(arr_r, 'dropna') else np.asarray(arr_r)
            gv = np.asarray(arr_g.dropna()) if hasattr(arr_g, 'dropna') else np.asarray(arr_g)
            rv = rv[~np.isnan(rv)] if rv.size else rv
            gv = gv[~np.isnan(gv)] if gv.size else gv
            if rv.size == 0 or gv.size == 0:
                continue

            rv, gv = _convert_units(rv, gv, feat)

            real_q = np.percentile(np.sort(rv), qs)
            gen_q = np.percentile(np.sort(gv), qs)

            color = cluster_color_map.get(translated_cluster)
            line = ax_joint.plot(real_q, gen_q, linestyle='-', linewidth=1, label=str(translated_cluster), alpha=1, color=color)

            real_qs.append(real_q)
            gen_qs.append(gen_q)
            colors.append(color)

            if line:
                handles.append(line[0])
                labels.append(f"$C_{{{translated_cluster}}}$")

        ax_joint.plot([vmin, vmax], [vmin, vmax], 'k--', linewidth=0.8, label='y=x')
        ax_joint.set_xlim(vmin, vmax)
        ax_joint.set_ylim(vmin, vmax)
        name_mapping = {'quantity_in_wh': 'Energy [kWh]', 'duration_sec': 'Duration [h]', 'start_hour': 'Start Hour [h]'}
        ax_joint.set_xlabel(f'Original {name_mapping.get(feat, feat)}', fontsize=8)
        ax_joint.set_ylabel(f'Generated {name_mapping.get(feat, feat)}', fontsize=8)
        if feat == 'duration_sec':
            ax_joint.set_xlim(0, 70)
            ax_joint.set_ylim(0, 70)
        ax_joint.tick_params(axis='both', labelsize=8)
        ax_joint.grid()

        if len(real_qs) > 0:
            ax_marg_x.hist(real_qs, bins=50, stacked=True, color=colors, alpha=1)
        if len(gen_qs) > 0:
            ax_marg_y.hist(gen_qs, bins=50, stacked=True, orientation='horizontal', color=colors, alpha=1)

        ax_marg_x.set_xlim(vmin, vmax)
        ax_marg_y.set_ylim(vmin, vmax)
        ax_marg_x.tick_params(axis='x', labelbottom=False)
        ax_marg_y.tick_params(axis='y', labelleft=False)
        ax_marg_x.tick_params(axis='y', labelsize=8)
        ax_marg_y.tick_params(axis='x', labelsize=8)
        ax_marg_x.set_ylabel('n [-]', fontsize=8)
        ax_marg_y.set_xlabel('n [-]', fontsize=8)

        if handles and labels:
            ax_joint.legend(handles=handles, labels=labels, fontsize=8, loc='lower right', bbox_to_anchor=(0.98, 0.02), ncol=2, framealpha=0.9, columnspacing=0.5, handletextpad=0.3)

        out_dir = os.path.join(basepath, 'results', experiment_name, 'analysis', 'ks_plots')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'ks_qq_all_clusters_{feat}_{method}.pdf')
        plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)


results = []
real_by_cluster = {}
gen_by_cluster = {}

features_global = ['duration_sec', 'quantity_in_wh', 'start_hour']

if 'AC' in experiment_name:
    is_AC_model = True
    is_DC_model = False
elif 'DC' in experiment_name:
    is_AC_model = False
    is_DC_model = True

for cluster in tqdm(model.final_clusters, desc="KS validation per cluster"):

    # Load real data for the same cluster
    real_data = dataloader.load_original_cdrs(cluster, is_AC_model=is_AC_model, is_DC_model=is_DC_model)
    real_data['start_hour'] = pd.to_datetime(real_data['start_time']).dt.hour + pd.to_datetime(real_data['start_time']).dt.minute / 60.0
    real_data['end_hour'] = pd.to_datetime(real_data['end_time']).dt.hour + pd.to_datetime(real_data['end_time']).dt.minute / 60.0
    real_data['start_hour_angle'] = real_data['start_hour'] * (2 * np.pi / 24)
    real_data['end_hour_angle'] = real_data['end_hour'] * (2 * np.pi / 24)
    real_data = real_data[real_data['duration_sec'] < 3*24*3600]

    n_cdrs = real_data.shape[0]
    try:
        cdrs_gen_filtered = pd.read_csv(os.path.join(basepath, 'results', experiment_name, 'analysis', f'{cluster}_generated_cdrs.csv'))
    except FileNotFoundError:
        cdr_generator = CDRGenerator(dataloader, cluster, is_AC_model=is_AC_model, is_DC_model=is_DC_model)
        cdrs_gen = cdr_generator.generate_cdrs(method, n_cdrs_per_week=n_cdrs, n_weeks=1)
        cdrs_gen['start_hour'] = pd.to_datetime(cdrs_gen['start_time']).dt.hour + pd.to_datetime(cdrs_gen['start_time']).dt.minute / 60.0
        cdrs_gen['end_hour'] = pd.to_datetime(cdrs_gen['end_time']).dt.hour + pd.to_datetime(cdrs_gen['end_time']).dt.minute / 60.0
        cdrs_gen_filtered = cdrs_gen[cdrs_gen['duration_sec'] > 15*60]

    real_data_filtered = real_data[real_data['duration_sec'] > 15*60]
    os.makedirs(os.path.join(basepath, 'results', experiment_name, 'analysis'), exist_ok=True)
    cdrs_gen_filtered.to_csv(os.path.join(basepath, 'results', experiment_name, 'analysis', f'{cluster}_generated_cdrs.csv'))

    try:
        real_by_cluster[cluster] = real_data_filtered.copy()
        gen_by_cluster[cluster] = cdrs_gen_filtered.copy()
    except Exception as e:
        print(f"Warning: could not store cluster {cluster} data: {e}")

    if real_data_filtered.empty or cdrs_gen_filtered.empty:
        continue

    acdc = 'AC' if 'AC' in experiment_name else 'DC'
    cluster_result = {'cluster': config['cluster_map'][acdc][cluster], 'n_cdrs_real': n_cdrs}

    for feat in features_global:
        if feat not in real_data.columns or feat not in cdrs_gen_filtered.columns:
            continue

        real_feat = real_data[feat].dropna()
        gen_feat = cdrs_gen_filtered[feat].dropna()
        real_feat_filtered = real_data_filtered[feat].dropna()
        gen_feat_filtered = cdrs_gen_filtered[feat].dropna()

        if len(real_feat) > 0 and len(gen_feat) > 0:
            D, p = ks_2samp(real_feat, gen_feat)
            cluster_result[f'{feat}_D'] = D
            cluster_result[f'{feat}_p'] = p

            ecdf_real_x, ecdf_real_y = ecdf(real_feat)
            ecdf_gen_x, ecdf_gen_y = ecdf(gen_feat)

            x_common = np.linspace(
                min(ecdf_real_x.min(), ecdf_gen_x.min()),
                max(ecdf_real_x.max(), ecdf_gen_x.max()),
                1000
            )
            y_r_i = np.interp(x_common, ecdf_real_x, ecdf_real_y)
            y_g_i = np.interp(x_common, ecdf_gen_x, ecdf_gen_y)

            r2 = r2_score(y_r_i, y_g_i)
            cluster_result[f'{feat}_R2'] = r2

            cluster_result[f'{feat}_distance_median'] = np.median(real_feat) - np.median(gen_feat)
            cluster_result[f'{feat}_distance_mean'] = np.mean(real_feat) - np.mean(gen_feat)
            cluster_result[f'{feat}_distance_median_filtered'] = np.median(real_feat_filtered) - np.median(gen_feat_filtered)
            cluster_result[f'{feat}_distance_mean_filtered'] = np.mean(real_feat_filtered) - np.mean(gen_feat_filtered)
            cluster_result[f'{feat}_distance_median_relative'] = (np.median(real_feat) - np.median(gen_feat)) / np.median(real_feat) if np.median(real_feat) != 0 else np.nan
            cluster_result[f'{feat}_distance_mean_relative'] = (np.mean(real_feat) - np.mean(gen_feat)) / np.mean(real_feat) if np.mean(real_feat) != 0 else np.nan

    results.append(cluster_result)


plot_qq_all_clusters(real_by_cluster, gen_by_cluster, features_global, model.final_clusters, basepath, experiment_name, method, config)

# --- Export ---
df_ks = pd.DataFrame(results)
out_path = os.path.join(basepath, 'results', experiment_name, 'analysis', f'ks_validation_{method}.csv')
df_ks.to_csv(out_path, index=False)
print(f"KS Validation results saved to {out_path}")

print('Finished')
