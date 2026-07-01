# -*- coding: utf-8 -*-
"""02_spatial_decision_robustness_dashboard.ipynb

built in Colab.


"""

# @title


# -*- coding: utf-8 -*-
"""02_spatial_decision_robustness_dashboard.ipynb
Built for Google Colab.
"""

# ====================================================================
# PHASE 2C: INTERACTIVE SDSS DASHBOARD (FLAT NATIVE MAP RENDERING)
# ====================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import rankdata, spearmanr
import os
import zipfile
import io
import re
import html
import requests
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML, Image as IPyImage
import folium

try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# Set plotting style
plt.style.use('default')
sns.set_theme(style="whitegrid", rc={"axes.edgecolor": "0.15", "xtick.bottom": True, "ytick.left": True})

# Global state
STATE = {
    'df_input': None,
    'dfs': {},
    'active_file': 'None',
    'source_year': 'Not specified',
    'ready': False,
    'geojson': None,
    'active_map_var': 'baseline_rank_balanced'
}

# Define outputs for tabs
out_qa = widgets.Output()
out_preproc = widgets.Output()
out_deterministic = widgets.Output()
out_stochastic = widgets.Output()
out_sensitivity = widgets.Output()
out_scatter = widgets.Output()
out_downloads = widgets.Output()
out_map = widgets.Output(layout=widgets.Layout(min_height='650px'))

# Status UI element
status_label = widgets.HTML("<p><b>Active Dataset:</b> None | <b>Source Year:</b> Not specified | <b>Status:</b> <span style='color:gray;'>Initializing...</span></p>")

# PATCH 4: Reset runtime state cleanly on failure / reload
def reset_runtime_state():
    STATE['dfs'] = {}
    STATE['ready'] = False
    STATE['source_year'] = 'Not specified'
    STATE['active_map_var'] = 'baseline_rank_balanced'

def update_status(filename=None, is_ready=False, error_msg=None):
    if filename:
        STATE['active_file'] = filename

    base_html = f"<b>Active Dataset:</b> {STATE['active_file']} | <b>Source Year:</b> {STATE['source_year']} | <b>Status:</b> "
    if error_msg:
        status_label.value = f"<p>{base_html}<span style='color:red;'>{error_msg}</span></p>"
    elif is_ready:
        status_label.value = f"<p>{base_html}<span style='color:green;'>Analysis Ready</span></p>"
    else:
        status_label.value = f"<p>{base_html}<span style='color:orange;'>Processing...</span></p>"

# PATCH 3: Prevent stale downloads by clearing out_downloads correctly
def clear_analysis_tabs():
    for out in [out_preproc, out_deterministic, out_stochastic, out_sensitivity, out_scatter, out_map]:
        with out:
            clear_output()
    with out_downloads:
        clear_output()
        display(HTML("<p><em>No valid outputs are currently loaded. Run a valid dataset to enable downloads.</em></p>"))

# ====================================================================
# 1. DATA STANDARDIZATION
# ====================================================================
def standardize_dataframe(df_raw):
    df = df_raw.copy()

    # tiny safe addition 1: strip whitespace from headers
    df.columns = df.columns.str.strip()

    if 'region' in df.columns and 'region_code' not in df.columns:
        df = df.rename(columns={'region': 'region_code'})

    detected_year = "Not specified"
    rename_map = {}

    # tiny safe addition 2: track duplicate mappings into canonical C1..C5
    canonical_mapping_tracker = {'C1': [], 'C2': [], 'C3': [], 'C4': [], 'C5': []}

    for col in df.columns:
        for c_base in ['C1', 'C2', 'C3', 'C4', 'C5']:
            if col == c_base:
                rename_map[col] = c_base
                canonical_mapping_tracker[c_base].append(col)
            elif re.match(rf"^{c_base}_.*(?:19|20)\d{{2}}$", col):
                rename_map[col] = c_base
                canonical_mapping_tracker[c_base].append(col)
                year_match = re.search(r'(20\d{2})', col)
                if year_match and detected_year == "Not specified":
                    detected_year = year_match.group(1)

    duplicates = []
    for c_base, mapped_cols in canonical_mapping_tracker.items():
        if len(mapped_cols) > 1:
            duplicates.append(f"{c_base} mapped from {mapped_cols}")

    if duplicates:
        raise ValueError("Duplicate canonical columns detected: " + "; ".join(duplicates))

    df = df.rename(columns=rename_map)

    # PATCH 1: Normalize Identifiers
    if 'region_code' in df.columns:
        df['region_code'] = df['region_code'].astype(str).str.strip().str.upper()
        df['region_code'] = df['region_code'].replace(['NAN', 'NONE', ''], np.nan)

    if 'country' in df.columns:
        df['country'] = df['country'].astype(str).str.strip().str.upper()
        df['country'] = df['country'].replace(['NAN', 'NONE', ''], np.nan)

    if 'source_year' in df.columns and not pd.isna(df['source_year'].iloc[0]):
        detected_year = str(df['source_year'].iloc[0])
    else:
        df['source_year'] = detected_year

    STATE['source_year'] = detected_year
    return df

# ====================================================================
# 2. CORE ANALYTICAL ENGINE
# ====================================================================
def run_analytical_engine(df_canonical):
    df_input = df_canonical.copy()
    df_input['C5_log10'] = np.log10(df_input['C5'])
    df_norm = pd.DataFrame({'region_code': df_input['region_code'], 'country': df_input['country']})

    df_preprocess_meta = pd.DataFrame({
        'criterion': ['C1', 'C2', 'C3', 'C4', 'C5'],
        'direction': ['Benefit (+)', 'Cost (-)', 'Benefit (+)', 'Cost (-)', 'Cost (-)'],
        'raw_unit': ['%', 'EUR/hour', '%', 'EUR/kWh', 'PER_KM2'],
        'transform': ['None', 'None', 'None', 'None', 'log10'],
        'normalization_rule': [
            'u = (x - min) / (max - min)',
            'u = (max - x) / (max - min)',
            'u = (x - min) / (max - min)',
            'u = (max - x) / (max - min)',
            'u = (max - x) / (max - min)'
        ]
    })

    def minmax_normalize(series, direction):
        s_min, s_max = series.min(), series.max()
        if s_max == s_min:
            raise ValueError(f"Normalization failed: constant criterion detected for {series.name}")
        if direction == 1: return (series - s_min) / (s_max - s_min)
        else: return (s_max - series) / (s_max - s_min)

    df_norm['C1_norm'] = minmax_normalize(df_input['C1'], 1)
    df_norm['C2_norm'] = minmax_normalize(df_input['C2'], -1)
    df_norm['C3_norm'] = minmax_normalize(df_input['C3'], 1)
    df_norm['C4_norm'] = minmax_normalize(df_input['C4'], -1)
    df_norm['C5_norm'] = minmax_normalize(df_input['C5_log10'], -1)

    scenarios = {
        'Balanced':     [0.20, 0.20, 0.20, 0.20, 0.20],
        'CostPressure': [0.10, 0.30, 0.10, 0.35, 0.15],
        'GreenFirst':   [0.15, 0.15, 0.40, 0.20, 0.10],
        'TalentFirst':  [0.40, 0.25, 0.15, 0.10, 0.10]
    }

    df_scenarios = pd.DataFrame.from_dict(scenarios, orient='index', columns=['w1', 'w2', 'w3', 'w4', 'w5']).reset_index()
    df_scenarios = df_scenarios.rename(columns={'index': 'scenario'})

    X_norm = df_norm[['C1_norm', 'C2_norm', 'C3_norm', 'C4_norm', 'C5_norm']].values
    X_raw = df_input[['C1', 'C2', 'C3', 'C4', 'C5_log10']].values
    dirs = np.array([1, -1, 1, -1, -1])
    n_regions = len(df_input)

    def run_wsm(W, X): return np.dot(X, W)
    def run_topsis(W, X, directions):
        norm_denom = np.sqrt(np.sum(X**2, axis=0))
        X_vec_norm = X / norm_denom
        V = X_vec_norm * W
        PIS = np.where(directions == 1, np.max(V, axis=0), np.min(V, axis=0))
        NIS = np.where(directions == 1, np.min(V, axis=0), np.max(V, axis=0))
        D_plus = np.sqrt(np.sum((V - PIS)**2, axis=1))
        D_minus = np.sqrt(np.sum((V - NIS)**2, axis=1))
        with np.errstate(invalid='ignore'): return D_minus / (D_plus + D_minus)
    def run_vikor(W, X, directions, v=0.5):
        f_star = np.where(directions == 1, np.max(X, axis=0), np.min(X, axis=0))
        f_minus = np.where(directions == 1, np.min(X, axis=0), np.max(X, axis=0))
        denom = f_star - f_minus
        dist = W * (f_star - X) / denom
        S, R = np.sum(dist, axis=1), np.max(dist, axis=1)
        S_star, S_minus = np.min(S), np.max(S)
        R_star, R_minus = np.min(R), np.max(R)
        return v * (S - S_star) / (S_minus - S_star) + (1 - v) * (R - R_star) / (R_minus - R_star)

    static_ranks_data = []
    spearman_data = []
    scores_wide_dict = {row['region_code']: {'region_code': row['region_code']} for _, row in df_input.iterrows()}

    for sc_name, weights in scenarios.items():
        W = np.array(weights)
        score_wsm = run_wsm(W, X_norm)
        score_topsis = run_topsis(W, X_raw, dirs)
        score_vikor = run_vikor(W, X_raw, dirs)

        rank_wsm = rankdata(-score_wsm, method='average')
        rank_topsis = rankdata(-score_topsis, method='average')
        rank_vikor = rankdata(score_vikor, method='average')

        rho_wt, _ = spearmanr(rank_wsm, rank_topsis)
        rho_wv, _ = spearmanr(rank_wsm, rank_vikor)
        rho_tv, _ = spearmanr(rank_topsis, rank_vikor)
        spearman_data.append({'scenario': sc_name, 'rho_WSM_TOPSIS': rho_wt, 'rho_WSM_VIKOR': rho_wv, 'rho_TOPSIS_VIKOR': rho_tv})

        for i, reg in enumerate(df_input['region_code']):
            static_ranks_data.extend([
                {'region_code': reg, 'scenario': sc_name, 'method': 'WSM', 'score': score_wsm[i], 'rank': rank_wsm[i]},
                {'region_code': reg, 'scenario': sc_name, 'method': 'TOPSIS', 'score': score_topsis[i], 'rank': rank_topsis[i]},
                {'region_code': reg, 'scenario': sc_name, 'method': 'VIKOR', 'score': score_vikor[i], 'rank': rank_vikor[i]}
            ])
            scores_wide_dict[reg][f'score_WSM_{sc_name}'] = score_wsm[i]

    df_static_ranks = pd.DataFrame(static_ranks_data)
    df_spearman_corr = pd.DataFrame(spearman_data)
    df_static_scores_wide = pd.DataFrame(list(scores_wide_dict.values()))

    # Stochastic Audit
    np.random.seed(42)
    n_iterations = 10000
    random_weights = np.random.dirichlet(np.ones(5), size=n_iterations)
    stochastic_scores = np.dot(random_weights, X_norm.T)
    temp_args = np.argsort(-stochastic_scores, axis=1)
    stochastic_ranks = np.empty_like(temp_args)
    for i in range(n_iterations): stochastic_ranks[i, temp_args[i]] = np.arange(1, n_regions + 1)

    rank_acceptability = np.zeros((n_regions, n_regions))
    for i in range(n_regions):
        counts = np.bincount(stochastic_ranks[:, i], minlength=n_regions+1)[1:]
        rank_acceptability[i, :] = counts / n_iterations

    regions_arr = df_input['region_code'].values
    baseline_ranks_bal = df_static_ranks[(df_static_ranks['scenario'] == 'Balanced') & (df_static_ranks['method'] == 'WSM')].set_index('region_code').loc[regions_arr, 'rank'].to_numpy()
    DRI = np.clip(1 - (np.mean(np.abs(stochastic_ranks - baseline_ranks_bal), axis=0) / (n_regions - 1)), 0, 1)

    df_smaa_summary = pd.DataFrame({
        'region_code': regions_arr,
        'baseline_rank_balanced': baseline_ranks_bal,
        'expected_rank': np.mean(stochastic_ranks, axis=0),
        'rank_sd': np.std(stochastic_ranks, axis=0),
        'top1_acceptability': rank_acceptability[:, 0],
        'top3_acceptability': np.sum(rank_acceptability[:, :3], axis=1),
        'DRI': DRI
    })

    smaa_ra_data = [{'region_code': reg, 'rank': r+1, 'b_ir': rank_acceptability[i, r]} for i, reg in enumerate(regions_arr) for r in range(n_regions)]
    df_smaa_rank_accept = pd.DataFrame(smaa_ra_data)

    # Local Dirichlet Sensitivity
    np.random.seed(4242)
    dirichlet_sens_data = []
    w_balanced = np.array(scenarios['Balanced'])
    for kappa in [20, 50, 100]:
        local_scores = np.dot(np.random.dirichlet(kappa * w_balanced, size=2000), X_norm.T)
        temp_args_loc = np.argsort(-local_scores, axis=1)
        local_ranks = np.empty_like(temp_args_loc)
        for i in range(2000): local_ranks[i, temp_args_loc[i]] = np.arange(1, n_regions + 1)
        exp_rk_loc, sd_rk_loc = np.mean(local_ranks, axis=0), np.std(local_ranks, axis=0)
        for i, reg in enumerate(regions_arr):
            dirichlet_sens_data.append({'region_code': reg, 'kappa': kappa, 'expected_rank_dirichlet': exp_rk_loc[i], 'rank_sd_dirichlet': sd_rk_loc[i]})

    df_dirichlet_sensitivity = pd.DataFrame(dirichlet_sens_data)

    # Meta
    cluster_map = {
        'IE06': 'FLAPD_Core', 'DE71': 'FLAPD_Core', 'NL32': 'FLAPD_Core', 'FR10': 'FLAPD_Core',
        'DE30': 'Western_Secondary', 'BE10': 'Western_Secondary', 'AT13': 'Western_Secondary', 'ITC4': 'Western_Secondary',
        'SE11': 'Nordic_Baltic', 'FI1B': 'Nordic_Baltic', 'DK01': 'Nordic_Baltic', 'EE00': 'Nordic_Baltic', 'LT01': 'Nordic_Baltic', 'LV00': 'Nordic_Baltic',
        'ES30': 'Southern', 'EL30': 'Southern', 'PT17': 'Southern',
        'CZ01': 'CEE_Emerging', 'PL91': 'CEE_Emerging', 'RO32': 'CEE_Emerging', 'RO11': 'CEE_Emerging', 'HU11': 'CEE_Emerging', 'BG41': 'CEE_Emerging', 'SK01': 'CEE_Emerging'
    }
    df_regions_meta = pd.DataFrame(list(cluster_map.items()), columns=['region_code', 'cluster'])
    df_smaa_summary = df_smaa_summary.merge(df_regions_meta, on='region_code', how='left')
    df_static_scores_wide = df_static_scores_wide.merge(df_regions_meta, on='region_code', how='left')

    # Ensure custom regions don't break hue mapping in scatter plot
    df_smaa_summary['cluster'] = df_smaa_summary['cluster'].fillna('Other / Unassigned')
    df_static_scores_wide['cluster'] = df_static_scores_wide['cluster'].fillna('Other / Unassigned')

    return {
        'df_input': df_input, 'df_preprocess_meta': df_preprocess_meta, 'df_norm': df_norm,
        'df_scenarios': df_scenarios, 'df_static_ranks': df_static_ranks,
        'df_spearman_corr': df_spearman_corr, 'df_smaa_summary': df_smaa_summary,
        'df_smaa_rank_accept': df_smaa_rank_accept, 'df_dirichlet_sensitivity': df_dirichlet_sensitivity,
        'df_static_scores_wide': df_static_scores_wide, 'df_regions_meta': df_regions_meta
    }

# ====================================================================
# 3. PLOTTING & EXPORTS (ORIGINAL `08c` VERSION PRESREVED)
# ====================================================================
def generate_and_save_figures(dfs):
    plt.close('all')

    fig1, ax1 = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        dfs['df_norm'].set_index('region_code')[['C1_norm', 'C2_norm', 'C3_norm', 'C4_norm', 'C5_norm']],
        annot=True, cmap="YlGnBu", fmt=".2f", cbar_kws={'label': 'Normalized Utility'}, ax=ax1,
        xticklabels=['C1 ICT\nshare', 'C2 labor\ncost', 'C3 renewable\nelectricity', 'C4 electricity\nprice', 'C5 population\ndensity']
    )
    plt.setp(ax1.get_xticklabels(), rotation=0, ha='center', fontsize=10)
    ax1.set_title('Figure 1: Normalized Decision Matrix', pad=25)
    fig1.tight_layout()
    fig1.savefig('Figure_1_Normalized_Heatmap.png', dpi=300)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(12, 10))
    ra_pivot = dfs['df_smaa_rank_accept'].pivot(index='region_code', columns='rank', values='b_ir')
    sorted_regions = dfs['df_smaa_summary'].sort_values('expected_rank')['region_code']
    ra_pivot = ra_pivot.reindex(sorted_regions)
    sns.heatmap(
        ra_pivot, cmap="Reds", vmin=0, linewidths=0.5, linecolor='lightgray',
        cbar_kws={'label': 'Acceptability (Probability)'}, ax=ax2
    )
    ax2.set_title('Figure 2: Rank Acceptability Heatmap (Stochastic SMAA)')
    ax2.set_ylabel('Region Code (Sorted by Expected Rank)')
    fig2.tight_layout()
    fig2.savefig('Figure_2_Rank_Acceptability.png', dpi=300)
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(12, 6))
    df_smaa_sorted = dfs['df_smaa_summary'].sort_values('expected_rank', ascending=True)
    ax3.bar(
        df_smaa_sorted['region_code'],
        df_smaa_sorted['expected_rank'],
        yerr=df_smaa_sorted['rank_sd'],
        capsize=4,
        color='skyblue',
        edgecolor='black'
    )
    ax3.set_title('Figure 3: Expected Rank with Standard Deviation')
    ax3.set_ylabel('Rank (Lower is Better)')
    ax3.set_xticks(np.arange(len(df_smaa_sorted['region_code'])))
    ax3.set_xticklabels(df_smaa_sorted['region_code'], rotation=45, ha='right')
    fig3.tight_layout()
    fig3.savefig('Figure_3_Expected_Rank.png', dpi=300)
    plt.close(fig3)

    fig4, ax4 = plt.subplots(figsize=(10, 8))
    sns.scatterplot(
        data=dfs['df_static_scores_wide'],
        x='score_WSM_CostPressure',
        y='score_WSM_GreenFirst',
        hue='cluster',
        s=100,
        edgecolor='k',
        ax=ax4
    )

    for _, row in dfs['df_static_scores_wide'].iterrows():
        reg = row['region_code']
        x_off, y_off = 5, 5

        if reg == 'HU11':
            x_off, y_off = 5, -12
        elif reg == 'FI1B':
            x_off, y_off = -25, -5
        elif reg == 'LV00':
            x_off, y_off = -15, -15

        ax4.annotate(
            reg,
            (row['score_WSM_CostPressure'], row['score_WSM_GreenFirst']),
            xytext=(x_off, y_off),
            textcoords='offset points',
            fontsize=8
        )

    ax4.set_title('Figure 4: WSM Score Comparison (CostPressure vs GreenFirst)')
    ax4.set_xlabel('WSM Score (CostPressure)')
    ax4.set_ylabel('WSM Score (GreenFirst)')
    ax4.grid(True, linestyle='--', alpha=0.7)
    ax4.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    fig4.tight_layout()
    fig4.savefig('Figure_4_Scenario_Scatter.png', dpi=300)
    plt.close(fig4)

    fig5, ax5 = plt.subplots(figsize=(12, 6))
    df_dri_sorted = dfs['df_smaa_summary'].sort_values('DRI', ascending=False)
    ax5.bar(df_dri_sorted['region_code'], df_dri_sorted['DRI'], color='mediumseagreen', edgecolor='black')
    ax5.axhline(y=df_dri_sorted['DRI'].mean(), color='r', linestyle='--', label='Average DRI')
    ax5.set_title('Figure 5: Decision Robustness Index (DRI)')
    ax5.set_xticks(np.arange(len(df_dri_sorted['region_code'])))
    ax5.set_xticklabels(df_dri_sorted['region_code'], rotation=45, ha='right')
    ax5.legend()
    fig5.tight_layout()
    fig5.savefig('Figure_5_DRI.png', dpi=300)
    plt.close(fig5)

    static_map_files = []
    try:
        import geopandas as gpd
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable

        url_geojson = "https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/NUTS_RG_20M_2021_4326_LEVL_2.geojson"
        response = requests.get(url_geojson, timeout=60)
        response.raise_for_status()

        gdf = gpd.read_file(io.BytesIO(response.content))
        gdf = gdf.to_crs(epsg=3035)

        df_map = dfs['df_smaa_summary'][
            ['region_code', 'baseline_rank_balanced', 'expected_rank', 'DRI', 'top3_acceptability']
        ].copy()

        gdf_study = gdf[gdf['NUTS_ID'].isin(df_map['region_code'])].copy()
        gdf_study = gdf_study.merge(df_map, left_on='NUTS_ID', right_on='region_code', how='left')

        if not gdf_study.empty:
            minx, miny, maxx, maxy = gdf_study.total_bounds
            margin_x = (maxx - minx) * 0.10
            margin_y = (maxy - miny) * 0.10

            xlim = (minx - margin_x, maxx + margin_x)
            ylim = (miny - margin_y, maxy + margin_y)

            map_specs = [
                (
                    'baseline_rank_balanced',
                    'Figure_6_Balanced_Rank_Choropleth.png',
                    'Balanced Scenario Rank',
                    'YlGnBu_r',
                    'Balanced rank (lower = better)'
                ),
                (
                    'expected_rank',
                    'Figure_7_Expected_Rank_Choropleth.png',
                    'Expected Rank under Weight Uncertainty',
                    'YlGnBu_r',
                    'Expected rank (lower = better)'
                ),
                (
                    'DRI',
                    'Figure_8_DRI_Choropleth.png',
                    'Decision Robustness Index',
                    'YlGn',
                    'DRI'
                ),
                (
                    'top3_acceptability',
                    'Figure_9_Top3_Acceptability_Choropleth.png',
                    'Top-3 Acceptability under Weight Uncertainty',
                    'YlOrRd',
                    'Top-3 acceptability probability'
                )
            ]

            for var, fname, title, cmap, cbar_label in map_specs:
                fig, ax = plt.subplots(figsize=(11, 9))

                gdf.plot(
                    ax=ax,
                    color='#e6e6e6',
                    edgecolor='#f7f7f7',
                    linewidth=0.35
                )

                gdf_study.plot(
                    ax=ax,
                    column=var,
                    cmap=cmap,
                    edgecolor='#2b2b2b',
                    linewidth=0.8,
                    legend=True,
                    legend_kwds={
                        'shrink': 0.72,
                        'label': cbar_label
                    }
                )

                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.set_axis_off()
                ax.set_title(title, fontsize=17, pad=12)

                fig.tight_layout()
                fig.savefig(fname, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close(fig)
                static_map_files.append(fname)

            composite_specs = [
    ('baseline_rank_balanced', '(a) Balanced Rank', 'YlGnBu_r', 'Balanced rank (lower = better)'),
    ('expected_rank', '(b) Expected Rank', 'YlGnBu_r', 'Expected rank (lower = better)'),
    ('DRI', '(c) DRI', 'YlGn', 'DRI (higher = better)'),
    ('top3_acceptability', '(d) Top-3 Acceptability', 'YlOrRd', 'Top-3 probability (higher = better)')
]

            fig = plt.figure(figsize=(15.2, 11.2))
            gs = fig.add_gridspec(
                2, 4,
                width_ratios=[1.00, 0.05, 1.00, 0.05],
                height_ratios=[1.00, 1.00],
                wspace=0.08,
                hspace=0.12
            )

            positions = [
                (0, 0, 0, 1),  # map stanga sus + colorbar
                (0, 2, 0, 3),  # map dreapta sus + colorbar
                (1, 0, 1, 1),  # map stanga jos + colorbar
                (1, 2, 1, 3)   # map dreapta jos + colorbar
            ]

            for (var, title, cmap, cbar_label), (r_map, c_map, r_cbar, c_cbar) in zip(composite_specs, positions):
                ax = fig.add_subplot(gs[r_map, c_map])
                cax = fig.add_subplot(gs[r_cbar, c_cbar])

                gdf.plot(
                    ax=ax,
                    color='#e6e6e6',
                    edgecolor='#f7f7f7',
                    linewidth=0.30
                )

                values = gdf_study[var].dropna()
                vmin = float(values.min())
                vmax = float(values.max())

                if vmin == vmax:
                    vmax = vmin + 1e-9

                gdf_study.plot(
                    ax=ax,
                    column=var,
                    cmap=cmap,
                    edgecolor='#2b2b2b',
                    linewidth=0.7,
                    legend=False
                )

                sm = ScalarMappable(
                    norm=Normalize(vmin=vmin, vmax=vmax),
                    cmap=plt.get_cmap(cmap)
                )
                sm._A = []

                cbar = fig.colorbar(sm, cax=cax)
                cbar.set_label(cbar_label, fontsize=8)
                cbar.ax.tick_params(labelsize=8)

                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.set_axis_off()
                ax.set_title(title, fontsize=13, pad=8)

            fig.suptitle('Spatial Robustness Maps', fontsize=18, y=0.975)
            fig.subplots_adjust(left=0.04, right=0.98, bottom=0.04, top=0.92)

            composite_fname = 'Figure_10_Composite_2x2_Spatial_Maps.png'
            fig.savefig(composite_fname, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            static_map_files.append(composite_fname)


            with zipfile.ZipFile('Static_Publication_Maps_PNG.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
                for f in static_map_files:
                    zf.write(f)

    except Exception as e:
        print(f"Static publication maps were not generated: {e}")

    excel_files = []
    for name, df in dfs.items():
        fname = f"{name}.xlsx"
        df.to_excel(fname, index=False)
        excel_files.append(fname)

    combined_xlsx = 'Phase2_all_tables_in_one_workbook.xlsx'
    with pd.ExcelWriter(combined_xlsx) as writer:
        for name, df in dfs.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
    excel_files.append(combined_xlsx)

    fig_files = [
        'Figure_1_Normalized_Heatmap.png',
        'Figure_2_Rank_Acceptability.png',
        'Figure_3_Expected_Rank.png',
        'Figure_4_Scenario_Scatter.png',
        'Figure_5_DRI.png'
    ] + static_map_files

    with zipfile.ZipFile('Phase2_tables_and_data.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in excel_files:
            zf.write(f)

    with zipfile.ZipFile('Phase2_figures_png.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in fig_files:
            zf.write(f)

    with zipfile.ZipFile('Phase2_all_outputs.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in excel_files + fig_files:
            zf.write(f)


# ====================================================================
# 4. MAP FACTORY (FLAT NATIVE RENDERING WITH CUSTOM REGION SUPPORT)
# ====================================================================

# UI Controls for the Map
dropdown_map = widgets.Dropdown(
    options=['baseline_rank_balanced', 'expected_rank', 'DRI', 'top3_acceptability'],
    value='baseline_rank_balanced',
    description='Variable:',
    disabled=False,
)

btn_dl_map = widgets.Button(description="Download Map HTML", button_style='info', icon='download')

def dl_map_action(b):
    if IN_COLAB:
        filepath = f"MCDA_Map_{STATE['active_map_var']}.html"
        if os.path.exists(filepath):
            files.download(filepath)
        else:
            print("Map file not generated yet.")
btn_dl_map.on_click(dl_map_action)

def fetch_geojson():
    if STATE['geojson'] is None:
        url = "https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/NUTS_RG_20M_2021_4326_LEVL_2.geojson"
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            STATE['geojson'] = response.json()
        except Exception as e:
            print(f"Failed to fetch GeoJSON: {e}")
            STATE['geojson'] = {"type": "FeatureCollection", "features": []}
    return STATE['geojson']

def build_folium_map(variable):
    df_map = STATE['dfs']['df_smaa_summary']
    geojson_data = fetch_geojson()

    import copy
    geo_data_copy = copy.deepcopy(geojson_data)

    metric_dict = df_map.set_index('region_code')[variable].to_dict()
    for feature in geo_data_copy['features']:
        region_id = feature['properties'].get('NUTS_ID')
        val = metric_dict.get(region_id, 'N/A')
        if isinstance(val, (int, float)):
            val = round(val, 4)
        feature['properties'][variable] = val

    m = folium.Map(
        location=[50, 15],
        zoom_start=4,
        tiles='CartoDB positron',
        attr='CARTO'
    )

    cmap = 'YlOrRd' if variable in ['baseline_rank_balanced', 'expected_rank'] else 'YlGnBu'

    folium.Choropleth(
        geo_data=geo_data_copy,
        name='choropleth',
        data=df_map,
        columns=['region_code', variable],
        key_on='feature.properties.NUTS_ID',
        fill_color=cmap,
        fill_opacity=0.75,
        line_opacity=0.3,
        legend_name=variable.replace('_', ' ').title()
    ).add_to(m)

    folium.GeoJson(
        geo_data_copy,
        style_function=lambda x: {'fillColor': '#ffffff', 'color':'#000000', 'fillOpacity': 0.0, 'weight': 0.1},
        highlight_function=lambda x: {'fillColor': '#000000', 'color':'#000000', 'fillOpacity': 0.20, 'weight': 1},
        tooltip=folium.features.GeoJsonTooltip(
            fields=['NUTS_ID', 'NAME_LATN', variable],
            aliases=['Region Code:', 'Name:', f'{variable}:'],
            labels=True,
            sticky=True,
            style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;")
        )
    ).add_to(m)

    return m

def update_map_tab(variable):
    STATE['active_map_var'] = variable
    with out_map:
        clear_output(wait=True)
        display(HTML("<h3>Geospatial Visualization</h3>"))

        controls = widgets.HBox([dropdown_map, btn_dl_map])
        display(controls)

        if STATE.get('ready') and 'df_smaa_summary' in STATE.get('dfs', {}):
            try:
                df_map = STATE['dfs']['df_smaa_summary']
                geo_data = fetch_geojson()
                geo_nuts = {f['properties'].get('NUTS_ID') for f in geo_data.get('features', [])}
                map_nuts = set(df_map['region_code'])

                matched = map_nuts.intersection(geo_nuts)
                unmatched = map_nuts - geo_nuts

                if len(matched) == 0:
                    display(HTML("""
                    <div style='padding: 15px; background-color: #ffe6e6; border-left: 5px solid red; margin-top: 15px;'>
                        <b>Warning:</b> None of the uploaded region codes match the official GISCO NUTS-2 GeoJSON format.
                        The map cannot be rendered, but the analytical results are still valid.
                    </div>
                    """))
                    return
                elif len(unmatched) > 0:
                    display(HTML(f"""
                    <div style='padding: 10px; background-color: #fff3cd; border-left: 5px solid orange; margin-top: 10px; margin-bottom: 10px;'>
                        <b>Note:</b> The following uploaded regions were not found in the NUTS-2 boundaries and will not appear on the map:<br>
                        <code>{', '.join(unmatched)}</code>
                    </div>
                    """))

                m = build_folium_map(variable)
                html_path = f"MCDA_Map_{variable}.html"
                m.save(html_path)
                display(m)
            except Exception as e:
                display(HTML(f"<div style='color:red;'><b>Error rendering map:</b> {str(e)}</div>"))
        else:
             display(HTML("<p>Please upload and process data first.</p>"))

def on_dropdown_map_change(change):
    if change['type'] == 'change' and change['name'] == 'value':
        update_map_tab(change['new'])

dropdown_map.observe(on_dropdown_map_change)

# ====================================================================
# 5. UI POPULATION & FLEXIBLE QA LOGIC
# ====================================================================
def render_qa(df, filename):
    with out_qa:
        clear_output()
        display(HTML("<h3>Data & QA Report</h3>"))
        display(HTML(""" <div style="margin:8px 0 14px 0; padding:8px 12px; background-color:#fff3cd; color:#856404; border-left:4px solid #ffc107; font-size:13px; line-height:1.35;">     The remaining tabs are populated after the analytical engine finishes. </div> """))
        if df is None:
            update_status(filename, False, "Failed to load dataframe.")
            display(HTML("<p style='color:red;'>Failed to load valid dataframe.</p>"))
            return

        required_cols = ['region_code', 'country', 'C1', 'C2', 'C3', 'C4', 'C5']
        missing_cols = [c for c in required_cols if c not in df.columns]

        num_rows = len(df)
        has_min_rows = (num_rows >= 2)

        has_missing_region = False
        has_missing_country = False

        if 'region_code' in df.columns:
            has_duplicates = df['region_code'].duplicated().any()
            has_missing_region = df['region_code'].isna().any()
        else:
            has_duplicates = False
            has_missing_region = True

        if 'country' in df.columns:
            has_missing_country = df['country'].isna().any()
        else:
            has_missing_country = True

        c5_non_positive = False
        has_non_finite = False

        if not missing_cols:
            df_numeric = df[['C1', 'C2', 'C3', 'C4', 'C5']].apply(pd.to_numeric, errors='coerce')
            has_nan = df_numeric.isna().sum().sum() > 0
            is_numeric_type = not has_nan
            if is_numeric_type:
                # PATCH 2: Finite check (reject inf, -inf)
                has_non_finite = not np.isfinite(df_numeric.to_numpy()).all()
                if not has_non_finite:
                    c5_non_positive = (df_numeric['C5'] <= 0).any()
        else:
            has_nan = True
            is_numeric_type = False

        # Is the dataset fundamentally valid for MCDA?
        is_valid = not missing_cols and has_min_rows and not has_duplicates and not has_nan and not has_missing_region and not has_missing_country and not c5_non_positive and not has_non_finite

        status_color = "green" if is_valid else "red"
        status_msg = "PASSED" if is_valid else "FAILED"

        display(HTML(f"<h4>Validation Status: <span style='color:{status_color};'>{status_msg}</span></h4>"))

        # Informational Study Configuration Note
        if num_rows == 24:
            study_note = "<span style='color:green;'>This may correspond to the 24-region study configuration.</span>"
        else:
            study_note = f"<span style='color:blue;'>Custom region set detected ({num_rows} regions).</span>"

        qa_html = f"""
        <ul>
            <li><b>File Name:</b> {filename}</li>
            <li><b>Source Year:</b> {STATE['source_year']}</li>
            <li><b>Study Profile:</b> {study_note}</li>
            <li><b>Total Rows:</b> {num_rows} {'(Pass)' if has_min_rows else '(<span style="color:red;">Fail - Expected >= 2</span>)'}</li>
            <li><b>Region/Country Identifiers:</b> {'<span style="color:red;">Fail - Missing values found</span>' if (has_missing_region or has_missing_country) else 'Pass'}</li>
            <li><b>Duplicate Regions:</b> {'<span style="color:red;">Fail - Duplicates found</span>' if has_duplicates else 'Pass (0 Duplicates)'}</li>
            <li><b>Missing Values / NaNs in C1-C5:</b> {'<span style="color:red;">Fail (NaNs found)</span>' if has_nan else 'Pass (0 NaNs)'}</li>
            <li><b>Numeric Types in C1-C5:</b> {'Pass' if is_numeric_type else '<span style="color:red;">Fail</span>'}</li>
            <li><b>Finite Values in C1-C5:</b> {'<span style="color:red;">Fail (inf/-inf found)</span>' if has_non_finite else 'Pass'}</li>
            <li><b>Strictly Positive C5:</b> {'<span style="color:red;">Fail (C5 <= 0)</span>' if c5_non_positive else 'Pass'}</li>
        </ul>
        """
        display(HTML(qa_html))

        # Hard failure triggers
        if missing_cols:
            display(HTML(f"<p style='color:red;'><b>Missing Required Canonical Columns:</b> {missing_cols}</p>"))
            STATE['ready'] = False
            update_status(filename, False, "Validation Failed (Missing Columns)")
        elif not has_min_rows:
            display(HTML("<p style='color:red;'><b>Error:</b> Dataset must contain at least 2 distinct regions for analysis.</p>"))
            STATE['ready'] = False
            update_status(filename, False, "Validation Failed (Not Enough Rows)")
        elif has_missing_region or has_missing_country:
            display(HTML("<p style='color:red;'><b>Error:</b> 'region_code' and 'country' columns must not contain missing values.</p>"))
            STATE['ready'] = False
            update_status(filename, False, "Validation Failed (Missing Identifiers)")
        elif has_duplicates:
            display(HTML("<p style='color:red;'><b>Error:</b> Dataset contains duplicate region codes. Each NUTS-2 row must be unique.</p>"))
            STATE['ready'] = False
            update_status(filename, False, "Validation Failed (Duplicates)")
        elif has_nan:
            display(HTML("<p style='color:red;'><b>Error:</b> Data contains non-numeric or missing values in criteria columns.</p>"))
            STATE['ready'] = False
            update_status(filename, False, "Validation Failed (NaN/Non-numeric)")
        elif has_non_finite:
            display(HTML("<p style='color:red;'><b>Error:</b> Data contains infinite values (inf or -inf) in criteria columns.</p>"))
            STATE['ready'] = False
            update_status(filename, False, "Validation Failed (Infinite values)")
        elif c5_non_positive:
            display(HTML("<p style='color:red;'><b>Error:</b> C5 must be strictly positive because log10(C5) is used in preprocessing.</p>"))
            STATE['ready'] = False
            update_status(filename, False, "Validation Failed (C5 <= 0)")
        else:
            STATE['ready'] = True
            display(HTML("<p style='color:green;'><b>Dataset passed all mandatory QA checks. Analytical engine successfully executed.</b></p>"))
            update_status(filename, True)

def render_all_tabs():
    if not STATE['ready']: return
    dfs = STATE['dfs']

    with out_preproc:
        clear_output()
        display(HTML("<h3>Normalized Decision Matrix</h3>"))
        display(dfs['df_norm'].head())
        display(IPyImage(filename='Figure_1_Normalized_Heatmap.png', width=750))

    with out_deterministic:
        clear_output()
        display(HTML("<h3>Spearman Rank Correlations</h3>"))
        display(dfs['df_spearman_corr'])
        display(HTML("<h3>Static Deterministic Ranks (Sample)</h3>"))
        display(dfs['df_static_ranks'].head(15))

    with out_stochastic:
        clear_output()
        top5_wsm = dfs['df_static_ranks'][(dfs['df_static_ranks']['scenario'] == 'Balanced') & (dfs['df_static_ranks']['method'] == 'WSM')].sort_values('rank').head(5)
        top5_exp = dfs['df_smaa_summary'].sort_values('expected_rank').head(5)
        most_vol = dfs['df_smaa_summary'].sort_values('rank_sd', ascending=False).head(3)
        most_stb = dfs['df_smaa_summary'].sort_values('rank_sd', ascending=True).head(3)

        summary_html = f"""
        <div style="background-color:#f9f9f9; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
            <h3>📊 Analysis Summary</h3>
            <div style="display:flex; gap: 40px;">
                <div><b>Top 5 WSM Balanced:</b><br>{'<br>'.join([f"{r.region_code} (Rank {r.rank})" for _, r in top5_wsm.iterrows()])}</div>
                <div><b>Top 5 Expected Rank:</b><br>{'<br>'.join([f"{r.region_code} (Rank {r.expected_rank:.1f})" for _, r in top5_exp.iterrows()])}</div>
                <div><b>Most Volatile (SD):</b><br>{'<br>'.join([f"{r.region_code} ({r.rank_sd:.2f})" for _, r in most_vol.iterrows()])}</div>
                <div><b>Most Stable (SD):</b><br>{'<br>'.join([f"{r.region_code} ({r.rank_sd:.2f})" for _, r in most_stb.iterrows()])}</div>
            </div>
        </div>
        """
        display(HTML(summary_html))
        display(IPyImage(filename='Figure_3_Expected_Rank.png', width=750))
        display(IPyImage(filename='Figure_2_Rank_Acceptability.png', width=750))
        display(HTML("<h3>SMAA Summary Table</h3>"))
        display(dfs['df_smaa_summary'].head())

    with out_sensitivity:
        clear_output()
        display(HTML("<h3>Decision Robustness Index (DRI)</h3>"))
        top5_dri = dfs['df_smaa_summary'].sort_values('DRI', ascending=False).head(5)
        display(HTML(f"<p><b>Top 5 Robust Regions (DRI):</b> {', '.join(top5_dri['region_code'].tolist())}</p>"))
        display(IPyImage(filename='Figure_5_DRI.png', width=750))
        display(HTML("<h3>Local Dirichlet Sensitivity (Sample)</h3>"))
        display(dfs['df_dirichlet_sensitivity'].head(10))

    with out_scatter:
        clear_output()
        display(HTML("<h3>Scenario Comparison Scatter</h3>"))
        display(IPyImage(filename='Figure_4_Scenario_Scatter.png', width=700))

    update_map_tab(dropdown_map.value)

# ====================================================================
# 6. DOWNLOAD CONTROLS (RESTORED BUTTONS + STATIC MAPS)
# ====================================================================
btn_layout = widgets.Layout(width='230px', height='40px')

btn_wb = widgets.Button(
    description="Download Excel Book",
    button_style='info',
    icon='download',
    layout=btn_layout
)
btn_tbl = widgets.Button(
    description="Download Tables ZIP",
    button_style='info',
    icon='download',
    layout=btn_layout
)
btn_fig = widgets.Button(
    description="Download Figures ZIP",
    button_style='info',
    icon='download',
    layout=btn_layout
)
btn_all = widgets.Button(
    description="Download All Outputs",
    button_style='success',
    icon='download',
    layout=btn_layout
)
btn_static_maps = widgets.Button(
    description="Download Static Maps ZIP",
    button_style='info',
    icon='download',
    layout=btn_layout
)


def dl_wb(b):
    if IN_COLAB:
        files.download('Phase2_all_tables_in_one_workbook.xlsx')
    else:
        print("Colab required.")

def dl_tbl(b):
    if IN_COLAB:
        files.download('Phase2_tables_and_data.zip')
    else:
        print("Colab required.")

def dl_fig(b):
    if IN_COLAB:
        files.download('Phase2_figures_png.zip')
    else:
        print("Colab required.")

def dl_all(b):
    if IN_COLAB:
        files.download('Phase2_all_outputs.zip')
    else:
        print("Colab required.")

def dl_static_maps(b):
    if IN_COLAB:
        if os.path.exists('Static_Publication_Maps_PNG.zip'):
            files.download('Static_Publication_Maps_PNG.zip')
        else:
            print("Static publication maps ZIP was not generated.")
    else:
        print("Colab required.")

btn_wb.on_click(dl_wb)
btn_tbl.on_click(dl_tbl)
btn_fig.on_click(dl_fig)
btn_all.on_click(dl_all)
btn_static_maps.on_click(dl_static_maps)

def render_downloads_tab():
    with out_downloads:
        clear_output()
        display(HTML("<h3>Export Dashboard Results</h3>"))
        display(
            widgets.GridBox(
                [btn_wb, btn_tbl, btn_fig, btn_all, btn_static_maps],
                layout=widgets.Layout(
                    grid_template_columns='repeat(2, 230px)',
                    grid_gap='10px 12px'
                )
            )
        )

# ====================================================================
# 7. MAIN PROCESSING WRAPPER
# ====================================================================
def process_dataframe(df_raw, filename):
    # PATCH 4 & 3: Reset cleanly on every new file logic run
    reset_runtime_state()
    clear_analysis_tabs()

    # only added behavior here: clean format-validation stop if standardization raises
    try:
        df_canonical = standardize_dataframe(df_raw)
    except ValueError as e:
        STATE['ready'] = False
        update_status(filename, False, "Format Validation Failed")
        with out_qa:
            clear_output()
            display(HTML(f"<h3>Data & QA Report</h3><p style='color:red;'><b>Validation Failed:</b> {str(e)}</p>"))
        return

    update_status(filename, False)
    render_qa(df_canonical, filename)

    if STATE['ready']:
        try:
            # Ensure safe conversion to float before analytics
            df_canonical[['C1','C2','C3','C4','C5']] = df_canonical[['C1','C2','C3','C4','C5']].astype(float)
            STATE['dfs'] = run_analytical_engine(df_canonical)
            generate_and_save_figures(STATE['dfs'])
            render_all_tabs()
            render_downloads_tab() # Deseneaza butoanele doar la finalul executiei corecte!
        except Exception as e:
            reset_runtime_state()
            clear_analysis_tabs()
            update_status(filename, False, "Engine Failed")
            with out_qa:
                display(HTML(f"<p style='color:red;'><b>Analytical Engine Error:</b> {str(e)}</p>"))

uploader = widgets.FileUpload(accept='.csv', multiple=False)
btn_reload = widgets.Button(description="Reload default reference dataset", button_style='warning', icon='refresh')

def on_upload(change):
    if uploader.value:
        uploaded_filename = list(uploader.value.keys())[0] if isinstance(uploader.value, dict) else uploader.value[0]['name']
        content = uploader.value[uploaded_filename]['content'] if isinstance(uploader.value, dict) else uploader.value[0]['content']
        df = pd.read_csv(io.BytesIO(content))
        process_dataframe(df, uploaded_filename)

def on_reload(b):
    default_file = 'Decision_Matrix_24x5_Eurostat_2022.csv'
    update_status(default_file, False)
    if os.path.exists(default_file):
        df_default = pd.read_csv(default_file)
        process_dataframe(df_default, default_file)
    else:
        reset_runtime_state()
        clear_analysis_tabs()
        msg = "Bundled reference dataset not found. Please upload a valid CSV."
        update_status(default_file, False, msg)
        with out_qa:
            clear_output()
            display(HTML(f"<h4 style='color:red;'>Error</h4><p>{msg}</p>"))

uploader.observe(on_upload, names='value')
btn_reload.on_click(on_reload)

# ====================================================================
# 8. INITIALIZE DASHBOARD
# ====================================================================
tabs = widgets.Tab(children=[
    out_qa, out_preproc, out_deterministic, out_stochastic, out_sensitivity, out_scatter, out_downloads, out_map
])
for i, title in enumerate(['Data & QA', 'Pre-processing', 'Deterministic MCDA', 'Stochastic Robustness', 'Sensitivity', 'Scatter', 'Downloads', 'Map']):
    tabs.set_title(i, title)

display(HTML("<h2>MCDA Interactive Spatial Decision Support System</h2>"))
display(status_label)
display(widgets.HBox([uploader, widgets.Label("  OR  "), btn_reload]))
display(tabs)

# Trigger reload sau taburi goale la inceput
if os.path.exists('Decision_Matrix_24x5_Eurostat_2022.csv'):
    on_reload(None)
else:
    clear_analysis_tabs()
    update_status("None", False, "No bundled dataset found. Please upload a CSV to begin.")