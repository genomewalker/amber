#!/usr/bin/env python3
"""
Plot ancient DNA damage "smiley" plots for AMBER bins.

Generates publication-quality damage pattern visualizations showing:
- C→T substitutions at 5' end (characteristic of aDNA deamination)
- G→A substitutions at 3' end (complementary damage on opposite strand)

Usage:
    # Using smiley_per_bin.tsv (requires --export-smiley during binning)
    python plot_smiley.py <smiley_per_bin.tsv> -o <output_dir>
    python plot_smiley.py <smiley_per_bin.tsv> --bins bin_0,bin_1 -o <output_dir>
    python plot_smiley.py <smiley_per_bin.tsv> --top 10 -o <output_dir>

    # Using bin_stats.tsv directly (summary plots only, no per-position curves)
    python plot_smiley.py --stats <bin_stats.tsv> -o <output_dir>

Damage parameters are auto-detected from bin_stats.tsv in the same directory.
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
plt.rcParams['mathtext.fontset'] = 'dejavusans'

PALETTE = {
    'ct_obs': '#C0392B',
    'ct_model': '#E74C3C',
    'ga_obs': '#2471A3',
    'ga_model': '#5DADE2',
    'grid': '#ECF0F1',
    'spine': '#BDC3C7',
    'text': '#2C3E50',
    'background': '#FFFFFF',
    'ancient': '#27AE60',
    'uncertain': '#F39C12',
    'modern': '#95A5A6',
}


def load_smiley_data(filepath: str) -> pd.DataFrame:
    """Load smiley_per_bin.tsv file."""
    df = pd.read_csv(filepath, sep='\t')
    return df


def load_damage_params(filepath: str) -> pd.DataFrame:
    """Load damage parameters from damage_per_bin.tsv or bin_stats.tsv."""
    if not filepath or not os.path.exists(filepath):
        return None

    df = pd.read_csv(filepath, sep='\t')

    # Check if this is bin_stats.tsv (has p_ancient column with bin_id)
    if 'bin_id' in df.columns and 'p_ancient' in df.columns:
        # Convert bin_stats.tsv format to damage_per_bin format
        cols = ['bin_id', 'p_ancient', 'amplitude_5p', 'lambda_5p',
                'amplitude_3p', 'lambda_3p']
        if 'damage_class' in df.columns:
            cols.append('damage_class')
        damage_df = df[cols].copy()
        damage_df = damage_df.rename(columns={'bin_id': 'bin'})
        return damage_df

    # Check if this is damage_per_bin.tsv format
    if 'bin' in df.columns and 'p_ancient' in df.columns:
        return df

    return None


def load_bin_stats(filepath: str) -> pd.DataFrame:
    """Load bin_stats.tsv for summary plotting without smiley data."""
    if not filepath or not os.path.exists(filepath):
        return None
    df = pd.read_csv(filepath, sep='\t')
    if 'bin_id' not in df.columns or 'p_ancient' not in df.columns:
        return None
    return df


def plot_single_bin(
    bin_data: pd.DataFrame,
    bin_name: str,
    damage_params: dict = None,
    ax: plt.Axes = None,
    show_model: bool = True,
    show_legend: bool = True,
    compact: bool = False,
) -> plt.Axes:
    """Plot smiley plot for a single bin."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    positions = bin_data['position'].values
    obs_5p = bin_data['obs_5p'].values * 100
    obs_3p = bin_data['obs_3p'].values * 100
    model_5p = bin_data['model_5p'].values * 100 if show_model else None
    model_3p = bin_data['model_3p'].values * 100 if show_model else None

    ax.set_facecolor(PALETTE['background'])

    marker_size = 40 if compact else 70
    ax.scatter(positions, obs_5p, c=PALETTE['ct_obs'], s=marker_size, alpha=0.85,
               edgecolors='white', linewidths=0.8, zorder=3, label="C→T (5')")
    ax.scatter(positions, obs_3p, c=PALETTE['ga_obs'], s=marker_size, alpha=0.85,
               edgecolors='white', linewidths=0.8, zorder=3, label="G→A (3')")

    if show_model and model_5p is not None:
        ax.plot(positions, model_5p, c=PALETTE['ct_model'], linewidth=2,
                linestyle='-', alpha=0.7, zorder=2)
        ax.plot(positions, model_3p, c=PALETTE['ga_model'], linewidth=2,
                linestyle='-', alpha=0.7, zorder=2)

    ax.axhline(y=1.0, color=PALETTE['spine'], linestyle=':', linewidth=1, alpha=0.5, zorder=1)

    fontsize = 9 if compact else 11
    ax.set_xlabel('Position', fontsize=fontsize, color=PALETTE['text'])
    ax.set_ylabel('Frequency (%)', fontsize=fontsize, color=PALETTE['text'])

    p_ancient = 0
    amp_5p = 0
    amp_3p = 0
    damage_class = 'uncertain'
    if damage_params:
        p_ancient = damage_params.get('p_ancient', 0)
        amp_5p = damage_params.get('amplitude_5p', 0) * 100
        amp_3p = damage_params.get('amplitude_3p', 0) * 100
        damage_class = damage_params.get('damage_class', 'uncertain')

    status_color = PALETTE.get(damage_class, PALETTE['uncertain'])
    status_text = damage_class.capitalize()

    title_fontsize = 10 if compact else 12
    ax.set_title(bin_name, fontsize=title_fontsize, color=PALETTE['text'],
                 fontweight='bold', pad=8 if compact else 12)

    info_fontsize = 7 if compact else 9
    if damage_params:
        avg_amp = (amp_5p + amp_3p) / 2
        info_text = f"Amp: {avg_amp:.1f}%"
        if p_ancient > 0:
            info_text += f"  |  P={p_ancient:.2f}"

        bbox_props = dict(boxstyle='round,pad=0.3', facecolor=status_color,
                         edgecolor='none', alpha=0.15)
        ax.text(0.98, 0.95, info_text, transform=ax.transAxes, fontsize=info_fontsize,
                ha='right', va='top', color=status_color, fontweight='bold',
                bbox=bbox_props)

    ax.set_xlim(0.5, max(positions) + 0.5)
    y_max = max(max(obs_5p), max(obs_3p)) * 1.2
    ax.set_ylim(-0.3, max(y_max, 3))

    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6 if compact else 10))
    tick_fontsize = 8 if compact else 10
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize, colors=PALETTE['text'])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(PALETTE['spine'])
    ax.spines['bottom'].set_color(PALETTE['spine'])
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)

    ax.grid(True, axis='y', linestyle='-', alpha=0.3, color=PALETTE['grid'], zorder=0)

    if show_legend and not compact:
        legend = ax.legend(loc='upper right', frameon=True, fancybox=True,
                          framealpha=0.95, edgecolor=PALETTE['spine'],
                          fontsize=8, borderpad=0.6, handletextpad=0.4)
        legend.get_frame().set_facecolor('white')

    return ax


def plot_grid(
    smiley_df: pd.DataFrame,
    damage_df: pd.DataFrame,
    bins: list,
    output_path: str,
    ncols: int = 5,
    show_model: bool = True,
):
    """Plot grid of smiley plots for multiple bins."""
    n_bins = len(bins)
    nrows = (n_bins + ncols - 1) // ncols

    fig_width = 3.2 * ncols
    fig_height = 2.8 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))

    if n_bins == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    damage_params_dict = {}
    if damage_df is not None:
        for _, row in damage_df.iterrows():
            damage_params_dict[row['bin']] = row.to_dict()

    for idx, bin_name in enumerate(bins):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]

        bin_data = smiley_df[smiley_df['bin'] == bin_name]
        if bin_data.empty:
            ax.set_visible(False)
            continue

        damage_params = damage_params_dict.get(bin_name, None)
        plot_single_bin(bin_data, bin_name, damage_params, ax,
                       show_model=show_model, show_legend=False, compact=True)

    for idx in range(n_bins, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].set_visible(False)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=PALETTE['ct_obs'],
               markersize=8, label="C→T (5' end)"),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=PALETTE['ga_obs'],
               markersize=8, label="G→A (3' end)"),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2,
              frameon=True, fancybox=True, fontsize=10, bbox_to_anchor=(0.5, 1.0))

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white',
                edgecolor='none', pad_inches=0.2)
    plt.close(fig)
    print(f"Saved grid plot: {output_path}")


def plot_summary(
    smiley_df: pd.DataFrame,
    damage_df: pd.DataFrame,
    output_path: str,
):
    """Plot summary figure showing damage distribution across all bins."""
    bins = smiley_df['bin'].unique()

    damage_params_dict = {}
    if damage_df is not None:
        for _, row in damage_df.iterrows():
            damage_params_dict[row['bin']] = row.to_dict()

    bin_stats = []
    for bin_name in bins:
        bin_data = smiley_df[smiley_df['bin'] == bin_name]
        if bin_data.empty:
            continue

        pos1_5p = bin_data[bin_data['position'] == 1]['obs_5p'].values
        pos1_3p = bin_data[bin_data['position'] == 1]['obs_3p'].values

        d1_5p = pos1_5p[0] * 100 if len(pos1_5p) > 0 else 0
        d1_3p = pos1_3p[0] * 100 if len(pos1_3p) > 0 else 0

        params = damage_params_dict.get(bin_name, {})
        p_ancient = params.get('p_ancient', 0)
        amp_5p = params.get('amplitude_5p', 0) * 100
        amp_3p = params.get('amplitude_3p', 0) * 100
        damage_class = params.get('damage_class', 'uncertain')

        bin_stats.append({
            'bin': bin_name,
            'd1_5p': d1_5p,
            'd1_3p': d1_3p,
            'd1_mean': (d1_5p + d1_3p) / 2,
            'amp_mean': (amp_5p + amp_3p) / 2,
            'p_ancient': p_ancient,
            'damage_class': damage_class,
        })

    if not bin_stats:
        print("No bins with data found")
        return

    stats_df = pd.DataFrame(bin_stats).sort_values('amp_mean', ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor('white')

    ax1 = axes[0]
    x = np.arange(len(stats_df))

    colors = [PALETTE.get(c, PALETTE['uncertain']) for c in stats_df['damage_class']]

    bars = ax1.bar(x, stats_df['amp_mean'], color=colors, alpha=0.85,
                   edgecolor='white', linewidth=0.8)

    ax1.axhline(y=5, color=PALETTE['spine'], linestyle='--', linewidth=1, alpha=0.5)
    ax1.text(len(x)-1, 5.5, '5% threshold', fontsize=8, ha='right', color=PALETTE['spine'])

    ax1.set_xlabel('Bin (sorted by amplitude)', fontsize=11, color=PALETTE['text'])
    ax1.set_ylabel('Damage Amplitude (%)', fontsize=11, color=PALETTE['text'])
    ax1.set_title('Terminal Damage Amplitude by Bin', fontsize=13, color=PALETTE['text'], fontweight='bold')

    if len(stats_df) <= 15:
        ax1.set_xticks(x)
        ax1.set_xticklabels(stats_df['bin'], rotation=45, ha='right', fontsize=8)
    else:
        step = max(1, len(x) // 8)
        ax1.set_xticks(x[::step])
        ax1.set_xticklabels(stats_df['bin'].values[::step], rotation=45, ha='right', fontsize=8)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=PALETTE['ancient'], edgecolor='white', label='Ancient'),
        Patch(facecolor=PALETTE['uncertain'], edgecolor='white', label='Uncertain'),
        Patch(facecolor=PALETTE['modern'], edgecolor='white', label='Modern'),
    ]
    ax1.legend(handles=legend_elements, loc='upper right', frameon=True, framealpha=0.95, fontsize=9)

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_linewidth(1.2)
    ax1.spines['bottom'].set_linewidth(1.2)
    ax1.grid(True, axis='y', linestyle='-', alpha=0.3)
    ax1.set_facecolor(PALETTE['background'])

    ax2 = axes[1]

    scatter = ax2.scatter(stats_df['d1_5p'], stats_df['d1_3p'],
                         c=stats_df['p_ancient'], cmap='RdYlGn',
                         s=100, alpha=0.85, edgecolors='white', linewidths=1,
                         vmin=0, vmax=1)

    max_val = max(stats_df['d1_5p'].max(), stats_df['d1_3p'].max()) * 1.1
    ax2.plot([0, max_val], [0, max_val], color=PALETTE['spine'], linestyle='--',
             linewidth=1.5, alpha=0.6, label='1:1 line')

    ax2.set_xlabel("C→T at position 1 (%)", fontsize=11, color=PALETTE['text'])
    ax2.set_ylabel("G→A at position 1 (%)", fontsize=11, color=PALETTE['text'])
    ax2.set_title("5' vs 3' Terminal Damage", fontsize=13, color=PALETTE['text'], fontweight='bold')

    cbar = plt.colorbar(scatter, ax=ax2, shrink=0.85, pad=0.02)
    cbar.set_label('P(ancient)', fontsize=10, color=PALETTE['text'])
    cbar.ax.tick_params(labelsize=9)

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_linewidth(1.2)
    ax2.spines['bottom'].set_linewidth(1.2)
    ax2.grid(True, linestyle='-', alpha=0.3)
    ax2.set_facecolor(PALETTE['background'])
    ax2.set_xlim(-0.5, max_val)
    ax2.set_ylim(-0.5, max_val)
    ax2.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white',
                edgecolor='none', pad_inches=0.2)
    plt.close(fig)
    print(f"Saved summary plot: {output_path}")


def plot_summary_from_stats(
    stats_df: pd.DataFrame,
    output_path: str,
):
    """Plot summary figure directly from bin_stats.tsv without smiley data."""
    from matplotlib.patches import Patch

    # Use amplitude values from bin_stats.tsv
    plot_data = []
    for _, row in stats_df.iterrows():
        bin_name = row['bin_id']
        p_ancient = row.get('p_ancient', 0)
        amp_5p = row.get('amplitude_5p', 0) * 100
        amp_3p = row.get('amplitude_3p', 0) * 100
        est_comp = row.get('est_completeness', 0)
        total_bp = row.get('total_bp', 0)
        damage_class = row.get('damage_class', 'uncertain')

        plot_data.append({
            'bin': bin_name,
            'amp_5p': amp_5p,
            'amp_3p': amp_3p,
            'amp_mean': (amp_5p + amp_3p) / 2,
            'p_ancient': p_ancient,
            'damage_class': damage_class,
            'est_completeness': est_comp,
            'total_bp': total_bp,
        })

    if not plot_data:
        print("No bins with data found")
        return

    df = pd.DataFrame(plot_data).sort_values('amp_mean', ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor('white')

    # Left plot: amplitude bar chart
    ax1 = axes[0]
    x = np.arange(len(df))

    colors = [PALETTE.get(c, PALETTE['uncertain']) for c in df['damage_class']]

    ax1.bar(x, df['amp_mean'], color=colors, alpha=0.85,
            edgecolor='white', linewidth=0.8)

    ax1.axhline(y=5, color=PALETTE['spine'], linestyle='--', linewidth=1, alpha=0.5)
    ax1.text(len(x)-1, 5.5, '5% threshold', fontsize=8, ha='right', color=PALETTE['spine'])

    ax1.set_xlabel('Bin (sorted by amplitude)', fontsize=11, color=PALETTE['text'])
    ax1.set_ylabel('Damage Amplitude (%)', fontsize=11, color=PALETTE['text'])
    ax1.set_title('Terminal Damage Amplitude by Bin', fontsize=13,
                  color=PALETTE['text'], fontweight='bold')

    if len(df) <= 15:
        ax1.set_xticks(x)
        ax1.set_xticklabels(df['bin'], rotation=45, ha='right', fontsize=8)
    else:
        step = max(1, len(x) // 8)
        ax1.set_xticks(x[::step])
        ax1.set_xticklabels(df['bin'].values[::step], rotation=45, ha='right', fontsize=8)

    legend_elements = [
        Patch(facecolor=PALETTE['ancient'], edgecolor='white', label='Ancient'),
        Patch(facecolor=PALETTE['uncertain'], edgecolor='white', label='Uncertain'),
        Patch(facecolor=PALETTE['modern'], edgecolor='white', label='Modern'),
    ]
    ax1.legend(handles=legend_elements, loc='upper right', frameon=True,
               framealpha=0.95, fontsize=9)

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_linewidth(1.2)
    ax1.spines['bottom'].set_linewidth(1.2)
    ax1.grid(True, axis='y', linestyle='-', alpha=0.3)
    ax1.set_facecolor(PALETTE['background'])

    # Right plot: 5' vs 3' amplitude scatter
    ax2 = axes[1]

    scatter = ax2.scatter(df['amp_5p'], df['amp_3p'],
                         c=df['p_ancient'], cmap='RdYlGn',
                         s=100, alpha=0.85, edgecolors='white', linewidths=1,
                         vmin=0, vmax=1)

    max_val = max(df['amp_5p'].max(), df['amp_3p'].max()) * 1.1
    if max_val > 0:
        ax2.plot([0, max_val], [0, max_val], color=PALETTE['spine'], linestyle='--',
                 linewidth=1.5, alpha=0.6, label='1:1 line')

    ax2.set_xlabel("5' Amplitude (%)", fontsize=11, color=PALETTE['text'])
    ax2.set_ylabel("3' Amplitude (%)", fontsize=11, color=PALETTE['text'])
    ax2.set_title("5' vs 3' Terminal Damage Amplitude", fontsize=13,
                  color=PALETTE['text'], fontweight='bold')

    cbar = plt.colorbar(scatter, ax=ax2, shrink=0.85, pad=0.02)
    cbar.set_label('P(ancient)', fontsize=10, color=PALETTE['text'])
    cbar.ax.tick_params(labelsize=9)

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_linewidth(1.2)
    ax2.spines['bottom'].set_linewidth(1.2)
    ax2.grid(True, linestyle='-', alpha=0.3)
    ax2.set_facecolor(PALETTE['background'])
    ax2.set_xlim(-0.5, max(max_val, 1))
    ax2.set_ylim(-0.5, max(max_val, 1))
    ax2.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white',
                edgecolor='none', pad_inches=0.2)
    plt.close(fig)
    print(f"Saved summary plot: {output_path}")

    # Print summary statistics
    n_ancient = sum(1 for c in df['damage_class'] if c == 'ancient')
    n_uncertain = sum(1 for c in df['damage_class'] if c == 'uncertain')
    n_modern = sum(1 for c in df['damage_class'] if c == 'modern')
    print(f"\nDamage Summary:")
    print(f"  Ancient bins: {n_ancient}")
    print(f"  Uncertain bins: {n_uncertain}")
    print(f"  Modern bins: {n_modern}")
    if n_ancient > 0:
        ancient_df = df[df['damage_class'] == 'ancient']
        print(f"  Mean ancient amplitude: {ancient_df['amp_mean'].mean():.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description='Plot ancient DNA damage smiley plots for AMBER bins',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('smiley_file', nargs='?', default=None,
                       help='Path to smiley_per_bin.tsv (optional if --stats provided)')
    parser.add_argument('--stats', type=str, default=None,
                       help='Path to bin_stats.tsv (summary plots only, no per-position curves)')
    parser.add_argument('-o', '--output', required=True,
                       help='Output directory for plots')
    parser.add_argument('--bins', type=str, default=None,
                       help='Comma-separated list of bins to plot (e.g., bin_0,bin_1)')
    parser.add_argument('--top', type=int, default=None,
                       help='Plot top N bins by damage level')
    parser.add_argument('--no-model', action='store_true',
                       help='Do not show model fit curves')
    parser.add_argument('--grid-cols', type=int, default=4,
                       help='Number of columns in grid layout (default: 4)')
    parser.add_argument('--individual', action='store_true',
                       help='Save individual plots for each bin')

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Mode 1: Use bin_stats.tsv directly (summary only, no smiley curves)
    if args.stats:
        print(f"Loading bin statistics from {args.stats}...")
        stats_df = load_bin_stats(args.stats)
        if stats_df is None:
            print("Error: Could not load bin_stats.tsv or missing required columns")
            sys.exit(1)

        print(f"Found {len(stats_df)} bins in data")
        summary_path = os.path.join(args.output, 'damage_summary.png')
        plot_summary_from_stats(stats_df, summary_path)
        print("Done!")
        return

    # Mode 2: Use smiley_per_bin.tsv with optional damage parameters
    if not args.smiley_file:
        print("Error: Either smiley_file or --stats is required")
        parser.print_help()
        sys.exit(1)

    print(f"Loading smiley data from {args.smiley_file}...")
    smiley_df = load_smiley_data(args.smiley_file)

    # Auto-detect damage parameters from bin_stats.tsv or damage_per_bin.tsv
    damage_df = None
    parent_dir = Path(args.smiley_file).parent

    # Try bin_stats.tsv first (new format with integrated damage params)
    bin_stats_path = str(parent_dir / 'bin_stats.tsv')
    if os.path.exists(bin_stats_path):
        print(f"Auto-detected damage parameters: {bin_stats_path}")
        damage_df = load_damage_params(bin_stats_path)

    # Fall back to damage_per_bin.tsv (legacy format)
    if damage_df is None:
        damage_file_path = str(parent_dir / 'damage_per_bin.tsv')
        if os.path.exists(damage_file_path):
            print(f"Auto-detected damage parameters: {damage_file_path}")
            damage_df = load_damage_params(damage_file_path)

    all_bins = sorted(smiley_df['bin'].unique())
    print(f"Found {len(all_bins)} bins in data")

    if args.bins:
        bins = [b.strip() for b in args.bins.split(',')]
        bins = [b for b in bins if b in all_bins]
    elif args.top:
        pos1_damage = []
        for bin_name in all_bins:
            bin_data = smiley_df[(smiley_df['bin'] == bin_name) & (smiley_df['position'] == 1)]
            if not bin_data.empty:
                d = (bin_data['obs_5p'].values[0] + bin_data['obs_3p'].values[0]) / 2
                pos1_damage.append((bin_name, d))
        pos1_damage.sort(key=lambda x: x[1], reverse=True)
        bins = [b[0] for b in pos1_damage[:args.top]]
    else:
        bins = all_bins

    print(f"Plotting {len(bins)} bins...")

    summary_path = os.path.join(args.output, 'damage_summary.png')
    plot_summary(smiley_df, damage_df, summary_path)

    if len(bins) > 1:
        grid_path = os.path.join(args.output, 'smiley_grid.png')
        plot_grid(smiley_df, damage_df, bins, grid_path,
                 ncols=args.grid_cols, show_model=not args.no_model)

    if args.individual or len(bins) <= 5:
        indiv_dir = os.path.join(args.output, 'individual')
        os.makedirs(indiv_dir, exist_ok=True)

        damage_params_dict = {}
        if damage_df is not None:
            for _, row in damage_df.iterrows():
                damage_params_dict[row['bin']] = row.to_dict()

        for bin_name in bins:
            bin_data = smiley_df[smiley_df['bin'] == bin_name]
            if bin_data.empty:
                continue

            fig, ax = plt.subplots(figsize=(8, 5))
            damage_params = damage_params_dict.get(bin_name, None)
            plot_single_bin(bin_data, bin_name, damage_params, ax,
                           show_model=not args.no_model)

            out_path = os.path.join(indiv_dir, f'{bin_name}_smiley.png')
            fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)

        print(f"Saved {len(bins)} individual plots to {indiv_dir}/")

    print("Done!")


if __name__ == '__main__':
    main()
