#!/usr/bin/env python3
"""
Aggregate per-contig smiley rates to per-bin for plot_smiley.py.

Usage:
    python aggregate_smiley.py <output_dir> [-o <plots_dir>]

Reads:
    <output_dir>/bins/smiley_plot_rates.tsv  (per-contig rates from AMBER)
    <output_dir>/bins/damage_per_bin.tsv     (per-bin damage summary from AMBER)
    <output_dir>/bins/checkm2/quality_report.tsv  (CheckM2 quality, optional)
    <output_dir>/bins/*.fa                   (bin FASTA files for contig membership)

Writes:
    <output_dir>/bins/smiley_per_bin.tsv     (expected format for plot_smiley.py)
    <output_dir>/bins/damage_params.tsv      (damage + CheckM2 quality, column-renamed)

Bins are sorted: HQ (≥90% comp, <5% cont) > MQ (≥50% comp, <10% cont) > rest,
then within each tier by damage amplitude descending.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def get_contig_to_bin(bins_dir: Path) -> dict:
    """Read bin FASTA files to build contig -> bin_name mapping."""
    contig_to_bin = {}
    for fa in sorted(bins_dir.glob("*.fa")):
        bin_name = fa.stem
        with open(fa) as f:
            for line in f:
                if line.startswith(">"):
                    contig = line[1:].split()[0]
                    contig_to_bin[contig] = bin_name
    return contig_to_bin


def aggregate_rates(rates_df: pd.DataFrame, contig_to_bin: dict,
                    dmg_df: pd.DataFrame = None) -> pd.DataFrame:
    """Aggregate per-contig rates to per-bin per-position DataFrame."""
    rates_df = rates_df.copy()
    rates_df["bin"] = rates_df["contig"].map(contig_to_bin)
    rates_df = rates_df.dropna(subset=["bin"])

    # Build damage param lookup for model curves
    dmg_lookup = {}
    if dmg_df is not None:
        for _, row in dmg_df.iterrows():
            dmg_lookup[row["bin"]] = row.to_dict()

    ct_cols = [c for c in rates_df.columns if c.startswith("ct_5p_")]
    ga_cols = [c for c in rates_df.columns if c.startswith("ga_3p_")]
    n_positions = len(ct_cols)

    rows = []
    for bin_name, group in rates_df.groupby("bin"):
        params = dmg_lookup.get(bin_name, {})
        amp5 = params.get("amplitude_5p", None)
        amp3 = params.get("amplitude_3p", None)
        lam5 = params.get("lambda_5p", None)
        lam3 = params.get("lambda_3p", None)
        for pos in range(n_positions):
            ct_vals = group[ct_cols[pos]].dropna()
            ga_vals = group[ga_cols[pos]].dropna()
            obs_5p = ct_vals.mean() if len(ct_vals) else 0.0
            obs_3p = ga_vals.mean() if len(ga_vals) else 0.0
            row = {"bin": bin_name, "position": pos + 1,
                   "obs_5p": obs_5p, "obs_3p": obs_3p}
            if amp5 is not None and lam5 is not None:
                row["model_5p"] = amp5 * np.exp(-lam5 * (pos + 1))
                row["model_3p"] = amp3 * np.exp(-lam3 * (pos + 1))
            rows.append(row)

    return pd.DataFrame(rows)


def fix_damage_params(dmg_df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to match plot_smiley.py expectations."""
    rename = {
        "bin_name": "bin",
        "p_ancient_mean": "p_ancient",
        "amp5": "amplitude_5p",
        "amp3": "amplitude_3p",
        "lambda5": "lambda_5p",
        "lambda3": "lambda_3p",
    }
    return dmg_df.rename(columns={k: v for k, v in rename.items() if k in dmg_df.columns})


def load_checkm2(bins_dir: Path) -> pd.DataFrame:
    """Load CheckM2 quality_report.tsv and compute quality tier."""
    qpath = bins_dir / "checkm2" / "quality_report.tsv"
    if not qpath.exists():
        return None
    df = pd.read_csv(qpath, sep="\t")
    # Standardize column names (CheckM2 uses 'Name', 'Completeness', 'Contamination')
    col_map = {}
    for col in df.columns:
        lc = col.lower()
        if lc == "name":
            col_map[col] = "bin"
        elif "completeness" in lc:
            col_map[col] = "completeness"
        elif "contamination" in lc:
            col_map[col] = "contamination"
    df = df.rename(columns=col_map)
    if "bin" not in df.columns:
        return None

    def quality_tier(row):
        comp = row.get("completeness", 0)
        cont = row.get("contamination", 100)
        if comp >= 90 and cont < 5:
            return 0  # HQ
        elif comp >= 50 and cont < 10:
            return 1  # MQ
        else:
            return 2  # LQ

    df["quality_tier"] = df.apply(quality_tier, axis=1)
    return df[["bin", "completeness", "contamination", "quality_tier"]]


def sort_bins(dmg_df: pd.DataFrame, checkm_df: pd.DataFrame) -> list:
    """Sort bins: HQ > MQ > LQ, then by damage amplitude descending within tier."""
    merged = dmg_df.copy()
    if checkm_df is not None:
        merged = merged.merge(checkm_df, on="bin", how="left")
        merged["quality_tier"] = merged["quality_tier"].fillna(2).astype(int)
    else:
        merged["quality_tier"] = 2

    amp_col = "amplitude_5p" if "amplitude_5p" in merged.columns else None
    if amp_col:
        merged["amp_mean"] = (merged["amplitude_5p"] + merged.get("amplitude_3p", merged["amplitude_5p"])) / 2
    else:
        merged["amp_mean"] = 0.0

    merged = merged.sort_values(["quality_tier", "amp_mean"], ascending=[True, False])
    return merged["bin"].tolist()


def main():
    parser = argparse.ArgumentParser(description="Aggregate smiley rates per bin.")
    parser.add_argument("output_dir", help="AMBER output directory (containing bins/)")
    parser.add_argument("-o", "--plots-dir", default=None,
                        help="Output directory for plots (default: <output_dir>/plots)")
    args = parser.parse_args()

    out = Path(args.output_dir)
    bins_dir = out / "bins"
    plots_dir = Path(args.plots_dir) if args.plots_dir else out / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Load per-contig rates
    rates_path = bins_dir / "smiley_plot_rates.tsv"
    if not rates_path.exists():
        print(f"ERROR: {rates_path} not found", file=sys.stderr)
        sys.exit(1)
    rates_df = pd.read_csv(rates_path, sep="\t")

    # Load damage params
    dmg_path = bins_dir / "damage_per_bin.tsv"
    if not dmg_path.exists():
        print(f"WARNING: {dmg_path} not found", file=sys.stderr)
        dmg_df = None
    else:
        dmg_df = fix_damage_params(pd.read_csv(dmg_path, sep="\t"))

    # Load CheckM2 quality
    checkm_df = load_checkm2(bins_dir)
    if checkm_df is not None:
        n_hq = (checkm_df["quality_tier"] == 0).sum()
        n_mq = (checkm_df["quality_tier"] == 1).sum()
        print(f"CheckM2: {n_hq} HQ, {n_mq} MQ bins")
        if dmg_df is not None:
            dmg_df = dmg_df.merge(checkm_df, on="bin", how="left")
    else:
        print("WARNING: CheckM2 quality_report.tsv not found — no quality annotations")

    # Write merged damage+quality params
    if dmg_df is not None:
        dmg_out = bins_dir / "damage_params.tsv"
        dmg_df.to_csv(dmg_out, sep="\t", index=False)
        print(f"Wrote {dmg_out}")

    # Build contig→bin map
    contig_to_bin = get_contig_to_bin(bins_dir)
    if not contig_to_bin:
        print("ERROR: no .fa files found in bins dir", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(contig_to_bin)} contigs across {len(set(contig_to_bin.values()))} bins")

    # Aggregate per-bin rates
    smiley_df = aggregate_rates(rates_df, contig_to_bin, dmg_df)

    # Sort bins by quality then damage
    if dmg_df is not None:
        sorted_bins = sort_bins(dmg_df, checkm_df)
    else:
        sorted_bins = sorted(smiley_df["bin"].unique())

    smiley_out = bins_dir / "smiley_per_bin.tsv"
    smiley_df.to_csv(smiley_out, sep="\t", index=False)
    print(f"Wrote {smiley_out} ({len(smiley_df['bin'].unique())} bins)")

    # Call plot_smiley.py with sorted bin order
    script_dir = Path(__file__).parent
    plot_script = script_dir / "plot_smiley.py"
    bins_arg = ",".join(sorted_bins) if sorted_bins else ""
    cmd = f"python {plot_script} {smiley_out} -o {plots_dir} --individual"
    if bins_arg:
        cmd += f" --bins {bins_arg}"
    print(f"Running: {cmd}")
    os.system(cmd)


if __name__ == "__main__":
    main()
