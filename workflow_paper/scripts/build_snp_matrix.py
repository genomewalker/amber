#!/usr/bin/env python3
"""
Build SNP alignment matrix from nucmer show-snps output.
Creates both all-SNP and transversion-only alignments.
"""

import sys
from collections import defaultdict
from pathlib import Path

def parse_snps_file(snps_file):
    """Parse show-snps output, return dict of {pos: (ref_base, query_base)}"""
    snps = {}
    with open(snps_file) as f:
        for line in f:
            if line.startswith('[') or line.startswith('P1') or not line.strip():
                continue
            fields = line.strip().split('\t')
            if len(fields) < 4:
                continue
            try:
                pos = int(fields[0])
                ref_base = fields[1].upper()
                query_base = fields[2].upper()
                if ref_base in 'ACGT' and query_base in 'ACGT':
                    snps[pos] = (ref_base, query_base)
            except (ValueError, IndexError):
                continue
    return snps

def is_transversion(base1, base2):
    """Check if substitution is a transversion (purine <-> pyrimidine)"""
    purines = set('AG')
    pyrimidines = set('CT')
    return (base1 in purines and base2 in pyrimidines) or \
           (base1 in pyrimidines and base2 in purines)

def main():
    # Get inputs from snakemake
    snps_files = snakemake.input.snps
    ref_file = snakemake.input.ref
    mask_file = getattr(snakemake.input, "mask", None)

    output_all = snakemake.output.all_snps
    output_tv = snakemake.output.tv_only
    output_stats = snakemake.output.stats

    taxa = snakemake.params.taxa
    ref_name = snakemake.params.ref_name

    # Parse all SNP files
    all_snps = {}  # {sample: {pos: (ref, query)}}
    for snp_file in snps_files:
        sample = Path(snp_file).stem
        all_snps[sample] = parse_snps_file(snp_file)

    # Find all variable positions across all samples
    all_positions = set()
    for sample_snps in all_snps.values():
        all_positions.update(sample_snps.keys())
    all_positions = sorted(all_positions)

    if mask_file:
        masked = set()
        with open(mask_file) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    masked.add(int(parts[1]))  # 0-based BED start
        all_positions = [p for p in all_positions if p not in masked]

    # Build alignment matrix
    # Reference has the reference base at each position
    alignment_all = defaultdict(list)
    alignment_tv = defaultdict(list)

    tv_count = 0
    ts_count = 0

    for pos in all_positions:
        # Get reference base (should be consistent across all samples)
        ref_base = None
        for sample in all_snps:
            if pos in all_snps[sample]:
                ref_base = all_snps[sample][pos][0]
                break

        if not ref_base:
            continue

        # Determine if this position has any transversions
        has_tv = False
        for sample in taxa:
            if sample in all_snps and pos in all_snps[sample]:
                query_base = all_snps[sample][pos][1]
                if is_transversion(ref_base, query_base):
                    has_tv = True
                    break

        if has_tv:
            tv_count += 1

        # Add to alignments
        for sample in taxa:
            if sample == ref_name:
                base = ref_base
            elif sample in all_snps and pos in all_snps[sample]:
                base = all_snps[sample][pos][1]
            else:
                base = ref_base  # No SNP = same as reference

            alignment_all[sample].append(base)

            if has_tv:
                alignment_tv[sample].append(base)

    ts_count = len(all_positions) - tv_count

    # Write alignments
    with open(output_all, 'w') as f:
        for sample in taxa:
            seq = ''.join(alignment_all.get(sample, []))
            if seq:
                f.write(f">{sample}\n{seq}\n")

    with open(output_tv, 'w') as f:
        for sample in taxa:
            seq = ''.join(alignment_tv.get(sample, []))
            if seq:
                f.write(f">{sample}\n{seq}\n")

    # Write stats
    with open(output_stats, 'w') as f:
        f.write("=== Stage 2 SNP Statistics ===\n\n")
        f.write(f"Taxa analyzed: {len(taxa)}\n")
        f.write(f"Reference: {ref_name}\n")
        f.write(f"\nSNP counts:\n")
        f.write(f"  Total variable positions: {len(all_positions)}\n")
        f.write(f"  Transitions (Ts): {ts_count}\n")
        f.write(f"  Transversions (Tv): {tv_count}\n")
        if tv_count > 0:
            f.write(f"  Ts/Tv ratio: {ts_count/tv_count:.2f}\n")
        f.write(f"\nAlignment lengths:\n")
        f.write(f"  All SNPs: {len(all_positions)} bp\n")
        f.write(f"  TV-only: {tv_count} bp\n")

        f.write(f"\nPer-sample SNP counts:\n")
        for sample in taxa:
            if sample in all_snps:
                f.write(f"  {sample}: {len(all_snps[sample])} SNPs\n")

if __name__ == "__main__":
    main()
