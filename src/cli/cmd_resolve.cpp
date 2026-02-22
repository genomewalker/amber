// amber resolve — cross-run probabilistic bin re-clustering
//
// Loads N run.abin files from independent amber bin runs. Builds a co-binning
// affinity graph: edge weight = P_cobin(i,j) × exp(-||E_avg[i]-E_avg[j]|| / bw)
// where P_cobin = fraction of runs that placed contigs i and j in the same bin,
// and E_avg = mean embedding across all runs. Runs Leiden with multiple restarts
// on this stable graph, eliminating encoder stochasticity from the final result.
//
// Usage:
//   amber resolve --runs run1.abin run2.abin run3.abin \
//                 --contigs contigs.fa --output resolved/

#include <amber/config.hpp>
#include "bin2/io/abin.h"
#include "bin2/clustering/clustering_types.h"
#include "bin2/clustering/leiden_backend.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace amber {

// ── FASTA loader ─────────────────────────────────────────────────────────────

static std::vector<std::pair<std::string,std::string>> load_fasta(const std::string& path) {
    std::vector<std::pair<std::string,std::string>> contigs;
    std::ifstream f(path);
    if (!f) throw std::runtime_error("resolve: cannot open contigs: " + path);
    std::string line, name, seq;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        if (line[0] == '>') {
            if (!name.empty()) contigs.emplace_back(name, seq);
            name = line.substr(1, line.find(' ') - 1);
            seq.clear();
        } else {
            seq += line;
        }
    }
    if (!name.empty()) contigs.emplace_back(name, seq);
    return contigs;
}

// ── main logic ────────────────────────────────────────────────────────────────

// ── SCG quality score for a labeling ─────────────────────────────────────────
// Computes an estimate of bin quality from SCG marker data in the .abin.
// Calibrated thresholds: internal SCG colocation underestimates CheckM2 by ~10-15pp
// due to aDNA fragmentation; 75% internal ≈ 90% CheckM2 HQ threshold.
// Score ordering: HQ > MQ > near-HQ > soft (identical to QualityLeidenBackend).
static float resolve_scg_score(
    const std::vector<int>& labels, int N,
    const std::vector<std::vector<int>>& scg,   // scg_markers from run[0]
    const std::vector<int>& union_to_run0,
    uint32_t total_markers)
{
    if (total_markers == 0 || scg.empty()) return 0.0f;

    std::unordered_map<int, std::vector<int>> clusters;
    for (int i = 0; i < N; i++)
        if (labels[i] >= 0) clusters[labels[i]].push_back(i);

    int   num_hq = 0, num_mq = 0, num_nhq = 0;
    float soft   = 0.0f;
    for (auto& [lbl, members] : clusters) {
        if (members.size() < 5) continue;
        std::unordered_map<int,int> mc;
        for (int ci : members) {
            int r0 = union_to_run0[ci];
            if (r0 < 0 || (size_t)r0 >= scg.size()) continue;
            for (int mid : scg[r0]) mc[mid]++;
        }
        int unique = (int)mc.size(), total_copies = 0;
        for (auto& [m,c] : mc) total_copies += c;
        float comp = (float)unique / total_markers;
        float cont = (float)(total_copies - unique) / total_markers;
        if (comp >= 0.75f && cont <= 0.05f) num_hq++;
        else if (comp >= 0.50f && cont <= 0.10f) num_mq++;
        else if (comp >= 0.30f && cont <= 0.05f) num_nhq++;
        soft += comp - 5.0f * cont;
    }
    return 1e6f * num_hq + 1e4f * num_mq + 1e2f * num_nhq + soft / 1e4f;
}

int cmd_resolve(int argc, char** argv) {
    std::vector<std::string> run_paths;
    std::string contigs_path, output_dir;
    float bandwidth      = 0.2f;
    float min_cobin_frac = 2.0f / 3.0f;  // majority vote: ≥2/3 of runs must agree
    int   n_restarts     = 25;            // total Leiden budget (spread across resolutions)
    int   base_seed      = 42;
    bool  help           = false;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--runs") {
            while (i + 1 < argc && argv[i+1][0] != '-') run_paths.push_back(argv[++i]);
        } else if (a == "--contigs"        && i+1 < argc) { contigs_path    = argv[++i]; }
        else if (a == "--output"           && i+1 < argc) { output_dir      = argv[++i]; }
        else if (a == "--bandwidth"        && i+1 < argc) { bandwidth       = std::stof(argv[++i]); }
        else if (a == "--min-cobin-frac"   && i+1 < argc) { min_cobin_frac  = std::stof(argv[++i]); }
        else if (a == "--restarts"         && i+1 < argc) { n_restarts      = std::stoi(argv[++i]); }
        else if (a == "--random-seed"      && i+1 < argc) { base_seed       = std::stoi(argv[++i]); }
        else if (a == "-h" || a == "--help") { help = true; }
    }

    if (help || run_paths.size() < 2 || contigs_path.empty() || output_dir.empty()) {
        std::cerr << "Usage: amber resolve --runs run1.abin run2.abin [run3.abin ...]\n"
                  << "                     --contigs contigs.fa --output out/\n\n"
                  << "Options:\n"
                  << "  --bandwidth       F   embedding similarity bandwidth (default 0.2)\n"
                  << "  --min-cobin-frac  F   min fraction of runs that must co-bin a pair (default 0.667)\n"
                  << "  --restarts        N   total Leiden budget spread across resolution sweep (default 25)\n"
                  << "  --random-seed     N   base seed for Leiden restarts (default 42)\n";
        return help ? 0 : 1;
    }

    std::filesystem::create_directories(output_dir);
    int R = static_cast<int>(run_paths.size());

    // ── 1. Load all .abin files ───────────────────────────────────────────────

    std::cerr << "[resolve] Loading " << R << " runs...\n";
    std::vector<AbinData> runs;
    runs.reserve(R);
    for (const auto& p : run_paths) {
        std::cerr << "  " << p << "\n";
        runs.push_back(abin_load(p));
    }

    // Build union namespace from ALL runs (order-independent, no contigs dropped)
    std::unordered_map<std::string,int> name_to_idx;
    for (int r = 0; r < R; r++)
        for (const auto& n : runs[r].names)
            name_to_idx.emplace(n, (int)name_to_idx.size());
    int N = static_cast<int>(name_to_idx.size());

    // Canonical name list in insertion order
    std::vector<std::string> all_names(N);
    for (const auto& [n, idx] : name_to_idx) all_names[idx] = n;

    // remap[r][abin_ci] = shared_ci (-1 if not found)
    std::vector<std::vector<int>> remap(R);
    for (int r = 0; r < R; r++) {
        remap[r].resize(runs[r].n_contigs, -1);
        for (uint32_t i = 0; i < runs[r].n_contigs; i++) {
            auto it = name_to_idx.find(runs[r].names[i]);
            if (it != name_to_idx.end()) remap[r][i] = it->second;
        }
    }

    // union_ci → run[0] local index (-1 if not in run[0]), for SCG lookup
    std::vector<int> union_to_run0(N, -1);
    for (uint32_t i = 0; i < runs[0].n_contigs; i++)
        union_to_run0[remap[0][i]] = static_cast<int>(i);

    // ── 2. Average embeddings across runs ─────────────────────────────────────

    int dim = runs[0].emb_dim;
    std::cerr << "[resolve] Averaging " << dim << "-dim embeddings across " << R << " runs...\n";
    std::vector<float> E_avg(N * dim, 0.0f);
    std::vector<int>   E_count(N, 0);

    for (int r = 0; r < R; r++) {
        if (runs[r].embeddings.empty()) continue;
        int rdim = runs[r].emb_dim;
        for (uint32_t ai = 0; ai < runs[r].n_contigs; ai++) {
            int si = remap[r][ai];
            if (si < 0) continue;
            const float* src = runs[r].embeddings.data() + ai * rdim;
            float* dst = E_avg.data() + si * dim;
            for (int d = 0; d < std::min(dim, rdim); d++) dst[d] += src[d];
            E_count[si]++;
        }
    }
    for (int i = 0; i < N; i++) {
        if (E_count[i] > 1) {
            float inv = 1.0f / E_count[i];
            float* e = E_avg.data() + i * dim;
            for (int d = 0; d < dim; d++) e[d] *= inv;
        }
    }

    // ── 3. Build co-binning count map ─────────────────────────────────────────

    std::cerr << "[resolve] Building cross-run co-binning graph...\n";
    // Key: pack pair (i,j) with i<j into uint64; value: run co-binning count
    std::unordered_map<uint64_t, uint32_t> cobin_count;
    cobin_count.reserve(1 << 22);  // pre-allocate ~4M buckets

    for (int r = 0; r < R; r++) {
        if (runs[r].bin_labels.empty()) continue;
        // Group abin-local contig indices by bin label
        std::unordered_map<int,std::vector<int>> bins;
        for (uint32_t ai = 0; ai < runs[r].n_contigs; ai++) {
            int lbl = runs[r].bin_labels[ai];
            if (lbl >= 0) bins[lbl].push_back(ai);
        }
        for (auto& [lbl, members] : bins) {
            // Map to shared indices, drop unmapped
            std::vector<int> shared;
            shared.reserve(members.size());
            for (int ai : members) {
                int si = remap[r][ai];
                if (si >= 0) shared.push_back(si);
            }
            std::sort(shared.begin(), shared.end());
            // Accumulate all pairs (i<j)
            for (size_t a = 0; a < shared.size(); a++) {
                for (size_t b = a + 1; b < shared.size(); b++) {
                    uint64_t key = (uint64_t(shared[a]) << 32) | uint64_t(shared[b]);
                    cobin_count[key]++;
                }
            }
        }
    }
    std::cerr << "[resolve] " << cobin_count.size() << " unique co-binned pairs\n";

    // ── 4. Build affinity edge list ───────────────────────────────────────────
    // Only keep pairs with P_cobin ≥ min_cobin_frac (majority vote).
    // w(i,j) = P_cobin(i,j) × exp(-||E_avg[i] - E_avg[j]|| / bw)
    // P_cobin denominator = min(runs_with_i, runs_with_j), not global R,
    // so partially-observed contigs are not systematically under-counted.

    std::vector<bin2::WeightedEdge> edges;
    edges.reserve(cobin_count.size() / 4);  // expect significant filtering

    size_t n_below_threshold = 0;
    for (const auto& [key, cnt] : cobin_count) {
        int i = static_cast<int>(key >> 32);
        int j = static_cast<int>(key & 0xFFFFFFFF);
        // Denominator: runs that observed BOTH contigs (max possible co-binnings)
        int denom = std::min(E_count[i], E_count[j]);
        if (denom == 0) continue;
        float p_cobin = static_cast<float>(cnt) / denom;

        // Majority-vote filter: skip pairs not agreed on by enough runs
        if (p_cobin < min_cobin_frac) { ++n_below_threshold; continue; }

        // L2 distance between averaged embeddings
        float dist = 0.0f;
        const float* ei = E_avg.data() + i * dim;
        const float* ej = E_avg.data() + j * dim;
        for (int d = 0; d < dim; d++) { float df = ei[d] - ej[d]; dist += df * df; }
        dist = std::sqrt(dist);

        float w = p_cobin * std::exp(-dist / bandwidth);
        if (w > 1e-6f) edges.emplace_back(i, j, w);
    }
    std::cerr << "[resolve] " << n_below_threshold << " pairs below min-cobin-frac=" << min_cobin_frac << " filtered\n";
    std::cerr << "[resolve] " << edges.size() << " affinity edges (bw=" << bandwidth << ")\n";

    // ── 5. SCG edge adjustments (from run[0] markers) ────────────────────────

    const auto& scg = runs[0].scg_markers;
    if (!scg.empty()) {
        constexpr float kComplementBonus = 0.3f;
        constexpr float kSharedPenalty   = 3.0f;
        for (auto& e : edges) {
            int r0u = union_to_run0[e.u];
            int r0v = union_to_run0[e.v];
            if (r0u < 0 || r0v < 0) continue;
            if ((size_t)r0u >= scg.size() || (size_t)r0v >= scg.size()) continue;
            const auto& mu = scg[r0u];
            const auto& mv = scg[r0v];
            if (mu.empty() || mv.empty()) continue;
            int shared = 0;
            size_t a = 0, b = 0;
            while (a < mu.size() && b < mv.size()) {
                if      (mu[a] == mv[b]) { ++shared; ++a; ++b; }
                else if (mu[a]  < mv[b]) ++a;
                else                      ++b;
            }
            if (shared == 0) e.w *= (1.0f + kComplementBonus);
            else             e.w *= std::exp(-kSharedPenalty * shared);
        }
        std::cerr << "[resolve] SCG edge adjustments applied\n";
    }

    // ── 6. Leiden resolution sweep, scored by SCG quality (PARALLEL) ─────────
    // Modularity is not comparable across resolutions, so we use an inline SCG
    // quality estimate (calibrated to CheckM2: 75%/5% internal ≈ 90%/5% CheckM2).
    // Budget: n_restarts spread evenly across 5 candidate resolutions.
    // All (res, seed) combinations run in parallel with OpenMP.

    const std::vector<float> res_candidates = {0.1f, 0.5f, 1.0f, 2.0f, 5.0f};
    int n_res = static_cast<int>(res_candidates.size());
    int seeds_per_res = std::max(1, n_restarts / n_res);
    int total_runs = n_res * seeds_per_res;

    std::cerr << "[resolve] Running Leiden sweep (" << n_res
              << " resolutions × " << seeds_per_res << " seeds = "
              << total_runs << " total, parallel)...\n";

    // Storage for all results (one per (res, seed) combination)
    struct RunResult {
        bin2::ClusteringResult result;
        float scg_score = -1e30f;
        float resolution = 0.0f;
    };
    std::vector<RunResult> all_results(total_runs);

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < total_runs; i++) {
        int res_idx = i / seeds_per_res;
        int seed_offset = i % seeds_per_res;
        float res = res_candidates[res_idx];

        // Each thread creates its own Leiden backend (thread-safe with TLS)
        auto leiden = amber::bin2::create_leiden_backend();
        bin2::LeidenConfig lcfg;
        lcfg.resolution = res;
        lcfg.random_seed = base_seed + i;

        auto result = leiden->cluster(edges, N, lcfg);
        float scg = resolve_scg_score(result.labels, N,
                                      runs[0].scg_markers, union_to_run0,
                                      runs[0].total_markers);

        all_results[i].result = std::move(result);
        all_results[i].scg_score = scg;
        all_results[i].resolution = res;
    }

    // Find best result per resolution, then overall best
    bin2::ClusteringResult best;
    float best_scg = -1e30f;
    float best_res = -1.0f;

    for (int r = 0; r < n_res; r++) {
        float res = res_candidates[r];
        float res_best_scg = -1e30f;
        int res_best_idx = -1;
        for (int s = 0; s < seeds_per_res; s++) {
            int idx = r * seeds_per_res + s;
            if (all_results[idx].scg_score > res_best_scg) {
                res_best_scg = all_results[idx].scg_score;
                res_best_idx = idx;
            }
        }
        std::cerr << "[resolve]   res=" << res << "  clusters="
                  << all_results[res_best_idx].result.num_clusters
                  << "  scg_score=" << res_best_scg << "\n";
        if (res_best_scg > best_scg) {
            best_scg = res_best_scg;
            best_res = res;
            best = std::move(all_results[res_best_idx].result);
        }
    }
    std::cerr << "[resolve] Best: res=" << best_res << " clusters=" << best.num_clusters
              << " scg_score=" << best_scg << "\n";

    // ── 7. Write output ───────────────────────────────────────────────────────

    auto contigs = load_fasta(contigs_path);
    std::unordered_map<std::string,int> fa_idx;
    fa_idx.reserve(contigs.size());
    for (int i = 0; i < (int)contigs.size(); i++)
        fa_idx[contigs[i].first] = i;

    // Map shared_ci → fasta index
    std::vector<int> shared_to_fa(N, -1);
    for (int i = 0; i < N; i++) {
        auto it = fa_idx.find(all_names[i]);
        if (it != fa_idx.end()) shared_to_fa[i] = it->second;
    }

    // Group shared contigs by cluster label
    std::unordered_map<int,std::vector<int>> cluster_members;
    for (int i = 0; i < N; i++) {
        int lbl = best.labels[i];
        if (lbl >= 0) cluster_members[lbl].push_back(i);
    }

    // Score each cluster by SCG completeness for stats output
    uint32_t total_markers = runs[0].total_markers;
    int bin_num = 0;
    std::ofstream stats(output_dir + "/resolve_stats.tsv");
    stats << "bin\tcluster_id\tn_contigs\tcompleteness\tcontamination\n";

    for (auto& [lbl, members] : cluster_members) {
        if (members.size() < 5) continue;

        // SCG completeness (index via union_to_run0)
        float comp = 0.0f, cont = 0.0f;
        if (total_markers > 0 && !scg.empty()) {
            std::unordered_map<int,int> marker_copies;
            for (int ci : members) {
                int r0ci = union_to_run0[ci];
                if (r0ci < 0 || (size_t)r0ci >= scg.size()) continue;
                for (int mid : scg[r0ci])
                    marker_copies[mid]++;
            }
            int unique = static_cast<int>(marker_copies.size());
            int total_copies = 0;
            for (auto& [m,c] : marker_copies) total_copies += c;
            comp = static_cast<float>(unique) / total_markers;
            cont = static_cast<float>(total_copies - unique) / total_markers;
        }

        bin_num++;
        std::string name = "bin_" + std::to_string(bin_num);
        std::ofstream out(output_dir + "/" + name + ".fa");
        for (int ci : members) {
            int fi = shared_to_fa[ci];
            if (fi < 0) continue;
            out << ">" << contigs[fi].first << "\n" << contigs[fi].second << "\n";
        }
        stats << name << "\t" << lbl << "\t" << members.size()
              << "\t" << comp << "\t" << cont << "\n";
    }

    std::cerr << "[resolve] Wrote " << bin_num << " bins to " << output_dir << "\n";

    // ── 8. Export damage stats per bin ───────────────────────────────────────────
    // Aggregate damage from abin files for smiley plot generation.
    // Damage layout (63-dim): p_anc, var_anc, n_eff, cov_anc, cov_mod, log_ratio,
    //   CT_rate[25], GA_rate[25], lambda5, lambda3, amp5, amp3, frag_mean, frag_std, frag_med

    constexpr int kDmgPAnc      = 0;
    constexpr int kDmgNEff      = 2;
    constexpr int kDmgCtStart   = 6;    // CT rates at positions 1-25 (5' end)
    constexpr int kDmgGaStart   = 31;   // GA rates at positions 1-25 (3' end)
    constexpr int kDmgLambda5   = 56;
    constexpr int kDmgLambda3   = 57;
    constexpr int kDmgAmp5      = 58;
    constexpr int kDmgAmp3      = 59;
    constexpr int kDmgFragMean  = 60;

    // Check if damage data available
    bool has_damage = false;
    int dmg_dim = 0;
    for (int r = 0; r < R; r++) {
        if (!runs[r].damage.empty() && runs[r].dmg_dim >= 63) {
            has_damage = true;
            dmg_dim = runs[r].dmg_dim;
            break;
        }
    }

    if (has_damage) {
        std::ofstream dmg_out(output_dir + "/damage_per_bin.tsv");
        dmg_out << "bin\tp_ancient\tlambda5\tlambda3\tamplitude_5p\tamplitude_3p\t"
                << "ct_1p\tga_1p\tfrag_mean\tdamage_class\n";

        std::ofstream rate_out(output_dir + "/smiley_plot_rates.tsv");
        rate_out << "bin\tposition\tct_rate\tga_rate\n";

        bin_num = 0;
        for (auto& [lbl, members] : cluster_members) {
            if (members.size() < 5) continue;
            bin_num++;
            std::string name = "bin_" + std::to_string(bin_num);

            // Aggregate damage stats (weighted by n_eff)
            double sum_p_anc = 0, sum_lambda5 = 0, sum_lambda3 = 0;
            double sum_amp5 = 0, sum_amp3 = 0, sum_frag = 0;
            std::vector<double> sum_ct(25, 0), sum_ga(25, 0);
            double total_weight = 0;

            for (int ci : members) {
                // Find this contig in any run with damage data
                for (int r = 0; r < R; r++) {
                    if (runs[r].damage.empty() || runs[r].dmg_dim < 63) continue;
                    // Get local index in this run
                    int local_idx = -1;
                    for (uint32_t ai = 0; ai < runs[r].n_contigs; ai++) {
                        if (remap[r][ai] == ci) { local_idx = ai; break; }
                    }
                    if (local_idx < 0) continue;

                    const float* d = runs[r].damage.data() + local_idx * runs[r].dmg_dim;
                    float w = std::max(d[kDmgNEff], 1.0f);  // weight by effective reads

                    sum_p_anc   += w * d[kDmgPAnc];
                    sum_lambda5 += w * d[kDmgLambda5];
                    sum_lambda3 += w * d[kDmgLambda3];
                    sum_amp5    += w * d[kDmgAmp5];
                    sum_amp3    += w * d[kDmgAmp3];
                    sum_frag    += w * d[kDmgFragMean];
                    for (int p = 0; p < 25; p++) {
                        sum_ct[p] += w * d[kDmgCtStart + p];
                        sum_ga[p] += w * d[kDmgGaStart + p];
                    }
                    total_weight += w;
                    break;  // only count once per contig
                }
            }

            if (total_weight < 1e-6) continue;
            double inv = 1.0 / total_weight;

            double p_anc   = sum_p_anc * inv;
            double lambda5 = sum_lambda5 * inv;
            double lambda3 = sum_lambda3 * inv;
            double amp5    = sum_amp5 * inv;
            double amp3    = sum_amp3 * inv;
            double frag    = sum_frag * inv;
            double ct_1p   = sum_ct[0] * inv;
            double ga_1p   = sum_ga[0] * inv;

            // Damage classification
            std::string dmg_class;
            double term_damage = (ct_1p + ga_1p) / 2.0;
            if (term_damage >= 0.10)       dmg_class = "ancient";
            else if (term_damage >= 0.03)  dmg_class = "moderate";
            else                           dmg_class = "modern";

            dmg_out << name << "\t" << p_anc << "\t" << lambda5 << "\t" << lambda3
                    << "\t" << amp5 << "\t" << amp3 << "\t" << ct_1p << "\t" << ga_1p
                    << "\t" << frag << "\t" << dmg_class << "\n";

            // Write per-position rates for smiley plot
            for (int p = 0; p < 25; p++) {
                rate_out << name << "\t" << (p + 1) << "\t"
                         << (sum_ct[p] * inv) << "\t" << (sum_ga[p] * inv) << "\n";
            }
        }
        std::cerr << "[resolve] Wrote damage stats to " << output_dir << "/damage_per_bin.tsv\n";
    }

    return 0;
}

}  // namespace amber
