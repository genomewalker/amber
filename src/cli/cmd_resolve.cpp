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

    // ── 7. Reverse lookup: union_ci → local index per run (O(1) damage access) ─
    std::vector<std::vector<int>> ci_to_local(R, std::vector<int>(N, -1));
    for (int r = 0; r < R; r++)
        for (uint32_t ai = 0; ai < runs[r].n_contigs; ai++) {
            int si = remap[r][ai];
            if (si >= 0) ci_to_local[r][si] = ai;
        }

    // ── 8. Load FASTA, compute per-contig GC ──────────────────────────────────
    auto contigs = load_fasta(contigs_path);
    std::unordered_map<std::string,int> fa_idx;
    fa_idx.reserve(contigs.size());
    for (int i = 0; i < (int)contigs.size(); i++)
        fa_idx[contigs[i].first] = i;

    std::vector<int> shared_to_fa(N, -1);
    for (int i = 0; i < N; i++) {
        auto it = fa_idx.find(all_names[i]);
        if (it != fa_idx.end()) shared_to_fa[i] = it->second;
    }

    std::vector<float> fa_gc(contigs.size(), 0.0f);
    for (int i = 0; i < (int)contigs.size(); i++) {
        const auto& s = contigs[i].second;
        if (s.empty()) continue;
        int gc = 0;
        for (char c : s) if (c=='G'||c=='g'||c=='C'||c=='c') gc++;
        fa_gc[i] = static_cast<float>(gc) / static_cast<float>(s.size());
    }

    // ── 9. Group contigs by cluster; sort by label for stable bin naming ──────
    std::unordered_map<int,std::vector<int>> cluster_map;
    for (int i = 0; i < N; i++) {
        int lbl = best.labels[i];
        if (lbl >= 0) cluster_map[lbl].push_back(i);
    }
    std::vector<std::pair<int,std::vector<int>>> clusters;
    clusters.reserve(cluster_map.size());
    for (auto& [lbl, members] : cluster_map)
        if ((int)members.size() >= 5)
            clusters.emplace_back(lbl, std::move(members));
    std::sort(clusters.begin(), clusters.end(),
              [](const auto& a, const auto& b){ return a.first < b.first; });

    // ── 10. Intra-bin mean edge weight (co-binning stability) ─────────────────
    std::unordered_map<int,double> bin_edge_sum;
    std::unordered_map<int,int>    bin_edge_cnt;
    for (const auto& e : edges) {
        int lu = best.labels[e.u];
        int lv = best.labels[e.v];
        if (lu >= 0 && lu == lv) { bin_edge_sum[lu] += e.w; bin_edge_cnt[lu]++; }
    }

    // ── 11. Damage availability ───────────────────────────────────────────────
    constexpr int kDmgPAnc     = 0;
    constexpr int kDmgNEff     = 2;
    constexpr int kDmgCtStart  = 6;   // CT rates positions 1-25 (5' end)
    constexpr int kDmgGaStart  = 31;  // GA rates positions 1-25 (3' end)
    constexpr int kDmgLambda5  = 56;
    constexpr int kDmgLambda3  = 57;
    constexpr int kDmgAmp5     = 58;
    constexpr int kDmgAmp3     = 59;
    constexpr int kDmgFragMean = 60;
    constexpr int kDmgFragStd  = 61;

    bool has_damage = false;
    for (int r = 0; r < R; r++)
        if (!runs[r].damage.empty() && runs[r].dmg_dim >= 63) { has_damage = true; break; }

    uint32_t total_markers = runs[0].total_markers;

    // ── 12. Write all outputs in one stable pass over sorted clusters ─────────
    std::ofstream f_stats(output_dir + "/bin_stats.tsv");
    f_stats << "bin\tcluster_id\tn_contigs\tgenome_size\tn50\tmax_contig\t"
            << "gc_content\tscg_completeness\tscg_contamination\tmean_cobin_weight\n";

    std::ofstream f_dmg, f_smiley;
    if (has_damage) {
        f_dmg.open(output_dir + "/damage_per_bin.tsv");
        f_dmg << "bin\tp_ancient\tlambda5\tlambda3\tamplitude_5p\tamplitude_3p\t"
              << "ct_1p\tga_1p\tfrag_mean\tfrag_std\tdamage_class\n";
        f_smiley.open(output_dir + "/smiley_plot_rates.tsv");
        f_smiley << "bin\tposition\tct_rate\tga_rate\n";
    }

    int bin_num = 0;
    for (auto& [lbl, members] : clusters) {
        bin_num++;
        std::string name = "bin_" + std::to_string(bin_num);

        // FASTA output
        {
            std::ofstream fa_out(output_dir + "/" + name + ".fa");
            for (int ci : members) {
                int fi = shared_to_fa[ci];
                if (fi >= 0) fa_out << ">" << contigs[fi].first << "\n" << contigs[fi].second << "\n";
            }
        }

        // Genome assembly stats (from FASTA lengths)
        std::vector<int> lens;
        lens.reserve(members.size());
        double gc_sum = 0.0;
        int64_t genome_size = 0;
        for (int ci : members) {
            int fi = shared_to_fa[ci];
            if (fi < 0) continue;
            int len = static_cast<int>(contigs[fi].second.size());
            lens.push_back(len);
            genome_size += len;
            gc_sum += static_cast<double>(fa_gc[fi]) * len;
        }
        float gc = genome_size > 0 ? static_cast<float>(gc_sum / genome_size) : 0.0f;
        int max_contig = lens.empty() ? 0 : *std::max_element(lens.begin(), lens.end());
        int n50 = 0;
        if (!lens.empty()) {
            std::sort(lens.begin(), lens.end(), std::greater<int>());
            int64_t half = genome_size / 2, cum = 0;
            for (int l : lens) { cum += l; if (cum >= half) { n50 = l; break; } }
        }

        // SCG completeness
        float comp = 0.0f, cont = 0.0f;
        if (total_markers > 0 && !scg.empty()) {
            std::unordered_map<int,int> mc;
            for (int ci : members) {
                int r0ci = union_to_run0[ci];
                if (r0ci < 0 || (size_t)r0ci >= scg.size()) continue;
                for (int mid : scg[r0ci]) mc[mid]++;
            }
            int unique = (int)mc.size(), total_copies = 0;
            for (auto& [m,c] : mc) total_copies += c;
            comp = static_cast<float>(unique) / total_markers;
            cont = static_cast<float>(total_copies - unique) / total_markers;
        }

        // Co-binning stability
        float mean_cobin = 0.0f;
        {
            auto it = bin_edge_cnt.find(lbl);
            if (it != bin_edge_cnt.end() && it->second > 0)
                mean_cobin = static_cast<float>(bin_edge_sum[lbl] / it->second);
        }

        f_stats << name << "\t" << lbl << "\t" << (int)members.size()
                << "\t" << genome_size << "\t" << n50 << "\t" << max_contig
                << "\t" << gc << "\t" << comp << "\t" << cont
                << "\t" << mean_cobin << "\n";

        // Damage stats
        if (has_damage) {
            double sum_p_anc = 0, sum_l5 = 0, sum_l3 = 0;
            double sum_a5 = 0, sum_a3 = 0, sum_fm = 0, sum_fs = 0;
            std::vector<double> sum_ct(25, 0), sum_ga(25, 0);
            double total_w = 0;

            for (int ci : members) {
                for (int r = 0; r < R; r++) {
                    if (runs[r].damage.empty() || runs[r].dmg_dim < 63) continue;
                    int li = ci_to_local[r][ci];
                    if (li < 0) continue;
                    const float* d = runs[r].damage.data() + li * runs[r].dmg_dim;
                    float w = std::max(d[kDmgNEff], 1.0f);
                    sum_p_anc += w * d[kDmgPAnc];
                    sum_l5    += w * d[kDmgLambda5];
                    sum_l3    += w * d[kDmgLambda3];
                    sum_a5    += w * d[kDmgAmp5];
                    sum_a3    += w * d[kDmgAmp3];
                    sum_fm    += w * d[kDmgFragMean];
                    sum_fs    += w * d[kDmgFragStd];
                    for (int p = 0; p < 25; p++) {
                        sum_ct[p] += w * d[kDmgCtStart + p];
                        sum_ga[p] += w * d[kDmgGaStart + p];
                    }
                    total_w += w;
                    break;  // one observation per contig
                }
            }

            if (total_w >= 1e-6) {
                double inv = 1.0 / total_w;
                double ct1 = sum_ct[0] * inv;
                double ga1 = sum_ga[0] * inv;
                double term = (ct1 + ga1) / 2.0;
                const char* dmg_class = term >= 0.10 ? "ancient"
                                      : term >= 0.03 ? "moderate" : "modern";

                f_dmg << name
                      << "\t" << sum_p_anc * inv
                      << "\t" << sum_l5 * inv << "\t" << sum_l3 * inv
                      << "\t" << sum_a5 * inv << "\t" << sum_a3 * inv
                      << "\t" << ct1 << "\t" << ga1
                      << "\t" << sum_fm * inv << "\t" << sum_fs * inv
                      << "\t" << dmg_class << "\n";

                for (int p = 0; p < 25; p++)
                    f_smiley << name << "\t" << (p+1) << "\t"
                             << (sum_ct[p] * inv) << "\t" << (sum_ga[p] * inv) << "\n";
            }
        }
    }

    std::cerr << "[resolve] Wrote " << bin_num << " bins to " << output_dir << "\n";
    std::cerr << "[resolve] Wrote bin_stats.tsv";
    if (has_damage) std::cerr << ", damage_per_bin.tsv, smiley_plot_rates.tsv";
    std::cerr << "\n";

    return 0;
}

}  // namespace amber
