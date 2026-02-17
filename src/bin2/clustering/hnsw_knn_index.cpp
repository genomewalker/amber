#include "hnsw_knn_index.h"
#include "../../../include/hnswlib.h"

#include <stdexcept>
#include <algorithm>

namespace amber::bin2 {

HnswKnnIndex::HnswKnnIndex(const KnnConfig& config)
    : config_(config) {}

HnswKnnIndex::~HnswKnnIndex() = default;

HnswKnnIndex::HnswKnnIndex(HnswKnnIndex&&) noexcept = default;
HnswKnnIndex& HnswKnnIndex::operator=(HnswKnnIndex&&) noexcept = default;

void HnswKnnIndex::build(const std::vector<std::vector<float>>& embeddings) {
    if (embeddings.empty()) {
        throw std::invalid_argument("Empty embeddings");
    }

    n_points_ = embeddings.size();
    dim_ = embeddings[0].size();

    data_.resize(n_points_ * dim_);
    for (size_t i = 0; i < n_points_; ++i) {
        if (embeddings[i].size() != dim_) {
            throw std::invalid_argument("Inconsistent embedding dimensions");
        }
        std::copy(embeddings[i].begin(), embeddings[i].end(),
                  data_.begin() + i * dim_);
    }

    build(data_.data(), n_points_, dim_);
}

void HnswKnnIndex::build(const float* data, size_t n, size_t dim) {
    if (n == 0 || dim == 0) {
        throw std::invalid_argument("Invalid dimensions");
    }

    n_points_ = n;
    dim_ = dim;

    if (data_.empty()) {
        data_.assign(data, data + n * dim);
    }

    space_ = std::make_unique<hnswlib::L2Space>(dim_);
    index_ = std::make_unique<hnswlib::HierarchicalNSW<float>>(
        space_.get(),
        n_points_,
        config_.M,
        config_.ef_construction,
        config_.random_seed
    );

    for (size_t i = 0; i < n_points_; ++i) {
        index_->addPoint(data_.data() + i * dim_, i);
    }

    index_->setEf(config_.ef_search);
}

NeighborList HnswKnnIndex::query_all() const {
    return query_all(config_.k);
}

NeighborList HnswKnnIndex::query_all(int k) const {
    if (!is_built()) {
        throw std::runtime_error("Index not built");
    }

    NeighborList result;
    result.resize(n_points_, k);

    int actual_k = std::min(k + 1, static_cast<int>(n_points_));

    // Use static scheduling for determinism (dynamic causes thread scheduling non-determinism)
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n_points_; ++i) {
        auto neighbors = index_->searchKnn(data_.data() + i * dim_, actual_k);

        result.ids[i].clear();
        result.dists[i].clear();

        std::vector<std::pair<float, size_t>> sorted;
        sorted.reserve(neighbors.size());

        while (!neighbors.empty()) {
            auto& top = neighbors.top();
            if (top.second != i) {
                sorted.emplace_back(top.first, top.second);
            }
            neighbors.pop();
        }

        std::sort(sorted.begin(), sorted.end());

        int count = std::min(static_cast<int>(sorted.size()), k);
        for (int j = 0; j < count; ++j) {
            result.ids[i].push_back(static_cast<int>(sorted[j].second));
            result.dists[i].push_back(sorted[j].first);
        }
    }

    return result;
}

std::vector<std::pair<float, int>> HnswKnnIndex::query(const float* point, int k) const {
    if (!is_built()) {
        throw std::runtime_error("Index not built");
    }

    auto neighbors = index_->searchKnn(point, k);

    std::vector<std::pair<float, int>> result;
    result.reserve(neighbors.size());

    while (!neighbors.empty()) {
        auto& top = neighbors.top();
        result.emplace_back(top.first, static_cast<int>(top.second));
        neighbors.pop();
    }

    std::sort(result.begin(), result.end());
    return result;
}

std::vector<std::pair<float, int>> HnswKnnIndex::query(const std::vector<float>& point, int k) const {
    if (point.size() != dim_) {
        throw std::invalid_argument("Point dimension mismatch");
    }
    return query(point.data(), k);
}

}  // namespace amber::bin2
