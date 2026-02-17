#pragma once

#include "clustering_types.h"
#include <memory>
#include <vector>

namespace hnswlib {
template<typename dist_t> class HierarchicalNSW;
template<typename MTYPE> class SpaceInterface;
}

namespace amber::bin2 {

class HnswKnnIndex {
public:
    explicit HnswKnnIndex(const KnnConfig& config = KnnConfig());
    ~HnswKnnIndex();

    HnswKnnIndex(const HnswKnnIndex&) = delete;
    HnswKnnIndex& operator=(const HnswKnnIndex&) = delete;
    HnswKnnIndex(HnswKnnIndex&&) noexcept;
    HnswKnnIndex& operator=(HnswKnnIndex&&) noexcept;

    void build(const std::vector<std::vector<float>>& embeddings);
    void build(const float* data, size_t n, size_t dim);

    NeighborList query_all() const;
    NeighborList query_all(int k) const;

    std::vector<std::pair<float, int>> query(const float* point, int k) const;
    std::vector<std::pair<float, int>> query(const std::vector<float>& point, int k) const;

    size_t size() const { return n_points_; }
    size_t dim() const { return dim_; }
    bool is_built() const { return index_ != nullptr; }

private:
    KnnConfig config_;
    std::unique_ptr<hnswlib::SpaceInterface<float>> space_;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> index_;
    size_t n_points_ = 0;
    size_t dim_ = 0;
    std::vector<float> data_;
};

}  // namespace amber::bin2
