// AMBER Binary format (.abin)
// Per-run encoder output for AMBER-DAS post-processing.
//
// Layout:
//   magic[8]       "AMBR\x01\x00\x00\n"
//   n_contigs[4]   uint32
//   chunks...      [tag:4][size:4][data:size]
//
// Chunks (any order, unknown tags skipped):
//   "NAME"  contig names   — offsets[n*4] + data_len[4] + packed null-terminated strings
//   "LNGT"  lengths        — int32[n]
//   "EMBD"  embeddings     — dim[2:uint16] + float32[n*dim]
//   "FEAT"  raw features   — dim[2:uint16] + float32[n*dim]
//   "DAMG"  damage profile — dim[2:uint16] + float32[n*dim]
//   "BINL"  bin labels     — int32[n]  (-1 = unbinned)
//   "SCGM"  SCG markers    — total_markers[4:uint32] + n_hits[4:uint32]
//                            + hits[n_hits * (contig_idx:uint32 + marker_id:uint16)]
//   "END\0" sentinel
//
// All multi-byte integers are little-endian.
#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace amber {

static constexpr uint8_t ABIN_MAGIC[8] = {'A','M','B','R',0x01,0x00,0x00,'\n'};

struct AbinData {
    uint32_t n_contigs = 0;
    std::vector<std::string>   names;
    std::vector<int32_t>       lengths;
    std::vector<float>         embeddings;   // n_contigs × emb_dim
    std::vector<float>         features;     // n_contigs × feat_dim
    std::vector<float>         damage;       // n_contigs × dmg_dim
    std::vector<int32_t>       bin_labels;   // n_contigs, -1 = unbinned
    uint16_t emb_dim  = 0;
    uint16_t feat_dim = 0;
    uint16_t dmg_dim  = 0;

    // SCG markers: sparse contig→marker mapping for bin scoring
    uint32_t total_markers = 0;             // total marker profiles (e.g. 206)
    std::vector<std::vector<int>> scg_markers;  // scg_markers[contig_idx] = list of marker IDs
};

// ── writer ────────────────────────────────────────────────────────────────────

class AbinWriter {
public:
    explicit AbinWriter(const std::string& path) : f_(path, std::ios::binary) {
        if (!f_) throw std::runtime_error("abin: cannot open for writing: " + path);
    }

    void write(const AbinData& d) {
        write_magic(d.n_contigs);
        if (!d.names.empty())      write_names(d);
        if (!d.lengths.empty())    write_i32_chunk("LNGT", d.lengths);
        if (!d.embeddings.empty()) write_f32_chunk("EMBD", d.embeddings, d.emb_dim);
        if (!d.features.empty())   write_f32_chunk("FEAT", d.features,   d.feat_dim);
        if (!d.damage.empty())     write_f32_chunk("DAMG", d.damage,     d.dmg_dim);
        if (!d.bin_labels.empty()) write_i32_chunk("BINL", d.bin_labels);
        if (d.total_markers > 0)   write_scgm(d);
        write_tag("END"); f_.put('\0');
        uint32_t zero = 0; write_u32(zero);
    }

private:
    std::ofstream f_;

    void write_magic(uint32_t n_contigs) {
        f_.write(reinterpret_cast<const char*>(ABIN_MAGIC), 8);
        write_u32(n_contigs);
    }

    void write_tag(const char* tag) { f_.write(tag, 4); }

    void write_u32(uint32_t v) { f_.write(reinterpret_cast<const char*>(&v), 4); }

    void write_names(const AbinData& d) {
        // Build packed name data
        std::vector<uint32_t> offsets(d.n_contigs);
        std::string packed;
        for (uint32_t i = 0; i < d.n_contigs; i++) {
            offsets[i] = static_cast<uint32_t>(packed.size());
            packed += d.names[i];
            packed += '\0';
        }
        uint32_t data_len = static_cast<uint32_t>(packed.size());
        uint32_t chunk_size = 4 * d.n_contigs + 4 + data_len;

        write_tag("NAME"); write_u32(chunk_size);
        f_.write(reinterpret_cast<const char*>(offsets.data()), 4 * d.n_contigs);
        write_u32(data_len);
        f_.write(packed.data(), data_len);
    }

    void write_f32_chunk(const char* tag, const std::vector<float>& v, uint16_t dim) {
        uint32_t chunk_size = 2 + static_cast<uint32_t>(v.size()) * 4;
        write_tag(tag); write_u32(chunk_size);
        f_.write(reinterpret_cast<const char*>(&dim), 2);
        f_.write(reinterpret_cast<const char*>(v.data()), v.size() * 4);
    }

    void write_i32_chunk(const char* tag, const std::vector<int32_t>& v) {
        uint32_t chunk_size = static_cast<uint32_t>(v.size()) * 4;
        write_tag(tag); write_u32(chunk_size);
        f_.write(reinterpret_cast<const char*>(v.data()), chunk_size);
    }

    void write_scgm(const AbinData& d) {
        // Count total hits
        uint32_t n_hits = 0;
        for (const auto& m : d.scg_markers) n_hits += static_cast<uint32_t>(m.size());
        uint32_t chunk_size = 4 + 4 + n_hits * 6;  // total_markers + n_hits + hits*(4+2)
        write_tag("SCGM"); write_u32(chunk_size);
        write_u32(d.total_markers);
        write_u32(n_hits);
        for (uint32_t ci = 0; ci < static_cast<uint32_t>(d.scg_markers.size()); ci++) {
            for (int mid : d.scg_markers[ci]) {
                write_u32(ci);
                uint16_t m = static_cast<uint16_t>(mid);
                f_.write(reinterpret_cast<const char*>(&m), 2);
            }
        }
    }
};

// ── reader ────────────────────────────────────────────────────────────────────

class AbinReader {
public:
    explicit AbinReader(const std::string& path) : f_(path, std::ios::binary) {
        if (!f_) throw std::runtime_error("abin: cannot open: " + path);
    }

    AbinData read() {
        read_magic();
        AbinData d;
        d.n_contigs = n_contigs_;

        char tag[4];
        uint32_t size;
        while (f_.read(tag, 4) && f_.read(reinterpret_cast<char*>(&size), 4)) {
            if (memcmp(tag, "END", 3) == 0) break;

            auto chunk_start = f_.tellg();

            if      (memcmp(tag, "NAME", 4) == 0) read_names(d, size);
            else if (memcmp(tag, "LNGT", 4) == 0) read_i32_chunk(d.lengths, size);
            else if (memcmp(tag, "EMBD", 4) == 0) read_f32_chunk(d.embeddings, d.emb_dim, size);
            else if (memcmp(tag, "FEAT", 4) == 0) read_f32_chunk(d.features,   d.feat_dim, size);
            else if (memcmp(tag, "DAMG", 4) == 0) read_f32_chunk(d.damage,     d.dmg_dim,  size);
            else if (memcmp(tag, "BINL", 4) == 0) read_i32_chunk(d.bin_labels, size);
            else if (memcmp(tag, "SCGM", 4) == 0) read_scgm(d);
            else f_.seekg(chunk_start + std::streamoff(size));  // skip unknown
        }
        return d;
    }

private:
    std::ifstream f_;
    uint32_t n_contigs_ = 0;

    void read_magic() {
        uint8_t magic[8];
        f_.read(reinterpret_cast<char*>(magic), 8);
        if (memcmp(magic, ABIN_MAGIC, 8) != 0)
            throw std::runtime_error("abin: bad magic bytes");
        f_.read(reinterpret_cast<char*>(&n_contigs_), 4);
    }

    void read_names(AbinData& d, uint32_t size) {
        std::vector<uint32_t> offsets(d.n_contigs);
        f_.read(reinterpret_cast<char*>(offsets.data()), 4 * d.n_contigs);
        uint32_t data_len;
        f_.read(reinterpret_cast<char*>(&data_len), 4);
        std::string packed(data_len, '\0');
        f_.read(packed.data(), data_len);
        d.names.resize(d.n_contigs);
        for (uint32_t i = 0; i < d.n_contigs; i++)
            d.names[i] = packed.c_str() + offsets[i];
        (void)size;
    }

    void read_f32_chunk(std::vector<float>& v, uint16_t& dim, uint32_t size) {
        f_.read(reinterpret_cast<char*>(&dim), 2);
        uint32_t n_floats = (size - 2) / 4;
        v.resize(n_floats);
        f_.read(reinterpret_cast<char*>(v.data()), n_floats * 4);
    }

    void read_i32_chunk(std::vector<int32_t>& v, uint32_t size) {
        v.resize(size / 4);
        f_.read(reinterpret_cast<char*>(v.data()), size);
    }

    void read_scgm(AbinData& d) {
        f_.read(reinterpret_cast<char*>(&d.total_markers), 4);
        uint32_t n_hits;
        f_.read(reinterpret_cast<char*>(&n_hits), 4);
        d.scg_markers.resize(d.n_contigs);
        for (uint32_t h = 0; h < n_hits; h++) {
            uint32_t ci; uint16_t mid;
            f_.read(reinterpret_cast<char*>(&ci), 4);
            f_.read(reinterpret_cast<char*>(&mid), 2);
            if (ci < d.n_contigs) d.scg_markers[ci].push_back(mid);
        }
    }
};

// ── convenience helpers ───────────────────────────────────────────────────────

inline void abin_save(const std::string& path, const AbinData& d) {
    AbinWriter w(path); w.write(d);
}

inline AbinData abin_load(const std::string& path) {
    AbinReader r(path); return r.read();
}

} // namespace amber
