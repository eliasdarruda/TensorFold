#pragma once

#include <hip/hip_runtime.h>

#include "decode/pages.hpp"

// A lane round's plan in device memory: no stream state is a launch argument, so one launch or graph serves any.
struct PlanArgs {
    const int* pos;                   // per row: its position
    const int* slot;                  // per row: its slot
    const int* first;                 // per slot: its first row
    const int* count;                 // per slot: its rows
    const unsigned long long* desc;   // per slot: the address of its descriptor
    const unsigned long long* snaps;  // per layer: the round's conv and DeltaNet snapshots, a row each
    unsigned pages;                   // words of a descriptor before its page table
    unsigned pool;                    // pages a head of a layer's pool holds
    // Trees (all null: every window a chain). A window's rows sit at cache positions in row order; a row's draw,
    // rotation and walk take its logical position, its window's start plus its depth, and its ancestors' keys.
    const int* depth;                 // per row: its depth in its window (a chain's row r - first)
    const unsigned long long* path;   // per row: its window's rows from depth 0 to its own, a nibble a depth
    const unsigned long long* kept;   // per slot: a keep's rows in order, a nibble each (a chain's 0, 1, 2, ...)
};

// Rows a nibble list names: a window's (lanes cap a window at 16 rows; padding rows past them are a chain).
constexpr int kTreeRows = 16;

// Window row `i` of a nibble list (a path's row at depth i, or a keep's i-th row); past 16 rows, row i itself.
__device__ __forceinline__ int plan_nibble(unsigned long long w, int i) {
    return i >= kTreeRows ? i : static_cast<int>((w >> (4 * i)) & 15);
}

// The cache position of row r's window start: its slot's first row's.
__device__ __forceinline__ int plan_base(const PlanArgs& p, int r) { return p.pos[p.first[p.slot[r]]]; }

// Row r's logical position: its window's start plus its depth (a chain's, its own cache position).
__device__ __forceinline__ int plan_see(const PlanArgs& p, int r) {
    return p.depth == nullptr ? p.pos[r] : plan_base(p, r) + p.depth[r];
}

// The cache position of the key row r sees at logical position `key`: past its window's start, its ancestor's.
__device__ __forceinline__ int plan_key(const unsigned long long path, bool tree, int base, int key) {
    return (!tree || key < base) ? key : base + plan_nibble(path, key - base);
}

// A row's window start and path for plan_key (tree false: no remap, every key its own position).
struct PlanSees {
    bool tree;
    int base;
    unsigned long long path;
};

__device__ __forceinline__ PlanSees plan_sees(const PlanArgs& p, int r) {
    if (p.path == nullptr) return {false, 0, 0ull};
    return {true, plan_base(p, r), p.path[r]};
}

__device__ __forceinline__ int plan_key(const PlanSees& s, int key) { return plan_key(s.path, s.tree, s.base, key); }

// Window row `i` of a slot's keep: its i-th listed row (a chain's i-th row).
__device__ __forceinline__ int plan_kept(const PlanArgs& p, int slot, int i) {
    return p.kept == nullptr ? i : plan_nibble(p.kept[slot], i);
}

// Window row r's parent (-1: the window's first row), from its path.
__device__ __forceinline__ int plan_parent(const PlanArgs& p, int first, int r) {
    if (p.depth == nullptr) return r - 1;
    const int d = p.depth[first + r];
    if (d == 0) return -1;
    return d > kTreeRows ? r - 1 : plan_nibble(p.path[first + r], d - 1);
}

// Slot descriptor: [0] cached positions, [1] last kept final row, two cache addresses a layer, then the page table.
__device__ __forceinline__ const unsigned long long* plan_desc(const PlanArgs& p, int slot) {
    return reinterpret_cast<const unsigned long long*>(p.desc[slot]);
}

__device__ __forceinline__ int plan_first(int layer) { return 2 + 2 * layer; }

__device__ __forceinline__ int plan_second(int layer) { return 3 + 2 * layer; }

__device__ __forceinline__ const unsigned* plan_pages(const PlanArgs& p, const unsigned long long* desc) {
    return reinterpret_cast<const unsigned*>(desc + p.pages);
}
