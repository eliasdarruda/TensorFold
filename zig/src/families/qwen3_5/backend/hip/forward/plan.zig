//! A lane round's plan on the device; launches depend only on its `Shape`, so one graph serves any streams and caches.

const std = @import("std");
const hip = @import("hip");
const state = @import("state.zig");

const round_shape = @import("round_shape.zig");

pub const buckets = round_shape.buckets;
pub const min_span = round_shape.min_span;
pub const bucketOf = round_shape.bucketOf;
pub const spanOf = round_shape.spanOf;
pub const Shape = round_shape.Shape;

/// Where each array sits in the plan buffer, in 4-byte words (the 8-byte arrays on even words).
pub const Layout = struct {
    layers: usize,
    tokens: usize,
    pos: usize,
    slot: usize,
    first: usize,
    count: usize,
    keep: usize,
    desc: usize,
    snaps: usize,
    words: usize,
    rope: usize,
    rotary: usize,
    /// Trees: each row's depth (i32) and path (u64 nibbles), each slot's kept rows at a keep (u64), the full layers.
    depth: usize,
    path: usize,
    kept: usize,
    full: usize,

    pub fn of(rows: usize, slots: usize, layers: usize) Layout {
        var l: Layout = undefined;
        l.layers = layers;
        l.tokens = 0;
        l.pos = rows;
        l.slot = 2 * rows;
        l.first = 3 * rows;
        l.count = l.first + slots;
        l.keep = l.count + slots;
        l.desc = std.mem.alignForward(usize, l.keep + slots, 2);
        l.snaps = l.desc + 2 * slots;
        l.rope = l.snaps + 4 * layers;
        l.rotary = l.rope + 3 * rows;
        l.depth = l.rotary + rows;
        l.path = std.mem.alignForward(usize, l.depth + rows, 2);
        l.kept = l.path + 2 * rows;
        l.full = l.kept + 2 * slots;
        l.words = l.full + layers;
        return l;
    }
};

/// A chain row's path: rows 0 .. i at depths 0 .. i (past 16 rows, the rows a nibble list holds).
fn chainRow(i: usize) u64 {
    const n = @min(i + 1, max_tree_rows);
    return if (n == max_tree_rows) chain_path else chain_path & ((@as(u64, 1) << @intCast(4 * n)) - 1);
}

fn put64(w: []u32, v: u64) void {
    w[0] = @truncate(v);
    w[1] = @truncate(v >> 32);
}

/// One stream's window of a round: its descriptor, the slot its first row takes, its tokens; `parents` makes it a
/// tree (each row's parent row, row 0's -1, parents before children): rows keep cache positions in row order, and
/// each row's rotation and walk take its window's start plus its depth.
pub const Window = struct { desc: u64, pos: usize, tokens: []const u32, shift: i32 = 0, parents: ?[]const i32 = null };

/// Rows a tree window may hold: a path's rows are nibbles.
pub const max_tree_rows = 16;

/// A chain's kept rows, 0, 1, 2, ... a nibble each: a keep that names no path.
pub const chain_path: u64 = 0xfedc_ba98_7654_3210;

/// Each row's depth and path (rows from depth 0 to its own, a nibble a depth) from its parents; a chain without.
pub fn treeRows(parents: ?[]const i32, rows: usize, depth: []u32, path: []u64) error{BadTree}!void {
    for (0..rows) |i| {
        const p: i32 = if (parents) |ps| ps[i] else @as(i32, @intCast(i)) - 1;
        if (i == 0) {
            if (p != -1) return error.BadTree;
            depth[0] = 0;
            path[0] = 0;
            continue;
        }
        if (p < 0 or p >= @as(i32, @intCast(i))) return error.BadTree;
        const d = depth[@intCast(p)] + 1;
        depth[i] = d;
        // past 16 rows a chain names no nibble: plan.hpp's rows past them are their own
        path[i] = if (d < max_tree_rows) path[@intCast(p)] | (@as(u64, i) << @intCast(4 * d)) else 0;
        if (i >= max_tree_rows and parents != null) return error.BadTree;
    }
}

/// A keep's rows in order as nibbles (a tree's path from row 0).
pub fn keptPath(rows: []const u32) error{BadTree}!u64 {
    if (rows.len == 0 or rows.len > max_tree_rows or rows[0] != 0) return error.BadTree;
    var w: u64 = 0;
    for (rows, 0..) |r, i| {
        if (r >= max_tree_rows or (i > 0 and r <= rows[i - 1])) return error.BadTree;
        w |= @as(u64, r) << @intCast(4 * i);
    }
    return w;
}

/// The plan buffer of one engine, as large as its biggest round, and the pinned words a round is written in.
pub const Buffer = struct {
    host: hip.HostBuffer,
    dev: hip.DeviceBuffer,
    /// Pages a head of a layer's pool holds: the kernels find a page's keys and values with it.
    pool_pages: u32,

    /// Device bytes a plan of `rows` rows over `layers` layers holds.
    pub fn deviceBytes(rows: usize, layers: usize) usize {
        return 4 * Layout.of(rows, rows + 1, layers).words;
    }

    pub fn init(d: *const hip.Runtime, rows: usize, layers: usize, pool_pages: usize) !Buffer {
        const bytes = deviceBytes(rows, layers);
        var host = try hip.HostBuffer.alloc(d, bytes);
        errdefer host.free();
        return .{ .host = host, .dev = try hip.DeviceBuffer.alloc(d, bytes), .pool_pages = @intCast(pool_pages) };
    }

    pub fn deinit(b: *Buffer) void {
        b.dev.free();
        b.host.free();
    }

    /// The device words of `shape` as launches read them.
    pub fn args(b: *const Buffer, l: Layout) hip.plan_ops.Args {
        const at = b.dev.base();
        return .{
            .pos = at + 4 * l.pos,
            .slot = at + 4 * l.slot,
            .first = at + 4 * l.first,
            .count = at + 4 * l.count,
            .desc = at + 4 * l.desc,
            .snaps = at + 4 * l.snaps,
            .pages = @intCast(state.tableWord(l.layers)),
            .pool = b.pool_pages,
            .depth = at + 4 * l.depth,
            .path = at + 4 * l.path,
            .kept = at + 4 * l.kept,
        };
    }

    /// Writes the round into the pinned words: windows, the scratch slot's `pad` rows, each linear layer's snapshots;
    /// `full` lists the full-attention layers (a tree keep moves their kept keys and values).
    pub fn fill(b: *Buffer, l: Layout, shape: Shape, windows: []const Window, scratch: u64, snaps: []const [2]u64, full: []const u32) error{BadTree}!void {
        const w = b.host.slice(u32);
        const slots: usize = shape.slots;
        var at: usize = 0;
        var depth: [max_tree_rows]u32 = undefined;
        var path: [max_tree_rows]u64 = undefined;
        for (windows, 0..) |win, s| {
            w[l.first + s] = @intCast(at);
            w[l.count + s] = @intCast(win.tokens.len);
            w[l.keep + s] = std.math.maxInt(u32);
            put64(w[l.desc + 2 * s ..], win.desc);
            put64(w[l.kept + 2 * s ..], chain_path);
            if (win.parents) |ps| {
                if (ps.len != win.tokens.len or ps.len > max_tree_rows) return error.BadTree;
                try treeRows(ps, ps.len, &depth, &path);
            }
            for (win.tokens, 0..) |t, i| {
                // a tree's row draws, rotates and walks at its window's start plus its depth
                const d: usize = if (win.parents != null) depth[i] else i;
                const p: u64 = if (win.parents != null) path[i] else chainRow(i);
                w[l.tokens + at + i] = t;
                w[l.pos + at + i] = @intCast(win.pos + i);
                w[l.slot + at + i] = @intCast(s);
                w[l.depth + at + i] = @intCast(d);
                put64(w[l.path + 2 * (at + i) ..], p);
                const rp = @as(i32, @intCast(win.pos + d)) + win.shift;
                for (0..3) |axis| w[l.rope + 3 * (at + i) + axis] = @bitCast(rp);
                w[l.rotary + at + i] = @bitCast(rp);
            }
            at += win.tokens.len;
        }
        // slots past the streams' are empty and the last is the scratch slot, where padding rows run from its start
        for (windows.len..slots - 1) |s| {
            w[l.first + s] = 0;
            w[l.count + s] = 0;
            w[l.keep + s] = std.math.maxInt(u32);
            put64(w[l.desc + 2 * s ..], 0);
            put64(w[l.kept + 2 * s ..], chain_path);
        }
        const pad = shape.rows - at;
        w[l.first + slots - 1] = @intCast(at);
        w[l.count + slots - 1] = @intCast(pad);
        w[l.keep + slots - 1] = std.math.maxInt(u32);
        put64(w[l.desc + 2 * (slots - 1) ..], scratch);
        put64(w[l.kept + 2 * (slots - 1) ..], chain_path);
        for (0..pad) |i| {
            w[l.tokens + at + i] = 0;
            w[l.pos + at + i] = @intCast(i);
            w[l.slot + at + i] = @intCast(slots - 1);
            w[l.depth + at + i] = @intCast(i);
            put64(w[l.path + 2 * (at + i) ..], chainRow(i));
            for (0..3) |axis| w[l.rope + 3 * (at + i) + axis] = @intCast(i);
            w[l.rotary + at + i] = @intCast(i);
        }
        for (snaps, 0..) |pair, i| {
            put64(w[l.snaps + 4 * i ..], pair[0]);
            put64(w[l.snaps + 4 * i + 2 ..], pair[1]);
        }
        for (0..l.layers) |i| w[l.full + i] = if (i < full.len) full[i] else 0;
    }

    /// The filled words up to the device in one copy (a graph holds the copy: it runs from the same pinned words).
    pub fn send(b: *const Buffer, stream: hip.abi.Stream, l: Layout) !void {
        try hip.raw.upload(b.dev, 0, b.host.bytes[0 .. 4 * l.words], stream);
    }
};

test "layout keeps the eight-byte arrays on even words" {
    const l = Layout.of(5, 4, 3);
    try std.testing.expect(l.desc % 2 == 0 and l.snaps % 2 == 0);
    try std.testing.expect(l.words >= l.snaps + 12);
}

test "round rotary metadata keeps each lane's shift separate from physical cache positions" {
    const l = Layout.of(5, 3, 0);
    var words: [128]u32 = undefined;
    var b: Buffer = .{ .host = .{ .r = undefined, .bytes = std.mem.sliceAsBytes(words[0..l.words]) }, .dev = undefined, .pool_pages = 0 };
    const windows = [_]Window{
        .{ .desc = 0x1000, .pos = 10, .tokens = &.{ 5, 6 }, .shift = -3 },
        .{ .desc = 0x2000, .pos = 20, .tokens = &.{7}, .shift = -9 },
    };
    try b.fill(l, .{ .rows = 5, .slots = 3, .span = 64 }, &windows, 0x3000, &.{}, &.{});
    try std.testing.expectEqualSlices(u32, &.{ 10, 11, 20, 0, 1 }, words[l.pos..][0..5]);
    try std.testing.expectEqualSlices(u32, &.{ 7, 7, 7, 8, 8, 8, 11, 11, 11, 0, 0, 0, 1, 1, 1 }, words[l.rope..][0..15]);
    try std.testing.expectEqualSlices(u32, &.{ 7, 8, 11, 0, 1 }, words[l.rotary..][0..5]);
}

test "a tree's rows: depths and paths from parents, cache positions in row order, rotation at start plus depth" {
    const l = Layout.of(6, 3, 2);
    var words: [256]u32 = undefined;
    var b: Buffer = .{ .host = .{ .r = undefined, .bytes = std.mem.sliceAsBytes(words[0..l.words]) }, .dev = undefined, .pool_pages = 0 };
    // row 0 the pending token; rows 1 and 2 its two children, row 3 row 1's child
    const windows = [_]Window{.{ .desc = 0x1000, .pos = 40, .tokens = &.{ 5, 6, 7, 8 }, .parents = &.{ -1, 0, 0, 1 } }};
    try b.fill(l, .{ .rows = 6, .slots = 3, .span = 64 }, &windows, 0x3000, &.{}, &.{ 3, 7 });
    try std.testing.expectEqualSlices(u32, &.{ 40, 41, 42, 43, 0, 1 }, words[l.pos..][0..6]);
    try std.testing.expectEqualSlices(u32, &.{ 0, 1, 1, 2, 0, 1 }, words[l.depth..][0..6]);
    try std.testing.expectEqualSlices(u32, &.{ 40, 41, 41, 42, 0, 1 }, words[l.rotary..][0..6]);
    const want = [_]u64{ 0x0, 0x10, 0x20, 0x310, 0x0, 0x10 };
    for (want, 0..) |x, i| try std.testing.expectEqual(x, @as(u64, words[l.path + 2 * i]) | @as(u64, words[l.path + 2 * i + 1]) << 32);
    try std.testing.expectEqualSlices(u32, &.{ 3, 7 }, words[l.full..][0..2]);
    for (0..3) |i| try std.testing.expectEqual(chain_path, @as(u64, words[l.kept + 2 * i]) | @as(u64, words[l.kept + 2 * i + 1]) << 32);
}

test "a tree's parents must precede their rows and fit 16 rows; a keep's path ascends from row 0" {
    var depth: [16]u32 = undefined;
    var path: [16]u64 = undefined;
    try std.testing.expectError(error.BadTree, treeRows(&.{ -1, 1 }, 2, &depth, &path));
    try std.testing.expectError(error.BadTree, treeRows(&.{ 0, 0 }, 2, &depth, &path));
    try treeRows(null, 16, &depth, &path);
    try std.testing.expectEqual(@as(u64, chain_path), path[15]);
    try std.testing.expectEqual(@as(u64, 0x310), try keptPath(&.{ 0, 1, 3 }));
    try std.testing.expectError(error.BadTree, keptPath(&.{ 1, 3 }));
    try std.testing.expectError(error.BadTree, keptPath(&.{ 0, 3, 2 }));
}
