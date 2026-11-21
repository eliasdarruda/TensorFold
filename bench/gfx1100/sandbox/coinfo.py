"""Per-kernel resources from HIP code objects (offload bundles or ELF), via llvm-readelf's AMDGPU notes.

usage: coinfo.py READELF DIR [DIR...]  -> JSON {dir: {kernel: {...}}} on stdout
"""
import json, os, re, subprocess, sys, tempfile

FIELDS = (".vgpr_count", ".sgpr_count", ".vgpr_spill_count", ".sgpr_spill_count", ".group_segment_fixed_size",
          ".private_segment_fixed_size", ".max_flat_workgroup_size", ".wavefront_size", ".kernarg_segment_size")


def elf_of(path):
    data = open(path, "rb").read()
    if data[:4] == b"\x7fELF":
        return data
    assert data[:24] == b"__CLANG_OFFLOAD_BUNDLE__", path
    n = int.from_bytes(data[24:32], "little")
    p = 32
    for _ in range(n):
        off, size, idlen = (int.from_bytes(data[p + 8 * i:p + 8 * i + 8], "little") for i in range(3))
        ident = data[p + 24:p + 24 + idlen].decode()
        p += 24 + idlen
        if "amdgcn" in ident:
            return data[off:off + size]
    raise ValueError(path)


def kernels(readelf, path):
    with tempfile.NamedTemporaryFile(suffix=".elf") as f:
        f.write(elf_of(path))
        f.flush()
        notes = subprocess.run([readelf, "--notes", f.name], capture_output=True, text=True).stdout
    out, cur = {}, None
    for line in notes.splitlines():
        if line.startswith("  - ") and ".args:" in line:
            cur = {}
            continue
        m = re.match(r"^    \.([a-z_]+):\s+(\S+)", line)
        if m and cur is not None:
            k, v = m.groups()
            if k == "name":
                cur["name"] = v
            elif "." + k in FIELDS:
                cur[k] = int(v)
            if k == "wavefront_size" and "name" in cur:
                out[cur.pop("name")] = cur
                cur = None
    return out


if __name__ == "__main__":
    readelf, dirs = sys.argv[1], sys.argv[2:]
    res = {}
    for d in dirs:
        res[d] = {}
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".hsaco"):
                for k, v in kernels(readelf, os.path.join(d, fn)).items():
                    v["group"] = fn
                    res[d][k] = v
    json.dump(res, sys.stdout, indent=0)
