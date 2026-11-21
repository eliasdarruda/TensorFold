// fakehip: a host-only stand-in for libamdhip64.so that presents one virtual gfx1100 card.
//
// It runs no kernels. It exists to drive TensorFold's real HIP host code (code object selection, kernel
// lookups, memory planning, the lane loop, DFlash2 attach) on a machine without a GPU, and to record every
// launch so the launches can be checked against the gfx1100 code objects' own limits afterwards.
//
//   FAKEHIP_ARCH       gfx name the card reports (default gfx1100)
//   FAKEHIP_NAME       marketing name (default "AMD Radeon RX 7900 XTX")
//   FAKEHIP_VRAM_GIB   device memory (default 24)
//   FAKEHIP_CUS        compute units (default 96)
//   FAKEHIP_LOG        file for the launch and lookup log (default fakehip.log)
//   FAKEHIP_COPY_MAX   largest host<->device copy or fill actually performed, bytes (default 1 MiB);
//                      larger ones are dropped so weights never commit host memory
#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <pthread.h>
#include <stdarg.h>

typedef int hipError_t;
enum { OK = 0, ERR_INVALID_VALUE = 1, ERR_OOM = 2, ERR_NO_BINARY = 209, ERR_INVALID_IMAGE = 200,
       ERR_NOT_FOUND = 500, ERR_INVALID_HANDLE = 400, ERR_LAUNCH = 719 };

static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static FILE *logf_;
static const char *arch = "gfx1100";
static const char *cardname = "AMD Radeon RX 7900 XTX";
static size_t vram_total, vram_used, copy_max = 1 << 20;
static int cus = 96;
static double clock_ms; // virtual time: launches and copies advance it

static void lg(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static void lg(const char *fmt, ...) {
    if (!logf_) return;
    va_list ap;
    va_start(ap, fmt);
    vfprintf(logf_, fmt, ap);
    va_end(ap);
    fputc('\n', logf_);
    fflush(logf_);
}

__attribute__((constructor)) static void init_env(void) {
    const char *e;
    if ((e = getenv("FAKEHIP_ARCH"))) arch = e;
    if ((e = getenv("FAKEHIP_NAME"))) cardname = e;
    double gib = (e = getenv("FAKEHIP_VRAM_GIB")) ? atof(e) : 24.0;
    vram_total = (size_t)(gib * 1073741824.0);
    if ((e = getenv("FAKEHIP_CUS"))) cus = atoi(e);
    if ((e = getenv("FAKEHIP_COPY_MAX"))) copy_max = strtoull(e, 0, 10);
    logf_ = fopen((e = getenv("FAKEHIP_LOG")) ? e : "fakehip.log", "a");
    lg("init arch=%s name=\"%s\" vram=%.1fGiB cus=%d", arch, cardname, gib, cus);
}

// ---------- device memory: virtual reservations, never committed unless small writes touch them

typedef struct Alloc { char *p; size_t n; struct Alloc *next; } Alloc;
static Alloc *allocs;

static Alloc *find_alloc(const void *p) {
    for (Alloc *a = allocs; a; a = a->next)
        if ((const char *)p >= a->p && (const char *)p < a->p + a->n) return a;
    return 0;
}

hipError_t hipMalloc(void **out, size_t n) {
    pthread_mutex_lock(&mu);
    size_t rounded = (n + 4095) & ~(size_t)4095;
    if (rounded == 0) rounded = 4096;
    if (vram_used + rounded > vram_total) {
        lg("malloc %zu FAILED: %zu of %zu used", n, vram_used, vram_total);
        pthread_mutex_unlock(&mu);
        return ERR_OOM;
    }
    void *p = mmap(0, rounded, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (p == MAP_FAILED) { pthread_mutex_unlock(&mu); return ERR_OOM; }
    Alloc *a = malloc(sizeof *a);
    *a = (Alloc){p, rounded, allocs};
    allocs = a;
    vram_used += rounded;
    *out = p;
    pthread_mutex_unlock(&mu);
    return OK;
}

hipError_t hipFree(void *p) {
    if (!p) return OK;
    pthread_mutex_lock(&mu);
    for (Alloc **pp = &allocs; *pp; pp = &(*pp)->next) {
        if ((*pp)->p == p) {
            Alloc *a = *pp;
            *pp = a->next;
            vram_used -= a->n;
            munmap(a->p, a->n);
            free(a);
            pthread_mutex_unlock(&mu);
            return OK;
        }
    }
    pthread_mutex_unlock(&mu);
    lg("free of unknown pointer %p", p);
    return ERR_INVALID_VALUE;
}

hipError_t hipMemGetInfo(size_t *free_, size_t *total) {
    *free_ = vram_total - vram_used;
    *total = vram_total;
    return OK;
}

hipError_t hipHostMalloc(void **out, size_t n, unsigned flags) {
    (void)flags;
    *out = calloc(1, n ? n : 1);
    return *out ? OK : ERR_OOM;
}
hipError_t hipHostFree(void *p) { free(p); return OK; }
hipError_t hipHostGetDevicePointer(void **d, void *h, unsigned f) { (void)f; *d = h; return OK; }

static void advance_copy(size_t n) { clock_ms += 0.005 + (double)n / 900e9 * 1e3; }

static hipError_t copy(void *dst, const void *src, size_t n, const char *kind) {
    pthread_mutex_lock(&mu);
    advance_copy(n);
    pthread_mutex_unlock(&mu);
    if (n <= copy_max) memcpy(dst, src, n);
    else if (strcmp(kind, "DtoH") == 0) memset(dst, 0, n); // the host sees zeros for a dropped region
    return OK;
}
hipError_t hipMemcpyHtoD(void *d, const void *h, size_t n) { return copy(d, h, n, "HtoD"); }
hipError_t hipMemcpyHtoDAsync(void *d, const void *h, size_t n, void *s) { (void)s; return copy(d, h, n, "HtoD"); }
hipError_t hipMemcpyDtoH(void *h, const void *d, size_t n) { return copy(h, d, n, "DtoH"); }
hipError_t hipMemcpyDtoHAsync(void *h, const void *d, size_t n, void *s) { (void)s; return copy(h, d, n, "DtoH"); }
hipError_t hipMemcpyDtoD(void *d, const void *s, size_t n) { return copy(d, s, n, "DtoD"); }
hipError_t hipMemcpyDtoDAsync(void *d, const void *s, size_t n, void *st) { (void)st; return copy(d, s, n, "DtoD"); }
hipError_t hipMemset(void *d, int v, size_t n) { if (n <= copy_max) memset(d, v, n); return OK; }
hipError_t hipMemsetAsync(void *d, int v, size_t n, void *s) { (void)s; return hipMemset(d, v, n); }
hipError_t hipMemsetD32(void *d, int v, size_t n) {
    if (n * 4 <= copy_max) for (size_t i = 0; i < n; i++) ((int *)d)[i] = v;
    return OK;
}
hipError_t hipMemsetD32Async(void *d, int v, size_t n, void *s) { (void)s; return hipMemsetD32(d, v, n); }

// ---------- device, context, streams, events

hipError_t hipInit(unsigned f) { return f == 0 ? OK : ERR_INVALID_VALUE; }
hipError_t hipGetDeviceCount(int *n) { *n = 1; return OK; }
hipError_t hipDeviceGet(int *d, int i) { if (i != 0) return ERR_INVALID_VALUE; *d = 0; return OK; }
static int ctx_token;
hipError_t hipDevicePrimaryCtxRetain(void **c, int d) { (void)d; *c = &ctx_token; return OK; }
hipError_t hipDevicePrimaryCtxRelease(int d) { (void)d; return OK; }
hipError_t hipCtxSetCurrent(void *c) { (void)c; return OK; }
hipError_t hipDeviceSynchronize(void) { return OK; }
hipError_t hipRuntimeGetVersion(int *v) { *v = 70253211; return OK; }
hipError_t hipDeviceGetName(char *out, int n, int d) { (void)d; snprintf(out, n, "%s", cardname); return OK; }

hipError_t hipGetDevicePropertiesR0600(void *props, int d) {
    if (d != 0) return ERR_INVALID_VALUE;
    char *p = props;
    snprintf(p, 256, "%s", cardname);
    // gcnArchName: TensorFold finds it by its "gfx" prefix past the first 256 bytes
    snprintf(p + 1160, 256, "%s", arch);
    return OK;
}

hipError_t hipDeviceGetAttribute(int *v, int attr, int d) {
    (void)d;
    switch (attr) {
    case 5: *v = 2500000; break;      // clock rate, kHz
    case 10: *v = 1; break;           // cooperative launch
    case 16: *v = 0; break;           // integrated: a discrete card
    case 19: *v = 6 << 20; break;     // L2
    case 56: *v = 1024; break;        // max threads per block
    case 63: *v = cus; break;         // compute units
    case 67: *v = 3; break;
    case 68: *v = 0; break;
    case 69: *v = 0; break;
    case 74: *v = 65536; break;       // LDS a workgroup
    case 75: *v = 65536; break;
    case 87: *v = 32; break;          // wave32
    case 10002: *v = 65536; break;
    default: *v = 0; lg("device attribute %d asked: 0", attr);
    }
    return OK;
}

typedef struct { double t; } Ev;
hipError_t hipStreamCreateWithFlags(void **s, unsigned f) { (void)f; *s = malloc(8); return OK; }
hipError_t hipStreamDestroy(void *s) { free(s); return OK; }
hipError_t hipStreamSynchronize(void *s) { (void)s; return OK; }
hipError_t hipStreamQuery(void *s) { (void)s; return OK; }
hipError_t hipStreamWaitEvent(void *s, void *e, unsigned f) { (void)s; (void)e; (void)f; return OK; }
hipError_t hipEventCreateWithFlags(void **e, unsigned f) { (void)f; *e = calloc(1, sizeof(Ev)); return OK; }
hipError_t hipEventDestroy(void *e) { free(e); return OK; }
hipError_t hipEventRecord(void *e, void *s) { (void)s; ((Ev *)e)->t = clock_ms; return OK; }
hipError_t hipEventSynchronize(void *e) { (void)e; return OK; }
hipError_t hipEventQuery(void *e) { (void)e; return OK; }
hipError_t hipEventElapsedTime(float *ms, void *a, void *b) { *ms = (float)(((Ev *)b)->t - ((Ev *)a)->t); return OK; }

static const char *names[] = {"hipSuccess", "hipErrorInvalidValue", "hipErrorOutOfMemory"};
const char *hipGetErrorName(hipError_t e) {
    if (e >= 0 && e < 3) return names[e];
    switch (e) {
    case ERR_NO_BINARY: return "hipErrorNoBinaryForGpu";
    case ERR_INVALID_IMAGE: return "hipErrorInvalidImage";
    case ERR_NOT_FOUND: return "hipErrorNotFound";
    case ERR_INVALID_HANDLE: return "hipErrorInvalidHandle";
    case ERR_LAUNCH: return "hipErrorLaunchFailure";
    default: return "hipErrorUnknown";
    }
}

// ---------- code objects: an offload bundle's (or a bare ELF's) entry for exactly this arch

typedef struct Kernel {
    char name[256];
    uint32_t lds, scratch, kernarg, vgprs, sgprs;
    unsigned long launches;
    unsigned max_block, max_dyn;
    struct Module *mod;
} Kernel;
typedef struct Global { char name[256]; size_t size; void *dev; } Global;
typedef struct Module { int id; Kernel *k; int nk; Global *g; int ng; } Module;
static int modules_loaded;
static Module *mods[512];

#define EM_AMDGPU 224
#define EF_AMDGPU_MACH 0x0ff
static unsigned mach_of(const char *a) {
    static const struct { const char *n; unsigned m; } t[] = {
        {"gfx1100", 0x041}, {"gfx1101", 0x046}, {"gfx1102", 0x047}, {"gfx1103", 0x044},
        {"gfx1150", 0x043}, {"gfx1151", 0x04a}, {"gfx1152", 0x055}, {"gfx1153", 0x058},
        {"gfx1030", 0x036}, {"gfx1200", 0x048}, {"gfx1201", 0x04e}};
    for (unsigned i = 0; i < sizeof t / sizeof *t; i++) if (!strcmp(t[i].n, a)) return t[i].m;
    return 0;
}

typedef struct { uint8_t ident[16]; uint16_t type, machine; uint32_t version; uint64_t entry, phoff, shoff;
                 uint32_t flags; uint16_t ehsize, phentsize, phnum, shentsize, shnum, shstrndx; } Ehdr;
typedef struct { uint32_t name, type; uint64_t flags, addr, offset, size; uint32_t link, info; uint64_t addralign, entsize; } Shdr;
typedef struct { uint32_t name; uint8_t info, other; uint16_t shndx; uint64_t value, size; } Sym;

static const uint8_t *select_image(const uint8_t *img, size_t *len_out) {
    if (!memcmp(img, "\x7f" "ELF", 4)) { *len_out = 0; return img; }
    if (memcmp(img, "__CLANG_OFFLOAD_BUNDLE__", 24)) return 0;
    uint64_t n;
    memcpy(&n, img + 24, 8);
    const uint8_t *p = img + 32;
    for (uint64_t i = 0; i < n; i++) {
        uint64_t off, size, idlen;
        memcpy(&off, p, 8); memcpy(&size, p + 8, 8); memcpy(&idlen, p + 16, 8);
        char id[256] = {0};
        memcpy(id, p + 24, idlen < 255 ? idlen : 255);
        p += 24 + idlen;
        const char *dash = strstr(id, "amdgcn-amd-amdhsa--");
        if (!dash) continue;
        const char *a = dash + strlen("amdgcn-amd-amdhsa--");
        size_t al = strcspn(a, ":");
        if (al == strlen(arch) && !strncmp(a, arch, al)) { *len_out = size; return img + off; }
        lg("bundle entry %s skipped", id);
    }
    return 0;
}

hipError_t hipModuleLoadData(void **out, const void *image) {
    size_t len;
    const uint8_t *elf = select_image(image, &len);
    if (!elf) { lg("module: no code object for %s in the image", arch); return ERR_NO_BINARY; }
    const Ehdr *eh = (const Ehdr *)elf;
    if (eh->machine != EM_AMDGPU) return ERR_INVALID_IMAGE;
    unsigned mach = eh->flags & EF_AMDGPU_MACH;
    if (mach != mach_of(arch)) { lg("module: ELF machine 0x%x is not %s", mach, arch); return ERR_NO_BINARY; }
    const Shdr *sh = (const Shdr *)(elf + eh->shoff);
    Module *m = calloc(1, sizeof *m);
    m->id = ++modules_loaded;
    if (m->id < 512) mods[m->id] = m;
    for (int s = 0; s < eh->shnum; s++) {
        if (sh[s].type != 2 /*SHT_SYMTAB*/) continue;
        const Sym *sym = (const Sym *)(elf + sh[s].offset);
        const char *str = (const char *)(elf + sh[sh[s].link].offset);
        int count = sh[s].size / sizeof(Sym);
        m->k = calloc(count, sizeof(Kernel));
        m->g = calloc(count, sizeof(Global));
        for (int i = 0; i < count; i++) {
            const char *nm = str + sym[i].name;
            size_t nl = strlen(nm);
            int bind = sym[i].info >> 4, type = sym[i].info & 0xf;
            if (nl > 3 && !strcmp(nm + nl - 3, ".kd") && sym[i].shndx && sym[i].shndx < eh->shnum) {
                Kernel *k = &m->k[m->nk++];
                snprintf(k->name, sizeof k->name, "%.*s", (int)(nl - 3), nm);
                const Shdr *in = &sh[sym[i].shndx];
                const uint8_t *kd = elf + in->offset + (sym[i].value - in->addr);
                memcpy(&k->lds, kd + 0, 4);
                memcpy(&k->scratch, kd + 4, 4);
                memcpy(&k->kernarg, kd + 8, 4);
                uint32_t rsrc1;
                memcpy(&rsrc1, kd + 0x30, 4);
                k->vgprs = ((rsrc1 & 0x3f) + 1) * 8; // gfx11 wave32: granules of 8
                k->mod = m;
            } else if (type == 1 /*STT_OBJECT*/ && bind == 1 /*GLOBAL*/ && sym[i].size) {
                Global *g = &m->g[m->ng++];
                snprintf(g->name, sizeof g->name, "%s", nm);
                g->size = sym[i].size;
            }
        }
    }
    lg("module %d: %d kernels, %d globals for %s", m->id, m->nk, m->ng, arch);
    *out = m;
    return OK;
}

hipError_t hipModuleUnload(void *mod) { (void)mod; return OK; }

hipError_t hipModuleGetFunction(void **f, void *mod, const char *name) {
    Module *m = mod;
    for (int i = 0; i < m->nk; i++)
        if (!strcmp(m->k[i].name, name)) { *f = &m->k[i]; return OK; }
    lg("MISSING kernel %s in module %d", name, m->id);
    return ERR_NOT_FOUND;
}

hipError_t hipModuleGetGlobal(void **d, size_t *size, void *mod, const char *name) {
    Module *m = mod;
    for (int i = 0; i < m->ng; i++)
        if (!strcmp(m->g[i].name, name)) {
            if (!m->g[i].dev) m->g[i].dev = calloc(1, m->g[i].size);
            if (d) *d = m->g[i].dev;
            if (size) *size = m->g[i].size;
            return OK;
        }
    lg("MISSING global %s in module %d", name, m->id);
    return ERR_NOT_FOUND;
}

hipError_t hipFuncGetAttribute(int *v, int attr, void *f) {
    Kernel *k = f;
    switch (attr) {
    case 0: *v = 1024; break;
    case 1: *v = k->lds; break;
    case 2: *v = 0; break;
    case 3: *v = k->scratch; break;
    case 4: *v = k->vgprs; break;
    case 8: *v = 65536 - k->lds; break;
    default: *v = 0;
    }
    return OK;
}

hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(int *n, void *f, int block, size_t dyn) {
    Kernel *k = f;
    // a gfx1100 CU pair holds 64 KiB of LDS a workgroup and 1536 VGPRs a lane a SIMD (wave32): bound by both
    int waves = (block + 31) / 32;
    int by_vgpr = k->vgprs ? (1536 / k->vgprs) * 2 / (waves ? waves : 1) : 16;
    size_t lds = k->lds + dyn;
    int by_lds = lds ? (int)(65536 / lds) : 16;
    *n = by_vgpr < by_lds ? by_vgpr : by_lds;
    if (*n < 1) *n = 1;
    return OK;
}

static hipError_t launch(Kernel *k, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by,
                         unsigned bz, unsigned shared, int coop) {
    if (!k) return ERR_INVALID_HANDLE;
    unsigned block = bx * by * bz;
    pthread_mutex_lock(&mu);
    k->launches++;
    if (block > k->max_block) k->max_block = block;
    if (shared > k->max_dyn) k->max_dyn = shared;
    clock_ms += 0.02;
    pthread_mutex_unlock(&mu);
    if (block == 0 || block > 1024 || gx == 0 || gy == 0 || gz == 0) {
        lg("BAD launch %s grid %ux%ux%u block %ux%ux%u", k->name, gx, gy, gz, bx, by, bz);
        return ERR_INVALID_VALUE;
    }
    if (k->lds + shared > 65536) {
        lg("BAD launch %s: %u static + %u dynamic LDS bytes past gfx11's 65536", k->name, k->lds, shared);
        return ERR_LAUNCH;
    }
    (void)coop;
    return OK;
}

static __thread struct Graph *capture_graph;
static __thread int capture_status;
static void capture_add(void *f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz, unsigned shared);
hipError_t hipModuleLaunchKernel(void *f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by,
                                 unsigned bz, unsigned shared, void *s, void **params, void **extra) {
    (void)s; (void)params; (void)extra;
    if (capture_status) { capture_add(f, gx, gy, gz, bx, by, bz, shared); return OK; }
    return launch(f, gx, gy, gz, bx, by, bz, shared, 0);
}
hipError_t hipModuleLaunchCooperativeKernel(void *f, unsigned gx, unsigned gy, unsigned gz, unsigned bx,
                                            unsigned by, unsigned bz, unsigned shared, void *s, void **params) {
    (void)s; (void)params;
    return launch(f, gx, gy, gz, bx, by, bz, shared, 1);
}

// ---------- graphs: kept as lists of kernel launches, replayed as launches

typedef struct { unsigned block[3]; void **extra; void *func; unsigned grid[3]; void **params; unsigned shared; } NodeParams;
typedef struct Node { NodeParams p; } Node;
typedef struct Graph { Node **nodes; size_t n, cap; } Graph;
typedef struct { int capturing; Graph *g; } StreamState;

static Node *add_node(Graph *g, const NodeParams *p) {
    if (g->n == g->cap) { g->cap = g->cap ? g->cap * 2 : 64; g->nodes = realloc(g->nodes, g->cap * sizeof *g->nodes); }
    Node *n = malloc(sizeof *n);
    n->p = *p;
    g->nodes[g->n++] = n;
    return n;
}
static void capture_add(void *f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz, unsigned shared) {
    NodeParams p = {{bx, by, bz}, 0, f, {gx, gy, gz}, 0, shared};
    add_node(capture_graph, &p);
}
hipError_t hipStreamBeginCapture(void *s, int mode) { (void)s; (void)mode; capture_graph = calloc(1, sizeof(Graph)); capture_status = 1; return OK; }
hipError_t hipStreamEndCapture(void *s, void **g) { (void)s; *g = capture_graph; capture_graph = 0; capture_status = 0; return OK; }
hipError_t hipStreamIsCapturing(void *s, int *st) { (void)s; *st = capture_status; return OK; }
hipError_t hipGraphCreate(void **g, unsigned f) { (void)f; *g = calloc(1, sizeof(Graph)); return OK; }
hipError_t hipGraphDestroy(void *g) { (void)g; return OK; }
hipError_t hipGraphAddKernelNode(void **node, void *g, void *const *deps, size_t nd, const NodeParams *p) {
    (void)deps; (void)nd;
    *node = add_node(g, p);
    return OK;
}
hipError_t hipGraphKernelNodeSetParams(void *node, const NodeParams *p) { ((Node *)node)->p = *p; return OK; }
hipError_t hipGraphAddDependencies(void *g, void *const *a, void *const *b, size_t n) { (void)g; (void)a; (void)b; (void)n; return OK; }
hipError_t hipGraphGetNodes(void *g, void **nodes, size_t *n) {
    Graph *gr = g;
    if (nodes) for (size_t i = 0; i < *n && i < gr->n; i++) nodes[i] = gr->nodes[i];
    *n = gr->n;
    return OK;
}
typedef struct { Graph *g; } Exec;
hipError_t hipGraphInstantiateWithFlags(void **e, void *g, unsigned long long f) {
    (void)f;
    Exec *x = malloc(sizeof *x);
    x->g = g;
    *e = x;
    return OK;
}
hipError_t hipGraphUpload(void *e, void *s) { (void)e; (void)s; return OK; }
hipError_t hipGraphLaunch(void *e, void *s) {
    (void)s;
    Graph *g = ((Exec *)e)->g;
    for (size_t i = 0; i < g->n; i++) {
        NodeParams *p = &g->nodes[i]->p;
        hipError_t r = launch(p->func, p->grid[0], p->grid[1], p->grid[2], p->block[0], p->block[1], p->block[2], p->shared, 0);
        if (r) return r;
    }
    return OK;
}
hipError_t hipGraphExecDestroy(void *e) { free(e); return OK; }
hipError_t hipGraphExecUpdate(void *e, void *g, void **err_node, int *result) {
    ((Exec *)e)->g = g;
    if (err_node) *err_node = 0;
    *result = 0;
    return OK;
}
hipError_t hipGraphExecKernelNodeSetParams(void *e, void *node, const NodeParams *p) { (void)e; ((Node *)node)->p = *p; return OK; }

// Every kernel a module held, with its launches and its widest block and dynamic LDS, at exit.
__attribute__((destructor)) static void report(void) {
    for (int i = 1; i <= modules_loaded && i < 512; i++) {
        Module *m = mods[i];
        for (int j = 0; j < m->nk; j++) {
            Kernel *k = &m->k[j];
            if (k->launches) lg("K %s launches=%lu block=%u dyn=%u lds=%u scratch=%u vgprs=%u", k->name, k->launches, k->max_block, k->max_dyn, k->lds, k->scratch, k->vgprs);
        }
    }
    lg("exit: %.1f GiB device memory still held, virtual clock %.1f ms", vram_used / 1073741824.0, clock_ms);
}

