// kprof.so: LD_PRELOAD kernel-time profiler on CUPTI activity records (no Nsight needed; works on kernels inside CUDA graphs).
// Sums GPU time per kernel name and launch config; writes a table to $KPROF_OUT (default kprof.txt) at exit.
// KPROF_FROM_ENV=1 starts recording only after a file $KPROF_START exists (so load and prefill can be left out of a decode profile).
#include <cupti.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>
#include <algorithm>
#include <unistd.h>

struct Agg { double ns = 0; unsigned long n = 0; unsigned grid = 0, block = 0, shm = 0, regs = 0; };
static std::map<std::string, Agg> g_agg;
static std::mutex g_mu;
static bool g_recording = true;

static void CUPTIAPI bufReq(uint8_t **b, size_t *sz, size_t *maxRec) { *sz = 8 << 20; *b = (uint8_t *) aligned_alloc(8, *sz); *maxRec = 0; }
static void CUPTIAPI bufDone(CUcontext, uint32_t, uint8_t *b, size_t, size_t valid) {
    CUpti_Activity *rec = nullptr;
    std::lock_guard<std::mutex> lk(g_mu);
    while (cuptiActivityGetNextRecord(b, valid, &rec) == CUPTI_SUCCESS) {
        if (rec->kind != CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL && rec->kind != CUPTI_ACTIVITY_KIND_KERNEL) continue;
        auto *k = (CUpti_ActivityKernel13 *) rec;
        if (!g_recording) continue;
        char key[900]; snprintf(key, sizeof key, "%s", k->name ? k->name : "?");
        Agg &a = g_agg[key];
        a.ns += (double) (k->end - k->start); a.n++; a.grid = k->gridX * k->gridY * k->gridZ; a.block = k->blockX * k->blockY * k->blockZ;
        a.shm = k->staticSharedMemory + k->dynamicSharedMemory; a.regs = k->registersPerThread;
    }
    free(b);
}
static void startRec() { std::lock_guard<std::mutex> lk(g_mu); g_agg.clear(); g_recording = true; }
static void *watcher(void *) {   // optional gate: record from the moment $KPROF_START appears
    const char *f = getenv("KPROF_START"); if (!f) return nullptr;
    g_recording = false;
    while (access(f, F_OK) != 0) usleep(20000);
    cuptiActivityFlushAll(1); startRec(); return nullptr;
}
__attribute__((constructor)) static void init() {
    cuptiActivityRegisterCallbacks(bufReq, bufDone);
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    if (getenv("KPROF_START")) { pthread_t t; pthread_create(&t, nullptr, watcher, nullptr); }
}
__attribute__((destructor)) static void fini() {
    cuptiActivityFlushAll(1);
    std::vector<std::pair<std::string, Agg>> v(g_agg.begin(), g_agg.end());
    std::sort(v.begin(), v.end(), [](auto &a, auto &b) { return a.second.ns > b.second.ns; });
    double tot = 0; for (auto &p : v) tot += p.second.ns;
    const char *o = getenv("KPROF_OUT"); FILE *f = fopen(o ? o : "kprof.txt", "w");
    if (!f) return;
    fprintf(f, "total kernel time %.1f ms over %zu kernels\n", tot / 1e6, v.size());
    fprintf(f, "%% total   ms      calls   avg_us  grid   block regs shm  name\n");
    for (auto &p : v) {
        const Agg &a = p.second;
        fprintf(f, "%5.1f %9.2f %8lu %8.1f %6u %5u %4u %6u  %.150s\n", 100 * a.ns / tot, a.ns / 1e6, a.n, a.ns / 1e3 / a.n, a.grid, a.block, a.regs, a.shm, p.first.c_str());
    }
    fclose(f);
}
