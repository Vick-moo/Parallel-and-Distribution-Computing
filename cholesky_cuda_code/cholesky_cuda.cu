// =============================================================================
//  cholesky_cuda.cu
//
//  AMCS2103 Parallel and Distributed Computing
//  Computational Problem : Cholesky Decomposition   A = L * L^T
//  Implementation        : CUDA (simple per-column kernels, single GPU)
//
//  Build (command line)  : nvcc -O3 -arch=sm_89 -o cholesky_cuda.exe cholesky_cuda.cu
//  Build (Visual Studio) : new CUDA project, add this .cu, build Release | x64
//
//  Run:
//      cholesky_cuda.exe              <-- 7 matrix sizes, 5 repetitions each
//      cholesky_cuda.exe 10           <-- 7 matrix sizes, 10 repetitions each
//
//  Method (right-looking, mirrors the sequential maths):
//      for k = 0 .. n-1
//          (1) L[k][k] = sqrt(L[k][k])            -- 1 thread
//          (2) L[i][k] /= L[k][k]  for i > k       -- 1D kernel over rows
//          (3) L[i][j] -= L[i][k]*L[j][k]          -- 2D kernel over the
//              for j > k, i >= j                     trailing lower triangle
//  The matrix starts as a copy of A; after step (3) of column k the trailing
//  submatrix holds the Schur complement, so column k+1 is ready.
//
//  Timing uses the SAME chrono steady_clock method as the sequential code:
//  the timed region covers the kernel launches + final synchronisation only.
//  Matrix generation, host<->device copies and verification stay outside,
//  exactly like the sequential version keeps generation/verify outside.
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <vector>
#include <chrono>
#include <algorithm>

#include <cuda_runtime.h>

using namespace std;

// =============================================================================
//  [1] Timer (identical to sequential)
// =============================================================================
static double now_sec()
{
    using namespace chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// =============================================================================
//  [2] Reproducible RNG (xorshift64*, bit-identical to sequential)
//  IMPORTANT: seed must match the sequential version so both factorise the
//  same matrices and any speedup figure is meaningful.
// =============================================================================
static uint64_t g_state = 12345ULL;     // must match the sequential version

static inline uint64_t rng_u64()
{
    uint64_t x = g_state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    g_state = x;
    return x * 2685821657736338717ULL;
}

static inline double rng_unit()
{
    return (double)(rng_u64() >> 11) * (1.0 / 9007199254740992.0);
}

// =============================================================================
//  [3] Generate a guaranteed positive-definite matrix (same as sequential)
// =============================================================================
static void make_positive_definite(int n, vector<double>& A)
{
    A.assign((size_t)n * (size_t)n, 0.0);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
            double v = rng_unit() * 2.0 - 1.0;
            A[(size_t)i * n + j] = v;
            A[(size_t)j * n + i] = v;
        }
    }

    for (int i = 0; i < n; i++) {
        double rowsum = 0.0;
        for (int j = 0; j < n; j++) {
            if (j != i) rowsum += fabs(A[(size_t)i * n + j]);
        }
        A[(size_t)i * n + i] = rowsum + 1.0;
    }
}

// =============================================================================
//  [4] CUDA kernels (one column per step)
// =============================================================================

// (1) Diagonal: single thread, because column k+1 must wait for it anyway.
__global__ static void k_diag(double* L, int n, int k, int* fail)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        double d = L[(size_t)k * n + k];
        if (d <= 0.0) {
            *fail = 1;          // mirrors the sequential "return false" guard
        }
        else {
            L[(size_t)k * n + k] = sqrt(d);
        }
    }
}

// (2) Scale the column below the diagonal: one thread per row i > k.
__global__ static void k_scale(double* L, int n, int k)
{
    int i = k + 1 + blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        L[(size_t)i * n + k] /= L[(size_t)k * n + k];
    }
}

// (3) Rank-1 update of the trailing lower triangle:
//     thread (j, i) handles j in (k, n), i in (j, n).
__global__ static void k_update(double* L, int n, int k)
{
    int j = k + 1 + blockIdx.x * blockDim.x + threadIdx.x;
    int i = j     + blockIdx.y * blockDim.y + threadIdx.y;
    if (j < n && i < n) {
        L[(size_t)i * n + j] -= L[(size_t)i * n + k] * L[(size_t)j * n + k];
    }
}

// Factorise in place on device. Returns false on non-SPD input or CUDA error.
static bool cholesky_cuda_factor(int n, double* dL)
{
    int h_fail = 0;
    int* d_fail = nullptr;
    if (cudaMalloc(&d_fail, sizeof(int)) != cudaSuccess) return false;
    if (cudaMemset(d_fail, 0, sizeof(int)) != cudaSuccess) {
        cudaFree(d_fail);
        return false;
    }

    const int B1 = 256;
    dim3 b2(16, 16);

    for (int k = 0; k < n; k++) {
        k_diag<<<1, 1>>>(dL, n, k, d_fail);

        int rows = n - k - 1;
        if (rows > 0) {
            int g1 = (rows + B1 - 1) / B1;
            k_scale<<<g1, B1>>>(dL, n, k);

            // j in [k+1, n): width W; i in [j, n): triangular rows
            int W = n - k - 1;
            dim3 g2((W + b2.x - 1) / b2.x, (W + b2.y - 1) / b2.y);
            k_update<<<g2, b2>>>(dL, n, k);
        }
    }

    // End of timed region boundary: wait for every column kernel.
    if (cudaDeviceSynchronize() != cudaSuccess) {
        cudaFree(d_fail);
        return false;
    }
    if (cudaMemcpy(&h_fail, d_fail, sizeof(int),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        cudaFree(d_fail);
        return false;
    }
    cudaFree(d_fail);
    return h_fail == 0;
}

// =============================================================================
//  [5] Correctness check: max |A - L*L^T| (same as sequential, runs on host)
// =============================================================================
static const double REL_TOL = 1e-9;

static double verify_factor(int n, const vector<double>& A, const vector<double>& L)
{
    double max_abs = 0.0;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double s = 0.0;
            for (int k = 0; k <= j; k++) {
                s += L[(size_t)i * n + k] * L[(size_t)j * n + k];
            }
            double e = fabs(s - A[(size_t)i * n + j]);
            if (e > max_abs) max_abs = e;
        }
    }

    return max_abs;
}

// =============================================================================
//  [6] Result of one experiment
// =============================================================================
struct Result {
    int    n;
    double best_ms;
    double avg_ms;
    double a_max;
    double current_err;
    bool   ok;
};

// =============================================================================
//  [7] Run one matrix size
// =============================================================================
static Result run_one(int n, int reps)
{
    Result r;
    r.n           = n;
    r.best_ms     = 0.0;
    r.avg_ms      = 0.0;
    r.a_max       = 0.0;
    r.current_err = 0.0;
    r.ok          = true;

    vector<double> A, L;
    make_positive_definite(n, A);
    L.assign((size_t)n * (size_t)n, 0.0);

    for (size_t i = 0; i < A.size(); i++) {
        double v = fabs(A[i]);
        if (v > r.a_max) r.a_max = v;
    }

    double* dW = nullptr;
    if (cudaMalloc(&dW, (size_t)n * (size_t)n * sizeof(double)) != cudaSuccess) {
        r.ok = false;
        return r;
    }
    // Initial upload, outside the timed region (like generation in sequential).
    if (cudaMemcpy(dW, A.data(), (size_t)n * (size_t)n * sizeof(double),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(dW);
        r.ok = false;
        return r;
    }

    // Warm-up: one untimed run (same reason as sequential).
    if (!cholesky_cuda_factor(n, dW)) {
        cudaFree(dW);
        r.ok = false;
        return r;
    }

    double t_best = 0.0;
    double t_sum  = 0.0;

    for (int rep = 0; rep < reps; rep++) {
        // Restore the pristine copy outside the timed region,
        // mirroring sequential's fill(L, 0) placement.
        cudaMemcpy(dW, A.data(), (size_t)n * (size_t)n * sizeof(double),
                   cudaMemcpyHostToDevice);

        double t0 = now_sec();

        bool ok = cholesky_cuda_factor(n, dW);

        double dt = now_sec() - t0;

        if (!ok) {
            cudaFree(dW);
            r.ok = false;
            return r;
        }

        t_sum += dt;
        if (rep == 0 || dt < t_best) t_best = dt;
    }

    // Bring the result back and check it, outside the timed region.
    if (cudaMemcpy(L.data(), dW, (size_t)n * (size_t)n * sizeof(double),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        cudaFree(dW);
        r.ok = false;
        return r;
    }
    cudaFree(dW);

    r.current_err = verify_factor(n, A, L);

    r.best_ms = t_best * 1000.0;
    r.avg_ms  = (t_sum / (double)reps) * 1000.0;

    return r;
}

// =============================================================================
//  [8] Print the results table
// =============================================================================
static void print_table(const Result* r, int count, int base_n)
{
    const char* sep =
        "+----------------------+--------------+---------------+---------------+----------------+-----------------+----------+";

    printf("\n%s\n", sep);
    printf("| %-20s | %-12s | %-13s | %-13s | %-14s | %-15s | %-8s |\n",
           "Reason for Selection", "Matrix Size", "Best Tp (ms)", "Average Tp (ms)",
           "Current Error", "Max Acceptable", "Status");
    printf("%s\n", sep);

    for (int i = 0; i < count; i++) {
        char sizebuf[32];
        char reason[64];

        snprintf(sizebuf, sizeof(sizebuf), "%d x %d", r[i].n, r[i].n);

        double ratio = (double)r[i].n / (double)base_n;
        if (i == 0) snprintf(reason, sizeof(reason), "Baseline");
        else        snprintf(reason, sizeof(reason), "%gx scale up", ratio);

        if (!r[i].ok) {
            printf("| %-20s | %-12s | %13s | %13s | %14s | %15s | %-8s |\n",
                   reason, sizebuf, "FAILED", "FAILED", "-", "-", "FAIL");
            continue;
        }

        double acceptable = REL_TOL * r[i].a_max;
        const char* status = (r[i].current_err <= acceptable) ? "PASS" : "FAIL";

        printf("| %-20s | %-12s | %13.3f | %13.3f | %14.3e | %15.3e | %-8s |\n",
               reason, sizebuf, r[i].best_ms, r[i].avg_ms,
               r[i].current_err, acceptable, status);
    }

    printf("%s\n", sep);
}

// =============================================================================
//  main
// =============================================================================
int main(int argc, char** argv)
{
    const int sizes[] = { 1000, 1500, 2000, 2500, 3000, 3500, 4000 };
    const int nSizes  = (int)(sizeof(sizes) / sizeof(sizes[0]));

    int reps = 5;
    if (argc > 1) reps = atoi(argv[1]);

    if (reps < 1) {
        printf("usage: %s [repetitions]\n", argv[0]);
        return 1;
    }

    int dev = 0;
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess) {
        printf("CUDA device: %s (cc %d.%d)\n", prop.name, prop.major, prop.minor);
        cudaSetDevice(dev);
    }
    else {
        printf("WARNING: no CUDA device found, will likely fail.\n");
    }

    vector<Result> results(nSizes);

    printf("CUDA Cholesky (simple per-column kernels)\n");
    printf("Matrix sizes: %d   Repetitions per size: %d\n\n", nSizes, reps);

    for (int s = 0; s < nSizes; s++) {
        printf("[%d/%d] n = %-5d ... ", s + 1, nSizes, sizes[s]);
        fflush(stdout);

        results[s] = run_one(sizes[s], reps);

        if (!results[s].ok) {
            printf("FAILED (matrix is not positive definite or CUDA error)\n");
        }
        else {
            printf("best %9.3f ms   avg %9.3f ms   max|A-LLt| = %.3e\n",
                   results[s].best_ms, results[s].avg_ms, results[s].current_err);
        }
    }

    print_table(results.data(), nSizes, sizes[0]);

    return 0;
}
