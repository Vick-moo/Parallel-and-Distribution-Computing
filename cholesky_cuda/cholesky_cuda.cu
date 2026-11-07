// =============================================================================
//  cholesky_cuda.cu
//
//  AMCS2103 Parallel and Distributed Computing
//  Computational Problem : Cholesky Decomposition   A = L * L^T
//  Implementation        : CUDA (column-oriented, right-looking)
//
//  Build:
//      nvcc -O3 -std=c++17 -arch=sm_89 -o cholesky_cuda.exe cholesky_cuda.cu
//
//  Run:
//      cholesky_cuda.exe            <-- 5 matrix sizes, 5 repetitions each
//      cholesky_cuda.exe 10         <-- 5 matrix sizes, 10 repetitions each
//
//  The CPU reference and the GPU version use the same RNG seed, so both
//  factorise exactly the same matrix. Every result is checked against
//  A = L * L^T before it is reported.
//
//  Dependences (Chapter 8, Bernstein conditions):
//    - Row i depends on rows < i, so the outer loop over i is NOT parallel.
//    - Inside one column j:
//        L[i][j] = A[i][j] / L[j][j]        for i > j   -> independent
//        A[i][k] -= L[i][j] * L[k][j]       for i,k > j -> independent
//      Both loops satisfy Bernstein's conditions and run on the GPU.
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

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess) {                                            \
            printf("CUDA error: %s at %s:%d\n",                               \
                   cudaGetErrorString(err_), __FILE__, __LINE__);             \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

// =============================================================================
//  [1] Timer
// =============================================================================
static double now_sec()
{
    using namespace chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// =============================================================================
//  [2] Reproducible RNG (must match the CPU baseline: seed 12345)
// =============================================================================
static uint64_t g_state = 12345ULL;

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
//  [3] Guaranteed positive-definite matrix (same construction as the baseline)
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
//  [4] Correctness check: max |A - L*L^T|
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
//  [5] CPU reference (same algorithm as the sequential baseline)
// =============================================================================
static double cholesky_cpu(int n, const vector<double>& A, vector<double>& L)
{
    fill(L.begin(), L.end(), 0.0);

    double t0 = now_sec();

    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = 0;
            for (int k = 0; k < j; k++) {
                sum += L[(size_t)i * n + k] * L[(size_t)j * n + k];
            }
            if (i == j) {
                L[(size_t)i * n + j] = sqrt(A[(size_t)i * n + j] - sum);
            }
            else {
                L[(size_t)i * n + j] =
                    (A[(size_t)i * n + j] - sum) / L[(size_t)j * n + j];
            }
        }
    }

    return now_sec() - t0;
}

// =============================================================================
//  [6] CUDA kernels -- one column at a time
// =============================================================================
//  Column j:
//    1. L[j][j] = sqrt(A[j][j])
//    2. L[i][j] = A[i][j] / L[j][j]              for i > j
//    3. A[i][k] -= L[i][j] * L[k][j]             for i,k > j
//
//  A[i][j] has already been updated by all previous columns, so no extra
//  dot product is needed here (right-looking form).
// =============================================================================

__global__ void kernel_diag(const double* __restrict__ A,
                            double* __restrict__ L,
                            int n, int j)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        double d = A[(size_t)j * n + j];
        L[(size_t)j * n + j] = sqrt(d);
    }
}

__global__ void kernel_div(const double* __restrict__ A,
                           double* __restrict__ L,
                           int n, int j)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + j + 1;

    if (i < n) {
        L[(size_t)i * n + j] =
            A[(size_t)i * n + j] / L[(size_t)j * n + j];
    }
}

//  threadIdx.x -> column k (contiguous -> coalesced stores to A[i][k])
//  threadIdx.y -> row i
__global__ void kernel_update(double* __restrict__ A,
                              const double* __restrict__ L,
                              int n, int j)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x + j + 1;
    int i = blockIdx.y * blockDim.y + threadIdx.y + j + 1;

    if (i < n && k < n) {
        A[(size_t)i * n + k] -= L[(size_t)i * n + j] * L[(size_t)k * n + j];
    }
}

// =============================================================================
//  [7] Launch the whole factorisation (host side)
// =============================================================================
static void cholesky_gpu_run(int n, double* dA, double* dL)
{
    for (int j = 0; j < n; j++) {

        kernel_diag<<<1, 1>>>(dA, dL, n, j);

        int rem = n - j - 1;
        if (rem <= 0) continue;

        // Divide the column below the diagonal.
        {
            const int threads = 256;
            int blocks = (rem + threads - 1) / threads;
            kernel_div<<<blocks, threads>>>(dA, dL, n, j);
        }

        // Rank-1 update of the trailing submatrix.
        {
            dim3 block(16, 16);
            dim3 grid((rem + block.x - 1) / block.x,
                      (rem + block.y - 1) / block.y);
            kernel_update<<<grid, block>>>(dA, dL, n, j);
        }
    }
}

// =============================================================================
//  [8] Result of one experiment
// =============================================================================
struct Result {
    int    n;
    double cpu_ms;       // CPU reference (one timed run)
    double gpu_best_ms;  // fastest GPU run
    double gpu_avg_ms;   // average GPU run
    double a_max;
    double current_err;
    bool   ok;
};

// =============================================================================
//  [9] Run one matrix size
// =============================================================================
static Result run_one(int n, int reps)
{
    Result r;
    r.n            = n;
    r.cpu_ms       = 0.0;
    r.gpu_best_ms  = 0.0;
    r.gpu_avg_ms   = 0.0;
    r.a_max        = 0.0;
    r.current_err  = 0.0;
    r.ok           = true;

    vector<double> A, L;
    make_positive_definite(n, A);
    L.assign((size_t)n * (size_t)n, 0.0);

    for (size_t i = 0; i < A.size(); i++) {
        double v = fabs(A[i]);
        if (v > r.a_max) r.a_max = v;
    }

    // ---- CPU reference: one warm-up run, then one timed run ----
    cholesky_cpu(n, A, L);
    r.cpu_ms = cholesky_cpu(n, A, L) * 1000.0;

    // ---- GPU ----
    const size_t bytes = (size_t)n * (size_t)n * sizeof(double);
    double* dA = nullptr;
    double* dL = nullptr;

    CUDA_CHECK(cudaMalloc((void**)&dA, bytes));
    CUDA_CHECK(cudaMalloc((void**)&dL, bytes));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warm-up (not timed).
    CUDA_CHECK(cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice));
    cholesky_gpu_run(n, dA, dL);
    CUDA_CHECK(cudaDeviceSynchronize());

    double best_ms = 0.0;
    double sum_ms  = 0.0;

    for (int rep = 0; rep < reps; rep++) {

        CUDA_CHECK(cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaEventRecord(start));
        cholesky_gpu_run(n, dA, dL);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

        sum_ms += ms;
        if (rep == 0 || ms < best_ms) best_ms = ms;
    }

    r.gpu_best_ms = best_ms;
    r.gpu_avg_ms  = sum_ms / (double)reps;

    // ---- correctness check (device -> host, outside the timed region) ----
    CUDA_CHECK(cudaMemcpy(L.data(), dL, bytes, cudaMemcpyDeviceToHost));
    r.current_err = verify_factor(n, A, L);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dL));

    return r;
}

// =============================================================================
//  [10] Print the results table
// =============================================================================
static void print_table(const Result* r, int count, int base_n)
{
    const char* sep =
        "+----------------------+--------------+---------------+---------------+---------------+----------------+-----------------+----------+";

    printf("\n%s\n", sep);
    printf("| %-20s | %-12s | %-13s | %-13s | %-13s | %-14s | %-15s | %-8s |\n",
           "Reason for Selection", "Matrix Size", "CPU Best (ms)", "GPU Best (ms)",
           "GPU Avg (ms)", "Speedup", "Current Error", "Status");
    printf("%s\n", sep);

    for (int i = 0; i < count; i++) {
        char sizebuf[32];
        char reason[64];

        snprintf(sizebuf, sizeof(sizebuf), "%d x %d", r[i].n, r[i].n);

        double ratio = (double)r[i].n / (double)base_n;
        if (i == 0) snprintf(reason, sizeof(reason), "Baseline");
        else        snprintf(reason, sizeof(reason), "%gx scale up", ratio);

        if (!r[i].ok) {
            printf("| %-20s | %-12s | %13s | %13s | %13s | %14s | %15s | %-8s |\n",
                   reason, sizebuf, "FAILED", "FAILED", "FAILED", "-", "-", "FAIL");
            continue;
        }

        double acceptable = REL_TOL * r[i].a_max;
        const char* status = (r[i].current_err <= acceptable) ? "PASS" : "FAIL";
        double speedup = (r[i].gpu_best_ms > 0.0)
                       ? (r[i].cpu_ms / r[i].gpu_best_ms) : 0.0;

        printf("| %-20s | %-12s | %13.3f | %13.3f | %13.3f | %13.2fx | %14.3e | %-8s |\n",
               reason, sizebuf, r[i].cpu_ms, r[i].gpu_best_ms, r[i].gpu_avg_ms,
               speedup, r[i].current_err, status);
    }

    printf("%s\n", sep);
}

// =============================================================================
//  main
// =============================================================================
int main(int argc, char** argv)
{
    const int sizes[] = { 1000, 1500, 2000, 2500, 3000 };
    const int nSizes  = (int)(sizeof(sizes) / sizeof(sizes[0]));

    int reps = 5;
    if (argc > 1) reps = atoi(argv[1]);

    if (reps < 1) {
        printf("usage: %s [repetitions]\n", argv[0]);
        return 1;
    }

    // Print device information once.
    int dev = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("CUDA device: %s  (sm_%d%d, %d MB global memory)\n",
           prop.name, prop.major, prop.minor,
           (int)(prop.totalGlobalMem / (1024 * 1024)));

    printf("CUDA Cholesky (column-oriented, right-looking)\n");
    printf("Matrix sizes: %d   Repetitions per size: %d\n\n", nSizes, reps);

    vector<Result> results(nSizes);

    for (int s = 0; s < nSizes; s++) {
        printf("[%d/%d] n = %-5d ... ", s + 1, nSizes, sizes[s]);
        fflush(stdout);

        results[s] = run_one(sizes[s], reps);

        if (!results[s].ok) {
            printf("FAILED\n");
        }
        else {
            double speedup = results[s].cpu_ms / results[s].gpu_best_ms;
            printf("CPU %9.3f ms   GPU %9.3f ms   speedup %6.2fx   max|A-LLt| = %.3e\n",
                   results[s].cpu_ms, results[s].gpu_best_ms,
                   speedup, results[s].current_err);
        }
    }

    print_table(results.data(), nSizes, sizes[0]);

    return 0;
}
