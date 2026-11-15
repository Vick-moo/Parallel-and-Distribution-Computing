// =============================================================================
//  cholesky_openmp.cpp
//
//  AMCS2103 Parallel and Distributed Computing
//  Computational Problem : Cholesky Decomposition   A = L * L^T
//  Implementation        : PARALLEL (OpenMP, blocked right-looking)
//
//  Build (command line, MSVC) :
//      cl /O2 /DNDEBUG /EHsc /std:c++20 /openmp cholesky_openmp.cpp
//  Build (command line, g++) :
//      g++ -O3 -fopenmp -std=c++17 -o cholesky_openmp cholesky_openmp.cpp
//
//  Build (Visual Studio) : this file defines its own main(), so it CANNOT be
//  added to the sequential project -- one project produces one .exe and would
//  fail to link with two main() functions. Either build it from the command
//  line as above, or add a second console project that contains only this file
//  and has "OpenMP Support" set to Yes (/openmp).
//
//  Run:
//      cholesky_openmp.exe                 <-- 6 sizes, 5 reps, all cores, B=128
//      cholesky_openmp.exe 10              <-- 6 sizes, 10 reps
//      cholesky_openmp.exe 10 4            <-- 6 sizes, 10 reps, 4 threads
//      cholesky_openmp.exe 5 4 256         <-- ... and block size 256
//
//  -----------------------------------------------------------------------------
//  WHY THIS IS NOT JUST "#pragma omp parallel for" ON THE OUTER i LOOP
//  -----------------------------------------------------------------------------
//  The textbook form of the algorithm is
//
//      for i = 0 .. n-1
//        for j = 0 .. i
//          sum = dot( L[i][0..j-1], L[j][0..j-1] )
//          i == j ? L[i][j] = sqrt( A[i][i] - sum )
//                 : L[i][j] = ( A[i][j] - sum ) / L[j][j]
//
//  Row i is computed from rows j < i, so rows have a strict dependency chain.
//  Two threads working on different values of i at the same time read rows that
//  may not be finished yet: that is a data race, and the result is silently
//  wrong (it does not crash, and it still prints plausible numbers). This is
//  why the outer loop is left alone here.
//
//  -----------------------------------------------------------------------------
//  WHAT THIS FILE DOES INSTEAD: BLOCKED RIGHT-LOOKING
//  -----------------------------------------------------------------------------
//  The matrix is cut into blocks of B columns. For each diagonal block k the
//  algorithm works "to the right" and only touches data that is already final:
//
//    (1) Factor the B x B diagonal block with the standard unblocked loop.
//        Inside it, for a fixed column j, every row i > j is independent, so
//        the work IS parallel -- but only over i, never over j.
//
//    (2) Triangular solve (TRSM): L[i][j] for i below the block. For a fixed
//        column j, every row i is independent -> parallel over i. Columns are
//        processed in order because column j needs columns < j.
//
//    (3) Rank-B update (SYRK/GEMM): A22 -= L21 * L21^T. This is (n-k)^2/2
//        independent dot products, and it is ~90% of the total work at these
//        sizes. Parallel over rows i; each thread writes only its own row.
//
//  Every element of L is still produced by one thread, and the accumulation
//  order inside each dot product is fixed by the loop structure rather than by
//  the number of threads. The output is therefore bit-identical for any thread
//  count, which makes the timings comparable and the verification stable.
//
//  Blocking also improves the single-thread cache behaviour: a B x B block of
//  doubles is 2*B*B*8 bytes (B = 128 -> 256 KB), which fits in L2, so the inner
//  kernels re-use data out of cache instead of streaming the whole matrix from
//  RAM every time.
//
//  Complexity: the same (n^3 - n)/6 multiply-adds as the sequential version
//  (= about n^3/3 FLOPs), Theta(n^2) memory.
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cerrno>
#include <vector>
#include <chrono>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

// =============================================================================
//  [1] Timer
// =============================================================================
// steady_clock is monotonic: it only moves forward and never jumps backwards
// when the operating system re-syncs its time.
// =============================================================================
static double now_sec()
{
    using namespace chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// =============================================================================
//  [2] Reproducible pseudo-random number generator (xorshift64*)
// =============================================================================
// This block is copied verbatim from cholesky_sequential.cpp, including the
// fact that the generator is NOT re-seeded between matrix sizes. The two
// programs therefore factor byte-for-byte the same matrix for every n, which is
// what makes the speedup numbers in the report meaningful.
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
//  [3] Generate a guaranteed positive-definite matrix
// =============================================================================
// Identical to the sequential version: mirrored random entries in [-1, 1), and
// each diagonal entry set to (sum of |off-diagonals| in that row) + 1. By the
// Gershgorin circle theorem every eigenvalue is then >= 1 > 0, so the matrix is
// positive definite and Cholesky cannot fail.
// =============================================================================
static void make_positive_definite(int n, vector<double>& A)
{
    A.assign((size_t)n * (size_t)n, 0.0);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
            double v = rng_unit() * 2.0 - 1.0;   // range [-1, 1)
            A[(size_t)i * n + j] = v;
            A[(size_t)j * n + i] = v;            // mirror into the upper half
        }
    }

    for (int i = 0; i < n; i++) {
        double rowsum = 0.0;
        for (int j = 0; j < n; j++) {
            if (j != i) rowsum += fabs(A[(size_t)i * n + j]);
        }
        A[(size_t)i * n + i] = rowsum + 1.0;     // the "+1" guarantees SPD
    }
}

// =============================================================================
//  [4] Parallel blocked Cholesky factorisation
// =============================================================================
// Factors in place: on entry W is the SPD matrix, on return the lower triangle
// of W holds L and the strict upper triangle is untouched garbage. Only the
// lower triangle is ever read, so no upper-triangle value can influence the
// result. Returns false if a pivot is not strictly positive.
//
// The three phases below correspond to the three comments in the file header.
// =============================================================================
static bool cholesky_blocked(int n, double* W, int B)
{
    for (int k = 0; k < n; k += B) {

        const int kb = (B < n - k) ? B : (n - k);   // last block may be smaller
        const int ke = k + kb;                      // one past the block

        // ---------------------------------------------------------------------
        //  (1) Factor the kb x kb diagonal block, unblocked.
        // ---------------------------------------------------------------------
        // Column j of the block depends on columns < j, so j stays sequential.
        // Row i only depends on columns <= j of itself and on column j of rows
        // above it, so the two loops over i are safe to run in parallel.
        for (int j = k; j < ke; j++) {

            double*       jr  = W + (size_t)j * n;
            const double  d   = jr[j];

            // A non-positive pivot means the matrix is not positive definite.
            // !(d > 0.0) rather than (d <= 0.0) so that NaN is also rejected.
            if (!(d > 0.0)) return false;

            const double piv = sqrt(d);
            jr[j] = piv;

            const int rows = ke - j - 1;
            if (rows <= 0) continue;

            // L[i][j] = W[i][j] / L[j][j]
            #pragma omp parallel for schedule(static) if (rows >= 64)
            for (int i = j + 1; i < ke; i++) {
                W[(size_t)i * n + j] /= piv;
            }

            // W[i][m] -= L[i][j] * L[m][j], for the lower triangle of the block
            #pragma omp parallel for schedule(dynamic, 4) if (rows >= 64)
            for (int i = j + 1; i < ke; i++) {
                double*      ir  = W + (size_t)i * n;
                const double lij = ir[j];           // must be read before writing
                for (int m = j + 1; m <= i; m++) {
                    ir[m] -= lij * W[(size_t)m * n + j];
                }
            }
        }

        // ---------------------------------------------------------------------
        //  (2) Triangular solve: the panel below the diagonal block.
        // ---------------------------------------------------------------------
        // L[i][j] = ( W[i][j] - sum_{m<j} L[i][m]*L[j][m] ) / L[j][j]
        // Columns are sequential (column j needs columns < j); rows are
        // independent, so the parallel loop is over i.
        for (int j = k; j < ke; j++) {

            const double* jr   = W + (size_t)j * n;
            const double  piv  = jr[j];
            const int     rows = n - ke;

            #pragma omp parallel for schedule(static) if (rows >= 64)
            for (int i = ke; i < n; i++) {
                double* ir = W + (size_t)i * n;
                double  s  = ir[j];
                for (int m = k; m < j; m++) s -= ir[m] * jr[m];
                ir[j] = s / piv;
            }
        }

        // ---------------------------------------------------------------------
        //  (3) Rank-kb update of the trailing submatrix: A22 -= L21 * L21^T.
        // ---------------------------------------------------------------------
        // Element (i,m) is a dot product of row i and row m of the panel, both
        // contiguous in memory. Row i is written only by the thread that owns
        // it, so there is no race. This is where nearly all the time is spent.
        const int rest = n - ke;
        if (rest > 0) {

            #pragma omp parallel for schedule(dynamic, 8) if (rest >= 64)
            for (int i = ke; i < n; i++) {

                double* ir = W + (size_t)i * n;

                for (int m = ke; m <= i; m++) {
                    const double* mr = W + (size_t)m * n;
                    double s = 0.0;
                    for (int j = k; j < ke; j++) s += ir[j] * mr[j];
                    ir[m] -= s;
                }
            }
        }
    }

    return true;
}

// =============================================================================
//  [5] Correctness check: max |A - L*L^T|
// =============================================================================
// Same check and same tolerance as the sequential version. It reads L straight
// out of the lower triangle of W, so no extra n x n matrix is needed.
//
// The pass/fail decision is relative:  max|A - LL^T| / max|A|  <=  REL_TOL
// =============================================================================
static const double REL_TOL = 1e-9;

static double verify_factor(int n, const vector<double>& A, const double* W)
{
    double max_abs = 0.0;

    for (int i = 0; i < n; i++) {

        const double* ir = W + (size_t)i * n;

        for (int j = 0; j <= i; j++) {          // L is lower triangular
            const double* jr = W + (size_t)j * n;

            double s = 0.0;
            for (int k = 0; k <= j; k++) {      // this is (L * L^T)[i][j]
                s += ir[k] * jr[k];
            }

            double e = fabs(s - A[(size_t)i * n + j]);

            // A NaN or Inf would compare false against max_abs and leave it at
            // 0.0, which would make a broken result print PASS. Reject it.
            if (!isfinite(e)) return INFINITY;
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
    double best_ms;      // fastest of the repetitions
    double avg_ms;       // average over all repetitions
    double a_max;        // max |A[i][j]|, scales the relative tolerance
    double current_err;  // max |A - L*L^T| for this matrix size
    bool   ok;           // false if the matrix was not positive definite
};

// =============================================================================
//  [7] Run one matrix size
// =============================================================================
static Result run_one(int n, int reps, int block)
{
    Result r{};
    r.n  = n;
    r.ok = true;

    vector<double> A, W;

    make_positive_definite(n, A);
    W.assign(A.size(), 0.0);

    // Largest entry of A, used to turn REL_TOL into an absolute limit.
    for (size_t i = 0; i < A.size(); i++) {
        double v = fabs(A[i]);
        if (v > r.a_max) r.a_max = v;
    }

    // Warm-up: one untimed run, so page faults, the CPU frequency ramp and
    // cold caches are not charged to the measured repetitions.
    copy(A.begin(), A.end(), W.begin());
    if (!cholesky_blocked(n, W.data(), block)) {
        r.ok = false;
        return r;
    }

    double t_best = 0.0;
    double t_sum  = 0.0;

    for (int rep = 0; rep < reps; rep++) {

        // W must start as a copy of A because the factorisation is in place.
        // The copy is outside the timed region, like the fill() in the
        // sequential version.
        copy(A.begin(), A.end(), W.begin());

        double t0 = now_sec();

        bool ok = cholesky_blocked(n, W.data(), block);

        double dt = now_sec() - t0;

        if (!ok) {
            r.ok = false;
            return r;
        }

        t_sum += dt;
        if (rep == 0 || dt < t_best) t_best = dt;
    }

    // Correctness check, outside the timed region.
    r.current_err = verify_factor(n, A, W.data());

    r.best_ms = t_best * 1000.0;
    r.avg_ms  = (t_sum / (double)reps) * 1000.0;

    return r;
}

// =============================================================================
//  [8] Print the results table (same layout as the sequential version)
// =============================================================================
static void print_table(const vector<Result>& r, int base_n)
{
    const char* sep =
        "+----------------------+--------------+---------------+---------------+----------------+-----------------+----------+";

    printf("\n%s\n", sep);
    printf("| %-20s | %-12s | %-13s | %-13s | %-14s | %-15s | %-8s |\n",
           "Reason for Selection", "Matrix Size", "Best Ts (ms)", "Average Ts (ms)",
           "Current Error", "Max Acceptable", "Status");
    printf("%s\n", sep);

    for (size_t i = 0; i < r.size(); i++) {

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
//  [9] Parse the optional arguments: [repetitions] [threads] [block size]
// =============================================================================
// strtol with a range check, so "abc", "5abc" and overflow are rejected instead
// of silently becoming 0 or the default.
// =============================================================================
static bool parse_int(const char* s, int lo, int hi, int& out)
{
    errno = 0;
    char* end = nullptr;
    long v = strtol(s, &end, 10);
    if (end == s || *end != '\0' || errno == ERANGE || v < lo || v > hi) return false;
    out = (int)v;
    return true;
}

// =============================================================================
//  main
// =============================================================================
int main(int argc, char** argv)
{
    const int sizes[] = { 1000, 1500, 2000, 2500, 3000, 3500 };
    const int nSizes  = (int)(sizeof(sizes) / sizeof(sizes[0]));

    int reps  = 5;
    int block = 128;
    int threads = 0;                            // 0 = use the OpenMP default

    if (argc > 1 && !parse_int(argv[1], 1, 1000, reps)) {
        printf("usage: %s [repetitions 1..1000] [threads 1..256] [block 16..1024]\n", argv[0]);
        return 1;
    }
    if (argc > 2 && !parse_int(argv[2], 1, 256, threads)) {
        printf("usage: %s [repetitions 1..1000] [threads 1..256] [block 16..1024]\n", argv[0]);
        return 1;
    }
    if (argc > 3 && !parse_int(argv[3], 16, 1024, block)) {
        printf("usage: %s [repetitions 1..1000] [threads 1..256] [block 16..1024]\n", argv[0]);
        return 1;
    }

#ifdef _OPENMP
    if (threads > 0) omp_set_num_threads(threads);
    const int nthreads = omp_get_max_threads();
#else
    // Compiled without /openmp (or without -fopenmp): every pragma is ignored,
    // so the program still runs and still verifies, but it is single-threaded.
    // Say so loudly rather than letting these timings reach a report.
    const int nthreads = 1;
    printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    printf("!!  COMPILED WITHOUT OpenMP - THIS RUN IS SINGLE-THREADED !!\n");
    printf("!!  MSVC: /openmp     g++: -fopenmp                        !!\n");
    printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
#endif

#ifdef NDEBUG
    printf("Build configuration : Release (/O2)   OK\n");
#else
    printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    printf("!!  DEBUG BUILD DETECTED - ALL TIMINGS ARE INVALID        !!\n");
    printf("!!  Switch to Release | x64 and rebuild before recording  !!\n");
    printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
#endif

    printf("Parallel Cholesky (OpenMP, blocked right-looking)\n");
    printf("Matrix sizes: %d   Repetitions per size: %d\n", nSizes, reps);
    printf("Threads: %d   Block size: %d\n\n", nthreads, block);

    vector<Result> results(nSizes);

    for (int s = 0; s < nSizes; s++) {
        printf("[%d/%d] n = %-5d ... ", s + 1, nSizes, sizes[s]);
        fflush(stdout);

        results[s] = run_one(sizes[s], reps, block);

        if (!results[s].ok) {
            printf("FAILED (matrix is not positive definite)\n");
        }
        else {
            // n^3/3 FLOPs for the whole factorisation.
            const double gflops =
                ((double)sizes[s] * sizes[s] * sizes[s] / 3.0) /
                (results[s].best_ms * 1.0e6);

            printf("best %9.3f ms   avg %9.3f ms   %6.2f GFLOP/s   max|A-LLt| = %.3e\n",
                   results[s].best_ms, results[s].avg_ms, gflops,
                   results[s].current_err);
        }
    }

    print_table(results, sizes[0]);

    printf("\nVerification: full lower triangle, max|A - L*L^T| compared with\n");
    printf("              REL_TOL * max|A|   (REL_TOL = %.0e).\n", REL_TOL);
    printf("Threads used: %d   Block size: %d\n", nthreads, block);

    return 0;
}
