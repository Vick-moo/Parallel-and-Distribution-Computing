// =============================================================================
//  cholesky_sequential.cpp
//
//  AMCS2103 Parallel and Distributed Computing
//  Computational Problem : Cholesky Decomposition   A = L * L^T
//  Implementation        : SEQUENTIAL (baseline, single-threaded)
//
//  Build (Visual Studio) : Release | x64, then Ctrl+Shift+B
//                          >>> Must use Release. Debug is about 14x slower. <<<
//  Build (command line)  : cl /O2 /EHsc /Fe:cholesky.exe cholesky_sequential.cpp
//                          g++ -O3 -std=c++17 -o cholesky cholesky_sequential.cpp
//
//  Run:
//      cholesky.exe              <-- 5 matrix sizes, 5 repetitions each
//      cholesky.exe 10           <-- 5 matrix sizes, 10 repetitions each
//
//  Each size gets one untimed warm-up run, then N timed repetitions. After
//  the timed runs the factorisation is checked against A = L * L^T and the
//  maximum absolute error is printed. The summary table reports both the
//  fastest run and the average of all runs.
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <vector>
#include <chrono>
#include <algorithm>

using namespace std;

// =============================================================================
//  [1] Timer
// =============================================================================
// Uses chrono's steady_clock, which is a monotonic clock: it only moves
// forward and never jumps backwards when the operating system re-syncs its
// time. Works on both Windows and Linux with no platform-specific code.
// =============================================================================
static double now_sec()
{
    using namespace chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// =============================================================================
//  [2] Reproducible pseudo-random number generator (xorshift64*)
// =============================================================================
// rand() is deliberately not used here because it produces a different
// sequence under MSVC and under g++/glibc. This generator produces
// bit-identical output on every platform, so the same seed always yields
// exactly the same matrix.
//
// IMPORTANT: this seed must be identical in the parallel version. Otherwise
// the two programs are not factorising the same matrix, and any speedup
// figure computed from them would be meaningless.
// =============================================================================
static uint64_t g_state = 12345ULL;     // must match the parallel version

static inline uint64_t rng_u64()
{
    uint64_t x = g_state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    g_state = x;
    return x * 2685821657736338717ULL;
}

// Converts the generator output to a double in [0, 1): keep the top 53 bits
// (a double has a 53-bit mantissa) and divide by 2^53.
static inline double rng_unit()
{
    return (double)(rng_u64() >> 11) * (1.0 / 9007199254740992.0);
}

// =============================================================================
//  [3] Generate a guaranteed positive-definite matrix
// =============================================================================
// Step 1: fill the off-diagonal entries with random values in [-1, 1),
//         mirrored so that A[i][j] == A[j][i].
// Step 2: set each diagonal entry to the sum of the absolute values of the
//         off-diagonal entries in that row, plus 1.
//
// Why is the result guaranteed to be positive definite? By the Gershgorin
// circle theorem every eigenvalue lies in a disc centred on A[i][i], with a
// radius equal to the sum of the absolute off-diagonal entries of that row.
// Since the centre is (radius + 1), every disc lies within [1, 2R + 1] and
// never reaches zero, so every eigenvalue is >= 1 > 0.
// =============================================================================
static void make_positive_definite(int n, vector<double>& A)
{
    A.assign((size_t)n * (size_t)n, 0.0);   // flat array: better cache behaviour

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
//  [4] Core Cholesky factorisation
// =============================================================================
// A = L * L^T, computed into L. Returns false if the input is not positive
// definite. Kept as a separate function so the warm-up run, the timed runs and
// the correctness check all execute exactly the same code.
// =============================================================================
static bool cholesky_factor(int n, const vector<double>& A, vector<double>& L)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {

            double sum = 0;

            for (int k = 0; k < j; k++) {
                sum += L[(size_t)i * n + k] * L[(size_t)j * n + k];
            }

            if (i == j) {
                // Guard against a non-SPD input; this is not a verification
                // step, it simply prevents sqrt() from receiving a negative
                // number and silently producing NaN.
                double d = A[(size_t)i * n + i] - sum;
                if (d <= 0) return false;
                L[(size_t)i * n + j] = sqrt(d);
            }
            else {
                L[(size_t)i * n + j] =
                    (A[(size_t)i * n + j] - sum) / L[(size_t)j * n + j];
            }
        }
    }
    return true;
}

// =============================================================================
//  [5] Correctness check: max |A - L*L^T|
// =============================================================================
// Runs after the timed region, so it never affects the benchmark numbers.
// A value around 1e-12 .. 1e-10 is normal floating-point rounding; 1e-3 or
// larger means the factorisation is wrong.
// =============================================================================
static double verify_factor(int n, const vector<double>& A, const vector<double>& L)
{
    double max_abs = 0.0;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {          // L is lower triangular
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
    double best_ms;      // fastest of the repetitions
    double avg_ms;       // average over all repetitions
    double max_err;      // max |A - L*L^T| after the last run
    bool   ok;           // false if the matrix was not positive definite
};

// =============================================================================
//  [7] Run one matrix size: generate the matrix, then time the repetitions
// =============================================================================
static Result run_one(int n, int reps)
{
    Result r;
    r.n       = n;
    r.best_ms = 0.0;
    r.avg_ms  = 0.0;
    r.max_err = 0.0;
    r.ok      = true;

    // The matrices are allocated inside this function, so they are released
    // when it returns and memory does not accumulate across matrix sizes.
    vector<double> A, L;

    make_positive_definite(n, A);
    L.assign((size_t)n * (size_t)n, 0.0);

    // Warm-up: one untimed run. Pulls the matrix into cache, triggers any
    // first-touch page faults and brings the CPU up to its sustained clock.
    if (!cholesky_factor(n, A, L)) {
        r.ok = false;
        return r;
    }

    double t_best = 0.0;
    double t_sum  = 0.0;

    for (int rep = 0; rep < reps; rep++) {

        fill(L.begin(), L.end(), 0.0);      // cleared outside the timed region

        double t0 = now_sec();

        bool ok = cholesky_factor(n, A, L);

        double dt = now_sec() - t0;

        if (!ok) {
            r.ok = false;
            return r;
        }

        t_sum += dt;
        if (rep == 0 || dt < t_best) t_best = dt;
    }

    // Correctness check, outside the timed region.
    r.max_err = verify_factor(n, A, L);

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
        "+----------------------+--------------+---------------+-----------------+";

    printf("\n%s\n", sep);
    printf("| %-20s | %-12s | %-13s | %-15s |\n",
           "Reason for Selection", "Matrix Size", "Best Ts (ms)", "Average Ts (ms)");
    printf("%s\n", sep);

    for (int i = 0; i < count; i++) {
        char sizebuf[32];
        char reason[64];

        snprintf(sizebuf, sizeof(sizebuf), "%d x %d", r[i].n, r[i].n);

        // "Reason for Selection": the baseline reads "Baseline", every other
        // row shows how much it was scaled up relative to the baseline.
        double ratio = (double)r[i].n / (double)base_n;
        if (i == 0) snprintf(reason, sizeof(reason), "Baseline");
        else        snprintf(reason, sizeof(reason), "%gx scale up", ratio);

        if (!r[i].ok) {
            printf("| %-20s | %-12s | %13s | %15s |\n",
                   reason, sizebuf, "FAILED", "FAILED");
            continue;
        }

        printf("| %-20s | %-12s | %13.3f | %15.3f |\n",
               reason, sizebuf, r[i].best_ms, r[i].avg_ms);
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

    int reps = 5;                                   // default repetitions
    if (argc > 1) reps = atoi(argv[1]);

    if (reps < 1) {
        printf("usage: %s [repetitions]\n", argv[0]);
        return 1;
    }

    vector<Result> results(nSizes);

    printf("Sequential Cholesky baseline\n");
    printf("Matrix sizes: %d   Repetitions per size: %d\n\n", nSizes, reps);

    for (int s = 0; s < nSizes; s++) {
        printf("[%d/%d] n = %-5d ... ", s + 1, nSizes, sizes[s]);
        fflush(stdout);                             // show progress immediately

        results[s] = run_one(sizes[s], reps);

        if (!results[s].ok) {
            printf("FAILED (matrix is not positive definite)\n");
        }
        else {
            printf("best %9.3f ms   avg %9.3f ms   max|A-LLt| = %.3e\n",
                   results[s].best_ms, results[s].avg_ms, results[s].max_err);
        }
    }

    print_table(results.data(), nSizes, sizes[0]);

    return 0;
}
