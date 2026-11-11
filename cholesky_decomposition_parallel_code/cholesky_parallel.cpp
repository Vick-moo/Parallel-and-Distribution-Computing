// =============================================================================
//  cholesky_parallel.cpp
//
//  AMCS2103 Parallel and Distributed Computing
//  Computational Problem : Cholesky Decomposition   A = L * L^T
//  Implementation        : SEQUENTIAL, plain three-loop (left-looking) form
//
//  Build (Visual Studio) : Release | x64, then Ctrl+Shift+B
//                          >>> Must use Release. Debug is about 14x slower. <<<
//  Build (command line)  : cl /O2 /EHsc /Fe:cholesky.exe cholesky_parallel.cpp
//                          g++ -O3 -std=c++17 -o cholesky cholesky_parallel.cpp
//
//  Run:
//      cholesky.exe              <-- 5 matrix sizes, 5 repetitions each
//      cholesky.exe 10           <-- 5 matrix sizes, 10 repetitions each
//
//  The factorisation itself is deliberately kept to the textbook three nested
//  loops -- no blocking, no vectorisation, no threading:
//
//      for i = 0 .. n-1
//        for j = 0 .. i
//          sum = dot( L[i][0..j-1], L[j][0..j-1] )
//          i == j ? L[i][j] = sqrt( A[i][i] - sum )
//                 : L[i][j] = ( A[i][j] - sum ) / L[j][j]
//
//  Everything around those three loops is measurement scaffolding:
//    - a reproducible SPD matrix generator (fixed seed per matrix size)
//    - one untimed warm-up run per size
//    - best / average / worst timing over N repetitions
//    - a PASS/FAIL verdict printed next to the measured error
//    - a full A = L * L^T correctness check with a NaN/Inf guard
//
//  Complexity: (n^3 - n)/6 multiply-adds, so Theta(n^3) time and Theta(n^2)
//  space. In FLOP terms that is about n^3/3 floating-point operations.
//
//  Note on A: this is the left-looking form, so A is read-only. It is never
//  modified, which is why the verification step can reuse the original A
//  directly without keeping a backup copy.
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cerrno>
#include <climits>
#include <limits>
#include <vector>
#include <chrono>
#include <algorithm>

using namespace std;

// =============================================================================
//  [1] Timer
// =============================================================================
// steady_clock is monotonic: it only moves forward and never jumps backwards
// when the operating system re-syncs its time. Works on Windows and Linux with
// no platform-specific code.
// =============================================================================
static double now_sec()
{
    using namespace chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// =============================================================================
//  [2] Reproducible pseudo-random number generator (xorshift64*)
// =============================================================================
// rand() is deliberately not used because it produces a different sequence
// under MSVC and under g++/glibc. This generator produces bit-identical output
// on every platform, so the same seed always yields exactly the same matrix.
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
// circle theorem every eigenvalue lies in a disc centred on A[i][i] with radius
// equal to the sum of the absolute off-diagonal entries of that row. Since the
// centre is (radius + 1), every disc lies inside [1, 2R + 1] and never reaches
// zero, so every eigenvalue is >= 1 > 0.
//
// The seed is derived from n, so the matrix for a given size depends only on
// that size. Inserting, removing or reordering sizes cannot silently change the
// matrix used for any other size, and re-running the program always reproduces
// exactly the same input.
// =============================================================================
static void make_positive_definite(int n, vector<double>& A)
{
    g_state = 12345ULL + (uint64_t)n;

    A.assign((size_t)n * (size_t)n, 0.0);   // flat array: better cache behaviour

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
            double v = rng_unit() * 2.0 - 1.0;   // range [-1, 1)
            A[(size_t)i * n + j] = v;
            A[(size_t)j * n + i] = v;            // mirror into the upper half
        }
    }

    for (int i = 0; i < n; i++) {
        const double* Ai = &A[(size_t)i * n];
        double rowsum = 0.0;
        for (int j = 0; j < n; j++) {
            if (j != i) rowsum += fabs(Ai[j]);
        }
        A[(size_t)i * n + i] = rowsum + 1.0;     // the "+1" guarantees SPD
    }
}

// =============================================================================
//  [4] Core Cholesky factorisation -- the three nested loops
// =============================================================================
// A = L * L^T, computed into L. Returns false if the input is not positive
// definite.
//
// Row pointers (Li, Lj) are hoisted out of the k loop. This is purely for
// readability: the compiler already performs this transformation, so it does
// not change the algorithm or its cost.
// =============================================================================
static bool cholesky_factor(int n, const vector<double>& A, vector<double>& L)
{
    for (int i = 0; i < n; i++) {

        const double* Ai = &A[(size_t)i * n];
        double*       Li = &L[(size_t)i * n];

        for (int j = 0; j <= i; j++) {

            const double* Lj = &L[(size_t)j * n];

            double sum = 0.0;
            for (int k = 0; k < j; k++) {
                sum += Li[k] * Lj[k];
            }

            if (i == j) {
                // The pivot. It must be strictly positive for a positive
                // definite matrix.
                //
                // If it is not, sqrt() would return NaN and every later entry
                // would silently become NaN as well, so the failure is caught
                // here instead. !(d > 0.0) is used rather than (d <= 0.0)
                // because NaN compares false against everything: the second
                // form would let a NaN pivot through.
                double d = Ai[i] - sum;
                if (!(d > 0.0)) return false;
                Li[j] = sqrt(d);
            }
            else {
                Li[j] = (Ai[j] - sum) / Lj[j];
            }
        }
    }
    return true;
}

// =============================================================================
//  [5] Correctness check: max |A - L*L^T|
// =============================================================================
// Runs after the timed region, so it never affects the benchmark numbers.
// A value around 1e-13 .. 1e-10 is normal floating-point rounding.
//
// The pass/fail decision uses the relative error:
//     max |A - L*L^T| / max|A|  <=  REL_TOL
//
// Only the lower triangle is checked, because L is lower triangular and A is
// symmetric. Note that this check costs (n^3)/6 multiply-adds -- as much as the
// factorisation itself -- which is why it runs once per size and never inside
// the timed loop.
// =============================================================================
static const double REL_TOL = 1e-9;

static double verify_factor(int n, const vector<double>& A, const vector<double>& L)
{
    double max_abs = 0.0;

    for (int i = 0; i < n; i++) {

        const double* Ai = &A[(size_t)i * n];
        const double* Li = &L[(size_t)i * n];

        for (int j = 0; j <= i; j++) {          // L is lower triangular

            const double* Lj = &L[(size_t)j * n];

            double s = 0.0;
            for (int k = 0; k <= j; k++) {      // this is (L * L^T)[i][j]
                s += Li[k] * Lj[k];
            }

            double e = fabs(s - Ai[j]);

            // A NaN or Inf error would compare false against max_abs and leave
            // it at 0.0, which would make a completely broken result print
            // PASS. Reject it outright instead.
            if (!std::isfinite(e)) {
                return std::numeric_limits<double>::infinity();
            }
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
    double worst_ms;     // slowest of the repetitions
    double a_max;        // max |A[i][j]|, scales the relative tolerance
    double current_err;  // max |A - L*L^T| for this matrix size
    bool   ok;           // false if the matrix was not positive definite
};

// =============================================================================
//  [7] Run one matrix size
// =============================================================================
static Result run_one(int n, int reps)
{
    Result r{};
    r.n  = n;
    r.ok = true;

    // The matrices are allocated inside this function, so they are released
    // when it returns and memory does not accumulate across matrix sizes.
    // A is read-only in this implementation, so no working copy is needed.
    vector<double> A, L;

    make_positive_definite(n, A);
    L.assign((size_t)n * (size_t)n, 0.0);

    // Largest entry of A. Used later to turn REL_TOL into an absolute limit.
    for (size_t i = 0; i < A.size(); i++) {
        double v = fabs(A[i]);
        if (v > r.a_max) r.a_max = v;
    }

    // Warm-up: one untimed run. Removes first-touch page faults, the CPU
    // frequency ramp and cold cache effects from the measured repetitions.
    if (!cholesky_factor(n, A, L)) {
        r.ok = false;
        return r;
    }

    double t_best  = 0.0;
    double t_worst = 0.0;
    double t_sum   = 0.0;

    for (int rep = 0; rep < reps; rep++) {

        fill(L.begin(), L.end(), 0.0);      // cleared outside the timed region

        double t0 = now_sec();

        bool ok = cholesky_factor(n, A, L); // <-- only the three loops are timed

        double dt = now_sec() - t0;

        if (!ok) {
            r.ok = false;
            return r;
        }

        t_sum += dt;
        if (rep == 0 || dt < t_best)  t_best  = dt;
        if (rep == 0 || dt > t_worst) t_worst = dt;
    }

    r.best_ms  = t_best * 1000.0;
    r.avg_ms   = (t_sum / (double)reps) * 1000.0;
    r.worst_ms = t_worst * 1000.0;

    // Correctness check, outside the timed region.
    r.current_err = verify_factor(n, A, L);

    return r;
}

// =============================================================================
//  [8] Print the results table
// =============================================================================
static void print_table(const vector<Result>& r, int base_n)
{
    const char* sep =
        "+----------------------+--------------+---------------+---------------+"
        "---------------+----------------+----------------+--------+";

    printf("\n%s\n", sep);
    printf("| %-20s | %-12s | %13s | %13s | %13s | %14s | %14s | %-6s |\n",
           "Reason for Selection", "Matrix Size",
           "Best Ts (ms)", "Average Ts (ms)", "Worst Ts (ms)",
           "Current Error", "Max Acceptable", "Result");
    printf("%s\n", sep);

    for (size_t i = 0; i < r.size(); i++) {

        char sizebuf[32];
        char reason[64];

        snprintf(sizebuf, sizeof(sizebuf), "%d x %d", r[i].n, r[i].n);

        // "Reason for Selection": the baseline reads "Baseline", every other
        // row shows how much it was scaled up relative to the baseline.
        double ratio = (double)r[i].n / (double)base_n;
        if (i == 0) snprintf(reason, sizeof(reason), "Baseline");
        else        snprintf(reason, sizeof(reason), "%gx scale up", ratio);

        if (!r[i].ok) {
            printf("| %-20s | %-12s | %13s | %13s | %13s | %14s | %14s | %-6s |\n",
                   reason, sizebuf, "FAILED", "FAILED", "FAILED", "-", "-", "FAIL");
            continue;
        }

        // The acceptance limit is REL_TOL * max|A|, printed so the check is
        // visible in the output itself: a row passes only when the measured
        // error is at or below that limit.
        double acceptable = REL_TOL * r[i].a_max;
        const char* status = (r[i].current_err <= acceptable) ? "PASS" : "FAIL";

        printf("| %-20s | %-12s | %13.3f | %13.3f | %13.3f | %14.3e | %14.3e | %-6s |\n",
               reason, sizebuf,
               r[i].best_ms, r[i].avg_ms, r[i].worst_ms,
               r[i].current_err, acceptable, status);
    }

    printf("%s\n", sep);
}

// =============================================================================
//  [9] Parse the optional repetition count
// =============================================================================
// strtol is used instead of atoi so that malformed input ("abc", "5abc",
// values that overflow long) is reported instead of silently becoming 0 or 5.
// =============================================================================
static bool parse_reps(int argc, char** argv, int& reps)
{
    reps = 5;                                   // default repetitions

    if (argc <= 1) return true;

    errno = 0;
    char* end = nullptr;
    long v = strtol(argv[1], &end, 10);

    if (end == argv[1] || *end != '\0' || errno == ERANGE ||
        v < 1 || v > 1000) {
        return false;
    }

    reps = (int)v;
    return true;
}

// =============================================================================
//  [10] main
// =============================================================================
int main(int argc, char** argv)
{
    const int sizes[] = { 1000, 1500, 2000, 2500, 3000 };
    const int nSizes  = (int)(sizeof(sizes) / sizeof(sizes[0]));

    int reps = 5;
    if (!parse_reps(argc, argv, reps)) {
        printf("usage: %s [repetitions]\n", argv[0]);
        printf("       repetitions must be an integer in [1, 1000]\n");
        return 1;
    }

    // Build-configuration guard. Debug is roughly 14x slower than Release, so
    // a Debug build produces meaningless timings while looking perfectly
    // normal. Say so loudly instead of letting those numbers reach a report.
#ifdef NDEBUG
    printf("Build configuration : Release (/O2)   OK\n");
#else
    printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    printf("!!  DEBUG BUILD DETECTED - ALL TIMINGS ARE INVALID        !!\n");
    printf("!!  Debug is about 14x slower than Release.               !!\n");
    printf("!!  Switch to Release | x64 and rebuild before recording  !!\n");
    printf("!!  any numbers for your report.                          !!\n");
    printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
#endif

    printf("Sequential Cholesky (plain three-loop form)\n");
    printf("Matrix sizes: %d   Repetitions per size: %d\n\n", nSizes, reps);

    vector<Result> results(nSizes);

    for (int s = 0; s < nSizes; s++) {
        printf("[%d/%d] n = %-5d ... ", s + 1, nSizes, sizes[s]);
        fflush(stdout);                             // show progress immediately

        results[s] = run_one(sizes[s], reps);

        if (!results[s].ok) {
            printf("FAILED (matrix is not positive definite)\n");
        }
        else {
            double acceptable = REL_TOL * results[s].a_max;
            const char* status = (results[s].current_err <= acceptable)
                               ? "PASS" : "FAIL";
            printf("best %9.3f ms   avg %9.3f ms   worst %9.3f ms   "
                   "max|A-LLt| = %.3e  (limit %.3e)  %s\n",
                   results[s].best_ms, results[s].avg_ms, results[s].worst_ms,
                   results[s].current_err, acceptable, status);
        }
    }

    print_table(results, sizes[0]);

    printf("\nVerification: full lower triangle, max|A - L*L^T| compared with\n");
    printf("              REL_TOL * max|A|   (REL_TOL = %.0e).\n", REL_TOL);
    printf("Note: the A = L*L^T check costs (n^3)/6 multiply-adds, as much as the\n");
    printf("      factorisation itself, and it runs outside the timed region.\n");

    return 0;
}
