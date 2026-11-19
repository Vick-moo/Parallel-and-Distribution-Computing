// =============================================================================
//  cholesky_mpi.cpp
//
//  AMCS2103 Parallel and Distributed Computing
//  Computational Problem : Cholesky Decomposition   A = L * L^T
//  Implementation        : PARALLEL (MPI, 1D cyclic row distribution,
//                            blocked right-looking, distributed memory)
//
//  Build (Windows, MS-MPI) :
//      1. Install Microsoft MPI v10+ (msmpisdk.msi for headers/libs,
//         msmpisetup.exe on every machine that runs the program).
//      2. Build this project in Visual Studio: Release | x64.
//         The .vcxproj already adds $(MSMPI_INC) / $(MSMPI_LIB64) and msmpi.lib.
//  Build (command line, MSVC + MS-MPI) :
//      cl /O2 /DNDEBUG /EHsc /std:c++20 /I"%MSMPI_INC%" cholesky_mpi.cpp ^
//         /link /LIBPATH:"%MSMPI_LIB64%" msmpi.lib
//  Build (Linux, OpenMPI / MPICH) :
//      mpicxx -O3 -DNDEBUG -std=c++17 -o cholesky_mpi cholesky_mpi.cpp
//
//  Run:
//      mpiexec -n 4 cholesky_mpi.exe              <-- 7 sizes, 5 reps, B = 128
//      mpiexec -n 4 cholesky_mpi.exe 10          <-- 7 sizes, 10 reps
//      mpiexec -n 8 cholesky_mpi.exe 5 256       <-- 7 sizes, 5 reps, B = 256
//  A single rank also runs (useful as a correctness baseline):
//      mpiexec -n 1 cholesky_mpi.exe 1
//
//  -----------------------------------------------------------------------------
//  WHY THIS IS NOT JUST "MPI_Scatter THE ROWS AND FACTOR LOCALLY"
//  -----------------------------------------------------------------------------
//  Cholesky factorisation has a strict dependency chain: row i needs every row
//  j < i to be finished first, and there is no shared address space between
//  ranks, so a row computed on one rank must be explicitly sent to the ranks
//  that need it. Scattering independent chunks of rows (the embarrassingly
//  parallel pattern) silently produces a wrong L here, for exactly the same
//  reason the naive "#pragma omp parallel for over i" is wrong in the OpenMP
//  version: it ignores the row dependency.
//
//  -----------------------------------------------------------------------------
//  WHAT THIS FILE DOES INSTEAD: BLOCKED RIGHT-LOOKING, 1D CYCLIC ROWS
//  -----------------------------------------------------------------------------
//  Global row i is owned by rank (i % P). Ownership never migrates, so every
//  element of L is still produced by exactly one rank, and the accumulation
//  order inside each dot product is fixed by the loop structure rather than by
//  the number of ranks. The output is therefore bit-identical for any rank
//  count, which makes the timings comparable and the verification stable.
//
//  The matrix is cut into panels of B columns. For each panel K = [k, ke) the
//  algorithm mirrors the three phases of the OpenMP version:
//
//    (1) Factor the kb x kb diagonal block. Its rows live on different ranks,
//        so they are collected with one MPI_Allgatherv; then every rank
//        factors the identical kb x kb block redundantly with the standard
//        unblocked loop (cheap: O(kb^3), no communication, same decision on
//        every rank, so no extra broadcast is needed to agree on failure).
//
//    (2) Triangular solve (TRSM): rows below the block. Columns are processed
//        in order (column j needs columns < j of the same panel); rows owned
//        by this rank are independent, so no communication happens here.
//
//    (3) Rank-kb update of the trailing submatrix: A22 -= L21 * L21^T. The
//        freshly solved panel is shared with one MPI_Allgatherv; then each
//        rank updates only the rows it owns. This is ~90% of the work.
//
//  Cost per panel: exactly 2 collectives, i.e. 2*n/B collectives per
//  factorisation (n = 4000, B = 128 -> 64 collectives). A has to be broadcast
//  once per matrix size; the final L is gathered once to rank 0 for the
//  correctness check. The gather/scatter buffers pack rows in rank order, and
//  a small offset table maps every global row to its position in the gathered
//  buffer (see build_row_offsets).
//
//  Load balance note: cyclic ownership (instead of contiguous block rows)
//  spreads the shrinking trailing-submatrix work evenly: with contiguous rows
//  the highest rank would own almost all of the late, expensive updates while
//  rank 0 idles. Cyclic keeps every rank busy until the last panels.
//
//  Complexity: the same (n^3 - n)/6 multiply-adds as the sequential version
//  (= about n^3/3 FLOPs), Theta(n^2) memory per rank (each rank keeps a full
//  n x n working copy; only owned rows are authoritative between collectives).
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cerrno>
#include <vector>
#include <chrono>
#include <algorithm>

#include <mpi.h>

using namespace std;

// =============================================================================
//  [1] Reproducible pseudo-random number generator (xorshift64*)
// =============================================================================
// Copied verbatim from cholesky_sequential.cpp, including the fact that the
// generator is NOT re-seeded between matrix sizes. Only rank 0 calls it, then
// broadcasts A, so every rank factorises byte-for-byte the same matrix as the
// sequential program for every n. That is what makes speedup figures in the
// report meaningful.
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
//  [2] Generate a guaranteed positive-definite matrix (rank 0 only)
// =============================================================================
// Identical to the sequential version: mirrored random entries in [-1, 1), and
// each diagonal entry set to (sum of |off-diagonals| in that row) + 1. By the
// Gershgorin circle theorem every eigenvalue is then >= 1 > 0.
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
//  [3] Ownership helpers (1D cyclic row distribution)
// =============================================================================
// Global row i belongs to rank (i % size) for the whole run. off[] maps a
// global row to its position (in doubles) inside a gathered buffer that holds
// one contiguous segment per rank, ranks in order, each rank's own rows in
// ascending global order. counts[r] / displs[r] are the matching MPI
// Allgatherv/Gatherv arguments (in doubles).
// =============================================================================
static inline int row_owner(int i, int size)
{
    return i % size;
}

// Fill counts/displs/off for the global row range [r0, r1), C doubles per row.
static void build_row_offsets(int r0, int r1, int C, int size,
                              vector<int>& counts, vector<int>& displs,
                              vector<int>& cnt, vector<int>& off)
{
    for (int r = 0; r < size; r++) counts[r] = 0;
    for (int i = r0; i < r1; i++) counts[row_owner(i, size)]++;

    displs[0] = 0;
    for (int r = 1; r < size; r++) displs[r] = displs[r - 1] + counts[r - 1] * C;

    for (int r = 0; r < size; r++) cnt[r] = 0;
    for (int i = r0; i < r1; i++) {
        int r = row_owner(i, size);
        off[i] = displs[r] + cnt[r] * C;
        cnt[r]++;
    }
}

// =============================================================================
//  [4] Parallel blocked Cholesky factorisation (distributed over MPI ranks)
// =============================================================================
// W is a full n x n working copy on EVERY rank (starts as a copy of A); only
// rows owned by this rank are authoritative between collectives. On return,
// the owned rows of the lower triangle hold L. Returns false if a pivot is not
// strictly positive (every rank sees the same diagonal block, so every rank
// reaches the same verdict with no extra communication).
//
// D/sbuf/rbuf/counts/displs/cnt/off are scratch workspace allocated once per
// matrix size by the caller and reused for every repetition.
// =============================================================================
static bool cholesky_mpi_factor(int n, double* W, int B, int rank, int size,
                                vector<double>& D,
                                vector<double>& sbuf, vector<double>& rbuf,
                                vector<int>& counts, vector<int>& displs,
                                vector<int>& cnt, vector<int>& off)
{
    for (int k = 0; k < n; k += B) {

        const int kb = (B < n - k) ? B : (n - k);   // last panel may be smaller
        const int ke = k + kb;                      // one past the panel

        // ---------------------------------------------------------------------
        //  (1) Factor the kb x kb diagonal block.
        // ---------------------------------------------------------------------
        // Gather its rows (columns k..ke-1 only) to every rank, then factor
        // redundantly with the standard unblocked loop.
        build_row_offsets(k, ke, kb, size, counts, displs, cnt, off);

        {
            int pos = 0;
            for (int i = k; i < ke; i++) {
                if (row_owner(i, size) == rank) {
                    for (int u = 0; u < kb; u++) sbuf[(size_t)pos + u] = W[(size_t)i * n + k + u];
                    pos += kb;
                }
            }
            // MPI counts are in doubles here.
            for (int r = 0; r < size; r++) counts[r] *= kb;
            MPI_Allgatherv(sbuf.data(), pos, MPI_DOUBLE,
                           rbuf.data(), counts.data(), displs.data(), MPI_DOUBLE,
                           MPI_COMM_WORLD);
            for (int r = 0; r < size; r++) counts[r] /= kb;   // restore row counts

            // Unpack into D in global row order, then factor in place.
            for (int i = k; i < ke; i++) {
                const double* src = rbuf.data() + off[i];
                double* dst = D.data() + (size_t)(i - k) * kb;
                for (int u = 0; u < kb; u++) dst[u] = src[u];
            }
        }

        for (int j = 0; j < kb; j++) {
            double d = D[(size_t)j * kb + j];
            // !(d > 0.0) rather than (d <= 0.0) so that NaN is also rejected.
            if (!(d > 0.0)) return false;
            double piv = sqrt(d);
            D[(size_t)j * kb + j] = piv;
            for (int i = j + 1; i < kb; i++) D[(size_t)i * kb + j] /= piv;
            for (int i = j + 1; i < kb; i++) {
                double lij = D[(size_t)i * kb + j];   // must be read before writing
                for (int m = j + 1; m <= i; m++) D[(size_t)i * kb + m] -= lij * D[(size_t)m * kb + j];
            }
        }

        // Write the factored block back into the owned rows of W.
        for (int i = k; i < ke; i++) {
            if (row_owner(i, size) == rank) {
                const double* src = D.data() + (size_t)(i - k) * kb;
                for (int u = 0; u <= i - k; u++) W[(size_t)i * n + k + u] = src[u];
            }
        }

        // ---------------------------------------------------------------------
        //  (2) Triangular solve: the panel below the diagonal block.
        // ---------------------------------------------------------------------
        // L[i][j] = ( W[i][j] - sum_{m<j} L[i][m]*L[j][m] ) / L[j][j].
        // Columns are sequential; rows owned by this rank are independent.
        // Purely local: the D row it needs was just factored above, and the
        // W[i][m] values are this rank's own rows.
        for (int j = k; j < ke; j++) {
            double piv = D[(size_t)(j - k) * kb + (j - k)];
            for (int i = ke; i < n; i++) {
                if (row_owner(i, size) != rank) continue;
                double s = W[(size_t)i * n + j];
                for (int m = k; m < j; m++) s -= W[(size_t)i * n + m] * D[(size_t)(j - k) * kb + (m - k)];
                W[(size_t)i * n + j] = s / piv;
            }
        }

        // ---------------------------------------------------------------------
        //  (3) Share the solved panel, then rank-kb update of the trailing
        //      submatrix: A22 -= L21 * L21^T.
        // ---------------------------------------------------------------------
        // One Allgatherv delivers every panel row to every rank; each rank
        // then updates only the rows it owns. This is where nearly all the
        // time is spent.
        const int rest = n - ke;
        if (rest > 0) {
            build_row_offsets(ke, n, kb, size, counts, displs, cnt, off);

            int pos = 0;
            for (int i = ke; i < n; i++) {
                if (row_owner(i, size) == rank) {
                    for (int u = 0; u < kb; u++) sbuf[(size_t)pos + u] = W[(size_t)i * n + k + u];
                    pos += kb;
                }
            }
            for (int r = 0; r < size; r++) counts[r] *= kb;
            MPI_Allgatherv(sbuf.data(), pos, MPI_DOUBLE,
                           rbuf.data(), counts.data(), displs.data(), MPI_DOUBLE,
                           MPI_COMM_WORLD);
            for (int r = 0; r < size; r++) counts[r] /= kb;   // restore row counts

            for (int i = ke; i < n; i++) {
                if (row_owner(i, size) != rank) continue;
                const double* ir = rbuf.data() + off[i];
                double* wr = W + (size_t)i * n;
                for (int m = ke; m <= i; m++) {
                    const double* mr = rbuf.data() + off[m];
                    double s = 0.0;
                    for (int t = 0; t < kb; t++) s += ir[t] * mr[t];
                    wr[m] -= s;
                }
            }
        }
    }

    return true;
}

// =============================================================================
//  [5] Correctness check: max |A - L*L^T| (rank 0 only)
// =============================================================================
// Same check and same tolerance as the sequential version. L arrives via one
// MPI_Gatherv after the timed region, so it never affects the benchmark.
// =============================================================================
static const double REL_TOL = 1e-9;

static double verify_factor(int n, const vector<double>& A, const vector<double>& L)
{
    double max_abs = 0.0;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {          // L is lower triangular
            double s = 0.0;
            for (int kk = 0; kk <= j; kk++) {    // this is (L * L^T)[i][j]
                s += L[(size_t)i * n + kk] * L[(size_t)j * n + kk];
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
//  [6] Result of one experiment (meaningful on rank 0 only)
// =============================================================================
struct Result {
    int    n;
    double best_ms;      // fastest of the repetitions (slowest rank each rep)
    double avg_ms;       // average over all repetitions
    double a_max;        // max |A[i][j]|, scales the relative tolerance
    double current_err;  // max |A - L*L^T| for this matrix size
    bool   ok;           // false if the matrix was not positive definite
};

// =============================================================================
//  [7] Run one matrix size
// =============================================================================
static Result run_one(int n, int reps, int block, int rank, int size)
{
    Result r{};
    r.n  = n;
    r.ok = true;

    vector<double> A, W;

    // Rank 0 generates the matrix (same RNG stream as sequential), then one
    // broadcast gives every rank an identical copy.
    if (rank == 0) {
        make_positive_definite(n, A);
        for (size_t i = 0; i < A.size(); i++) {
            double v = fabs(A[i]);
            if (v > r.a_max) r.a_max = v;
        }
    } else {
        A.assign((size_t)n * (size_t)n, 0.0);
    }
    MPI_Bcast(A.data(), (int)((size_t)n * (size_t)n), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&r.a_max, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    W.assign((size_t)n * (size_t)n, 0.0);

    // Scratch workspace, allocated once and reused for every repetition.
    vector<double> D((size_t)block * block);
    vector<double> sbuf((size_t)n * block);
    vector<double> rbuf((size_t)n * block);
    vector<int> counts(size), displs(size), cnt(size), off(n);

    // Warm-up: one untimed run, so page faults, collectives setup and cold
    // caches are not charged to the measured repetitions.
    copy(A.begin(), A.end(), W.begin());
    MPI_Barrier(MPI_COMM_WORLD);
    int okflag = cholesky_mpi_factor(n, W.data(), block, rank, size,
                                    D, sbuf, rbuf, counts, displs, cnt, off) ? 1 : 0;
    int allok = 0;
    MPI_Reduce(&okflag, &allok, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    if (rank == 0 && allok != 1) {
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
        MPI_Barrier(MPI_COMM_WORLD);   // align ranks so the max-time is honest

        double t0 = MPI_Wtime();

        okflag = cholesky_mpi_factor(n, W.data(), block, rank, size,
                                    D, sbuf, rbuf, counts, displs, cnt, off) ? 1 : 0;

        double dt = MPI_Wtime() - t0;

        // Parallel time of this repetition = the slowest rank's time.
        double dt_max = 0.0;
        MPI_Reduce(&dt, &dt_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&okflag, &allok, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            if (allok != 1) {
                r.ok = false;
                return r;
            }
            t_sum += dt_max;
            if (rep == 0 || dt_max < t_best) t_best = dt_max;
        }
    }

    // Gather the owned rows to rank 0 and check correctness there, outside the
    // timed region. The gather packs full rows, so the send buffer must grow
    // from n*B (panel phases) to n*n here.
    sbuf.resize((size_t)n * (size_t)n);
    vector<double> L;
    if (rank == 0) L.assign((size_t)n * (size_t)n, 0.0);
    {
        build_row_offsets(0, n, n, size, counts, displs, cnt, off);
        int pos = 0;
        for (int i = 0; i < n; i++) {
            if (row_owner(i, size) == rank) {
                for (int u = 0; u < n; u++) sbuf[(size_t)pos + u] = W[(size_t)i * n + u];
                pos += n;
            }
        }
        for (int rr = 0; rr < size; rr++) counts[rr] *= n;
        double* recv = (rank == 0) ? L.data() : nullptr;
        MPI_Gatherv(sbuf.data(), pos, MPI_DOUBLE,
                    recv, counts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
        if (rank == 0) {
            // Reorder rank-grouped rows back to global order.
            vector<double> tmp((size_t)n * (size_t)n);
            for (int i = 0; i < n; i++) {
                const double* src = L.data() + off[i];
                double* dst = tmp.data() + (size_t)i * n;
                for (int u = 0; u < n; u++) dst[u] = src[u];
            }
            L.swap(tmp);
        }
    }

    if (rank == 0) {
        r.current_err = verify_factor(n, A, L);
        r.best_ms = t_best * 1000.0;
        r.avg_ms  = (t_sum / (double)reps) * 1000.0;
    }

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
           "Reason for Selection", "Matrix Size", "Best Tp (ms)", "Average Tp (ms)",
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
//  [9] Parse the optional arguments: [repetitions] [block size]
// =============================================================================
// The rank count is NOT an argument here: it comes from mpiexec -n <P>.
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
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int sizes[] = { 1000, 1500, 2000, 2500, 3000, 3500, 4000 };
    const int nSizes  = (int)(sizeof(sizes) / sizeof(sizes[0]));

    int reps  = 5;
    int block = 128;

    if (argc > 1 && !parse_int(argv[1], 1, 1000, reps)) {
        if (rank == 0) printf("usage: %s [repetitions 1..1000] [block 16..1024]\n", argv[0]);
        MPI_Finalize();
        return 1;
    }
    if (argc > 2 && !parse_int(argv[2], 16, 1024, block)) {
        if (rank == 0) printf("usage: %s [repetitions 1..1000] [block 16..1024]\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
#ifdef NDEBUG
        printf("Build configuration : Release (/O2)   OK\n");
#else
        printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        printf("!!  DEBUG BUILD DETECTED - ALL TIMINGS ARE INVALID        !!\n");
        printf("!!  Switch to Release | x64 and rebuild before recording  !!\n");
        printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
#endif

        printf("Parallel Cholesky (MPI, 1D cyclic rows, blocked right-looking)\n");
        printf("Matrix sizes: %d   Repetitions per size: %d\n", nSizes, reps);
        printf("MPI ranks: %d   Block size: %d\n\n", size, block);
    }

    vector<Result> results(nSizes);

    for (int s = 0; s < nSizes; s++) {
        if (rank == 0) {
            printf("[%d/%d] n = %-5d ... ", s + 1, nSizes, sizes[s]);
            fflush(stdout);
        }

        results[s] = run_one(sizes[s], reps, block, rank, size);

        if (rank == 0) {
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
    }

    if (rank == 0) {
        print_table(results, sizes[0]);

        printf("\nVerification: full lower triangle, max|A - L*L^T| compared with\n");
        printf("              REL_TOL * max|A|   (REL_TOL = %.0e).\n", REL_TOL);
        printf("MPI ranks used: %d   Block size: %d\n", size, block);
    }

    MPI_Finalize();
    return 0;
}
