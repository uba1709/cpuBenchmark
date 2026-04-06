#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <random>
using Clock = std::chrono::high_resolution_clock;

// indeksowanie 2D → 1D
inline int idx(int i, int j, int n) {
    return i * n + j;
}

// generowanie macierzy
std::vector<double> generateMatrix(int n) {
    std::vector<double> m(n * n);
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (int i = 0; i < n * n; i++)
        m[i] = dist(rng);

    return m;
}

// transpozycja macierzy
std::vector<double> transpose(const std::vector<double>& B, int n) {
    std::vector<double> BT(n * n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            BT[idx(j, i, n)] = B[idx(i, j, n)];
    return BT;
}

// worker – liczy fragment macierzy
void worker(const std::vector<double>& A,
            const std::vector<double>& BT,
            std::vector<double>& C,
            int n,
            int startRow,
            int endRow,
            std::chrono::time_point<Clock> endTime,
            long long& localCount)
{
    localCount = 0;

    while (Clock::now() < endTime) {

        for (int i = startRow; i < endRow; i++) {
            for (int j = 0; j < n; j++) {

                double sum = 0.0;

                // cache-friendly
                for (int k = 0; k < n; k++) {
                    sum += A[idx(i, k, n)] * BT[idx(j, k, n)];
                }

                C[idx(i, j, n)] = sum;
            }
        }

        localCount++; // policzono fragment macierzy
    }
}

int main() {
    int n = 600;              
    int durationSeconds = 20;

    unsigned int threads = std::thread::hardware_concurrency();
    std::cout << "Threads: " << threads << "\n";

    auto A = generateMatrix(n);
    auto B = generateMatrix(n);
    auto BT = transpose(B, n);
    std::vector<double> C(n * n, 0);

    std::vector<std::thread> workers;
    std::vector<long long> counts(threads, 0);

    int rowsPerThread = n / threads;

    auto endTime = Clock::now() + std::chrono::seconds(durationSeconds);

    for (unsigned int t = 0; t < threads; t++) {
        int startRow = t * rowsPerThread;
        int endRow = (t == threads - 1) ? n : startRow + rowsPerThread;

        workers.emplace_back(worker,
                             std::cref(A),
                             std::cref(BT),
                             std::ref(C),
                             n,
                             startRow,
                             endRow,
                             endTime,
                             std::ref(counts[t]));
    }

    for (auto& th : workers)
        th.join();

    long long total = 0;
    for (auto c : counts)
        total += c;

    std::cout << "Matrices computed: " << total << "\n";
    std::cout << "Matrices per second: "
              << (double)total / threads / durationSeconds << "\n";

    return 0;
}