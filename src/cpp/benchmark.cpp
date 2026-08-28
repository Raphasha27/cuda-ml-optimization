#include <iostream>
#include <vector>
#include <chrono>
#include <random>

void matrixMultiplyCPU(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main() {
    const int N = 1024;
    std::vector<float> A(N * N), B(N * N), C(N * N);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (auto& x : A) x = dis(gen);
    for (auto& x : B) x = dis(gen);

    auto start = std::chrono::high_resolution_clock::now();
    matrixMultiplyCPU(A.data(), B.data(), C.data(), N);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end - start;
    std::cout << "CPU Matrix Multiply (" << N << "x" << N << "): " 
              << diff.count() * 1000 << " ms" << std::endl;

    return 0;
}
