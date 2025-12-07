#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>
#include <random>
#include <functional>
#include <iomanip>

using namespace std::chrono;

// Option 1: Matrix multiplication (CPU intensive)
void matrixMultiplication(int size) {
    std::vector<std::vector<double>> a(size, std::vector<double>(size));
    std::vector<std::vector<double>> b(size, std::vector<double>(size));
    std::vector<std::vector<double>> c(size, std::vector<double>(size, 0.0));
    
    // Initialize with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            a[i][j] = dis(gen);
            b[i][j] = dis(gen);
        }
    }
    
    // Matrix multiplication
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            for (int k = 0; k < size; k++) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    
    std::cout << "Matrix result sample: " << c[0][0] << std::endl;
}

// Option 2: Prime number calculation
void primeCalculation(int limit) {
    std::vector<bool> isPrime(limit, true);
    isPrime[0] = isPrime[1] = false;
    
    for (int i = 2; i * i < limit; i++) {
        if (isPrime[i]) {
            for (int j = i * i; j < limit; j += i) {
                isPrime[j] = false;
            }
        }
    }
    
    int primeCount = 0;
    for (bool prime : isPrime) {
        if (prime) primeCount++;
    }
    
    std::cout << "Found " << primeCount << " primes up to " << limit << std::endl;
}

// Option 3: Mathematical series calculation
void mathematicalSeries(int iterations) {
    double pi = 0.0;
    double factorial = 1.0;
    
    for (int i = 0; i < iterations; i++) {
        // Calculate pi using Leibniz formula and other heavy math
        pi += (i % 2 == 0 ? 1 : -1) / (2.0 * i + 1);
        
        // Heavy trigonometric calculations
        double result = 0.0;
        for (int j = 0; j < 1000; j++) {
            result += sin(i * j * 0.001) * cos(i * j * 0.002) * exp(-0.001 * j);
        }
        
        if (i < 100) {
            factorial *= (i + 1);
        }
    }
    
    std::cout << "Pi approximation: " << pi * 4 << ", Factorial sample: " << factorial << std::endl;
}

// Option 4: Monte Carlo simulation
void monteCarlo(int samples) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    
    int inside = 0;
    for (int i = 0; i < samples; i++) {
        double x = dis(gen);
        double y = dis(gen);
        if (x * x + y * y <= 1.0) {
            inside++;
        }
        
        // Add some extra computation
        double z = sin(x) * cos(y) * exp(-x*x - y*y);
    }
    
    double pi_estimate = 4.0 * inside / samples;
    std::cout << "Monte Carlo Pi estimate: " << pi_estimate << std::endl;
}

int main() {
    std::cout << "Testing different calculations to find one that takes ~100ms:\n" << std::endl;
    
    // Test different workload sizes
    std::vector<std::pair<std::string, std::function<void()>>> tests = {
        {"Matrix 150x150", []() { matrixMultiplication(150); }},
        {"Matrix 200x200", []() { matrixMultiplication(200); }},
        {"Matrix 250x250", []() { matrixMultiplication(250); }},
        {"Primes to 500K", []() { primeCalculation(500000); }},
        {"Primes to 1M", []() { primeCalculation(1000000); }},
        {"Math Series 10K", []() { mathematicalSeries(10000); }},
        {"Math Series 20K", []() { mathematicalSeries(20000); }},
        {"Monte Carlo 1M", []() { monteCarlo(1000000); }},
        {"Monte Carlo 2M", []() { monteCarlo(2000000); }},
        {"Monte Carlo 5M", []() { monteCarlo(5000000); }}
    };
    
    for (auto& test : tests) {
        std::cout << "\n🔥 Testing: " << test.first << std::endl;
        auto start = high_resolution_clock::now();
        
        test.second();
        
        auto duration = high_resolution_clock::now() - start;
        double ms = duration_cast<microseconds>(duration).count() / 1000.0;
        
        std::cout << "⏱️  Time: " << std::fixed << std::setprecision(2) << ms << "ms";
        if (ms >= 90 && ms <= 110) {
            std::cout << " ✅ TARGET ACHIEVED!";
        }
        std::cout << std::endl;
    }
    
    return 0;
}