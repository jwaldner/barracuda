#include "src/barracuda_engine.hpp"
#include <iostream>
#include <vector>
#include <cstring>

using namespace barracuda;

int main() {
    std::cout << "Testing CUDA audit message functionality..." << std::endl;
    
    // Create engine
    BarracudaEngine engine;
    
    // Check if CUDA is available
    bool cudaAvailable = engine.IsCudaAvailable();
    std::cout << "CUDA Available: " << (cudaAvailable ? "YES" : "NO") << std::endl;
    std::cout << "Device Count: " << engine.GetDeviceCount() << std::endl;
    
    // Create test contracts
    std::vector<OptionContract> contracts;
    OptionContract contract;
    contract.symbol = "TEST";
    contract.strike_price = 100.0;
    contract.underlying_price = 105.0;
    contract.time_to_expiration = 0.25; // 3 months
    contract.risk_free_rate = 0.05;
    contract.volatility = 0.2;
    contract.option_type = 'C';
    contract.market_close_price = 0.0;
    
    contracts.push_back(contract);
    
    std::cout << "Running Black-Scholes calculation with audit symbol 'TEST'..." << std::endl;
    
    // Call with audit symbol - this should write to audit.json
    auto results = engine.CalculateBlackScholes(contracts, "TEST");
    
    std::cout << "Calculation completed." << std::endl;
    std::cout << "Result price: " << results[0].theoretical_price << std::endl;
    std::cout << "Check audit.json for 'hello from cuda!' or 'hello from cpu!' message" << std::endl;
    
    return 0;
}