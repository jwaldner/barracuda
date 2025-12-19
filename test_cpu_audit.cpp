#include "src/barracuda_engine.hpp"
#include <iostream>
#include <vector>

using namespace barracuda;

int main() {
    std::cout << "Testing CPU audit message functionality..." << std::endl;
    
    // Create engine
    BarracudaEngine engine;
    
    // Force CPU mode by setting execution mode
    engine.SetExecutionMode(ExecutionMode::CPU);
    
    std::cout << "Execution mode set to CPU" << std::endl;
    std::cout << "CUDA Available: " << (engine.IsCudaAvailable() ? "YES" : "NO") << std::endl;
    
    // Create test contracts
    std::vector<OptionContract> contracts;
    OptionContract contract;
    contract.symbol = "CPU_TEST";
    contract.strike_price = 100.0;
    contract.underlying_price = 105.0;
    contract.time_to_expiration = 0.25; // 3 months
    contract.risk_free_rate = 0.05;
    contract.volatility = 0.2;
    contract.option_type = 'C';
    contract.market_close_price = 0.0;
    
    contracts.push_back(contract);
    
    std::cout << "Running Black-Scholes calculation with audit symbol 'CPU_TEST' in CPU mode..." << std::endl;
    
    // Call with audit symbol - this should force CPU path and write CPU audit message
    auto results = engine.CalculateBlackScholes(contracts, "CPU_TEST");
    
    std::cout << "Calculation completed." << std::endl;
    std::cout << "Result price: " << results[0].theoretical_price << std::endl;
    std::cout << "Check audit.json for 'hello from cpu! (symbol: CPU_TEST)' message" << std::endl;
    
    return 0;
}