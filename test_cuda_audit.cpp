#include <iostream>
#include <fstream>
#include <cstring>
#include "src/barracuda_engine.hpp"

using namespace barracuda;

int main() {
    std::cout << "Testing CUDA audit functionality..." << std::endl;
    
    // Create engine
    auto engine = barracuda_create_engine();
    if (!engine) {
        std::cout << "Failed to create engine!" << std::endl;
        return 1;
    }
    
    // Check if audit.json exists
    std::ifstream audit_check("audit.json");
    if (!audit_check.is_open()) {
        std::cout << "No audit.json found - creating one for TSLA..." << std::endl;
        // Create a basic audit.json for testing
        std::ofstream audit_file("audit.json");
        audit_file << R"({
  "header": {
    "ticker": "TSLA",
    "expiry_date": "2026-01-16",
    "start_time": "2025-12-26T10:00:00-06:00"
  },
  "entries": []
})";
        audit_file.close();
    } else {
        audit_check.close();
    }
    
    // Create a test contract for TSLA
    CompleteOptionContract contract;
    memset(&contract, 0, sizeof(contract));
    
    // Set contract data
    strcpy(contract.symbol, "TSLA");
    contract.strike_price = 450.0;
    contract.underlying_price = 460.0;
    contract.time_to_expiration = 0.0274; // 10 days
    contract.risk_free_rate = 0.05;
    contract.volatility = 0.25;
    contract.option_type = 'C';
    contract.market_close_price = 15.50;
    
    std::cout << "Calling barracuda_calculate_options_complete..." << std::endl;
    
    // Call the complete processing function
    int result = barracuda_calculate_options_complete(engine, &contract, 1, 10000.0, 10);
    
    if (result == 0) {
        std::cout << "SUCCESS! Function returned 0" << std::endl;
        std::cout << "Contract processed:" << std::endl;
        std::cout << "  Symbol: " << contract.symbol << std::endl;
        std::cout << "  Theoretical Price: " << contract.theoretical_price << std::endl;
        std::cout << "  Delta: " << contract.delta << std::endl;
        std::cout << "  Implied Vol: " << contract.implied_volatility << std::endl;
        
        // Check if audit.json was updated
        std::ifstream audit_read("audit.json");
        if (audit_read.is_open()) {
            std::string line;
            std::cout << "\nChecking audit.json contents:" << std::endl;
            while (std::getline(audit_read, line)) {
                if (line.find("BlackScholesCalculation") != std::string::npos ||
                    line.find("CUDA") != std::string::npos ||
                    line.find("theoretical_price") != std::string::npos) {
                    std::cout << "FOUND AUDIT ENTRY: " << line << std::endl;
                }
            }
            audit_read.close();
        }
    } else {
        std::cout << "FAILED! Function returned: " << result << std::endl;
    }
    
    // Cleanup
    barracuda_destroy_engine(engine);
    
    return 0;
}