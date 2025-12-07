#include <stdio.h>
#include <stddef.h>

typedef struct {
    char symbol[32];
    double strike_price;
    double underlying_price;
    double time_to_expiration;
    double risk_free_rate;
    double volatility;
    char option_type;
    double delta;
    double gamma;
    double theta;
    double vega;
    double rho;
    double theoretical_price;
} COptionContract;

int main() {
    printf("sizeof(COptionContract) = %zu\n", sizeof(COptionContract));
    printf("offsetof(symbol) = %zu\n", offsetof(COptionContract, symbol));
    printf("offsetof(strike_price) = %zu\n", offsetof(COptionContract, strike_price));
    printf("offsetof(underlying_price) = %zu\n", offsetof(COptionContract, underlying_price));
    printf("offsetof(time_to_expiration) = %zu\n", offsetof(COptionContract, time_to_expiration));
    printf("offsetof(risk_free_rate) = %zu\n", offsetof(COptionContract, risk_free_rate));
    printf("offsetof(volatility) = %zu\n", offsetof(COptionContract, volatility));
    printf("offsetof(option_type) = %zu\n", offsetof(COptionContract, option_type));
    printf("offsetof(delta) = %zu\n", offsetof(COptionContract, delta));
    printf("offsetof(gamma) = %zu\n", offsetof(COptionContract, gamma));
    printf("offsetof(theta) = %zu\n", offsetof(COptionContract, theta));
    printf("offsetof(vega) = %zu\n", offsetof(COptionContract, vega));
    printf("offsetof(rho) = %zu\n", offsetof(COptionContract, rho));
    printf("offsetof(theoretical_price) = %zu\n", offsetof(COptionContract, theoretical_price));
    return 0;
}
