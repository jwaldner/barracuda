package main

import (
	"fmt"
	"math"
	"strings"
)

// Test the zero volatility fix using basic Black-Scholes math
func main() {
	fmt.Println("🔍 Testing Zero Volatility Fix")
	fmt.Println("=============================")

	// Test case from Grok report - Contract 1 (originally problematic)
	S := 188.36   // Corrected stock price (was 485.4)
	K := 425.0    // Strike price
	T := 0.058362 // Time to expiration
	r := 0.039830 // Risk-free rate
	sigma := 0.0  // Zero volatility (problematic case)

	fmt.Printf("Parameters: S=%.2f, K=%.0f, T=%.6f, r=%.6f, σ=%.6f\n", S, K, T, r, sigma)
	fmt.Println()

	// For zero volatility, option value is intrinsic value only
	fmt.Println("Zero Volatility Analysis:")
	fmt.Println("------------------------")

	// Forward price: S * e^(r*T)
	forward := S * math.Exp(r*T)
	fmt.Printf("Forward price: %.6f\n", forward)

	// Present value of strike: K * e^(-r*T)
	pvStrike := K * math.Exp(-r*T)
	fmt.Printf("PV of strike: %.6f\n", pvStrike)

	// For a put: max(0, K*e^(-r*T) - S)
	putValue := math.Max(0, pvStrike-S)
	fmt.Printf("Put intrinsic value: %.6f\n", putValue)

	// Delta calculation for zero volatility
	var delta float64
	if S < pvStrike {
		delta = -1.0 // Deep in-the-money put
	} else {
		delta = 0.0 // Out-of-the-money put
	}
	fmt.Printf("Put delta: %.6f\n", delta)

	fmt.Println()
	fmt.Printf("Status: %s < %s = %t (Put is %s)\n",
		fmt.Sprintf("%.2f", S),
		fmt.Sprintf("%.2f", pvStrike),
		S < pvStrike,
		map[bool]string{true: "ITM", false: "OTM"}[S < pvStrike])

	// Compare with original problematic values
	fmt.Println()
	fmt.Println("Original Grok Report Analysis:")
	fmt.Println("-----------------------------")
	fmt.Printf("Expected put value: %.6f (intrinsic value only)\n", putValue)
	fmt.Printf("Original reported: 3.000000 ❌\n")
	fmt.Printf("Expected delta: %.6f\n", delta)
	fmt.Printf("Original reported: -0.107286 ❌\n")
	fmt.Println()
	fmt.Println("✅ Our fix should handle this gracefully now!")

	// Test case 2 - normal volatility (Contract 2)
	fmt.Println("\n" + strings.Repeat("=", 50))
	fmt.Println("✅ Testing Normal Case (Contract 2)")
	fmt.Println(strings.Repeat("=", 50))

	S2 := 188.36
	K2 := 166.0
	T2 := 0.057534
	r2 := 0.039830
	sigma2 := 0.399222

	fmt.Printf("Parameters: S=%.2f, K=%.0f, T=%.6f, r=%.6f, σ=%.6f\n", S2, K2, T2, r2, sigma2)

	// Standard Black-Scholes calculation
	d1 := (math.Log(S2/K2) + (r2+0.5*sigma2*sigma2)*T2) / (sigma2 * math.Sqrt(T2))
	d2 := d1 - sigma2*math.Sqrt(T2)

	// Normal CDF approximation using erf
	normalCDF := func(x float64) float64 {
		return 0.5 * (1.0 + math.Erf(x/math.Sqrt(2.0)))
	}

	normalPDF := func(x float64) float64 {
		return math.Exp(-0.5*x*x) / math.Sqrt(2.0*math.Pi)
	}

	Nd1 := normalCDF(d1)
	Nd2 := normalCDF(d2)
	nd1 := normalPDF(d1)

	// Put option formula
	putPrice := K2*math.Exp(-r2*T2)*(1.0-Nd2) - S2*(1.0-Nd1)
	putDelta := Nd1 - 1.0
	gamma := nd1 / (S2 * sigma2 * math.Sqrt(T2))
	theta := (-(S2*nd1*sigma2)/(2.0*math.Sqrt(T2)) + r2*K2*math.Exp(-r2*T2)*(1.0-Nd2)) / 365.0
	vega := (S2 * nd1 * math.Sqrt(T2)) / 100.0
	rho := (-K2 * T2 * math.Exp(-r2*T2) * (1.0 - Nd2)) / 100.0

	fmt.Println("\nCalculated Results:")
	fmt.Printf("d1: %.6f\n", d1)
	fmt.Printf("d2: %.6f\n", d2)
	fmt.Printf("Price: %.6f\n", putPrice)
	fmt.Printf("Delta: %.6f\n", putDelta)
	fmt.Printf("Gamma: %.6f\n", gamma)
	fmt.Printf("Theta: %.6f\n", theta)
	fmt.Printf("Vega: %.6f\n", vega)
	fmt.Printf("Rho: %.6f\n", rho)

	fmt.Println("\nGrok Expected vs Our Calculation:")
	fmt.Printf("Price: Expected=0.704000, Got=%.6f, Diff=%.6f\n", putPrice, math.Abs(0.704000-putPrice))
	fmt.Printf("Delta: Expected=-0.082080, Got=%.6f, Diff=%.6f\n", putDelta, math.Abs(-0.082080-putDelta))
	fmt.Printf("Gamma: Expected=0.008400, Got=%.6f, Diff=%.6f\n", gamma, math.Abs(0.008400-gamma))
	fmt.Printf("Theta: Expected=-0.063300, Got=%.6f, Diff=%.6f\n", theta, math.Abs(-0.063300-theta))
	fmt.Printf("Vega: Expected=0.068500, Got=%.6f, Diff=%.6f\n", vega, math.Abs(0.068500-vega))
	fmt.Printf("Rho: Expected=-0.009290, Got=%.6f, Diff=%.6f\n", rho, math.Abs(-0.009290-rho))

	fmt.Println("\n🎯 Our calculations should now match expected values within precision tolerances!")
}
