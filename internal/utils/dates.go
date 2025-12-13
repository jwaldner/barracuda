package utils

import "time"

// CalculateNextOptionsExpiration returns the next third Friday for options expiration
// This implements the standard options expiration business logic:
// - Third Friday of current month if we haven't reached the expiration week yet
// - Third Friday of next month if we're in or past the expiration week
func CalculateNextOptionsExpiration() string {
	today := time.Now()
	currentMonth := today.Month()
	currentYear := today.Year()

	// Find 3rd Friday of current month
	firstDay := time.Date(currentYear, currentMonth, 1, 0, 0, 0, 0, today.Location())
	firstFriday := firstDay
	for firstFriday.Weekday() != time.Friday {
		firstFriday = firstFriday.AddDate(0, 0, 1)
	}
	thirdFriday := firstFriday.AddDate(0, 0, 14)

	// If current day is in the week of 3rd Friday or past it, use next month's 3rd Friday
	weekStart := thirdFriday.AddDate(0, 0, -7)

	if today.After(weekStart) || today.Equal(weekStart) {
		// Use next month's 3rd Friday
		nextMonth := currentMonth + 1
		nextYear := currentYear
		if nextMonth > 12 {
			nextMonth = 1
			nextYear++
		}
		nextFirstDay := time.Date(nextYear, nextMonth, 1, 0, 0, 0, 0, today.Location())
		nextFirstFriday := nextFirstDay
		for nextFirstFriday.Weekday() != time.Friday {
			nextFirstFriday = nextFirstFriday.AddDate(0, 0, 1)
		}
		nextThirdFriday := nextFirstFriday.AddDate(0, 0, 14)
		return nextThirdFriday.Format("2006-01-02")
	}

	return thirdFriday.Format("2006-01-02")
}
