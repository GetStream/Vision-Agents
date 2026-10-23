package phone

func (s *PhoneSuite) TestCountryReadsTheRegionOfANumber() {
	for number, country := range map[string]string{
		"+15125550123":  "US",
		"+14165550123":  "CA",
		"+16045550123":  "CA",
		"+442071838750": "GB",
		"+390212345678": "IT",
		"393123456789":  "IT",
		"15125550123":   "US",
		// An international network belongs to no country, and its vendors price it apart.
		"+882345678901":   "",
		"":                "",
		"user@icloud.com": "",
		"+1":              "",
	} {
		s.Equal(country, Country(number), number)
	}
}
