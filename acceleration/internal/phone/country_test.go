package phone

import "github.com/GetStream/Vision-Agents/acceleration/internal/routing"

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

func (s *PhoneSuite) TestASearchOutsideTheUSAndCanadaIsRefusedBeforeAVendorIsAsked() {
	vendor := &stub{vendor: "twilio", offers: []Available{{E164: "+442071838750", Vendor: "twilio"}}}
	service := s.serviceWith(vendor)

	_, err := service.SearchAll(s.ctx, Search{Country: "GB"})
	s.ErrorIs(err, ErrInvalid)
	s.ErrorContains(err, "US")
	s.ErrorContains(err, "CA")

	_, err = service.Search(s.ctx, "twilio", Search{Country: "GB"})
	s.ErrorIs(err, ErrInvalid)

	s.Zero(vendor.searched)
}

func (s *PhoneSuite) TestASearchInCanadaStillReachesTheVendor() {
	vendor := &stub{vendor: "twilio", offers: []Available{{E164: "+14165550123", Vendor: "twilio"}}}
	service := s.serviceWith(vendor)

	offers, err := service.SearchAll(s.ctx, Search{Country: "CA"})
	s.Require().NoError(err)
	s.Equal([]string{"+14165550123"}, numbers(offers))

	_, err = service.Search(s.ctx, "twilio", Search{Country: "ca"})
	s.Require().NoError(err)
}

func (s *PhoneSuite) TestANumberOutsideTheUSAndCanadaIsNotBought() {
	vendor := &stub{vendor: "twilio"}
	service := s.serviceWith(vendor)

	_, err := service.Buy(s.ctx, Purchase{Vendor: "twilio", E164: "+442071838750", Country: "GB", Owner: routing.Owner{CustomerID: "c"}})

	s.ErrorIs(err, ErrInvalid)
	s.ErrorContains(err, "US")
	s.Zero(vendor.ordered)
}

func (s *PhoneSuite) TestANumberSharingTheUSDialCodeIsNotBoughtWhateverCountryTheRequestNames() {
	vendor := &stub{vendor: "twilio"}
	service := s.serviceWith(vendor)

	// Jamaica, Puerto Rico and the Bahamas dial +1 like the US.
	for _, e164 := range []string{"+18765551234", "+17875551234", "+12425551234"} {
		_, err := service.Buy(s.ctx, Purchase{Vendor: "twilio", E164: e164, Country: "US", Owner: routing.Owner{CustomerID: "c"}})
		s.ErrorIs(err, ErrInvalid, e164)
	}
	s.Zero(vendor.ordered)
}

func (s *PhoneSuite) TestAUSNumberOfferedFromAnotherCountryIsNotBought() {
	vendor := &stub{vendor: "twilio"}
	service := s.serviceWith(vendor)

	_, err := service.Buy(s.ctx, Purchase{Vendor: "twilio", E164: "+15125550123", Country: "GB", Owner: routing.Owner{CustomerID: "c"}})

	s.ErrorIs(err, ErrInvalid)
	s.Zero(vendor.ordered)
}

func (s *PhoneSuite) TestNumbersInTheUSAndCanadaAreStillBought() {
	vendor := &stub{vendor: "twilio"}
	service := s.serviceWith(vendor)

	for _, purchase := range []Purchase{
		{Vendor: "twilio", E164: "+15125550123", Country: "US"},
		{Vendor: "twilio", E164: "+14165550123", Country: "CA"},
		{Vendor: "twilio", E164: "+16045550123"},
	} {
		purchase.Owner = routing.Owner{CustomerID: "c"}
		bought, err := service.Buy(s.ctx, purchase)
		s.Require().NoError(err, purchase.E164)
		s.Equal(purchase.E164, bought.E164)
	}
	s.Equal(3, vendor.ordered)
}
