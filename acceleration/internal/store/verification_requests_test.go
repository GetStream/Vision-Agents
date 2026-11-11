//go:build integration

package store

import "time"

func (s *StoreSuite) verificationRequest(organizationID, legalName string, at time.Time) VerificationRequest {
	return VerificationRequest{
		OrganizationID: organizationID,
		LegalName:      legalName,
		Country:        "NL",
		Website:        "https://example.com",
		TaxID:          "NL123",
		UseCase:        "support",
		CallsPerMonth:  "1000",
		Email:          "ops@example.com",
		SubmittedBy:    "user",
		SubmittedAt:    at,
	}
}

func (s *StoreSuite) TestASecondVerificationRequestReplacesTheFirst() {
	_, err := s.store.SaveVerificationRequest(s.ctx, s.verificationRequest("one", "A", s.base))
	s.Require().NoError(err)
	saved, err := s.store.SaveVerificationRequest(s.ctx, s.verificationRequest("one", "B", s.base.Add(time.Hour)))
	s.Require().NoError(err)
	s.Equal("B", saved.LegalName)

	found, err := s.store.VerificationRequestOf(s.ctx, "one")
	s.Require().NoError(err)
	s.Require().NotNil(found)
	s.Equal("B", found.LegalName)
	s.True(s.base.Add(time.Hour).Equal(found.SubmittedAt))

	listed, err := s.store.ListVerificationRequests(s.ctx, 10, nil)
	s.Require().NoError(err)
	s.Len(listed, 1)
}

func (s *StoreSuite) TestAnOrganizationWithNoVerificationRequestHasNone() {
	found, err := s.store.VerificationRequestOf(s.ctx, "nobody")
	s.Require().NoError(err)
	s.Nil(found)
}

func (s *StoreSuite) TestVerificationRequestsListNewestFirstAndTiesByOrganization() {
	for _, request := range []VerificationRequest{
		s.verificationRequest("a", "A", s.base),
		s.verificationRequest("b", "B", s.base),
		s.verificationRequest("c", "C", s.base.Add(time.Minute)),
	} {
		_, err := s.store.SaveVerificationRequest(s.ctx, request)
		s.Require().NoError(err)
	}

	all, err := s.store.ListVerificationRequests(s.ctx, 10, nil)
	s.Require().NoError(err)
	var order []string
	for _, entry := range all {
		order = append(order, entry.OrganizationID)
	}
	s.Equal([]string{"c", "b", "a"}, order)

	var walked []string
	var after *VerificationPosition
	for range 4 {
		page, err := s.store.ListVerificationRequests(s.ctx, 1, after)
		s.Require().NoError(err)
		if len(page) == 0 {
			break
		}
		walked = append(walked, page[0].OrganizationID)
		after = &VerificationPosition{SubmittedAt: page[0].SubmittedAt, OrganizationID: page[0].OrganizationID}
	}
	s.Equal([]string{"c", "b", "a"}, walked)
}

func (s *StoreSuite) TestAVerificationRequestCarriesTheOrganizationsLastTier() {
	s.Require().NoError(s.store.SetOrganizationTier(s.ctx, "one", "verified", s.base))
	for _, id := range []string{"one", "two"} {
		_, err := s.store.SaveVerificationRequest(s.ctx, s.verificationRequest(id, id, s.base))
		s.Require().NoError(err)
	}

	one, err := s.store.VerificationRequestOf(s.ctx, "one")
	s.Require().NoError(err)
	s.Equal("verified", one.Tier)
	two, err := s.store.VerificationRequestOf(s.ctx, "two")
	s.Require().NoError(err)
	s.Equal("", two.Tier)

	tiers := map[string]string{}
	listed, err := s.store.ListVerificationRequests(s.ctx, 10, nil)
	s.Require().NoError(err)
	for _, entry := range listed {
		tiers[entry.OrganizationID] = entry.Tier
	}
	s.Equal(map[string]string{"one": "verified", "two": ""}, tiers)
}
