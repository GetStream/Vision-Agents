//go:build integration

package store

import (
	"errors"
)

// useCase creates a draft use case for a customer.
func (s *StoreSuite) useCase(customerID, name string) UseCase {
	useCase := UseCase{CustomerID: customerID, Name: name, Status: "draft", UseCaseType: "CUSTOMER_CARE"}
	s.Require().NoError(s.store.CreateUseCase(s.ctx, &useCase))
	return useCase
}

// A number assigned to no use case still has to send as something.
func (s *StoreSuite) TestAnAppsFirstUseCaseIsItsDefaultUntilAnotherTakesOver() {
	customerID := newID()
	first := s.useCase(customerID, "support")
	second := s.useCase(customerID, "reminders")
	s.True(first.IsDefault)
	s.False(second.IsDefault)

	second.IsDefault = true
	s.Require().NoError(s.store.UpdateUseCase(s.ctx, &second))

	first, err := s.store.UseCase(s.ctx, customerID, first.ID)
	s.Require().NoError(err)
	s.False(first.IsDefault)
	_, err = s.store.UseCase(s.ctx, newID(), second.ID)
	s.ErrorIs(err, ErrUnknownUseCase, "another app does not see it")
}

// Two reviewers acting on the same use case at once must not both win.
func (s *StoreSuite) TestMovingAUseCaseFromAStatusItLeftWritesNothing() {
	useCase := s.useCase(newID(), "support")

	useCase.Status = "submitted"
	s.Require().NoError(s.store.MoveUseCase(s.ctx, &useCase, "draft", &ReviewLog{Actor: "app"}))
	useCase.Status = "approved"
	err := s.store.MoveUseCase(s.ctx, &useCase, "draft", &ReviewLog{Actor: "staff"})
	s.ErrorIs(err, ErrUseCaseMoved)

	logs, err := s.store.ReviewLogs(s.ctx, useCase.CustomerID, useCase.ID, 0, nil)
	s.Require().NoError(err)
	s.Require().Len(logs, 1)
	s.Equal("draft", logs[0].FromStatus)
	s.Equal("submitted", logs[0].ToStatus)
	held, err := s.store.UseCase(s.ctx, useCase.CustomerID, useCase.ID)
	s.Require().NoError(err)
	s.Equal("submitted", held.Status)
}

func (s *StoreSuite) TestTheDefaultUseCaseSendsForEveryNumberNotAssignedElsewhere() {
	customerID := newID()
	fallback := s.useCase(customerID, "support")
	reminders := s.useCase(customerID, "reminders")
	var numbers []string
	for range 2 {
		number := PhoneNumber{CustomerID: customerID, E164: "+1555" + newID()[:7], Vendor: "telnyx"}
		s.Require().NoError(s.store.RecordNumber(s.ctx, &number))
		numbers = append(numbers, number.E164)
	}

	s.Require().NoError(s.store.AssignNumbers(s.ctx, customerID, reminders.ID, numbers[:1]))

	assigned, err := s.store.UseCaseNumbers(s.ctx, reminders)
	s.Require().NoError(err)
	rest, err := s.store.UseCaseNumbers(s.ctx, fallback)
	s.Require().NoError(err)
	s.Require().Len(assigned, 1)
	s.Require().Len(rest, 1)
	s.Equal(numbers[0], assigned[0].E164)
	s.Equal(numbers[1], rest[0].E164)
	s.Error(s.store.AssignNumbers(s.ctx, customerID, reminders.ID, []string{"+15550000000"}),
		"a number the app does not hold")
}

// START lifts what STOP did, and the record of both stays.
func (s *StoreSuite) TestAnOptOutFromEveryChannelBlocksEachUntilRevoked() {
	customerID := newID()
	first := OptOut{CustomerID: customerID, Recipient: "+15551230000", Channel: OptOutAll, Source: "api"}
	again := first
	s.Require().NoError(s.store.OptOut(s.ctx, &first))
	s.Require().NoError(s.store.OptOut(s.ctx, &again))
	s.Equal(first.ID, again.ID, "asking twice keeps the first")

	blocked, err := s.store.OptedOut(s.ctx, customerID, "+15551230000", "sms")
	s.Require().NoError(err)
	s.True(blocked)

	s.Require().NoError(s.store.RevokeOptOuts(s.ctx, customerID, "+15551230000", "sms"))
	blocked, err = s.store.OptedOut(s.ctx, customerID, "+15551230000", "voice")
	s.Require().NoError(err)
	s.False(blocked)
	s.True(errors.Is(s.store.RevokeOptOut(s.ctx, customerID, first.ID), ErrUnknownOptOut), "already revoked")
}

// The brand is the vendor's: saving the profile again must not forget it was registered.
func (s *StoreSuite) TestSavingABusinessProfileAgainKeepsItsBrand() {
	customerID := newID()
	profile := BusinessProfile{CustomerID: customerID, LegalBusinessName: "Acme Inc", Address: PostalAddress{Country: "US"}}
	s.Require().NoError(s.store.SaveBusinessProfile(s.ctx, &profile))
	s.Require().NoError(s.store.SetBrand(s.ctx, customerID, "brand-1", "VERIFIED"))

	profile = BusinessProfile{CustomerID: customerID, LegalBusinessName: "Acme Corp"}
	s.Require().NoError(s.store.SaveBusinessProfile(s.ctx, &profile))

	held, err := s.store.BusinessProfile(s.ctx, customerID)
	s.Require().NoError(err)
	s.Equal("Acme Corp", held.LegalBusinessName)
	s.Equal("brand-1", held.VendorBrandID)
	s.Equal("brand-1", profile.VendorBrandID, "the saved row is handed back")
}

func (s *StoreSuite) TestSandboxRecipientsAreReplacedWhole() {
	customerID := newID()
	s.Require().NoError(s.store.SetSandboxRecipients(s.ctx, customerID, []string{"+15551110000", "+15552220000"}))
	s.Require().NoError(s.store.SetSandboxRecipients(s.ctx, customerID, []string{"+15553330000"}))

	recipients, err := s.store.SandboxRecipients(s.ctx, customerID)
	s.Require().NoError(err)
	s.Equal([]string{"+15553330000"}, recipients)
}
