package dlc

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

type DLCSuite struct {
	suite.Suite
}

func TestDLCSuite(t *testing.T) {
	suite.Run(t, new(DLCSuite))
}

func (s *DLCSuite) TestAUseCaseMissingWhatTheRegistryAsksForIsNotSubmittable() {
	err := ValidateUseCase(store.UseCase{UseCaseType: "CUSTOMER_CARE", MessageSamples: []string{"hi"}})
	s.ErrorIs(err, ErrInvalid)
	s.ErrorContains(err, "message_samples")
	s.ErrorContains(err, "opt_out_message")
}

func (s *DLCSuite) TestACompleteUseCaseIsSubmittable() {
	s.NoError(ValidateUseCase(store.UseCase{
		UseCaseType:    "CUSTOMER_CARE",
		Description:    "Order updates and support replies for Acme customers who wrote in.",
		MessageFlow:    "Customers opt in by ticking a box on the checkout page at acme.example.",
		MessageSamples: []string{"Your order 123 has shipped.", "Reply HELP for help, STOP to stop."},
		HelpMessage:    "Acme support: help@acme.example",
		OptOutMessage:  "You will receive no more messages from Acme.",
	}))
}

// A sole proprietor has no tax id to give, and a public company has a ticker to.
func (s *DLCSuite) TestWhatAProfileNeedsDependsOnWhatKindOfBusinessItIs() {
	sole := store.BusinessProfile{LegalEntityType: "sole_proprietor"}
	public := store.BusinessProfile{LegalEntityType: "corporation", OrganizationType: "public"}
	s.NotContains(ValidateProfile(sole).Error(), "tax_id")
	s.ErrorContains(ValidateProfile(sole), "authorized_contact_first_name")
	s.ErrorContains(ValidateProfile(public), "tax_id")
	s.ErrorContains(ValidateProfile(public), "stock_symbol")
}

func (s *DLCSuite) TestOnlyAUseCaseTheVendorDoesNotHoldMayBeEditedOrDeleted() {
	s.True(Editable(ChangesRequested))
	s.False(Editable(Submitted))
	s.False(Deletable(VendorPending))
	s.True(Deletable(Rejected))
}
