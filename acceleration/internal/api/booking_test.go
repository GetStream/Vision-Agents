//go:build integration

package api

import (
	"context"
	"net/http"
	"testing"
	"time"

	"github.com/uptrace/bun"

	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/dlc"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone/telnyx"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone/vendors"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// BookingSuite runs the router with ROUTER_LIMITS_MODE=off and the real usage service: a
// purchase still books its price out of the organization's credit first.
type BookingSuite struct {
	RouterSuite
}

func TestBookingSuite(t *testing.T) {
	runSuite(t, new(BookingSuite))
}

func (s *BookingSuite) SetupSuite() {
	s.limited, s.limitsMode = true, config.LimitsOff
	phones, err := phone.DefaultConfig()
	s.Require().NoError(err)
	for i, vendor := range phones.Vendors {
		if vendor.Vendor == "telnyx" {
			phones.Vendors[i].Credentials = nil
		}
	}
	registry := vendors.Registry(phones)
	registry.Register("telnyx", func() (phone.Provider, error) {
		return telnyx.New(telnyx.Options{APIKey: "k", BaseURL: unreachableVendor})
	})
	s.phoneVendors = registry
	s.RouterSuite.SetupSuite()
}

func (s *BookingSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

func (s *BookingSuite) TestANumberIsNotBoughtWithNoCreditEvenWithLimitsOff() {
	status, payload := s.serverClient.call(http.MethodPost, "/v1/phone/numbers",
		BuyNumberRequest{Vendor: "telnyx", E164: "+15125551234"})

	s.Require().Equal(http.StatusPaymentRequired, status, string(payload))
	s.Contains(string(payload), codeInsufficientCredits)
	s.Contains(string(payload), "add credit first")
}

func (s *BookingSuite) TestANumberIsNotBoughtOnTrialCredit() {
	s.usage.Trial(s.T(), s.app.organization.ID, 5_000_000)

	status, payload := s.serverClient.call(http.MethodPost, "/v1/phone/numbers",
		BuyNumberRequest{Vendor: "telnyx", E164: "+15125551234"})

	s.Require().Equal(http.StatusPaymentRequired, status, string(payload))
	s.Contains(string(payload), codeTopUpRequired)
}

func (s *BookingSuite) TestANumberWithCreditIsAskedOfTheVendor() {
	s.usage.Credit(s.T(), s.app.organization.ID, 5_000_000)

	status, payload := s.serverClient.call(http.MethodPost, "/v1/phone/numbers",
		BuyNumberRequest{Vendor: "telnyx", E164: "+15125551234"})

	s.NotEqual(http.StatusPaymentRequired, status, "the vendor, not the credit, answers: %s", payload)
}

// DLCBookingSuite registers for 10DLC with ROUTER_LIMITS_MODE=off and the real usage service:
// the brand's and campaign's fees are booked before they are registered.
type DLCBookingSuite struct {
	dlcRegistration
}

func TestDLCBookingSuite(t *testing.T) {
	runSuite(t, new(DLCBookingSuite))
}

func (s *DLCBookingSuite) SetupSuite() {
	s.limited, s.limitsMode = true, config.LimitsOff
	s.dlcRegistration.SetupSuite()
}

func (s *DLCBookingSuite) TestAUseCaseWhoseFeesTheCreditDoesNotCoverIsNotSubmitted() {
	s.completeProfile()
	var created UseCase
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/phone/use-cases", completeUseCase("support"), &created))

	status, payload := s.serverClient.call(http.MethodPost, "/v1/phone/use-cases/"+created.Id+"/submit", nil)

	s.Require().Equal(http.StatusPaymentRequired, status, string(payload))
	s.Contains(string(payload), codeInsufficientCredits)
}

func (s *DLCBookingSuite) TestAUseCaseIsNotSubmittedOnTrialCredit() {
	s.usage.Trial(s.T(), s.app.organization.ID, 100_000_000)
	s.completeProfile()
	var created UseCase
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/phone/use-cases", completeUseCase("support"), &created))

	status, payload := s.serverClient.call(http.MethodPost, "/v1/phone/use-cases/"+created.Id+"/submit", nil)

	s.Require().Equal(http.StatusPaymentRequired, status, string(payload))
	s.Contains(string(payload), codeTopUpRequired)
}

func (s *DLCBookingSuite) TestAUseCaseWhoseFeesAreCoveredIsRegisteredAndBilled() {
	s.usage.Credit(s.T(), s.app.organization.ID, 100_000_000)
	sent := s.submitted()

	status, reviewed := s.review(sent.Id, dlc.Approve, "")

	s.Require().Equal(http.StatusOK, status)
	s.NotNil(reviewed.VendorCampaignId)
	s.Require().Eventually(func() bool {
		var billed int
		s.Require().NoError(s.store.ShardDB().NewSelect().Model((*store.Request)(nil)).ColumnExpr("count(*)").
			Where("customer_id = ?", s.customerID()).Where("model IN (?)", bun.In([]string{dlc.ModelBrand, dlc.ModelCampaign})).
			Scan(context.Background(), &billed))
		return billed == 2
	}, settleFor, 50*time.Millisecond, "the brand and the campaign were billed")
}
