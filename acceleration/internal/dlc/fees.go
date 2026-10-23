package dlc

import (
	"context"
	"time"

	"github.com/google/uuid"

	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// The models a registrar's fees are priced as, under the registrar's name.
const (
	// ModelBrand is registering a business, a fee charged once per brand.
	ModelBrand = "dlc_brand"
	// ModelCampaign is the carriers' review of a campaign, charged per submission.
	ModelCampaign = "dlc_campaign"
	// ModelCampaignMonth is a month of a registered campaign, by its use case type.
	ModelCampaignMonth = "dlc_campaign_month"
)

// The attributes those fees are priced by.
const (
	AttributeEntityType  = "entity_type"
	AttributeUseCaseType = "use_case_type"
)

// UpfrontMonths is how many of a campaign's months a registrar charges when it registers it.
const UpfrontMonths = 3

// OriginFees is the origin of a registrar's fees when no request asked for the work: the
// poller registering what a report or a review left waiting, and each month of a campaign.
var OriginFees = routing.JobOrigin("dlc_fees")

// Used is the one registration a brand is billed for.
func (Brand) Used() pricing.Quantities { return pricing.Quantities{pricing.Requests: 1} }

// UsageID names the brand, so a brand reported twice is billed once.
func (b Brand) UsageID() uuid.UUID {
	if b.ID == "" {
		return uuid.Nil
	}
	return routing.UsageID("dlc", "brand", b.ID)
}

// Used is the one review a campaign is billed for.
func (Campaign) Used() pricing.Quantities { return pricing.Quantities{pricing.Requests: 1} }

// UsageID names the campaign, so a campaign reported twice is billed once.
func (c Campaign) UsageID() uuid.UUID {
	if c.ID == "" {
		return uuid.Nil
	}
	return routing.UsageID("dlc", "campaign", c.ID)
}

// feesOwner is who a registration for an app is billed to: the request that asked for it, or
// the app under OriginFees.
func feesOwner(ctx context.Context, customerID string) routing.Owner {
	attribution := routing.AttributionFrom(ctx)
	if attribution.APIKey == "" {
		attribution = routing.Attribution{APIKey: OriginFees}
	}
	return routing.Owner{CustomerID: customerID, Attribution: attribution}
}

// feeMeter records one of the registrar's fees for an app.
func (s *Service) feeMeter(ctx context.Context, model, customerID string, attributes map[string]string) routing.Meter {
	return routing.Meter{
		Recorder:   s.recorder,
		Config:     routing.ProviderConfig{Provider: s.registrar.Name(), Model: model},
		Owner:      feesOwner(ctx, customerID),
		Attributes: attributes,
	}
}

// campaignMonth is the meter of one calendar month of a use case's campaign: the same month
// is billed once, however many times it is recorded.
func (s *Service) campaignMonth(useCase store.UseCase, month time.Time) routing.Meter {
	month = time.Date(month.Year(), month.Month(), 1, 0, 0, 0, 0, time.UTC)
	return routing.Meter{
		Recorder: s.recorder,
		Config:   routing.ProviderConfig{Provider: useCase.Vendor, Model: ModelCampaignMonth},
		Owner:    routing.Owner{CustomerID: useCase.CustomerID, Attribution: routing.Attribution{APIKey: OriginFees}},
		UsageID: routing.UsageID("dlc_campaign", useCase.Vendor, useCase.VendorCampaignID,
			month.Format("2006-01")),
		BilledAt:   month,
		Attributes: map[string]string{AttributeUseCaseType: useCase.UseCaseType},
		Unmeasured: true,
	}
}

// billUpfront bills the months a registrar charges for when it registers a campaign: this
// one and the next UpfrontMonths-1, each named as the monthly job names it, so the job does
// not bill them again.
func (s *Service) billUpfront(useCase store.UseCase, registered time.Time) {
	for k := range UpfrontMonths {
		month := time.Date(registered.Year(), registered.Month()+time.Month(k), 1, 0, 0, 0, 0, time.UTC)
		meter := s.campaignMonth(useCase, month)
		// Charged now, though it bills a month to come.
		meter.BilledAt = registered
		s.settle(meter)
	}
}

// BillCampaigns bills the current month of every campaign a vendor holds that has no row for
// it yet, and reports how many it billed.
func (s *Service) BillCampaigns(ctx context.Context) (int, error) {
	if s.recorder == nil {
		return 0, nil
	}
	useCases, err := s.store.UseCasesWithCampaigns(ctx, VendorPending, Approved)
	if err != nil {
		return 0, err
	}
	now := s.now().UTC()
	ids := make([]uuid.UUID, 0, len(useCases))
	for _, useCase := range useCases {
		ids = append(ids, s.campaignMonth(useCase, now).UsageID)
	}
	recorded, err := s.store.RecordedUsage(ctx, ids)
	if err != nil {
		return 0, err
	}
	billed := 0
	for _, useCase := range useCases {
		meter := s.campaignMonth(useCase, now)
		if recorded[meter.UsageID] {
			continue
		}
		s.settle(meter)
		billed++
	}
	return billed, nil
}

// RunFees bills the campaigns' months at once and then every interval until ctx ends.
func (s *Service) RunFees(ctx context.Context, every time.Duration) {
	ticker := time.NewTicker(every)
	defer ticker.Stop()
	for {
		if billed, err := s.BillCampaigns(ctx); err != nil {
			s.logger.Error("could not bill the months of registered campaigns", "error", err)
		} else if billed > 0 {
			s.logger.Info("billed the months of registered campaigns", "months", billed)
		}
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}
	}
}

// settle records a fee billed with no request in hand.
func (s *Service) settle(meter routing.Meter) {
	if err := routing.Settle(meter, pricing.Quantities{pricing.Requests: 1}); err != nil {
		s.logger.Error("could not bill a 10DLC fee", "model", meter.Config.Model, "error", err)
	}
}

// now is the service's clock, which a campaign's months are counted by.
func (s *Service) now() time.Time {
	if s.clock != nil {
		return s.clock()
	}
	return time.Now()
}
