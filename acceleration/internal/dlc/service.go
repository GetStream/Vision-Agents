package dlc

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// HookPath is where a registrar reports on brands and campaigns.
const HookPath = "/v1/phone/hooks/10dlc"

// Service moves use cases through review and registration.
type Service struct {
	store     *store.Store
	registrar Registrar
	hookURL   string
	logger    *slog.Logger
}

// Options configures a Service. Without a registrar Stream's approval is the last word,
// which is what a deployment registering nothing with any vendor has.
type Options struct {
	Store     *store.Store
	Registrar Registrar
	// PublicURL is where the registrar's reports reach this router.
	PublicURL string
	Logger    *slog.Logger
}

// NewService returns a Service.
func NewService(options Options) (*Service, error) {
	if options.Store == nil {
		return nil, errors.New("dlc: a store is required")
	}
	if options.Logger == nil {
		options.Logger = slog.Default()
	}
	hookURL := ""
	if options.PublicURL != "" {
		hookURL = options.PublicURL + HookPath
	}
	return &Service{store: options.Store, registrar: options.Registrar, hookURL: hookURL, logger: options.Logger}, nil
}

// Run checks the use cases a vendor holds every interval until ctx ends, for the reports a
// hook missed.
func (s *Service) Run(ctx context.Context, every time.Duration) {
	ticker := time.NewTicker(every)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			s.Poll(ctx)
		}
	}
}

// Poll asks the vendor where the use cases it holds stand, a page of them, those that
// moved longest ago first. One that moves goes to the back of the queue.
func (s *Service) Poll(ctx context.Context) {
	if s.registrar == nil {
		return
	}
	useCases, err := s.store.UseCasesInStatus(ctx, VendorPending, 0, nil)
	if err != nil {
		s.logger.Error("could not list the use cases a vendor holds", "error", err)
		return
	}
	for _, useCase := range useCases {
		if err := s.refresh(ctx, useCase, nil); err != nil {
			s.logger.Error("could not refresh a use case", "use_case", useCase.ID, "error", err)
		}
	}
}

// Create records a new draft use case and the numbers that send as it.
func (s *Service) Create(ctx context.Context, useCase *store.UseCase, numbers []string) error {
	useCase.Status = Draft
	if err := s.store.CreateUseCase(ctx, useCase); err != nil {
		return err
	}
	return s.assign(ctx, *useCase, numbers)
}

// Update replaces what an app wrote on a use case it may still edit. numbers nil leaves
// the assignment as it was.
func (s *Service) Update(ctx context.Context, useCase *store.UseCase, numbers []string) error {
	if !Editable(useCase.Status) {
		return stack.Wrap(fmt.Errorf("%w: a %s use case cannot be edited", ErrLocked, useCase.Status))
	}
	if err := s.store.UpdateUseCase(ctx, useCase); err != nil {
		return err
	}
	if numbers == nil {
		return nil
	}
	return s.assign(ctx, *useCase, numbers)
}

func (s *Service) assign(ctx context.Context, useCase store.UseCase, numbers []string) error {
	if err := s.store.AssignNumbers(ctx, useCase.CustomerID, useCase.ID, numbers); err != nil {
		return stack.Wrap(fmt.Errorf("%w: %w", ErrInvalid, err))
	}
	return nil
}

// Delete removes a use case the vendor does not hold.
func (s *Service) Delete(ctx context.Context, customerID, id string) error {
	useCase, err := s.store.UseCase(ctx, customerID, id)
	if err != nil {
		return err
	}
	if !Deletable(useCase.Status) {
		return stack.Wrap(fmt.Errorf("%w: a %s use case cannot be deleted", ErrLocked, useCase.Status))
	}
	return s.store.DeleteUseCase(ctx, customerID, id)
}

// Submit sends a use case to Stream's review, once it and the app's profile are complete.
func (s *Service) Submit(ctx context.Context, customerID, id string) (store.UseCase, error) {
	useCase, err := s.store.UseCase(ctx, customerID, id)
	if err != nil {
		return store.UseCase{}, err
	}
	if !Editable(useCase.Status) {
		return store.UseCase{}, stack.Wrap(fmt.Errorf("%w: a %s use case cannot be submitted", ErrLocked, useCase.Status))
	}
	profile, err := s.store.BusinessProfile(ctx, customerID)
	if errors.Is(err, store.ErrNoBusinessProfile) {
		return store.UseCase{}, stack.Wrap(fmt.Errorf("%w: save a business profile first", ErrInvalid))
	}
	if err != nil {
		return store.UseCase{}, err
	}
	if err := errors.Join(ValidateProfile(profile), ValidateUseCase(useCase)); err != nil {
		return store.UseCase{}, stack.Wrap(err)
	}

	from := useCase.Status
	now := time.Now().UTC()
	useCase.Status, useCase.SubmittedAt = Submitted, &now
	err = s.store.MoveUseCase(ctx, &useCase, from, &store.ReviewLog{Actor: ActorApp, SubmittedAt: &now})
	return useCase, err
}

// Review records Stream's decision on a submitted use case. Approving sends it to the
// vendor, or approves it outright with no vendor to send it to.
func (s *Service) Review(ctx context.Context, id, decision, notes, reviewer string) (store.UseCase, error) {
	useCase, err := s.store.UseCaseByID(ctx, id)
	if err != nil {
		return store.UseCase{}, err
	}
	if useCase.Status != Submitted {
		return store.UseCase{}, stack.Wrap(fmt.Errorf("%w: only a submitted use case is reviewed, this one is %s", ErrLocked, useCase.Status))
	}
	log := &store.ReviewLog{Actor: ActorStaff, ActorName: reviewer, Notes: notes}
	switch decision {
	case Reject:
		useCase.Status = Rejected
	case RequestChanges:
		useCase.Status = ChangesRequested
	case Approve:
		if s.registrar == nil {
			approve(&useCase, log)
			break
		}
		useCase.Status = VendorPending
		useCase.Vendor = s.registrar.Name()
	default:
		return store.UseCase{}, stack.Wrap(fmt.Errorf("%w: decision %q", ErrInvalid, decision))
	}
	if err := s.store.MoveUseCase(ctx, &useCase, Submitted, log); err != nil {
		return store.UseCase{}, err
	}
	if useCase.Status != VendorPending {
		return useCase, nil
	}
	if err := s.refresh(ctx, useCase, nil); err != nil {
		// The poller tries again: the review itself is recorded, and the vendor being down
		// is no reason to make the reviewer decide twice.
		s.logger.Error("could not register a use case with its vendor", "use_case", id, "error", err)
	}
	return s.store.UseCaseByID(ctx, id)
}

// Hook takes a vendor's report on a brand or a campaign, verified, and moves whichever use
// cases it is about.
func (s *Service) Hook(ctx context.Context, header http.Header, body []byte) error {
	if s.registrar == nil {
		return fmt.Errorf("%w: no registrar is configured", ErrRefused)
	}
	if err := s.registrar.VerifyHook(header, body, time.Now()); err != nil {
		return err
	}
	campaignID, brandID := s.registrar.HookSubject(body)
	if campaignID != "" {
		useCase, err := s.store.UseCaseByCampaign(ctx, s.registrar.Name(), campaignID)
		if errors.Is(err, store.ErrUnknownUseCase) {
			return nil
		}
		if err != nil {
			return err
		}
		return s.refresh(ctx, useCase, body)
	}
	if brandID != "" {
		// A brand report is the moment to register what waited on it, which Poll does.
		s.Poll(ctx)
	}
	return nil
}

// refresh asks the vendor where a use case it holds stands, registering the brand and the
// campaign it is still missing, and moves it to where the vendor says.
func (s *Service) refresh(ctx context.Context, useCase store.UseCase, report []byte) error {
	if s.registrar == nil || useCase.Status != VendorPending {
		return nil
	}
	if useCase.VendorCampaignID == "" {
		return s.register(ctx, useCase)
	}
	campaign, err := s.registrar.Campaign(ctx, useCase.VendorCampaignID)
	if err != nil {
		return err
	}
	if report == nil {
		report = campaign.Raw
	}
	log := &store.ReviewLog{Actor: ActorVendor, ActorName: s.registrar.Name(), VendorPayload: report}
	switch campaign.Outcome {
	case Accepted:
		if err := s.assignAtVendor(ctx, useCase); err != nil {
			return err
		}
		useCase.VendorStatus = campaign.Status
		approve(&useCase, log)
	case Failed:
		useCase.VendorStatus, useCase.Status = campaign.Status, VendorRejected
		log.Notes = "The vendor answered " + campaign.Status + "."
	default:
		if campaign.Status == useCase.VendorStatus {
			return nil
		}
		useCase.VendorStatus = campaign.Status
	}
	return s.store.MoveUseCase(ctx, &useCase, VendorPending, log)
}

// register registers the app's brand if it has none, and the campaign once the brand is
// verified. A brand the vendor refused leaves nothing to register the campaign under.
func (s *Service) register(ctx context.Context, useCase store.UseCase) error {
	profile, err := s.store.BusinessProfile(ctx, useCase.CustomerID)
	if err != nil {
		return err
	}
	var brand Brand
	if profile.VendorBrandID == "" {
		brand, err = s.registrar.RegisterBrand(ctx, profile)
	} else {
		brand, err = s.registrar.Brand(ctx, profile.VendorBrandID)
	}
	if err != nil {
		return err
	}
	if brand.ID != profile.VendorBrandID || brand.Status != profile.BrandStatus {
		if err := s.store.SetBrand(ctx, useCase.CustomerID, brand.ID, brand.Status); err != nil {
			return err
		}
	}

	log := &store.ReviewLog{Actor: ActorVendor, ActorName: s.registrar.Name()}
	switch brand.Outcome {
	case Pending:
		return nil
	case Failed:
		useCase.Status, useCase.VendorStatus = VendorRejected, brand.Status
		log.Notes, log.VendorPayload = "The vendor refused the brand: "+brand.Status+".", brand.Raw
		return s.store.MoveUseCase(ctx, &useCase, VendorPending, log)
	}
	campaign, err := s.registrar.RegisterCampaign(ctx, brand.ID, useCase, s.hookURL)
	if err != nil {
		return err
	}
	useCase.VendorCampaignID, useCase.VendorStatus = campaign.ID, campaign.Status
	log.Notes, log.VendorPayload = "Registered as campaign "+campaign.ID+".", campaign.Raw
	return s.store.MoveUseCase(ctx, &useCase, VendorPending, log)
}

// assignAtVendor tells the vendor which of its numbers send as the campaign. A number from
// another vendor is not the registrar's to assign.
func (s *Service) assignAtVendor(ctx context.Context, useCase store.UseCase) error {
	numbers, err := s.store.UseCaseNumbers(ctx, useCase)
	if err != nil {
		return err
	}
	for _, number := range numbers {
		if number.Vendor != s.registrar.Name() {
			continue
		}
		if err := s.registrar.AssignNumber(ctx, useCase.VendorCampaignID, number.E164); err != nil {
			return err
		}
	}
	return nil
}

func approve(useCase *store.UseCase, log *store.ReviewLog) {
	now := time.Now().UTC()
	useCase.Status, useCase.ApprovedAt = Approved, &now
	log.ApprovedAt = &now
}
