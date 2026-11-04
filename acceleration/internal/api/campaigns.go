package api

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"slices"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/campaign"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/danielgtaylor/huma/v2"
)

// errNoCampaigns is what the campaign paths say on a deployment that cannot run one. A
// campaign is a phone call, a conversation and a row, so it needs all three.
var (
	errNoCampaigns     = notConfigured("campaigns are not available: this deployment has no database, telephony or sessions")
	errUnknownCampaign = APIError{
		Type: ErrorTypeNotFound, Code: codeCampaignNotFound,
		Message: "no such campaign",
	}
	errCampaignLocked = APIError{
		Type: ErrorTypeConflict, Code: codeCampaignLocked,
		Message: "this campaign has been launched; only its name, number and concurrency can change",
	}
	errConsentRequired = APIError{
		Type: ErrorTypeInvalidRequest, Code: codeConsentRequired,
		Message: "launching a campaign needs consent: true, saying the people on it agreed to be called",
	}
	errAutoEnrollUnavailable = invalidRequest("auto-enroll campaigns are not available yet")
)

// campaignNotReady is a launch refused for what the campaign still lacks.
func campaignNotReady(message string) APIError {
	return APIError{Type: ErrorTypeInvalidRequest, Code: codeCampaignNotReady, Message: message}
}

// campaignInState is a move the campaign's state does not allow, such as resuming a running one.
func campaignInState(state, move string) APIError {
	return APIError{Type: ErrorTypeConflict, Code: codeCampaignLocked, Message: fmt.Sprintf("a %s campaign cannot be %s", state, move)}
}

// launchSkew is how far in the past a start_at may be and still mean now: the clock of whoever
// picked it is not the router's.
const launchSkew = time.Minute

const defaultCampaignPage = 50

// listCampaigns returns a page of the calling customer's campaigns, newest first.
func (s *Server) listCampaigns(ctx context.Context, request *listCampaignsRequest) (*listCampaignsResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if s.store == nil {
		return nil, errNoCampaigns
	}
	after, err := decodeCursor[store.CreatedPosition](&request.Cursor)
	if err != nil {
		return nil, invalidRequest(err.Error())
	}

	limit := cmp.Or(request.Limit, defaultCampaignPage)
	found, err := s.store.QueryCampaigns(ctx, customerID, store.CampaignFilter{State: string(request.State), Limit: limit + 1, After: after})
	if err != nil {
		return nil, err
	}
	kept, more := page(found, limit)
	items, err := s.campaignsOf(ctx, kept)
	if err != nil {
		return nil, err
	}
	rendered := CampaignPage{Items: items, HasMore: more}
	if more {
		last := kept[len(kept)-1]
		rendered.NextCursor = encodeCursor(store.CreatedPosition{At: last.CreatedAt, ID: last.ID})
	}
	return &listCampaignsResponse{Body: rendered}, nil
}

// createCampaign makes a draft. What it leaves out is defaulted, so the draft can be filled in
// a piece at a time.
func (s *Server) createCampaign(ctx context.Context, request *createCampaignRequest) (*campaignResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if s.store == nil {
		return nil, errNoCampaigns
	}
	if request.Body == nil {
		return nil, invalidRequest("a request body is required")
	}
	body := request.Body
	name := strings.TrimSpace(value(body.Name))
	if name == "" {
		return nil, invalidRequest("a campaign needs a name")
	}
	if body.Type != nil && *body.Type != CampaignTypeStandard {
		return nil, errAutoEnrollUnavailable
	}

	if from := value(body.FromNumber); from != "" {
		if err := s.holdsNumber(ctx, customerID, from); err != nil {
			return nil, err
		}
	}
	configID, sequence, err := s.configAndSequence(ctx, customerID, body, "", campaign.DefaultSequence(""))
	if err != nil {
		return nil, err
	}
	if sequence == nil {
		sequence, _ = json.Marshal(campaign.DefaultSequence(""))
	}
	hours, err := callingHoursOf(body.CallingHours)
	if err != nil {
		return nil, err
	}
	if hours == nil {
		hours, _ = json.Marshal(campaign.DefaultCallingHours())
	}

	// Launching names who owns the calls; until then the rows name who made it.
	maker := routing.AttributionFrom(ctx)
	created := store.Campaign{
		CustomerID:      customerID,
		Name:            name,
		Type:            store.Standard,
		ConfigID:        value(configID),
		FromNumber:      value(body.FromNumber),
		DefaultCountry:  value(body.DefaultCountry),
		Concurrency:     value(body.Concurrency),
		Sequence:        sequence,
		CallingHours:    hours,
		APIKey:          maker.APIKey,
		UserID:          maker.UserID,
		UsageCategories: maker.UsageCategories,
	}
	if body.Tags != nil {
		created.Tags = *body.Tags
	}
	if err := routing.Tags(created.Tags).Validate(); err != nil {
		return nil, invalidRequest(err.Error())
	}
	if err := s.store.CreateCampaign(ctx, &created); err != nil {
		return nil, err
	}
	return s.campaignAnswer(ctx, created)
}

// getCampaign returns one campaign.
func (s *Server) getCampaign(ctx context.Context, request *campaignRequest) (*campaignResponse, error) {
	_, found, err := s.campaign(ctx, request.Id)
	if err != nil {
		return nil, err
	}
	return s.campaignAnswer(ctx, found)
}

// campaign reads one of the calling customer's campaigns.
func (s *Server) campaign(ctx context.Context, id string) (string, store.Campaign, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return "", store.Campaign{}, errMissingCustomer
	}
	if s.store == nil {
		return "", store.Campaign{}, errNoCampaigns
	}
	found, err := s.store.Campaign(ctx, customerID, id)
	if errors.Is(err, store.ErrNoCampaign) {
		return "", store.Campaign{}, errUnknownCampaign
	}
	if err != nil {
		return "", store.Campaign{}, err
	}
	return customerID, found, nil
}

// configAndSequence is the config and the sequence a request leaves a campaign with, nil for
// one it leaves as it is. The step rings with the campaign's config, so setting either one
// sets both.
func (s *Server) configAndSequence(ctx context.Context, customerID string, body *CampaignRequest, configID string, current campaign.Sequence) (*string, json.RawMessage, error) {
	var sequence *campaign.Sequence
	if body.Sequence != nil {
		parsed, err := sequenceOf(*body.Sequence)
		if err != nil {
			return nil, nil, err
		}
		sequence = &parsed
	}

	config := body.ConfigId
	switch {
	case config != nil && sequence != nil:
		if sequence.First().ConfigID != *config {
			return nil, nil, invalidRequest("config_id and the sequence's step's config_id differ: send the same one, or only one of them")
		}
	case config != nil:
		changed := current
		changed.Steps = slices.Clone(current.Steps)
		if len(changed.Steps) > 0 {
			changed.Steps[0].ConfigID = *config
		}
		sequence = &changed
	case sequence != nil:
		stepConfig := sequence.First().ConfigID
		config = &stepConfig
	default:
		return nil, nil, nil
	}

	// A campaign that names a config nobody has would fail one call at a time, at whatever
	// hour it was launched.
	if *config != "" && *config != configID {
		_, err := s.configs.AgentConfig(ctx, customerID, *config)
		if errors.Is(err, store.ErrNoAgentConfig) {
			return nil, nil, errUnknownConfig
		}
		if err != nil {
			return nil, nil, err
		}
	}
	raw, err := json.Marshal(sequence)
	if err != nil {
		return nil, nil, stack.Wrap(fmt.Errorf("api: campaign sequence: %w", err))
	}
	return config, raw, nil
}

// sequenceOf reads a sequence sent on the wire as a draft may hold it.
func sequenceOf(sent CampaignSequence) (campaign.Sequence, error) {
	raw, err := json.Marshal(sent)
	if err != nil {
		return campaign.Sequence{}, stack.Wrap(fmt.Errorf("api: campaign sequence: %w", err))
	}
	sequence, err := campaign.ParseSequence(raw)
	if err != nil {
		return campaign.Sequence{}, invalidRequest(err.Error())
	}
	if err := sequence.Validate(); err != nil {
		return campaign.Sequence{}, invalidRequest(err.Error())
	}
	return sequence, nil
}

// callingHoursOf reads calling hours sent on the wire, nil when none were.
func callingHoursOf(sent *CallingHours) (json.RawMessage, error) {
	if sent == nil {
		return nil, nil
	}
	raw, err := json.Marshal(sent)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("api: calling hours: %w", err))
	}
	hours, err := campaign.ParseCallingHours(raw)
	if err != nil {
		return nil, invalidRequest(err.Error())
	}
	if err := hours.Validate(); err != nil {
		return nil, invalidRequest(err.Error())
	}
	raw, err = json.Marshal(hours)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("api: calling hours: %w", err))
	}
	return raw, nil
}

// campaignAnswer renders one campaign with its counts.
func (s *Server) campaignAnswer(ctx context.Context, found store.Campaign) (*campaignResponse, error) {
	rendered, err := s.campaignsOf(ctx, []store.Campaign{found})
	if err != nil {
		return nil, err
	}
	return &campaignResponse{Body: rendered[0]}, nil
}

// campaignsOf renders campaigns for the wire, counting the contacts of all of them at once.
func (s *Server) campaignsOf(ctx context.Context, campaigns []store.Campaign) ([]Campaign, error) {
	ids := make([]string, 0, len(campaigns))
	for _, one := range campaigns {
		ids = append(ids, one.ID)
	}
	stages, err := s.store.ContactStages(ctx, ids)
	if err != nil {
		return nil, err
	}
	totals, err := s.store.CampaignTotals(ctx, ids)
	if err != nil {
		return nil, err
	}

	rendered := make([]Campaign, 0, len(campaigns))
	for _, one := range campaigns {
		item, err := campaignOf(one, stages[one.ID], totals[one.ID])
		if err != nil {
			return nil, err
		}
		rendered = append(rendered, item)
	}
	return rendered, nil
}

// campaignOf renders a campaign with its contacts counted by stage.
func campaignOf(found store.Campaign, stages map[string]int, totals store.CampaignTotal) (Campaign, error) {
	rendered := Campaign{
		Id:             found.ID,
		Name:           found.Name,
		Type:           CampaignType(found.Type),
		State:          CampaignState(found.State),
		ConfigId:       found.ConfigID,
		FromNumber:     found.FromNumber,
		DefaultCountry: found.DefaultCountry,
		Concurrency:    found.Concurrency,
		Tags:           found.Tags,
		PausedReason:   optional(found.PausedReason),
		StartAt:        found.StartAt,
		LaunchedAt:     found.LaunchedAt,
		StartedAt:      found.StartedAt,
		StoppedAt:      found.StoppedAt,
		CompletedAt:    found.FinishedAt,
		CreatedAt:      found.CreatedAt,
		Contacts: ContactCounts{
			Enrolled:   stages[store.Enrolled],
			InProgress: stages[store.InProgress],
			Completed:  stages[store.ContactCompleted],
			Rejected:   stages[store.Rejected],
			Stopped:    stages[store.ContactStopped],
			Failed:     stages[store.Failed],
			Dialed:     totals.Dialed,
			Converted:  totals.Converted,
		},
	}
	if rendered.Tags == nil {
		rendered.Tags = map[string]string{}
	}
	for _, count := range stages {
		rendered.Contacts.Total += count
	}
	if err := json.Unmarshal(found.Sequence, &rendered.Sequence); err != nil {
		return Campaign{}, stack.Wrap(fmt.Errorf("api: campaign %s sequence: %w", found.ID, err))
	}
	if err := json.Unmarshal(found.CallingHours, &rendered.CallingHours); err != nil {
		return Campaign{}, stack.Wrap(fmt.Errorf("api: campaign %s calling hours: %w", found.ID, err))
	}
	return rendered, nil
}

// updateCampaign changes what a request sends. Once launched, only the name, number and
// concurrency may change; the rest may be sent back as it is.
func (s *Server) updateCampaign(ctx context.Context, request *updateCampaignRequest) (*campaignResponse, error) {
	customerID, current, err := s.campaign(ctx, request.Id)
	if err != nil {
		return nil, err
	}
	if request.Body == nil {
		return nil, invalidRequest("a request body is required")
	}
	body := request.Body
	if body.Type != nil && string(*body.Type) != current.Type {
		return nil, invalidRequest("a campaign's type never changes")
	}

	changes := store.CampaignChanges{
		FromNumber: body.FromNumber, DefaultCountry: body.DefaultCountry, Concurrency: body.Concurrency,
	}
	if body.Name != nil {
		name := strings.TrimSpace(*body.Name)
		if name == "" {
			return nil, invalidRequest("a campaign needs a name")
		}
		changes.Name = &name
	}
	// A launched campaign rings from its number: one it cannot call from refuses every call,
	// and each contact runs out of attempts without anybody being rung.
	if from := body.FromNumber; from != nil && *from != current.FromNumber {
		if *from == "" && current.State != store.Draft {
			return nil, invalidRequest("a launched campaign needs one of the app's numbers to call from")
		}
		if *from != "" {
			if err := s.holdsNumber(ctx, customerID, *from); err != nil {
				return nil, err
			}
		}
	}
	sequence, err := campaign.ParseSequence(current.Sequence)
	if err != nil {
		return nil, stack.Wrap(err)
	}
	changes.ConfigID, changes.Sequence, err = s.configAndSequence(ctx, customerID, body, current.ConfigID, sequence)
	if err != nil {
		return nil, err
	}
	if changes.CallingHours, err = callingHoursOf(body.CallingHours); err != nil {
		return nil, err
	}
	if body.Tags != nil {
		if err := routing.Tags(*body.Tags).Validate(); err != nil {
			return nil, invalidRequest(err.Error())
		}
		changes.Tags = *body.Tags
	}

	updated, err := s.store.UpdateCampaign(ctx, customerID, current.ID, changes)
	if errors.Is(err, store.ErrCampaignLocked) {
		return nil, errCampaignLocked
	}
	if err != nil {
		return nil, err
	}
	return s.campaignAnswer(ctx, updated)
}

// deleteCampaign removes a campaign that is not ringing anybody, with its contacts.
func (s *Server) deleteCampaign(ctx context.Context, request *campaignRequest) (*struct{}, error) {
	customerID, current, err := s.campaign(ctx, request.Id)
	if err != nil {
		return nil, err
	}
	err = s.store.DeleteCampaign(ctx, customerID, current.ID)
	if errors.Is(err, store.ErrCampaignLocked) {
		return nil, campaignInState(current.State, "deleted")
	}
	if err != nil {
		return nil, err
	}
	return nil, nil
}

// launchCampaign takes a draft out of draft, once everything it needs to ring people is there
// and the customer has said the people on it agreed to be called.
func (s *Server) launchCampaign(ctx context.Context, request *launchCampaignRequest) (*campaignResponse, error) {
	customerID, current, err := s.campaign(ctx, request.Id)
	if err != nil {
		return nil, err
	}
	if s.phone == nil {
		return nil, errNoCampaigns
	}
	if request.Body == nil {
		return nil, invalidRequest("a request body is required")
	}
	if current.State != store.Draft {
		return nil, campaignInState(current.State, "launched")
	}
	if !request.Body.Consent {
		return nil, errConsentRequired
	}

	now := time.Now().UTC()
	start := now
	if at := request.Body.StartAt; at != nil {
		if at.Before(now.Add(-launchSkew)) {
			return nil, campaignNotReady("start_at is in the past: leave it out to start now")
		}
		start = at.UTC()
	}
	if err := s.readyToLaunch(ctx, customerID, current); err != nil {
		return nil, err
	}
	if err := s.mayCall(ctx, customerID); err != nil {
		return nil, err
	}

	// The calls it places later, by no request, are owned by whoever launched it.
	launcher := routing.AttributionFrom(ctx)
	launched, err := s.store.LaunchCampaign(ctx, customerID, current.ID, store.Launch{
		StartAt: start, Now: now,
		APIKey: launcher.APIKey, UserID: launcher.UserID, UsageCategories: launcher.UsageCategories,
	})
	if errors.Is(err, store.ErrCampaignLocked) {
		return nil, errCampaignLocked
	}
	if err != nil {
		return nil, err
	}
	return s.campaignAnswer(ctx, launched)
}

// readyToLaunch says what a draft still lacks before it can ring anybody, the first thing it
// finds.
func (s *Server) readyToLaunch(ctx context.Context, customerID string, draft store.Campaign) error {
	contacts, err := s.store.QueryContacts(ctx, draft.ID, store.ContactFilter{Limit: 1})
	if err != nil {
		return err
	}
	if len(contacts) == 0 {
		return campaignNotReady("a campaign needs contacts to ring before it can launch")
	}
	if draft.FromNumber == "" {
		return campaignNotReady("a campaign needs one of the app's numbers to call from before it can launch")
	}
	_, err = s.store.Number(ctx, customerID, draft.FromNumber)
	if isNumberNotHeld(err) {
		return campaignNotReady(draft.FromNumber + " is not one of this app's numbers, so the campaign cannot call from it")
	}
	if err != nil {
		return err
	}
	if draft.ConfigID == "" {
		return campaignNotReady("a campaign needs an agent config to make its calls with before it can launch")
	}
	config, err := s.configs.AgentConfig(ctx, customerID, draft.ConfigID)
	if errors.Is(err, store.ErrNoAgentConfig) {
		return campaignNotReady("the agent config " + draft.ConfigID + " the campaign calls with no longer exists")
	}
	if err != nil {
		return err
	}

	sequence, err := campaign.ParseSequence(draft.Sequence)
	if err != nil {
		return stack.Wrap(err)
	}
	if err := sequence.Ready(); err != nil {
		return campaignNotReady(err.Error())
	}
	if sequence.First().Voicemail.Action == campaign.VoicemailLeaveMessage && session.FromConfig(config).Native() {
		return campaignNotReady("the voicemail cannot leave a message with a speech-to-speech agent: use off or hang_up, or an agent config with a speech-to-text and a voice")
	}
	hours, err := campaign.ParseCallingHours(draft.CallingHours)
	if err != nil {
		return stack.Wrap(err)
	}
	if err := hours.Validate(); err != nil {
		return campaignNotReady(err.Error())
	}
	return nil
}

// mayCall refuses a customer who may not place calls now, as each call would be refused.
func (s *Server) mayCall(ctx context.Context, customerID string) error {
	err := s.phone.Suspended(ctx, customerID)
	if err == nil {
		err = s.phone.Admit(ctx, customerID)
	}
	if errors.Is(err, phone.ErrSuspended) {
		return errTelephonySuspended
	}
	if answer, refused := moneyError(err); refused {
		return answer
	}
	return err
}

// holdsNumber refuses a number the app does not hold, which no call can be placed from.
func (s *Server) holdsNumber(ctx context.Context, customerID, e164 string) error {
	_, err := s.store.Number(ctx, customerID, e164)
	if isNumberNotHeld(err) {
		return invalidRequest(e164 + " is not one of this app's numbers")
	}
	return err
}

// pauseCampaign stops ringing anybody new. The calls already happening are left to finish.
func (s *Server) pauseCampaign(ctx context.Context, request *campaignRequest) (*campaignResponse, error) {
	return s.moveCampaign(ctx, request.Id, "paused", func(customerID string, current store.Campaign) error {
		return s.store.PauseCampaign(ctx, customerID, current.ID, "")
	})
}

// resumeCampaign rings people again, from whoever is left, once the customer may place calls.
func (s *Server) resumeCampaign(ctx context.Context, request *campaignRequest) (*campaignResponse, error) {
	if s.phone == nil {
		return nil, errNoCampaigns
	}
	return s.moveCampaign(ctx, request.Id, "resumed", func(customerID string, current store.Campaign) error {
		if current.State != store.Paused {
			return campaignInState(current.State, "resumed")
		}
		// A launched campaign's config cannot change, so one deleted since leaves it nothing to
		// ring with: the dialer would pause it again at its first call.
		_, err := s.configs.AgentConfig(ctx, customerID, current.ConfigID)
		if errors.Is(err, store.ErrNoAgentConfig) {
			return campaignNotReady("the agent config " + current.ConfigID + " the campaign calls with was deleted: duplicate the campaign with another")
		}
		if err != nil {
			return err
		}
		if err := s.mayCall(ctx, customerID); err != nil {
			return err
		}
		return s.store.ResumeCampaign(ctx, customerID, current.ID)
	})
}

// stopCampaign ends a campaign for good. Nobody waiting is rung; calls in progress finish.
func (s *Server) stopCampaign(ctx context.Context, request *campaignRequest) (*campaignResponse, error) {
	return s.moveCampaign(ctx, request.Id, "stopped", func(customerID string, current store.Campaign) error {
		return s.store.StopCampaign(ctx, customerID, current.ID, time.Now().UTC())
	})
}

// moveCampaign moves a campaign to another state and returns it there.
func (s *Server) moveCampaign(ctx context.Context, id, move string, apply func(customerID string, current store.Campaign) error) (*campaignResponse, error) {
	customerID, current, err := s.campaign(ctx, id)
	if err != nil {
		return nil, err
	}
	err = apply(customerID, current)
	if errors.Is(err, store.ErrCampaignLocked) {
		return nil, campaignInState(current.State, move)
	}
	if err != nil {
		return nil, err
	}
	moved, err := s.store.Campaign(ctx, customerID, current.ID)
	if err != nil {
		return nil, err
	}
	return s.campaignAnswer(ctx, moved)
}

// duplicateCampaign makes a draft with a campaign's settings and, as asked, its contacts.
func (s *Server) duplicateCampaign(ctx context.Context, request *duplicateCampaignRequest) (*campaignResponse, error) {
	customerID, current, err := s.campaign(ctx, request.Id)
	if err != nil {
		return nil, err
	}
	if request.Body == nil {
		return nil, invalidRequest("a request body is required")
	}
	copied, err := s.store.DuplicateCampaign(ctx, customerID, current.ID, request.Body.Contacts)
	if errors.Is(err, store.ErrCampaignLocked) {
		return nil, APIError{Type: ErrorTypeConflict, Code: codeCampaignLocked,
			Message: "the contacts never rung are known once a campaign is stopped or completed; this one is " + current.State}
	}
	if err != nil {
		return nil, err
	}
	return s.campaignAnswer(ctx, copied)
}

// getCampaignAnalytics is how a campaign's calls went.
func (s *Server) getCampaignAnalytics(ctx context.Context, request *campaignRequest) (*campaignAnalyticsResponse, error) {
	_, current, err := s.campaign(ctx, request.Id)
	if err != nil {
		return nil, err
	}
	counted, err := s.store.CampaignAnalytics(ctx, current.ID)
	if err != nil {
		return nil, err
	}
	rendered := CampaignAnalytics{
		Dialed: counted.Dialed, Connected: counted.Connected, Converted: counted.Converted,
		Outcomes: counted.Outcomes, CallOutcomes: counted.CallOutcomes, DialsPerDay: make([]CampaignDialDay, 0, len(counted.Days)),
	}
	if counted.Dialed > 0 {
		rendered.ConversionRate = float64(counted.Converted) / float64(counted.Dialed)
	}
	for _, day := range counted.Days {
		rendered.DialsPerDay = append(rendered.DialsPerDay, CampaignDialDay{
			Date: day.Day.Format(time.DateOnly), Dials: day.Dials, Answered: day.Answered, Voicemail: day.Voicemail,
		})
	}
	return &campaignAnalyticsResponse{Body: rendered}, nil
}

// serverSideOnly closes the description of every campaign operation.
const serverSideOnly = "Server-side only: it needs a server-side token, so it cannot be reached from an end user's device."

// registerCampaigns declares the operations served in campaigns.go.
func (s *Server) registerCampaigns(api huma.API) {
	failures := []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound}
	huma.Register(api, huma.Operation{
		OperationID: "listCampaigns",
		Method:      http.MethodGet,
		Path:        "/v1/agents/campaigns",
		Summary:     "The campaigns the calling customer has",
		Description: "Newest first.\n" + serverSideOnly,
		Responses: map[string]*huma.Response{
			"200": {Description: "A page of the customer's campaigns"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden},
	}, s.listCampaigns)
	huma.Register(api, huma.Operation{
		OperationID: "createCampaign",
		Method:      http.MethodPost,
		Path:        "/v1/agents/campaigns",
		Summary:     "Make a draft campaign",
		Description: "A campaign rings a list of people with one agent config. It is made a draft, with the " +
			"default sequence and calling hours for whatever it leaves out: add the people to ring, then " +
			"launch it.\n" + serverSideOnly,
		DefaultStatus: http.StatusCreated,
		Responses: map[string]*huma.Response{
			"201": {Description: "The draft"},
		},
		Errors: failures,
	}, s.createCampaign)
	huma.Register(api, huma.Operation{
		OperationID: "getCampaign",
		Method:      http.MethodGet,
		Path:        "/v1/agents/campaigns/{id}",
		Summary:     "One campaign, with its contacts counted by stage",
		Description: serverSideOnly,
		Responses: map[string]*huma.Response{
			"200": {Description: "The campaign"},
		},
		Errors: failures,
	}, s.getCampaign)
	huma.Register(api, huma.Operation{
		OperationID: "updateCampaign",
		Method:      http.MethodPatch,
		Path:        "/v1/agents/campaigns/{id}",
		Summary:     "Change a campaign's settings",
		Description: "Sets only what it sends. Once a campaign is launched only its name, number and " +
			"concurrency change: any other field sent must be what it already is, or the change is " +
			"refused with 409 campaign_locked.\n" + serverSideOnly,
		Responses: map[string]*huma.Response{
			"200": {Description: "The campaign as it is now"},
		},
		Errors: append(failures, http.StatusConflict),
	}, s.updateCampaign)
	huma.Register(api, huma.Operation{
		OperationID:   "deleteCampaign",
		Method:        http.MethodDelete,
		Path:          "/v1/agents/campaigns/{id}",
		Summary:       "Delete a campaign and its contacts",
		Description:   "A draft, stopped or completed campaign only. Its calls stay.\n" + serverSideOnly,
		DefaultStatus: http.StatusNoContent,
		Responses: map[string]*huma.Response{
			"204": {Description: "The campaign is gone"},
		},
		Errors: append(failures, http.StatusConflict),
	}, s.deleteCampaign)
	huma.Register(api, huma.Operation{
		OperationID: "launchCampaign",
		Method:      http.MethodPost,
		Path:        "/v1/agents/campaigns/{id}/launch",
		Summary:     "Launch a draft",
		Description: "Records the customer's consent and rings the contacts from start_at. A draft missing " +
			"what it needs (contacts, one of the app's numbers, an agent config, a ready sequence) is refused " +
			"with campaign_not_ready, saying what. An organization that may not pay for calls is refused here " +
			"as a call would be, and a campaign refused midway is paused with paused_reason set. Contacts who " +
			"opted out of calls are rejected without a call.\n" + serverSideOnly,
		Responses: map[string]*huma.Response{
			"200": {Description: "The campaign, scheduled or running"},
		},
		Errors: append(failures, http.StatusPaymentRequired, http.StatusConflict, http.StatusTooManyRequests, http.StatusServiceUnavailable),
	}, s.launchCampaign)
	huma.Register(api, huma.Operation{
		OperationID: "pauseCampaign",
		Method:      http.MethodPost,
		Path:        "/v1/agents/campaigns/{id}/pause",
		Summary:     "Stop ringing anybody new",
		Description: "A running or scheduled campaign only. The calls already happening are left alone: a campaign is " +
			"paused to stop ringing people, not to hang up on the ones who answered. A scheduled one paused " +
			"does not start at its start_at, and resuming it before then returns it to scheduled.\n" + serverSideOnly,
		Responses: map[string]*huma.Response{
			"200": {Description: "The campaign is paused"},
		},
		Errors: append(failures, http.StatusConflict),
	}, s.pauseCampaign)
	huma.Register(api, huma.Operation{
		OperationID: "resumeCampaign",
		Method:      http.MethodPost,
		Path:        "/v1/agents/campaigns/{id}/resume",
		Summary:     "Ring people again",
		Description: "A paused campaign only. It carries on from whoever is left, and clears paused_reason. " +
			"A campaign whose start time has not come goes back to scheduled rather than running. " +
			"An organization that may not pay for calls is refused as a call would be.\n" + serverSideOnly,
		Responses: map[string]*huma.Response{
			"200": {Description: "The campaign is running, or scheduled when its start time has not come"},
		},
		Errors: append(failures, http.StatusPaymentRequired, http.StatusConflict, http.StatusTooManyRequests, http.StatusServiceUnavailable),
	}, s.resumeCampaign)
	huma.Register(api, huma.Operation{
		OperationID: "stopCampaign",
		Method:      http.MethodPost,
		Path:        "/v1/agents/campaigns/{id}/stop",
		Summary:     "End a campaign for good",
		Description: "Nobody waiting is rung; calls in progress finish.\n" + serverSideOnly,
		Responses: map[string]*huma.Response{
			"200": {Description: "The campaign is stopped"},
		},
		Errors: append(failures, http.StatusConflict),
	}, s.stopCampaign)
	huma.Register(api, huma.Operation{
		OperationID:   "duplicateCampaign",
		Method:        http.MethodPost,
		Path:          "/v1/agents/campaigns/{id}/duplicate",
		Summary:       "Make a draft with a campaign's settings",
		Description:   "Its contacts are copied as asked, starting over with no attempts.\n" + serverSideOnly,
		DefaultStatus: http.StatusCreated,
		Responses: map[string]*huma.Response{
			"201": {Description: "The new draft"},
		},
		Errors: append(failures, http.StatusConflict),
	}, s.duplicateCampaign)
	huma.Register(api, huma.Operation{
		OperationID: "getCampaignAnalytics",
		Method:      http.MethodGet,
		Path:        "/v1/agents/campaigns/{id}/analytics",
		Summary:     "How a campaign's calls went",
		Description: "A dial is a call placed: answered, unanswered, taken by an answering machine or still happening. dialed, connected and " +
			"converted count contacts, not dials.\n" + serverSideOnly,
		Responses: map[string]*huma.Response{
			"200": {Description: "The campaign's analytics"},
		},
		Errors: failures,
	}, s.getCampaignAnalytics)
}

type listCampaignsRequest struct {
	State  CampaignState `query:"state" doc:"Only the campaigns in this state."`
	Limit  int           `query:"limit" minimum:"1" maximum:"200" doc:"Up to 200. Omitted is 50."`
	Cursor string        `query:"cursor" doc:"The next_cursor of the previous page, sent with the same state."`
}

type listCampaignsResponse struct {
	Body CampaignPage
}

type createCampaignRequest struct {
	Body *CampaignRequest `required:"true"`
}

type campaignRequest struct {
	Id string `path:"id" doc:"The resource, as returned when it was created."`
}

type campaignResponse struct {
	Body Campaign
}

type updateCampaignRequest struct {
	Id   string           `path:"id" doc:"The resource, as returned when it was created."`
	Body *CampaignRequest `required:"true"`
}

type launchCampaignRequest struct {
	Id   string                 `path:"id" doc:"The resource, as returned when it was created."`
	Body *LaunchCampaignRequest `required:"true"`
}

type duplicateCampaignRequest struct {
	Id   string                    `path:"id" doc:"The resource, as returned when it was created."`
	Body *DuplicateCampaignRequest `required:"true"`
}

type campaignAnalyticsResponse struct {
	Body CampaignAnalytics
}

// LaunchCampaignRequest is when a campaign starts ringing, and the customer's word that it may.
type LaunchCampaignRequest struct {
	StartAt *time.Time `json:"start_at,omitempty" doc:"When to start ringing. Omitted is now; up to a minute in the past is now too."`
	Consent bool       `json:"consent" doc:"Must be true: the people on the campaign agreed to be called by it. Recorded with the time."`
}

// DuplicateCampaignRequest is which contacts a copy of a campaign takes.
type DuplicateCampaignRequest struct {
	Contacts string `json:"contacts" enum:"none,all,not_contacted" doc:"none, all, or the contacts never rung, which a stopped or completed campaign alone knows."`
}

// CampaignAnalytics is how a campaign's calls went.
type CampaignAnalytics struct {
	Dialed         int               `json:"dialed" doc:"Contacts with at least one call placed."`
	Connected      int               `json:"connected" doc:"Contacts who answered at least one call."`
	Converted      int               `json:"converted" doc:"Contacts that ended successfully."`
	ConversionRate float64           `json:"conversion_rate" doc:"converted / dialed, 0 when nobody was dialed."`
	Outcomes       map[string]int    `json:"outcomes" nullable:"false" doc:"Contacts that are done, by outcome id, no_response included."`
	CallOutcomes   map[string]int    `json:"call_outcomes" nullable:"false" doc:"Calls placed that ended, by how the phone says they ended: completed, busy, no_answer, failed, canceled or unknown. A call an answering machine took is completed."`
	DialsPerDay    []CampaignDialDay `json:"dials_per_day" nullable:"false" doc:"Oldest first, only the days with a dial."`
}

// CampaignDialDay is the calls placed on one UTC day.
type CampaignDialDay struct {
	Date      string `json:"date" format:"date" doc:"YYYY-MM-DD, in UTC."`
	Dials     int    `json:"dials" doc:"Calls placed, answering machines included."`
	Answered  int    `json:"answered" doc:"Calls a person picked up and spoke on."`
	Voicemail int    `json:"voicemail" doc:"Calls an answering machine took."`
}

// Campaign is a list of people to ring with one agent, and how.
type Campaign struct {
	Id             string            `json:"id"`
	Name           string            `json:"name"`
	Type           CampaignType      `json:"type"`
	State          CampaignState     `json:"state"`
	ConfigId       string            `json:"config_id" doc:"The agent config the calls are made with, the same as the sequence's step names."`
	FromNumber     string            `json:"from_number" doc:"One of the app's own numbers, which is what the person sees."`
	DefaultCountry string            `json:"default_country" doc:"The region, as ISO 3166-1 alpha-2, a contact's number without a country code is read in."`
	Concurrency    int               `json:"concurrency" doc:"How many of these calls may be happening at once."`
	Sequence       CampaignSequence  `json:"sequence"`
	CallingHours   CallingHours      `json:"calling_hours"`
	Tags           map[string]string `json:"tags" nullable:"false"`
	PausedReason   *string           `json:"paused_reason,omitempty" doc:"Why the router paused the campaign itself: the code of what refused its next call, for example telephony_suspended, account_suspended, insufficient_credits, system_limit_reached, spend_limit_reached, user_limit_reached or spend_check_unavailable, config_missing when its agent config was deleted, session_refused when its agent config makes a session that cannot open, or voicemail_needs_exact_words when its step leaves a voicemail message and its agent config became speech-to-speech. The contact that call was for is rung once the campaign is resumed. Absent for a campaign somebody paused, and cleared by resuming it."`
	StartAt        *time.Time        `json:"start_at,omitempty" doc:"When it was launched to start ringing."`
	LaunchedAt     *time.Time        `json:"launched_at,omitempty"`
	StartedAt      *time.Time        `json:"started_at,omitempty" doc:"When it first rang anybody."`
	StoppedAt      *time.Time        `json:"stopped_at,omitempty"`
	CompletedAt    *time.Time        `json:"completed_at,omitempty" doc:"When it had nobody left to ring."`
	CreatedAt      time.Time         `json:"created_at"`
	Contacts       ContactCounts     `json:"contacts"`
}

func (*Campaign) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Properties["concurrency"].Format = ""
	return schema
}

// CampaignType is what a campaign rings: standard rings the contacts it was given.
type CampaignType string

const (
	CampaignTypeStandard   CampaignType = store.Standard
	CampaignTypeAutoEnroll CampaignType = store.AutoEnroll
)

func (CampaignType) Schema(registry huma.Registry) *huma.Schema {
	return namedEnum(registry, "CampaignType", "What a campaign rings. standard rings the contacts it was "+
		"given; auto_enroll is refused for now. Fixed once the campaign is created.",
		string(CampaignTypeStandard), string(CampaignTypeAutoEnroll))
}

// CampaignState is where a campaign is in its life.
type CampaignState string

const (
	CampaignStateDraft     CampaignState = store.Draft
	CampaignStateScheduled CampaignState = store.Scheduled
	CampaignStateRunning   CampaignState = store.Running
	CampaignStatePaused    CampaignState = store.Paused
	CampaignStateStopped   CampaignState = store.Stopped
	CampaignStateCompleted CampaignState = store.Completed
)

func (CampaignState) Schema(registry huma.Registry) *huma.Schema {
	return namedEnum(registry, "CampaignState", "Where a campaign is: draft until it is launched, "+
		"scheduled until its start_at, running while it rings people, paused while it rings nobody "+
		"new, stopped for good, completed once it has nobody left to ring.",
		string(CampaignStateDraft), string(CampaignStateScheduled), string(CampaignStateRunning),
		string(CampaignStatePaused), string(CampaignStateStopped), string(CampaignStateCompleted))
}

// CampaignSequence is what a campaign does with each contact.
type CampaignSequence struct {
	Steps      []CampaignStep `json:"steps" nullable:"false" doc:"Exactly one step for now."`
	DoubleDial bool           `json:"double_dial" doc:"Ring again at once when the first call of an attempt is not answered."`
}

// CampaignStep is the calls one step makes, and where each outcome leads.
type CampaignStep struct {
	Id         string             `json:"id" doc:"step_1, for now."`
	ConfigId   string             `json:"config_id" doc:"The agent config the step rings with, the same as the campaign's config_id."`
	Goal       string             `json:"goal" required:"false" maxLength:"2000" doc:"What this conversation is for, told to the agent on every call of the step. {{variable}} placeholders are filled from the contact. Required to launch."`
	Attempts   []CampaignAttempt  `json:"attempts" nullable:"false" doc:"1 to 10. The first has a delay of 0."`
	Outcomes   []CampaignOutcome  `json:"outcomes" nullable:"false" doc:"What an answered call can end as. do_not_call is built in."`
	Other      CampaignRoute      `json:"other" doc:"Where an answered call none of the outcomes fit leads."`
	NoResponse CampaignRoute      `json:"no_response" doc:"Where running out of attempts leads. It ends unsuccessfully."`
	Voicemail  *CampaignVoicemail `json:"voicemail,omitempty" required:"false" doc:"What the agent does when an answering machine picks up. Absent is off: the agent talks to a machine as to a person. Leaving a message needs an agent that is not speech-to-speech."`
}

// CampaignVoicemail is what a step's call does on an answering machine.
type CampaignVoicemail struct {
	Action  CampaignVoicemailAction `json:"action"`
	Message string                  `json:"message,omitempty" maxLength:"1000" doc:"{{variable}} placeholders are filled from the contact, then it is said word for word once the greeting has ended, and the call ends. Required to launch for leave_message, and not only spaces."`
}

// CampaignAttempt is one try at reaching a contact.
type CampaignAttempt struct {
	DelayMinutes int `json:"delay_minutes" doc:"Minutes after the previous attempt started, pushed into the calling hours."`
}

// CampaignOutcome is one way an answered call can end.
type CampaignOutcome struct {
	Id          string       `json:"id" doc:"Lowercase letters, digits and underscores, starting with a letter."`
	Name        string       `json:"name"`
	Description string       `json:"description" doc:"What the call has to have been for this outcome, which the model reading the transcript goes by."`
	Next        CampaignNext `json:"next"`
}

// CampaignRoute is where an answer leads.
type CampaignRoute struct {
	Next CampaignNext `json:"next"`
}

// CampaignNext ends the sequence for a contact.
type CampaignNext struct {
	End  *string `json:"end,omitempty" enum:"successful,unsuccessful" doc:"How the contact ends: successful counts as converted."`
	Step *string `json:"step,omitempty" doc:"Reserved for branching to another step, which is refused for now."`
}

// CallingHours is when a campaign may start a call, in the contact's local time.
type CallingHours struct {
	Windows       map[string][]CallingWindow `json:"windows" doc:"Keyed monday to sunday. A day left out rings nobody."`
	TimeZone      CampaignTimeZone           `json:"time_zone"`
	ExcludedDates []string                   `json:"excluded_dates" nullable:"false" doc:"YYYY-MM-DD, the contact's local date, on which nobody is rung."`
}

// CallingWindow is one stretch of a day.
type CallingWindow struct {
	Start string `json:"start" doc:"HH:MM, on a 15-minute step."`
	End   string `json:"end" doc:"HH:MM, up to 24:00. A window does not cross midnight."`
}

// CampaignTimeZone is the zone the calling hours are read in.
type CampaignTimeZone struct {
	Mode     string  `json:"mode" enum:"contact,fixed" doc:"contact reads the hours where each contact is: their time_zone, else their number's, else fallback. fixed reads them in fixed."`
	Fallback string  `json:"fallback" doc:"An IANA zone, used by contact mode."`
	Fixed    *string `json:"fixed,omitempty" doc:"An IANA zone, used by fixed mode."`
}

// ContactCounts is a campaign's contacts by stage. A call awaiting its outcome is in progress.
type ContactCounts struct {
	Total      int `json:"total"`
	Enrolled   int `json:"enrolled"`
	InProgress int `json:"in_progress"`
	Completed  int `json:"completed"`
	Rejected   int `json:"rejected"`
	Stopped    int `json:"stopped"`
	Failed     int `json:"failed"`
	Dialed     int `json:"dialed" doc:"Contacts with at least one call placed."`
	Converted  int `json:"converted" doc:"Contacts that ended successfully."`
}

// CampaignRequest is the settings of a campaign. A create needs a name; a change sets only
// what it sends.
type CampaignRequest struct {
	Name           *string            `json:"name,omitempty" minLength:"1" doc:"Required on create."`
	Type           *CampaignType      `json:"type,omitempty" doc:"Omitted is standard on create. It never changes."`
	ConfigId       *string            `json:"config_id,omitempty" doc:"The agent config the calls are made with. Setting it sets the sequence's step's too."`
	FromNumber     *string            `json:"from_number,omitempty" doc:"One of the app's own numbers, which is what the person sees."`
	DefaultCountry *string            `json:"default_country,omitempty" doc:"The region, as ISO 3166-1 alpha-2, a contact's number without a country code is read in."`
	Concurrency    *int               `json:"concurrency,omitempty" minimum:"1" maximum:"50" doc:"How many of these calls may be happening at once. Omitted is 1 on create."`
	Sequence       *CampaignSequence  `json:"sequence,omitempty" doc:"Omitted is four attempts over twenty days, with double dial, on create. Its step's config_id is the campaign's: sending both, they must agree."`
	CallingHours   *CallingHours      `json:"calling_hours,omitempty" doc:"Omitted is weekdays 09:00 to 18:00 where the contact is, on create."`
	Tags           *map[string]string `json:"tags,omitempty"`
}

func (*CampaignRequest) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Properties["concurrency"].Format = ""
	return schema
}

// CampaignPage is one page of a customer's campaigns.
type CampaignPage struct {
	Items      []Campaign `json:"items" nullable:"false"`
	HasMore    bool       `json:"has_more"`
	NextCursor *string    `json:"next_cursor,omitempty" doc:"Pass as cursor for the next page. Absent on the last."`
}
