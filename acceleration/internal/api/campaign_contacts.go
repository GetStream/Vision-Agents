package api

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"net/http"
	"slices"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/danielgtaylor/huma/v2"
	"github.com/nyaruka/phonenumbers"

	"github.com/GetStream/Vision-Agents/acceleration/internal/campaign"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

var errUnknownContact = APIError{
	Type: ErrorTypeNotFound, Code: codeContactNotFound,
	Message: "no such contact",
}

const (
	defaultContactPage = 50
	maxVariableLength  = 1000
	maxVariables       = 50
)

// contactPosition is the last contact of a page.
type contactPosition struct {
	Seq int64 `json:"s"`
}

// listCampaignContacts returns a page of a campaign's contacts, in the order they are rung.
func (s *Server) listCampaignContacts(ctx context.Context, request *listCampaignContactsRequest) (*listCampaignContactsResponse, error) {
	_, current, err := s.campaign(ctx, request.Id)
	if err != nil {
		return nil, err
	}
	after, err := decodeCursor[contactPosition](&request.Cursor)
	if err != nil {
		return nil, invalidRequest(err.Error())
	}
	filter := store.ContactFilter{Stage: string(request.Stage), Number: request.Number}
	if after != nil {
		filter.After = &after.Seq
	}
	limit := cmp.Or(request.Limit, defaultContactPage)
	filter.Limit = limit + 1
	found, err := s.store.QueryContacts(ctx, current.ID, filter)
	if err != nil {
		return nil, err
	}
	kept, more := page(found, limit)
	items, err := s.contactsOf(ctx, current, kept)
	if err != nil {
		return nil, err
	}
	rendered := ContactPage{Items: items, HasMore: more}
	if more {
		rendered.NextCursor = encodeCursor(contactPosition{Seq: kept[len(kept)-1].Seq})
	}
	return &listCampaignContactsResponse{Body: rendered}, nil
}

// addCampaignContacts adds people to ring to a draft. A row that cannot be rung is refused and
// the rest are added.
func (s *Server) addCampaignContacts(ctx context.Context, request *addCampaignContactsRequest) (*addCampaignContactsResponse, error) {
	_, current, err := s.campaign(ctx, request.Id)
	if err != nil {
		return nil, err
	}
	if request.Body == nil {
		return nil, invalidRequest("a request body is required")
	}
	if current.State != store.Draft {
		return nil, errCampaignLocked
	}
	country := strings.ToUpper(cmp.Or(value(request.Body.DefaultCountry), current.DefaultCountry, "US"))
	if phonenumbers.GetCountryCodeForRegion(country) == 0 {
		return nil, invalidRequest(country + " is not a region a number can be read in: send an ISO 3166-1 alpha-2 code such as US")
	}

	sent := request.Body.Contacts
	refused := []RefusedContact{}
	contacts := make([]store.Contact, 0, len(sent))
	// sentAt is each contact's index in the request, which a duplicate is reported by.
	sentAt := make([]int, 0, len(sent))
	for index, input := range sent {
		contact, reason := contactFrom(input, country)
		if reason != "" {
			refused = append(refused, RefusedContact{Index: index, ToNumber: input.ToNumber, Reason: reason})
			continue
		}
		contacts = append(contacts, contact)
		sentAt = append(sentAt, index)
	}

	answer := AddContactsResponse{Refused: refused}
	if len(contacts) > 0 {
		added, duplicates, err := s.store.AddContacts(ctx, current.ID, contacts)
		if errors.Is(err, store.ErrCampaignLocked) {
			return nil, errCampaignLocked
		}
		if errors.Is(err, store.ErrTooManyContacts) {
			return nil, invalidRequest(fmt.Sprintf("a campaign holds at most %d contacts, and these would take it past that", store.MaxCampaignContacts))
		}
		if err != nil {
			return nil, err
		}
		answer.Added = len(added)
		for _, duplicate := range duplicates {
			index := sentAt[duplicate]
			answer.Refused = append(answer.Refused, RefusedContact{Index: index, ToNumber: sent[index].ToNumber, Reason: RefusedReasonDuplicate})
		}
		slices.SortFunc(answer.Refused, func(a, b RefusedContact) int { return a.Index - b.Index })
	}
	return &addCampaignContactsResponse{Body: answer}, nil
}

// contactFrom reads one row of a request, or says why it cannot be rung.
func contactFrom(input ContactInput, country string) (store.Contact, RefusedReason) {
	number, err := phonenumbers.Parse(input.ToNumber, country)
	if err != nil || !phonenumbers.IsPossibleNumber(number) {
		return store.Contact{}, RefusedReasonInvalidNumber
	}
	if len(input.Variables) > maxVariables {
		return store.Contact{}, RefusedReasonInvalidVariable
	}
	for name, text := range input.Variables {
		// A value is put into the agent's prompt, where a line of its own reads as an instruction.
		if !campaign.VariableName.MatchString(name) || utf8.RuneCountInString(text) > maxVariableLength || strings.ContainsAny(text, "\r\n") {
			return store.Contact{}, RefusedReasonInvalidVariable
		}
	}
	zone := value(input.TimeZone)
	if zone != "" {
		// Local is the router's own zone, which says nothing about where the person is.
		if _, err := time.LoadLocation(zone); err != nil || zone == "Local" {
			return store.Contact{}, RefusedReasonInvalidTimeZone
		}
	}
	return store.Contact{
		ToNumber:     phonenumbers.Format(number, phonenumbers.E164),
		Variables:    input.Variables,
		TimeZone:     zone,
		Instructions: value(input.Instructions),
	}, ""
}

// deleteCampaignContacts removes every contact of a draft.
func (s *Server) deleteCampaignContacts(ctx context.Context, request *campaignRequest) (*struct{}, error) {
	_, current, err := s.campaign(ctx, request.Id)
	if err != nil {
		return nil, err
	}
	err = s.store.DeleteContacts(ctx, current.ID)
	if errors.Is(err, store.ErrCampaignLocked) {
		return nil, errCampaignLocked
	}
	if err != nil {
		return nil, err
	}
	return nil, nil
}

// getCampaignContact returns one contact with its attempts.
func (s *Server) getCampaignContact(ctx context.Context, request *campaignContactRequest) (*campaignContactResponse, error) {
	_, current, err := s.campaign(ctx, request.Id)
	if err != nil {
		return nil, err
	}
	return s.contactAnswer(ctx, current, request.ContactId)
}

// stopCampaignContact leaves one contact unrung.
func (s *Server) stopCampaignContact(ctx context.Context, request *campaignContactRequest) (*campaignContactResponse, error) {
	_, current, err := s.campaign(ctx, request.Id)
	if err != nil {
		return nil, err
	}
	err = s.store.StopContact(ctx, current.ID, request.ContactId)
	if errors.Is(err, store.ErrNoCampaignContact) {
		return nil, errUnknownContact
	}
	if err != nil {
		return nil, err
	}
	return s.contactAnswer(ctx, current, request.ContactId)
}

// contactAnswer renders one contact of a campaign with every dial made to them.
func (s *Server) contactAnswer(ctx context.Context, current store.Campaign, contactID string) (*campaignContactResponse, error) {
	contact, attempts, err := s.store.CampaignContact(ctx, current.ID, contactID)
	if errors.Is(err, store.ErrNoCampaignContact) {
		return nil, errUnknownContact
	}
	if err != nil {
		return nil, err
	}
	rendered, err := s.contactsOf(ctx, current, []store.Contact{contact})
	if err != nil {
		return nil, err
	}
	answer := rendered[0]
	answer.Attempts = make([]ContactAttempt, 0, len(attempts))
	for _, attempt := range attempts {
		answer.Attempts = append(answer.Attempts, ContactAttempt{
			Attempt:   attempt.Attempt,
			Dial:      attempt.Dial,
			CallId:    optional(attempt.CallID),
			Result:    attempt.Result,
			Outcome:   optional(attempt.Outcome),
			StartedAt: attempt.StartedAt,
			EndedAt:   attempt.EndedAt,
		})
	}
	return &campaignContactResponse{Body: answer}, nil
}

// contactsOf renders a campaign's contacts for the wire, counting the attempts of all of them
// at once.
func (s *Server) contactsOf(ctx context.Context, current store.Campaign, contacts []store.Contact) ([]Contact, error) {
	hours, err := campaign.ParseCallingHours(current.CallingHours)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("api: campaign %s calling hours: %w", current.ID, err))
	}
	sequence, err := campaign.ParseSequence(current.Sequence)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("api: campaign %s sequence: %w", current.ID, err))
	}
	ids := make([]string, 0, len(contacts))
	for _, contact := range contacts {
		ids = append(ids, contact.ID)
	}
	made, err := s.store.AttemptsMade(ctx, ids)
	if err != nil {
		return nil, err
	}

	total := len(sequence.First().Attempts)
	rendered := make([]Contact, 0, len(contacts))
	for _, contact := range contacts {
		stage := ContactStage(contact.Stage)
		if contact.Stage == store.AwaitingOutcome {
			stage = ContactStageInProgress
		}
		// A claim leaves the time the contact was due, which is past once a call is under way.
		var nextAttempt *time.Time
		if contact.Stage == store.Enrolled {
			nextAttempt = contact.NextAttemptAt
		}
		variables := contact.Variables
		if variables == nil {
			variables = map[string]string{}
		}
		rendered = append(rendered, Contact{
			Id:            contact.ID,
			ToNumber:      contact.ToNumber,
			Variables:     variables,
			TimeZone:      optional(contact.TimeZone),
			Zone:          hours.Zone(contact.TimeZone, contact.ToNumber).String(),
			Stage:         stage,
			AttemptsMade:  made[contact.ID],
			AttemptsTotal: total,
			NextAttemptAt: nextAttempt,
			Outcome:       optional(contact.Outcome),
			Converted:     contact.Converted,
			Reason:        optional(contact.Reason),
			CallId:        optional(contact.CallID),
			CreatedAt:     contact.CreatedAt,
		})
	}
	return rendered, nil
}

// registerCampaignContacts declares the operations served in campaign_contacts.go.
func (s *Server) registerCampaignContacts(api huma.API) {
	failures := []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound}
	huma.Register(api, huma.Operation{
		OperationID: "listCampaignContacts",
		Method:      http.MethodGet,
		Path:        "/v1/agents/campaigns/{id}/contacts",
		Summary:     "Who a campaign rings, and how far it has got",
		Description: "In the order they are rung.\n" + serverSideOnly,
		Responses: map[string]*huma.Response{
			"200": {Description: "A page of the campaign's contacts"},
		},
		Errors: failures,
	}, s.listCampaignContacts)
	huma.Register(api, huma.Operation{
		OperationID: "addCampaignContacts",
		Method:      http.MethodPost,
		Path:        "/v1/agents/campaigns/{id}/contacts",
		Summary:     "Add people to ring",
		Description: "A draft only: once launched, the list is fixed (409 campaign_locked). Up to 1,000 a " +
			"request. Each number is stored in E.164, read in default_country when it has no country code. " +
			"A row that cannot be rung, or repeats a number the campaign holds, is refused and the rest " +
			"are added: refused says which and why.\n" + serverSideOnly,
		DefaultStatus: http.StatusCreated,
		Responses: map[string]*huma.Response{
			"201": {Description: "How many were added, and which were refused"},
		},
		Errors: append(failures, http.StatusConflict),
	}, s.addCampaignContacts)
	huma.Register(api, huma.Operation{
		OperationID:   "deleteCampaignContacts",
		Method:        http.MethodDelete,
		Path:          "/v1/agents/campaigns/{id}/contacts",
		Summary:       "Remove every contact of a draft",
		Description:   "A draft only: once launched, the list is fixed (409 campaign_locked).\n" + serverSideOnly,
		DefaultStatus: http.StatusNoContent,
		Responses: map[string]*huma.Response{
			"204": {Description: "The campaign has no contacts"},
		},
		Errors: append(failures, http.StatusConflict),
	}, s.deleteCampaignContacts)
	huma.Register(api, huma.Operation{
		OperationID: "getCampaignContact",
		Method:      http.MethodGet,
		Path:        "/v1/agents/campaigns/{id}/contacts/{contact_id}",
		Summary:     "One contact, with every dial made to them",
		Description: serverSideOnly,
		Responses: map[string]*huma.Response{
			"200": {Description: "The contact"},
		},
		Errors: failures,
	}, s.getCampaignContact)
	huma.Register(api, huma.Operation{
		OperationID: "stopCampaignContact",
		Method:      http.MethodPost,
		Path:        "/v1/agents/campaigns/{id}/contacts/{contact_id}/stop",
		Summary:     "Ring a contact no more",
		Description: "One waiting for their next attempt is stopped now. One on a call is not cut off: they " +
			"are stopped when the call ends. One the sequence is done with is left as it is.\n" + serverSideOnly,
		Responses: map[string]*huma.Response{
			"200": {Description: "The contact"},
		},
		Errors: failures,
	}, s.stopCampaignContact)
}

type listCampaignContactsRequest struct {
	Id     string       `path:"id" doc:"The resource, as returned when it was created."`
	Limit  int          `query:"limit" minimum:"1" maximum:"200" doc:"Up to 200. Omitted is 50."`
	Cursor string       `query:"cursor" doc:"The next_cursor of the previous page, sent with the same filters."`
	Stage  ContactStage `query:"stage" doc:"Only the contacts at this stage."`
	Number string       `query:"number" doc:"Only the contacts whose number contains this."`
}

type listCampaignContactsResponse struct {
	Body ContactPage
}

type addCampaignContactsRequest struct {
	Id   string              `path:"id" doc:"The resource, as returned when it was created."`
	Body *AddContactsRequest `required:"true"`
}

type addCampaignContactsResponse struct {
	Body AddContactsResponse
}

type campaignContactRequest struct {
	Id        string `path:"id" doc:"The campaign."`
	ContactId string `path:"contact_id" doc:"The contact, as the campaign's contacts list it."`
}

type campaignContactResponse struct {
	Body Contact
}

// Contact is one person a campaign rings, and how far it has got with them.
type Contact struct {
	Id            string            `json:"id"`
	ToNumber      string            `json:"to_number" doc:"E.164."`
	Variables     map[string]string `json:"variables" nullable:"false" doc:"Filled into the agent's {{placeholders}} on their calls."`
	TimeZone      *string           `json:"time_zone,omitempty" doc:"The IANA zone they were added with."`
	Zone          string            `json:"zone" doc:"The IANA zone the calling hours are read in for them: the campaign's fixed zone, else their time_zone, else their number's, else the campaign's fallback."`
	Stage         ContactStage      `json:"stage"`
	AttemptsMade  int               `json:"attempts_made" doc:"Attempts that placed at least one call."`
	AttemptsTotal int               `json:"attempts_total" doc:"The attempts the sequence makes at most."`
	NextAttemptAt *time.Time        `json:"next_attempt_at,omitempty" doc:"When they are next rung, while they are waiting to be."`
	Outcome       *string           `json:"outcome,omitempty" doc:"The id of the outcome they ended with, once they have."`
	Converted     bool              `json:"converted" doc:"They ended successfully."`
	Reason        *string           `json:"reason,omitempty" doc:"Why they were rejected, failed or stopped."`
	CallId        *string           `json:"call_id,omitempty" doc:"The last call placed to them, which is what the call paths take."`
	CreatedAt     time.Time         `json:"created_at"`
	Attempts      []ContactAttempt  `json:"attempts,omitempty" doc:"Every dial made to them, oldest first. Only on a contact read alone."`
}

func (*Contact) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Properties["attempts_made"].Format = ""
	schema.Properties["attempts_total"].Format = ""
	return schema
}

// ContactAttempt is one dial made to a contact.
type ContactAttempt struct {
	Attempt   int        `json:"attempt" doc:"Which attempt of the sequence, from 1."`
	Dial      int        `json:"dial" doc:"1, or 2 for the second call of a double dial."`
	CallId    *string    `json:"call_id,omitempty"`
	Result    string     `json:"result" doc:"answered, unanswered, invalid_number, refused, deferred, paused, canceled or abandoned; empty while the call is happening. Only answered and unanswered placed a call that ended."`
	Outcome   *string    `json:"outcome,omitempty" doc:"The outcome an answered call was read as."`
	StartedAt time.Time  `json:"started_at"`
	EndedAt   *time.Time `json:"ended_at,omitempty"`
}

func (*ContactAttempt) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Properties["attempt"].Format = ""
	schema.Properties["dial"].Format = ""
	return schema
}

// ContactStage is where a contact is in the campaign.
type ContactStage string

const (
	ContactStageEnrolled   ContactStage = "enrolled"
	ContactStageInProgress ContactStage = "in_progress"
	ContactStageCompleted  ContactStage = "completed"
	ContactStageRejected   ContactStage = "rejected"
	ContactStageStopped    ContactStage = "stopped"
	ContactStageFailed     ContactStage = "failed"
)

func (ContactStage) Schema(registry huma.Registry) *huma.Schema {
	return namedEnum(registry, "ContactStage", "Where a contact is: enrolled while waiting to be rung, "+
		"in_progress while a call to them happens or is being read, completed once the sequence is done "+
		"with them, rejected when their number was refused before any call, stopped when they were left "+
		"unrung, failed when a call could not be made.",
		string(ContactStageEnrolled), string(ContactStageInProgress), string(ContactStageCompleted),
		string(ContactStageRejected), string(ContactStageStopped), string(ContactStageFailed))
}

// ContactPage is one page of a campaign's contacts.
type ContactPage struct {
	Items      []Contact `json:"items" nullable:"false"`
	HasMore    bool      `json:"has_more"`
	NextCursor *string   `json:"next_cursor,omitempty" doc:"Pass as cursor for the next page. Absent on the last."`
}

// AddContactsRequest is people to add to a draft.
type AddContactsRequest struct {
	DefaultCountry *string        `json:"default_country,omitempty" doc:"The region, as ISO 3166-1 alpha-2, a number without a country code is read in. Omitted is the campaign's default_country, else US."`
	Contacts       []ContactInput `json:"contacts" nullable:"false" minItems:"1" maxItems:"1000" doc:"Up to 1,000."`
}

// ContactInput is one person to ring.
type ContactInput struct {
	ToNumber     string            `json:"to_number" doc:"With or without a country code."`
	Variables    map[string]string `json:"variables,omitempty" doc:"Filled into the agent's {{placeholders}}. Names are lowercase letters, digits and underscores, starting with a letter; up to 50, each value up to 1,000 characters on one line."`
	TimeZone     *string           `json:"time_zone,omitempty" doc:"An IANA zone, which a campaign reading its hours where each contact is goes by before the number's."`
	Instructions *string           `json:"instructions,omitempty" maxLength:"2000" doc:"What to say to this person, added to whatever the config already says. Up to 2,000 characters; longer fails the whole request with 400."`
}

// AddContactsResponse is how many people were added, and which were not.
type AddContactsResponse struct {
	Added   int              `json:"added"`
	Refused []RefusedContact `json:"refused" nullable:"false" doc:"In the order sent."`
}

func (*AddContactsResponse) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Properties["added"].Format = ""
	return schema
}

// RefusedContact is a row that was not added, and why.
type RefusedContact struct {
	Index    int           `json:"index" doc:"Its place in contacts, from 0."`
	ToNumber string        `json:"to_number" doc:"As it was sent."`
	Reason   RefusedReason `json:"reason"`
}

func (*RefusedContact) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Properties["index"].Format = ""
	return schema
}

// RefusedReason is why a contact was not added.
type RefusedReason string

const (
	RefusedReasonInvalidNumber   RefusedReason = "invalid_number"
	RefusedReasonDuplicate       RefusedReason = "duplicate"
	RefusedReasonInvalidVariable RefusedReason = "invalid_variable"
	RefusedReasonInvalidTimeZone RefusedReason = "invalid_time_zone"
)

func (RefusedReason) Schema(registry huma.Registry) *huma.Schema {
	return namedEnum(registry, "RefusedReason", "Why a contact was not added: invalid_number is a number "+
		"that cannot be read, duplicate one the campaign already holds or the request repeats, "+
		"invalid_variable more than 50 variables, a variable with a bad name, or a value too long or over more than one line, "+
		"invalid_time_zone a time zone that is not an IANA one.",
		string(RefusedReasonInvalidNumber), string(RefusedReasonDuplicate),
		string(RefusedReasonInvalidVariable), string(RefusedReasonInvalidTimeZone))
}
