package server

import (
	"bytes"
	"context"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"

	"github.com/danielgtaylor/huma/v2"

	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage/ledger"
)

const (
	// manageScheme is the security scheme of the management API.
	manageScheme = "ManageKey"
	// manageKeyHeader carries the manage key.
	manageKeyHeader = "X-Manage-Key"
	// actorIDHeader and actorNameHeader name who a management call is made by.
	actorIDHeader   = "X-Stream-Actor-Id"
	actorNameHeader = "X-Stream-Actor-Name"
	// actorIDLimit is the longest actor a call may name, in characters; actorNameLimit is how
	// much of a name is kept, in bytes.
	actorIDLimit   = 256
	actorNameLimit = 200
	// idempotencyHeader carries the key a credit is written once under.
	idempotencyHeader = "Idempotency-Key"
	// maxTeamLength is the longest a team name may be.
	maxTeamLength = 64
)

var (
	errNoManagement = problem(http.StatusNotFound, codeNotConfigured, "the management API is not available: no manage key configured")
	errManageKey    = problem(http.StatusUnauthorized, codeUnauthenticated, "the management API needs its key: "+manageKeyHeader)
	errNoActor      = invalidRequest("a management call names its actor in " + actorIDHeader + ", 1 to 256 characters")
	errActorID      = invalidRequest(actorIDHeader + " holds a control character or is not UTF-8")
	errActorName    = invalidRequest(actorNameHeader + " holds a % that is not percent-encoded UTF-8")
	errCursor       = invalidRequest("cursor is not one this list answered")
)

// manageSecurity marks an operation as the management API's, reached with the manage key.
var manageSecurity = []map[string][]string{{manageScheme: {}}}

// actorKey holds who a management call is made by.
type actorKey struct{}

// actorFrom is who the management call is made by, as requireManageKey read it.
func actorFrom(ctx context.Context) ledger.Actor {
	actor, _ := ctx.Value(actorKey{}).(ledger.Actor)
	return actor
}

// requireManageKey answers a management call without the manage key 401, and every one 404
// not_configured when the service has none. A call with the key must name its actor, whom
// every change it makes is recorded against. The ingest token opens none of it.
func (s *Server) requireManageKey(ctx huma.Context, next func(huma.Context)) {
	if !s.manageOn {
		writeProblem(ctx, errNoManagement)
		return
	}
	// Hashed first, so the comparison takes as long whatever the length sent.
	digest := sha256.Sum256([]byte(ctx.Header(manageKeyHeader)))
	if subtle.ConstantTimeCompare(digest[:], s.manageKey[:]) != 1 {
		writeProblem(ctx, errManageKey)
		return
	}
	actor, failure := manageActor(ctx)
	if failure != nil {
		writeProblem(ctx, failure)
		return
	}
	next(huma.WithValue(ctx, actorKey{}, actor))
}

// manageActor reads who a management call is made by, as the router's management API does.
// The id is stored as sent, so one that is not UTF-8 or holds a control character is refused.
// A name that is not printable ASCII, or holds a %, is sent percent-encoded as UTF-8, and one
// that does not decode is refused rather than stored as sent.
func manageActor(ctx huma.Context) (ledger.Actor, *apiError) {
	id := strings.TrimSpace(ctx.Header(actorIDHeader))
	if id == "" || utf8.RuneCountInString(id) > actorIDLimit {
		return ledger.Actor{}, errNoActor
	}
	if !utf8.ValidString(id) || strings.ContainsFunc(id, unicode.IsControl) {
		return ledger.Actor{}, errActorID
	}
	name := ctx.Header(actorNameHeader)
	if strings.Contains(name, "%") {
		decoded, err := url.PathUnescape(name)
		if err != nil || !utf8.ValidString(decoded) {
			return ledger.Actor{}, errActorName
		}
		name = decoded
	}
	name = strings.TrimSpace(strings.Map(func(r rune) rune {
		if unicode.IsControl(r) {
			return -1
		}
		return r
	}, name))
	for len(name) > actorNameLimit {
		_, size := utf8.DecodeLastRuneInString(name)
		name = name[:len(name)-size]
	}
	return ledger.Actor{ID: id, Name: name}, nil
}

// failed maps a ledger error to its answer. One it does not know is the database failing.
func (s *Server) failed(err error, doing string) error {
	var refused *ledger.InvalidError
	switch {
	case errors.As(err, &refused):
		return invalidRequest(refused.Problem)
	case errors.Is(err, ledger.ErrDefaultTier):
		return problem(http.StatusConflict, codeDefaultTier, "the default tier answers for every tier string with no row of its own, and cannot be deleted")
	case errors.Is(err, ledger.ErrIdempotencyKeyReused):
		return problem(http.StatusConflict, codeKeyReused, "the "+idempotencyHeader+" already holds a credit entry of another organization, kind or amount")
	}
	s.logger.Error("a management call failed", "doing", doing, "error", err)
	return huma.Error503ServiceUnavailable("the service's database failed; try again")
}

// encodeCursor is the opaque cursor of a page's last position.
func encodeCursor(position any) string {
	encoded, _ := json.Marshal(position)
	return base64.RawURLEncoding.EncodeToString(encoded)
}

// decodeCursor reads a cursor encodeCursor made into position. Every position is an object, so
// anything else (null would decode to the zero position) is not one this API answered.
func decodeCursor(cursor string, position any) error {
	decoded, err := base64.RawURLEncoding.DecodeString(cursor)
	if err != nil || !bytes.HasPrefix(bytes.TrimSpace(decoded), []byte("{")) || json.Unmarshal(decoded, position) != nil {
		return errCursor
	}
	return nil
}

// checkOrganization refuses an organization id no event could carry. Postgres would store a
// NUL stripped and invalid UTF-8 replaced, so the call would reach another organization.
func checkOrganization(id string) error {
	if id == "" || len(id) > usage.MaxIDLength || strings.ContainsAny(id, "|{}") ||
		!utf8.ValidString(id) || strings.ContainsFunc(id, unicode.IsControl) {
		return invalidRequest(fmt.Sprintf("organization_id is 1 to %d bytes of UTF-8, without | { } or a control character", usage.MaxIDLength))
	}
	return nil
}

// checkTierName refuses a tier name no row can have.
func checkTierName(name string) error {
	if !usage.ValidTierName(name) {
		return invalidRequest(fmt.Sprintf("tier name %q is not 1 to %d characters of A-Z, a-z, 0-9, _ . : -", name, usage.MaxTierNameLength))
	}
	return nil
}

// validTeam reports whether team is 1 to 64 characters of [A-Za-z0-9_.-].
func validTeam(team string) bool {
	if team == "" || len(team) > maxTeamLength {
		return false
	}
	for _, r := range team {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '_', r == '.', r == '-':
		default:
			return false
		}
	}
	return true
}

// registerManage declares the management API.
func (s *Server) registerManage(api huma.API) {
	failures := []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusNotFound, http.StatusServiceUnavailable}
	register := func(operation huma.Operation, description string) huma.Operation {
		operation.Tags = []string{tag}
		operation.Security = manageSecurity
		operation.Description = description + "\n\n" + manageNote
		if operation.Errors == nil {
			operation.Errors = failures
		}
		return operation
	}
	huma.Register(api, register(huma.Operation{
		OperationID: "listTiers", Method: http.MethodGet, Path: usage.PathManage + "/tiers",
		Summary: "List the tiers",
	}, "Every tier row, by name. A tier string with no row answers the `default` row's settings."), s.listTiers)
	huma.Register(api, register(huma.Operation{
		OperationID: "getTier", Method: http.MethodGet, Path: usage.PathManage + "/tiers/{name}",
		Summary: "Read a tier",
	}, "The tier's row, or, for a name with no row, the `default` tier's settings with `fallback_tier` true."), s.getTier)
	huma.Register(api, register(huma.Operation{
		OperationID: "listTierOverrides", Method: http.MethodGet, Path: usage.PathManage + "/tiers/{name}/overrides",
		Summary: "List the organizations overriding a tier",
	}, "The organizations whose newest event carried this tier and that override any of its settings, the spend "+
		"hard limit among them, by organization id."), s.listTierOverrides)
	huma.Register(api, register(huma.Operation{
		OperationID: "setTier", Method: http.MethodPut, Path: usage.PathManage + "/tiers/{name}",
		Summary: "Write a tier",
	}, "Creates the tier or replaces its settings. A setting left out is no limit; no tier needs a spend hard limit. "+
		"Every router sees it at its next standings read."), s.setTier)
	huma.Register(api, register(huma.Operation{
		OperationID: "deleteTier", Method: http.MethodDelete, Path: usage.PathManage + "/tiers/{name}",
		Summary: "Delete a tier", DefaultStatus: http.StatusNoContent,
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusNotFound, http.StatusConflict, http.StatusServiceUnavailable},
	}, "Deletes the tier's row, if it has one: its organizations get the `default` tier's settings. The `default` "+
		"tier cannot be deleted (409 `default_tier`)."), s.deleteTier)
	huma.Register(api, register(huma.Operation{
		OperationID: "getAccount", Method: http.MethodGet, Path: usage.PathManage + "/accounts/{organization_id}",
		Summary: "Read an organization's account",
	}, "The organization's standing: its tier's settings, its own overrides and the settings in force, its status, "+
		"what was granted and spent, its balance, and its spend in the current UTC day and calendar month. Every "+
		"organization has an account: one nothing has changed is active with nothing granted or spent."), s.getAccount)
	huma.Register(api, register(huma.Operation{
		OperationID: "setAccountLimits", Method: http.MethodPut, Path: usage.PathManage + "/accounts/{organization_id}/limits",
		Summary: "Set an organization's own settings",
	}, "Replaces the organization's overrides of its tier's settings, the spend hard limit included: a setting left "+
		"out is the tier's again, and `{}` clears them all. An amount of 0 in the spend hard limit is no limit for "+
		"that interval, whatever the tier's. Answers the standing."), s.setAccountLimits)
	huma.Register(api, register(huma.Operation{
		OperationID: "addCredit", Method: http.MethodPost, Path: usage.PathManage + "/accounts/{organization_id}/credits",
		Summary: "Add to an organization's credit", DefaultStatus: http.StatusCreated,
		Responses: map[string]*huma.Response{"200": {
			Description: "The entry already written under the Idempotency-Key; nothing changed",
			Content:     map[string]*huma.MediaType{"application/json": {Schema: &huma.Schema{Ref: "#/components/schemas/CreditEntry"}}},
		}},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusNotFound, http.StatusConflict, http.StatusServiceUnavailable},
	}, "Writes a credit entry and adds its amount to the organization's credit, once per `"+idempotencyHeader+"` "+
		"(required, 1 to 255 characters): 201 with the entry written, or 200 with the one already written under the key, "+
		"which changes nothing. A key that holds another organization's entry, or one of another kind or amount, is a "+
		"409 `idempotency_key_reused`. Only the sign is checked: `top_up`, `grant` and `dispute_reversal` are above 0, "+
		"`refund` and `dispute` below, an `adjustment` either but not 0. Any account takes credit, a suspended one "+
		"included: payment rules are the writer's."), s.addCredit)
	huma.Register(api, register(huma.Operation{
		OperationID: "listCredits", Method: http.MethodGet, Path: usage.PathManage + "/accounts/{organization_id}/credits",
		Summary: "List an organization's credit entries",
	}, "Newest first."), s.listCredits)
	huma.Register(api, register(huma.Operation{
		OperationID: "suspendAccount", Method: http.MethodPost, Path: usage.PathManage + "/accounts/{organization_id}/suspend",
		Summary: "Suspend an organization",
	}, "Every request of a suspended organization is refused, free work included, until it is restored. The reason "+
		"(required, 1 to 500 characters) is the account's `status_reason` and is audited; suspending a suspended "+
		"organization replaces it. Credit can still be written. Answers the standing."), s.suspendAccount)
	huma.Register(api, register(huma.Operation{
		OperationID: "restoreAccount", Method: http.MethodPost, Path: usage.PathManage + "/accounts/{organization_id}/restore",
		Summary: "Restore a suspended organization",
	}, "Clears the suspension and its `status_reason`; the reason (required, 1 to 500 characters) is kept in the "+
		"audit alone. Restoring an active organization changes nothing and is audited all the same. Answers the "+
		"standing."), s.restoreAccount)
	huma.Register(api, register(huma.Operation{
		OperationID: "listUsage", Method: http.MethodGet, Path: usage.PathManage + "/accounts/{organization_id}/usage",
		Summary: "List an organization's usage events",
	}, "The organization's events, newest first, each with its tier, app, key, user and categories. `team` is the "+
		"category `team:<name>`, so it cannot be given with `category`; `user_id` needs `app_id`, since a user id is "+
		"only unique within its app."), s.listUsage)
}

// manageNote ends every management operation's description.
const manageNote = "Management API: it takes the manage key in `" + manageKeyHeader + "`, never the ingest token, and " +
	"answers 404 `not_configured` when the service has none. Every call names its actor in `" + actorIDHeader +
	"` (1 to 256 characters) and may name a person in `" + actorNameHeader + "`, percent-encoded when it is not " +
	"printable ASCII; every change is recorded against them."

// The parameters several operations share, embedded in their inputs. Exported, because Huma
// reads no parameter of an unexported embedded struct.

// PageQuery is a list's position: at most limit items after the cursor a page answered.
type PageQuery struct {
	Limit  int    `query:"limit" default:"50" minimum:"1" maximum:"200" doc:"The most items one page answers."`
	Cursor string `query:"cursor" doc:"The next_cursor of the page before; absent for the first page."`
}

// TierPath names a tier.
type TierPath struct {
	Name string `path:"name" doc:"The tier string: 1 to 64 characters of A-Z, a-z, 0-9, _ . : -."`
}

// OrganizationPath names an organization.
type OrganizationPath struct {
	OrganizationID string `path:"organization_id" doc:"The organization: 1 to 255 bytes of UTF-8, without | { } or a control character."`
}

// TierList is every tier row.
type TierList struct {
	Tiers []usage.Tier `json:"tiers"`
}

type tierListOutput struct {
	Body TierList
}

func (s *Server) listTiers(ctx context.Context, _ *struct{}) (*tierListOutput, error) {
	tiers, err := s.ledger.Tiers(ctx)
	if err != nil {
		return nil, s.failed(err, "list tiers")
	}
	return &tierListOutput{Body: TierList{Tiers: tiers}}, nil
}

// TierAnswer is the tier a name answers.
type TierAnswer struct {
	Tier         usage.Tier `json:"tier"`
	FallbackTier bool       `json:"fallback_tier" doc:"The name has no row: tier is the default tier's."`
}

type tierOutput struct {
	Body TierAnswer
}

func (s *Server) getTier(ctx context.Context, input *TierPath) (*tierOutput, error) {
	if err := checkTierName(input.Name); err != nil {
		return nil, err
	}
	tier, found, err := s.ledger.Tier(ctx, input.Name)
	if err != nil {
		return nil, s.failed(err, "get tier")
	}
	return &tierOutput{Body: TierAnswer{Tier: tier, FallbackTier: !found}}, nil
}

// TierOverride is one organization's overrides of its tier's settings.
type TierOverride struct {
	OrganizationID string       `json:"organization_id"`
	LimitOverrides usage.Limits `json:"limit_overrides"`
}

// TierOverridePage is one page of a tier's organizations with overrides.
type TierOverridePage struct {
	Items      []TierOverride `json:"items"`
	HasMore    bool           `json:"has_more"`
	NextCursor string         `json:"next_cursor,omitempty"`
}

// overridePosition is where a page of tier overrides ends.
type overridePosition struct {
	OrganizationID string `json:"o"`
}

type tierOverridesInput struct {
	TierPath
	PageQuery
}

type tierOverridesOutput struct {
	Body TierOverridePage
}

func (s *Server) listTierOverrides(ctx context.Context, input *tierOverridesInput) (*tierOverridesOutput, error) {
	if err := checkTierName(input.Name); err != nil {
		return nil, err
	}
	var after overridePosition
	if input.Cursor != "" {
		if err := decodeCursor(input.Cursor, &after); err != nil {
			return nil, err
		}
	}
	accounts, more, err := s.ledger.TierOverrides(ctx, input.Name, after.OrganizationID, input.Limit)
	if err != nil {
		return nil, s.failed(err, "list tier overrides")
	}
	page := TierOverridePage{Items: make([]TierOverride, 0, len(accounts)), HasMore: more}
	for _, account := range accounts {
		page.Items = append(page.Items, TierOverride{OrganizationID: account.OrganizationID, LimitOverrides: account.LimitOverrides})
	}
	if more {
		page.NextCursor = encodeCursor(overridePosition{OrganizationID: accounts[len(accounts)-1].OrganizationID})
	}
	return &tierOverridesOutput{Body: page}, nil
}

// TierSettings are what a tier decides.
type TierSettings struct {
	_        struct{}       `additionalProperties:"false"`
	FailMode usage.FailMode `json:"fail_mode" enum:"closed,open" doc:"What admission does with billable work while it cannot read the organization's balance: closed refuses it, open admits it."`
	Limits   usage.Limits   `json:"limits,omitempty" doc:"The tier's settings; a setting left out is no limit."`
}

type setTierInput struct {
	TierPath
	Body TierSettings
}

type setTierOutput struct {
	Body usage.Tier
}

func (s *Server) setTier(ctx context.Context, input *setTierInput) (*setTierOutput, error) {
	if err := checkTierName(input.Name); err != nil {
		return nil, err
	}
	tier, err := s.ledger.SetTier(ctx, usage.Tier{Name: input.Name, FailMode: input.Body.FailMode, Limits: input.Body.Limits}, actorFrom(ctx))
	if err != nil {
		return nil, s.failed(err, "set tier")
	}
	return &setTierOutput{Body: tier}, nil
}

func (s *Server) deleteTier(ctx context.Context, input *TierPath) (*struct{}, error) {
	if err := checkTierName(input.Name); err != nil {
		return nil, err
	}
	if _, err := s.ledger.DeleteTier(ctx, input.Name, actorFrom(ctx)); err != nil {
		return nil, s.failed(err, "delete tier")
	}
	return nil, nil
}

type standingOutput struct {
	Body usage.Standing
}

// standing is the organization's standing on tier, or, when tier is empty, on the tier its
// newest event carried.
func (s *Server) standing(ctx context.Context, organization, tier string) (*standingOutput, error) {
	if tier == "" {
		var err error
		if tier, _, err = s.ledger.LastTier(ctx, organization); err != nil {
			return nil, s.failed(err, "read last tier")
		}
	}
	standings, err := s.ledger.Standings(ctx, []usage.StandingOrganization{{OrganizationID: organization, OrganizationTier: tier}})
	if err != nil {
		return nil, s.failed(err, "read standing")
	}
	return &standingOutput{Body: standings[0]}, nil
}

type getAccountInput struct {
	OrganizationPath
	OrganizationTier string `query:"organization_tier" doc:"The tier to answer the standing on; absent is the tier the organization's newest event carried."`
}

func (s *Server) getAccount(ctx context.Context, input *getAccountInput) (*standingOutput, error) {
	if err := checkOrganization(input.OrganizationID); err != nil {
		return nil, err
	}
	return s.standing(ctx, input.OrganizationID, input.OrganizationTier)
}

type setAccountLimitsInput struct {
	OrganizationPath
	Body usage.Limits
}

func (s *Server) setAccountLimits(ctx context.Context, input *setAccountLimitsInput) (*standingOutput, error) {
	if err := checkOrganization(input.OrganizationID); err != nil {
		return nil, err
	}
	if _, err := s.ledger.SetAccountLimits(ctx, input.OrganizationID, input.Body, actorFrom(ctx)); err != nil {
		return nil, s.failed(err, "set account limits")
	}
	return s.standing(ctx, input.OrganizationID, "")
}

// CreditRequest is a credit entry to write.
type CreditRequest struct {
	_            struct{}         `additionalProperties:"false"`
	Kind         usage.CreditKind `json:"kind" enum:"top_up,grant,refund,dispute,dispute_reversal,adjustment" doc:"What the entry is, which fixes its sign."`
	AmountMicros int64            `json:"amount_micros" doc:"Signed, in micros: above 0 for top_up, grant and dispute_reversal, below 0 for refund and dispute, either for adjustment."`
	Reference    string           `json:"reference,omitempty" maxLength:"255" doc:"The writer's own reference, such as a payment id."`
	Note         string           `json:"note,omitempty" maxLength:"1000"`
}

type addCreditInput struct {
	OrganizationPath
	IdempotencyKey string `header:"Idempotency-Key" doc:"Required: 1 to 255 characters. The entry is written once under it."`
	Body           CreditRequest
}

type addCreditOutput struct {
	Status int
	Body   usage.CreditEntry
}

func (s *Server) addCredit(ctx context.Context, input *addCreditInput) (*addCreditOutput, error) {
	if err := checkOrganization(input.OrganizationID); err != nil {
		return nil, err
	}
	if n := utf8.RuneCountInString(input.IdempotencyKey); n == 0 || n > ledger.MaxIdempotencyKeyLength {
		return nil, invalidRequest(fmt.Sprintf("a credit is written once under its %s: 1 to %d characters", idempotencyHeader, ledger.MaxIdempotencyKeyLength))
	}
	actor := actorFrom(ctx)
	entry, written, err := s.ledger.AddCredit(ctx, ledger.CreditEntry{
		OrganizationID: input.OrganizationID, Kind: input.Body.Kind, AmountMicros: input.Body.AmountMicros,
		Reference: input.Body.Reference, IdempotencyKey: input.IdempotencyKey, Note: input.Body.Note,
		Actor: actor.ID, ActorName: actor.Name,
	})
	if err != nil {
		return nil, s.failed(err, "add credit")
	}
	status := http.StatusOK
	if written {
		status = http.StatusCreated
	}
	return &addCreditOutput{Status: status, Body: entry.Contract()}, nil
}

// CreditEntryPage is one page of an organization's credit entries.
type CreditEntryPage struct {
	Items      []usage.CreditEntry `json:"items"`
	HasMore    bool                `json:"has_more"`
	NextCursor string              `json:"next_cursor,omitempty"`
}

type listCreditsInput struct {
	OrganizationPath
	PageQuery
}

type listCreditsOutput struct {
	Body CreditEntryPage
}

func (s *Server) listCredits(ctx context.Context, input *listCreditsInput) (*listCreditsOutput, error) {
	if err := checkOrganization(input.OrganizationID); err != nil {
		return nil, err
	}
	var after *ledger.CreditPosition
	if input.Cursor != "" {
		after = &ledger.CreditPosition{}
		if err := decodeCursor(input.Cursor, after); err != nil {
			return nil, err
		}
	}
	entries, more, err := s.ledger.CreditEntries(ctx, input.OrganizationID, after, input.Limit)
	if err != nil {
		return nil, s.failed(err, "list credits")
	}
	page := CreditEntryPage{Items: make([]usage.CreditEntry, 0, len(entries)), HasMore: more}
	for _, entry := range entries {
		page.Items = append(page.Items, entry.Contract())
	}
	if more {
		last := entries[len(entries)-1]
		page.NextCursor = encodeCursor(ledger.CreditPosition{CreatedAt: last.CreatedAt, ID: last.ID})
	}
	return &listCreditsOutput{Body: page}, nil
}

// StatusChange is why an organization is suspended or restored.
type StatusChange struct {
	_      struct{} `additionalProperties:"false"`
	Reason string   `json:"reason,omitempty" doc:"Required: 1 to 500 characters."`
}

type statusInput struct {
	OrganizationPath
	Body StatusChange
}

func (s *Server) suspendAccount(ctx context.Context, input *statusInput) (*standingOutput, error) {
	return s.setStatus(ctx, input, usage.AccountSuspended)
}

func (s *Server) restoreAccount(ctx context.Context, input *statusInput) (*standingOutput, error) {
	return s.setStatus(ctx, input, usage.AccountActive)
}

func (s *Server) setStatus(ctx context.Context, input *statusInput, status usage.AccountStatus) (*standingOutput, error) {
	if err := checkOrganization(input.OrganizationID); err != nil {
		return nil, err
	}
	if _, err := s.ledger.SetAccountStatus(ctx, input.OrganizationID, status, input.Body.Reason, actorFrom(ctx)); err != nil {
		return nil, s.failed(err, "set account status")
	}
	return s.standing(ctx, input.OrganizationID, "")
}

// UsageEventPage is one page of an organization's usage events.
type UsageEventPage struct {
	Items      []usage.Event `json:"items"`
	HasMore    bool          `json:"has_more"`
	NextCursor string        `json:"next_cursor,omitempty"`
}

type listUsageInput struct {
	OrganizationPath
	PageQuery
	From     time.Time `query:"from" doc:"Events that happened at or after this instant."`
	To       time.Time `query:"to" doc:"Events that happened before this instant."`
	Meter    string    `query:"meter" doc:"The event's kind of work, such as llm or runtime."`
	AppID    string    `query:"app_id"`
	APIKey   string    `query:"api_key"`
	UserID   string    `query:"user_id" doc:"Needs app_id: a user id is only unique within its app."`
	Team     string    `query:"team" doc:"A team name, 1 to 64 characters of A-Z, a-z, 0-9, _ . -: the events with the category team:<name>. Not with category."`
	Category string    `query:"category" doc:"Events with this usage category among theirs."`
}

type listUsageOutput struct {
	Body UsageEventPage
}

func (s *Server) listUsage(ctx context.Context, input *listUsageInput) (*listUsageOutput, error) {
	if err := checkOrganization(input.OrganizationID); err != nil {
		return nil, err
	}
	filter := ledger.UsageFilter{
		OrganizationID: input.OrganizationID, From: input.From, To: input.To, Meter: input.Meter,
		AppID: input.AppID, APIKey: input.APIKey, UserID: input.UserID, Category: input.Category, Limit: input.Limit,
	}
	switch {
	case input.UserID != "" && input.AppID == "":
		return nil, invalidRequest("user_id needs app_id: a user id is only unique within its app")
	case input.Team != "" && input.Category != "":
		return nil, invalidRequest("team is the category team:<name>: give team or category, not both")
	case input.Team != "":
		if !validTeam(input.Team) {
			return nil, invalidRequest(fmt.Sprintf("team %q is not 1 to %d characters of A-Z, a-z, 0-9, _ . -", input.Team, maxTeamLength))
		}
		filter.Category = "team:" + input.Team
	}
	if input.Cursor != "" {
		filter.After = &ledger.UsagePosition{}
		if err := decodeCursor(input.Cursor, filter.After); err != nil {
			return nil, err
		}
	}
	rows, more, err := s.ledger.Usage(ctx, filter)
	if err != nil {
		return nil, s.failed(err, "list usage")
	}
	page := UsageEventPage{Items: make([]usage.Event, 0, len(rows)), HasMore: more}
	for _, row := range rows {
		event := row.Event()
		event.OccurredAt = event.OccurredAt.UTC()
		page.Items = append(page.Items, event)
	}
	if more {
		last := rows[len(rows)-1]
		page.NextCursor = encodeCursor(ledger.UsagePosition{OccurredAt: last.OccurredAt, UsageID: last.UsageID})
	}
	return &listUsageOutput{Body: page}, nil
}
