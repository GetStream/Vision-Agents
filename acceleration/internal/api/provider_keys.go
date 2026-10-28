package api

import (
	"context"
	"errors"
	"net/http"
	"slices"
	"strings"
	"time"

	"github.com/danielgtaylor/huma/v2"

	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/providerkeys"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// errNoProviderKeys is what the provider key paths say on a deployment without a database.
var errNoProviderKeys = notConfigured("provider keys are not available: no database configured")

// errNoProviderKeySealer is a provider key put on a deployment that could only store it in the
// clear.
var errNoProviderKeySealer = notConfigured("a provider key cannot be stored: no key encryption key configured")

// errUnknownProviderKey is a delete of a key the customer does not keep.
var errUnknownProviderKey = APIError{
	Type: ErrorTypeNotFound, Code: codeProviderKeyNotFound,
	Message: "there is no key for that provider in that region",
}

// listProviderKeys returns a page of the keys the calling customer keeps, by provider and then
// region. A key itself is never returned, only its last four characters.
func (s *Server) listProviderKeys(ctx context.Context, request *listProviderKeysRequest) (*listProviderKeysResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if s.configs == nil {
		return nil, errNoProviderKeys
	}
	after, err := decodeCursor[store.ProviderKeyPosition](request.Cursor.ptr())
	if err != nil {
		return nil, invalidRequest(err.Error())
	}
	limit := store.ProviderKeyLimit(value(request.Limit.ptr()))
	rows, err := s.store.ProviderKeyPage(ctx, customerID, limit, after)
	if err != nil {
		return nil, err
	}
	rows, more := page(rows, limit)
	listed := ProviderKeyPage{Items: make([]ProviderKey, 0, len(rows)), HasMore: more}
	for _, row := range rows {
		listed.Items = append(listed.Items, providerKeyOf(row))
	}
	if more {
		last := rows[len(rows)-1]
		listed.NextCursor = encodeCursor(store.ProviderKeyPosition{Provider: last.Provider, Region: last.Region})
	}
	return &listProviderKeysResponse{Body: listed}, nil
}

// putProviderKey stores the key the calling customer keeps for a provider in a region, sealed,
// replacing the one they kept there.
func (s *Server) putProviderKey(ctx context.Context, request *putProviderKeyRequest) (*putProviderKeyResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if s.configs == nil {
		return nil, errNoProviderKeys
	}
	sent := request.Body
	if sent == nil {
		return nil, invalidRequest("a request body is required")
	}
	region := value(sent.Region)
	if region == "" {
		region = pricing.DefaultRegion
	}
	if !pricing.ValidRegion(region) {
		return nil, invalidRequest("region is a letter and up to 31 of a-z, 0-9 and hyphens, such as us or eu")
	}
	if failure := s.keyComplaint(sent.Provider); failure != nil {
		return nil, failure
	}
	if !providerkeys.Valid(sent.ApiKey) {
		return nil, invalidRequest("api_key is up to 1024 characters, with no space, comma or control character")
	}
	if s.secrets == nil {
		return nil, errNoProviderKeySealer
	}
	sealed, err := providerkeys.Seal(s.secrets, customerID, sent.Provider, region, sent.ApiKey)
	if err != nil {
		return nil, err
	}
	was, replaced, err := s.configs.PutProviderKey(ctx, &sealed)
	if err != nil {
		return nil, err
	}
	record := auditRecord{
		ResourceType: store.AuditProviderKey, ResourceID: providerKeyName(sealed), ResourceName: providerKeyName(sealed),
		Action: store.AuditCreated, Changes: auditDiff(nil, auditedProviderKey(sealed)),
	}
	if replaced {
		// A key replaced by one ending in the same four characters is still a change.
		record.Action = store.AuditUpdated
		record.Changes = []store.AuditChange{{Field: "api_key", Before: maskedKey(was.KeyHint), After: maskedKey(sealed.KeyHint)}}
	}
	s.audit(ctx, record)
	return &putProviderKeyResponse{Body: providerKeyOf(sealed)}, nil
}

// deleteProviderKey forgets the key the calling customer keeps for a provider in a region. Work
// is routed on the deployment's own key from then on.
func (s *Server) deleteProviderKey(ctx context.Context, request *deleteProviderKeyRequest) (*struct{}, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if s.configs == nil {
		return nil, errNoProviderKeys
	}
	region := value(request.Region.ptr())
	if region == "" {
		region = pricing.DefaultRegion
	}
	if !pricing.ValidRegion(region) {
		return nil, invalidRequest("region is a letter and up to 31 of a-z, 0-9 and hyphens, such as us or eu")
	}
	deleted, err := s.configs.DeleteProviderKey(ctx, customerID, request.Provider, region)
	if errors.Is(err, store.ErrNoProviderKey) {
		return nil, errUnknownProviderKey
	}
	if err != nil {
		return nil, err
	}
	s.audit(ctx, auditRecord{
		ResourceType: store.AuditProviderKey, ResourceID: providerKeyName(deleted), ResourceName: providerKeyName(deleted),
		Action: store.AuditDeleted, Changes: auditDiff(auditedProviderKey(deleted), nil),
	})
	return nil, nil
}

// keyComplaint refuses a provider whose work is never routed on a customer's own key here.
func (s *Server) keyComplaint(provider string) error {
	keyed := s.keyedProviders()
	if slices.Contains(keyed, provider) {
		return nil
	}
	return APIError{
		Type: ErrorTypeInvalidRequest, Code: codeProviderTakesNoKey,
		Message: "no work is routed on a key of your own at " + provider + " here: a key is kept for one of " +
			strings.Join(keyed, ", "),
	}
}

// keyedProviders is every provider a customer may keep a key for on this deployment, sorted:
// each language model vendor it routes whose factory builds with a key of the instance's own,
// turbopuffer for knowledge and Daytona for sandboxes.
func (s *Server) keyedProviders() []string {
	keyed := []string{providerkeys.Turbopuffer, providerkeys.Daytona}
	for _, router := range s.routers {
		takes, ok := router.(interface{ TakesCustomerKey(provider string) bool })
		if !ok {
			continue
		}
		for _, provider := range router.Config().Providers {
			if takes.TakesCustomerKey(provider.Provider) && !slices.Contains(keyed, provider.Provider) {
				keyed = append(keyed, provider.Provider)
			}
		}
	}
	slices.Sort(keyed)
	return keyed
}

// providerKeyName is what a key's audit entries name it by: its provider and region.
func providerKeyName(key store.ProviderKey) string { return key.Provider + "/" + key.Region }

// maskedKey is a key as an audit entry shows it: its last four characters at most.
func maskedKey(hint string) string { return "****" + hint }

// auditedProviderKey is a key as its audit entries show it, the key itself masked.
func auditedProviderKey(key store.ProviderKey) map[string]string {
	return map[string]string{"provider": key.Provider, "region": key.Region, "api_key": maskedKey(key.KeyHint)}
}

// providerKeyOf renders a key for the wire. The key itself never goes back out.
func providerKeyOf(key store.ProviderKey) ProviderKey {
	return ProviderKey{
		Provider:  key.Provider,
		Region:    key.Region,
		KeyHint:   key.KeyHint,
		CreatedAt: key.CreatedAt,
		UpdatedAt: key.UpdatedAt,
	}
}

// registerProviderKeys declares the operations served in provider_keys.go.
func (s *Server) registerProviderKeys(api huma.API) {
	huma.Register(api, huma.Operation{
		OperationID: "listProviderKeys",
		Method:      http.MethodGet,
		Path:        "/v1/agents/provider-keys",
		Summary:     "The keys the calling customer keeps for their own accounts at the vendors",
		Description: "Each key's provider, region, last four characters and when it was put. A key itself is " +
			"never returned.\nServer-side only: it needs a server-side token, so it cannot be reached from an " +
			"end user's device.",
		Responses: map[string]*huma.Response{
			"200": {Description: "The customer's keys, by provider and then region"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden},
	}, s.listProviderKeys)
	huma.Register(api, huma.Operation{
		OperationID: "putProviderKey",
		Method:      http.MethodPut,
		Path:        "/v1/agents/provider-keys",
		Summary:     "Keep a key for the customer's own account at a vendor",
		Description: "Bring your own key (BYOK). The key replaces the one kept for the provider in the region, " +
			"and is stored sealed and never returned. From the next request, each model of that provider in " +
			"that region is tried on the customer's key first, immediately before the deployment's own instance " +
			"of it, which is asked next when the customer's fails, as any next candidate is. Work on the " +
			"customer's key is billed to them by the vendor, and by the router at its BYOK price, a fee of 20% " +
			"of what the work costs at the vendor's rates. A turbopuffer key keeps the knowledge bases the " +
			"customer first writes from then on in their own turbopuffer account, and a daytona key makes their " +
			"sandboxes in their own Daytona organization. A provider whose work is never routed on a customer's " +
			"key is refused with provider_takes_no_key, naming the ones that are. Only a key is taken, never an " +
			"endpoint: the deployment's own is used.\nServer-side only: it needs a server-side token, so it " +
			"cannot be reached from an end user's device.",
		Responses: map[string]*huma.Response{
			"200": {Description: "The key is kept"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden},
	}, s.putProviderKey)
	huma.Register(api, huma.Operation{
		OperationID: "deleteProviderKey",
		Method:      http.MethodDelete,
		Path:        "/v1/agents/provider-keys/{provider}",
		Summary:     "Forget a key the customer keeps for a vendor",
		Description: "Work is routed on the deployment's own keys from the next request. A knowledge base kept " +
			"in the customer's own turbopuffer account is refused until a key for its region is put back: it " +
			"is never read from the deployment's account.\nServer-side only: it needs a server-side token, so " +
			"it cannot be reached from an end user's device.",
		DefaultStatus: http.StatusNoContent,
		Responses: map[string]*huma.Response{
			"204": {Description: "The key is gone"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.deleteProviderKey)
}

type listProviderKeysRequest struct {
	Limit  optionalParam[int]    `query:"limit" doc:"Up to 200. Omitted is 25." minimum:"1" maximum:"200"`
	Cursor optionalParam[string] "query:\"cursor\" doc:\"The `next_cursor` of the previous page. Omitted is the first page.\""
}

type listProviderKeysResponse struct {
	Body ProviderKeyPage
}

type putProviderKeyRequest struct {
	Body *ProviderKeyRequest `required:"true"`
}

type putProviderKeyResponse struct {
	Body ProviderKey
}

type deleteProviderKeyRequest struct {
	Provider string                `path:"provider" doc:"The provider the key is for, such as anthropic."`
	Region   optionalParam[string] `query:"region" doc:"The region the key is kept for. Omitted is us."`
}

// ProviderKeyRequest is the ProviderKeyRequest schema.
type ProviderKeyRequest struct {
	Provider string  `json:"provider" doc:"The vendor the key is for, as the router names it: a language model vendor whose models take a key of their own here (openai, anthropic, gemini, xai, deepseek where routed), turbopuffer or daytona." pattern:"^[a-z0-9][a-z0-9._-]{0,63}$" example:"anthropic"`
	Region   *string `json:"region,omitempty" pattern:"^[a-z][a-z0-9-]{0,31}$" patternDescription:"a letter and up to 31 of a-z, 0-9 and hyphens, such as us or eu" example:"eu" doc:"The region whose work the key is used for: only the instances running there are tried on it. Omitted is us."`
	ApiKey   string  `json:"api_key" doc:"The key, as the vendor issued it. Stored sealed and never returned." minLength:"1" maxLength:"1024" writeOnly:"true"`
}

// ProviderKey is the ProviderKey schema.
type ProviderKey struct {
	Provider  string    `json:"provider" example:"anthropic"`
	Region    string    `json:"region" example:"us"`
	KeyHint   string    `json:"key_hint" doc:"The key's last four characters, so one key can be told from another; empty for a key under 16 characters. The key itself is never returned." example:"x9Qa"`
	CreatedAt time.Time `json:"created_at" doc:"When a key for the provider in the region was first put."`
	UpdatedAt time.Time `json:"updated_at" doc:"When the key now kept was put."`
}

// ProviderKeyPage is the ProviderKeyPage schema.
type ProviderKeyPage struct {
	HasMore    bool          `json:"has_more"`
	Items      []ProviderKey `json:"items" nullable:"false"`
	NextCursor *string       "json:\"next_cursor,omitempty\" doc:\"Pass as `cursor` for the next page. Absent on the last one.\""
}
