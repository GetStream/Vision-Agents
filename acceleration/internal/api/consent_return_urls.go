package api

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strings"

	"github.com/danielgtaylor/huma/v2"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// The bounds of an app's list. The settings document it is kept in is part of the API key
// cache entry read on every authenticated request (appconfig.Store.APIKey,
// store.LiveAPIKey), so the list stays small: 10 URLs of 2048 bytes is 20 KB at most. 2048
// is the length the router takes for every other URL it keeps (ConnectorEventDestinationRequest.url,
// McpServer.url).
const (
	maxConsentReturnURLs      = 10
	maxConsentReturnURLLength = 2048
)

// errNoAppSettings is every deployment but api_key: only there is the caller's app a row of
// its own (README "Authentication"). proxy and noauth name a customer that no row stands
// behind, and noauth believes whoever names one, so a list written there would be anybody's.
var errNoAppSettings = notImplemented("this deployment keeps no app settings: consent return URLs are " +
	"registered on a router that runs with ROUTER_AUTH_MODE=api_key, where the app is a row of its own")

// ConsentReturnUrls are the pages of an app's own that may finish a connector consent.
type ConsentReturnUrls struct {
	Urls []string `json:"urls" maxItems:"10" nullable:"false" doc:"Absolute URLs of at most 2048 bytes each, kept exactly as sent. Each is https, or http on a loopback host (localhost, 127.0.0.0/8 or [::1]) for local development (RFC 8252 sections 7.3 and 8.3). None carries a user name, a password or a fragment (RFC 6749 section 3.1.2), and none is listed twice. Empty when none are registered."`
}

func (*ConsentReturnUrls) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "The pages of the app's own that a connector consent may return to, " +
		"besides the dashboard. Nothing reads the list yet: every consent still finishes on " +
		"the dashboard."
	return schema
}

type setConsentReturnURLsRequest struct {
	Body ConsentReturnUrls
}

type consentReturnURLsResponse struct {
	Body ConsentReturnUrls
}

// registerConsentReturnURLs declares the app's list of consent return URLs. Both operations
// are server-side only: the list says where a consent may send an end user back to, which is
// the app's backend's to decide.
func (s *Server) registerConsentReturnURLs(api huma.API) {
	huma.Register(api, huma.Operation{
		OperationID: "setConsentReturnUrls",
		Method:      http.MethodPut,
		Path:        "/v1/agents/consent-return-urls",
		Summary:     "Set the pages that may finish the app's connector consents",
		Description: "Replaces the app's whole list of consent return URLs with the one sent, and " +
			"answers it. An empty list removes them all. Other app settings are left as they are.\n\n" +
			"Answered 501 by a router that does not run with ROUTER_AUTH_MODE=api_key, which " +
			"keeps no settings for the app.\n\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an " +
			"end user's device.",
		Responses: map[string]*huma.Response{"200": {Description: "The list as it now is"}},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden,
			http.StatusNotImplemented},
	}, s.setConsentReturnURLs)
	huma.Register(api, huma.Operation{
		OperationID: "getConsentReturnUrls",
		Method:      http.MethodGet,
		Path:        "/v1/agents/consent-return-urls",
		Summary:     "Read the pages that may finish the app's connector consents",
		Description: "The app's consent return URLs as they were registered, an empty list when " +
			"there are none.\n\n" +
			"Answered 501 by a router that does not run with ROUTER_AUTH_MODE=api_key, which " +
			"keeps no settings for the app.\n\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an " +
			"end user's device.",
		Responses: map[string]*huma.Response{"200": {Description: "The list"}},
		Errors:    []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotImplemented},
	}, s.getConsentReturnURLs)
}

func (s *Server) setConsentReturnURLs(ctx context.Context, request *setConsentReturnURLsRequest) (*consentReturnURLsResponse, error) {
	appID, err := s.appWithSettings(ctx)
	if err != nil {
		return nil, err
	}
	urls := request.Body.Urls
	if err := checkConsentReturnURLs(urls); err != nil {
		return nil, invalidRequest(err.Error())
	}
	err = s.store.SetConsentReturnURLs(ctx, appID, urls)
	if errors.Is(err, store.ErrNoApp) {
		return nil, errNoAppSettings
	}
	if err != nil {
		return nil, err
	}
	return &consentReturnURLsResponse{Body: ConsentReturnUrls{Urls: nonNil(urls)}}, nil
}

func (s *Server) getConsentReturnURLs(ctx context.Context, _ *struct{}) (*consentReturnURLsResponse, error) {
	appID, err := s.appWithSettings(ctx)
	if err != nil {
		return nil, err
	}
	settings, err := s.store.AppSettingsFor(ctx, appID)
	if err != nil {
		return nil, err
	}
	return &consentReturnURLsResponse{Body: ConsentReturnUrls{Urls: nonNil(settings.ConsentReturnURLs)}}, nil
}

// appWithSettings is the caller's app, on a deployment where an app has settings to keep.
func (s *Server) appWithSettings(ctx context.Context) (string, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return "", errMissingCustomer
	}
	if s.authMode != auth.APIKey || s.store == nil {
		return "", errNoAppSettings
	}
	return customerID, nil
}

// checkConsentReturnURLs is why the list cannot be registered, or nil.
func checkConsentReturnURLs(urls []string) error {
	seen := make(map[string]int, len(urls))
	for i, raw := range urls {
		if err := checkConsentReturnURL(raw); err != nil {
			return fmt.Errorf("urls[%d]: %w", i, err)
		}
		// Refused rather than collapsed, so the list stored is the list sent.
		if first, ok := seen[raw]; ok {
			return fmt.Errorf("urls[%d]: the same URL as urls[%d]", i, first)
		}
		seen[raw] = i
	}
	return nil
}

func checkConsentReturnURL(raw string) error {
	if len(raw) > maxConsentReturnURLLength {
		return fmt.Errorf("longer than %d bytes", maxConsentReturnURLLength)
	}
	parsed, err := url.Parse(raw)
	if err != nil || !parsed.IsAbs() || parsed.Opaque != "" || parsed.Hostname() == "" {
		return errors.New("not an absolute URL with a host")
	}
	if parsed.User != nil {
		return errors.New("a URL carrying a user name or password is refused")
	}
	// An empty fragment ("…#") leaves parsed.Fragment empty, so the string is what is checked.
	if strings.Contains(raw, "#") {
		return errors.New("a URL with a fragment is refused (RFC 6749 section 3.1.2)")
	}
	switch parsed.Scheme {
	case "https":
		return nil
	case "http":
		// RFC 8252 sections 7.3 and 8.3: plain http is for the loopback interface alone,
		// which never leaves the machine.
		if loopbackHost(parsed.Hostname()) {
			return nil
		}
		return errors.New("http is taken for a loopback host alone (localhost, 127.0.0.0/8 or [::1]); use https")
	}
	return errors.New("the scheme must be https")
}

// nonNil is a list a JSON answer sends as [] rather than null.
func nonNil(urls []string) []string {
	if urls == nil {
		return []string{}
	}
	return urls
}
