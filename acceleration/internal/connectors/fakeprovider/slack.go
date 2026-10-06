package fakeprovider

import (
	"encoding/json"
	"net/http"
	"strings"
	"time"
	"unicode/utf8"
)

// PathSlackAPI is where the fake Slack's app manifest API and configuration token rotation
// answer: Server.URL + PathSlackAPI + method, as Slack's are https://slack.com/api/<method>
// (https://docs.slack.dev/reference/methods/apps.manifest.create).
const PathSlackAPI = "/api/"

// ConfigTokenTTL is how long a configuration token lives: «Each app configuration token will
// expire 12 hours after it has been generated»
// (https://docs.slack.dev/app-manifests/configuring-apps-with-app-manifests#config-tokens).
const ConfigTokenTTL = 12 * time.Hour

// slowRotationDelay is how long SlowConfigRotation holds a rotation: long enough that a second
// caller that does not wait for the first one's lock reaches the server while the first one's
// refresh token is still being spent. Not a Slack value.
const slowRotationDelay = 200 * time.Millisecond

// Limits the fake holds a manifest to, from the app manifest reference
// (https://docs.slack.dev/reference/app-manifest): display_information.name «Maximum length
// is 35 characters», settings.allowed_ip_address_ranges «Maximum 10 items».
const (
	slackMaxName     = 35
	slackMaxIPRanges = 10
)

// SlackApp is one app the fake Slack holds, as apps.manifest.create made it.
type SlackApp struct {
	AppID         string
	ClientID      string
	ClientSecret  string
	SigningSecret string
	// Manifest is the manifest the app was last created or updated with, as sent.
	Manifest json.RawMessage
	// Updates counts apps.manifest.update calls that changed it.
	Updates int
	Deleted bool
}

// configToken is one configuration token the fake issued.
type configToken struct {
	expires time.Time
}

// NewConfigToken is a workspace admin generating an app configuration token in Slack's app
// settings («Under Your App Configuration Tokens, click Generate Token»). It returns the
// refresh token, which is what the router is given and rotates. The access token beside it
// is never handed out, so the router's first call is a rotation.
func (s *Server) NewConfigToken() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	refresh := synthetic("xoxe")
	s.configRefresh[refresh] = false
	return refresh
}

// ConfigTokenRotations is how many tooling.tokens.rotate calls succeeded.
func (s *Server) ConfigTokenRotations() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.configRotations
}

// SlackApps are the apps the fake holds, deleted ones included, in the order they were made.
func (s *Server) SlackApps() []SlackApp {
	s.mu.Lock()
	defer s.mu.Unlock()
	apps := make([]SlackApp, 0, len(s.slackOrder))
	for _, id := range s.slackOrder {
		apps = append(apps, *s.slackApps[id])
	}
	return apps
}

// slackRoutes adds the fake Slack's methods to mux.
func (s *Server) slackRoutes(mux *http.ServeMux) {
	mux.HandleFunc("POST "+PathSlackAPI+"tooling.tokens.rotate", s.rotateConfigToken)
	mux.HandleFunc("POST "+PathSlackAPI+"apps.manifest.create", s.createSlackApp)
	mux.HandleFunc("POST "+PathSlackAPI+"apps.manifest.update", s.updateSlackApp)
	mux.HandleFunc("POST "+PathSlackAPI+"apps.manifest.delete", s.deleteSlackApp)
}

// rotateConfigToken is tooling.tokens.rotate
// (https://docs.slack.dev/reference/methods/tooling.tokens.rotate): a refresh token for a new
// configuration token, a new refresh token and the new token's exp. Slack's pages do not say
// whether the spent refresh token keeps working; the fake takes the strict reading and
// answers a second use with invalid_refresh_token, so a test sees a refresh token that was
// lost or spent twice. # unverified against Slack
func (s *Server) rotateConfigToken(w http.ResponseWriter, r *http.Request) {
	if s.isOn(SlowConfigRotation) {
		time.Sleep(slowRotationDelay)
	}
	refresh := r.PostFormValue("refresh_token")
	s.mu.Lock()
	defer s.mu.Unlock()
	spent, known := s.configRefresh[refresh]
	if !known || spent {
		slackError(w, "invalid_refresh_token")
		return
	}
	s.configRefresh[refresh] = true
	token, next := synthetic("xoxe.xoxp"), synthetic("xoxe")
	issued := s.now()
	s.configTokens[token] = &configToken{expires: issued.Add(ConfigTokenTTL)}
	s.configRefresh[next] = false
	s.configRotations++
	slackOK(w, map[string]any{
		"token": token, "refresh_token": next, "team_id": s.TeamID, "user_id": s.UserID,
		"iat": issued.Unix(), "exp": issued.Add(ConfigTokenTTL).Unix(),
	})
}

// createSlackApp is apps.manifest.create: a new app with an id and credentials, from a
// manifest held to the reference's limits.
func (s *Server) createSlackApp(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if code := s.configTokenRefusal(r); code != "" {
		slackError(w, code)
		return
	}
	manifest := r.PostFormValue("manifest")
	if code := checkSlackManifest(manifest); code != "" {
		slackError(w, code)
		return
	}
	// The shape of the page's example app id, A012ABCD0A0.
	app := &SlackApp{
		AppID:         "A" + strings.ToUpper(synthetic("app")[4:14]),
		ClientID:      syntheticDigits(13) + "." + syntheticDigits(13),
		ClientSecret:  synthetic("slack-client-secret"),
		SigningSecret: synthetic("slack-signing-secret"),
		Manifest:      json.RawMessage(manifest),
	}
	s.slackApps[app.AppID] = app
	s.slackOrder = append(s.slackOrder, app.AppID)
	slackOK(w, map[string]any{
		"app_id": app.AppID,
		"credentials": map[string]string{
			"client_id": app.ClientID, "client_secret": app.ClientSecret,
			"verification_token": synthetic("verification"), "signing_secret": app.SigningSecret,
		},
	})
}

// updateSlackApp is apps.manifest.update
// (https://docs.slack.dev/reference/methods/apps.manifest.update): app_not_found for an app it
// does not hold.
func (s *Server) updateSlackApp(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if code := s.configTokenRefusal(r); code != "" {
		slackError(w, code)
		return
	}
	app, found := s.slackApps[r.PostFormValue("app_id")]
	if !found || app.Deleted {
		slackError(w, "app_not_found")
		return
	}
	manifest := r.PostFormValue("manifest")
	if code := checkSlackManifest(manifest); code != "" {
		slackError(w, code)
		return
	}
	app.Manifest = json.RawMessage(manifest)
	app.Updates++
	slackOK(w, map[string]any{"app_id": app.AppID, "permissions_updated": false})
}

// deleteSlackApp is apps.manifest.delete
// (https://docs.slack.dev/reference/methods/apps.manifest.delete).
func (s *Server) deleteSlackApp(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if code := s.configTokenRefusal(r); code != "" {
		slackError(w, code)
		return
	}
	app, found := s.slackApps[r.PostFormValue("app_id")]
	if !found || app.Deleted {
		slackError(w, "app_not_found")
		return
	}
	app.Deleted = true
	slackOK(w, map[string]any{})
}

// configTokenRefusal is the error code for a request whose bearer configuration token is not
// one the fake issued (not_authed without one, invalid_auth for an unknown one) or has
// expired (token_expired), or "" for a good one. Call it with mu held.
func (s *Server) configTokenRefusal(r *http.Request) string {
	token, found := strings.CutPrefix(r.Header.Get("Authorization"), "Bearer ")
	if !found || token == "" {
		return "not_authed"
	}
	issued, known := s.configTokens[token]
	if !known {
		return "invalid_auth"
	}
	if !s.now().Before(issued.expires) {
		return "token_expired"
	}
	return ""
}

// checkSlackManifest is invalid_manifest for a manifest that is not JSON, has no name or a
// name past the limit, too many IP ranges, or a request URL that is not https.
func checkSlackManifest(raw string) string {
	var manifest struct {
		DisplayInformation struct {
			Name string `json:"name"`
		} `json:"display_information"`
		Settings struct {
			EventSubscriptions *struct {
				RequestURL string `json:"request_url"`
			} `json:"event_subscriptions"`
			AllowedIPAddressRanges []string `json:"allowed_ip_address_ranges"`
		} `json:"settings"`
	}
	if err := json.Unmarshal([]byte(raw), &manifest); err != nil {
		return "invalid_manifest"
	}
	name := manifest.DisplayInformation.Name
	if name == "" || utf8.RuneCountInString(name) > slackMaxName || len(manifest.Settings.AllowedIPAddressRanges) > slackMaxIPRanges {
		return "invalid_manifest"
	}
	if events := manifest.Settings.EventSubscriptions; events != nil && !strings.HasPrefix(events.RequestURL, "https://") {
		return "invalid_manifest"
	}
	return ""
}

// slackOK answers {"ok": true} with fields, as every Slack method does on success.
func slackOK(w http.ResponseWriter, fields map[string]any) {
	fields["ok"] = true
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(fields)
}

// slackError answers {"ok": false, "error": code} with HTTP 200: «For failure results, the
// error property will contain a short machine-readable error code»
// (https://docs.slack.dev/apis/web-api/). The status of an error is not on that page; 200 is
// what core's Slack fixtures record (CommaScopes). # unverified for these methods
func slackError(w http.ResponseWriter, code string) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{"ok": false, "error": code})
}
