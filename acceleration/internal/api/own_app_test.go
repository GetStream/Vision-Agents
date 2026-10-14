//go:build integration

package api

import (
	"context"
	"encoding/json"
	"math/rand/v2"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"time"

	"github.com/golang-jwt/jwt/v5"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation/chattest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

const (
	// ownAppSecret signs every token the suite's gateway mints. The suite's Stream checks
	// no signature, so one secret serves every app.
	ownAppSecret = "the-apps-own-secret"
	// suiteMintBearer is what the suite's gateway expects of the router.
	suiteMintBearer = "the-routers-bearer"
)

// streamAppID is an app id nothing else in the database has.
func streamAppID() int64 { return 1_000_000_000 + rand.Int64N(8_000_000_000) }

// numberedApp is an app of its own whose id is a Stream app id.
func (s *RouterSuite) numberedApp(id int64) testApp {
	ctx := context.Background()
	organization := store.Organization{Name: "organization-" + s.utils.uuid()}
	s.Require().NoError(s.store.CreateOrganization(ctx, &organization))
	app := store.App{ID: strconv.FormatInt(id, 10), OrganizationID: organization.ID, Name: "app-" + s.utils.uuid()}
	s.Require().NoError(s.store.CreateApp(ctx, &app))
	return s.data.keyed(organization, app)
}

func (s *RouterSuite) appID() int64 {
	id, err := strconv.ParseInt(s.customerID(), 10, 64)
	s.Require().NoError(err)
	return id
}

// answerAs makes the suite's Stream say the key given is app id's, set up as the router needs.
func (s *RouterSuite) answerAs(apiKey string, id int64) {
	s.chat.SetAppFor(apiKey, chattest.App{
		ID:           id,
		ChannelTypes: map[string]map[string][]string{"agent": {"channel_member": {"read-channel", "create-message"}}},
		CallTypes:    []string{"agent"},
	})
}

// forwarded holds a server token for the calling app's own Stream app, minted for the key
// given, as the gateway forwards one with each request it vouches for.
func (s *RouterSuite) forwarded(apiKey string) {
	s.Require().NoError(s.stream.KeepToken(context.Background(), s.customerID(), apiKey,
		appToken(jwt.MapClaims{"exp": time.Now().Add(24 * time.Hour).Unix()})))
}

// vouched marks a hook from an app as the gateway marks one whose signature it checked.
func vouched(request *http.Request, app int64, apiKey string) {
	request.Header.Set(auth.HookAppHeader, strconv.FormatInt(app, 10))
	request.Header.Set(auth.ServerTokenKeyHeader, apiKey)
	request.Header.Set(auth.ServerTokenHeader, appToken(jwt.MapClaims{"exp": time.Now().Add(24 * time.Hour).Unix()}))
}

func appToken(claims jwt.MapClaims) string {
	token, err := jwt.NewWithClaims(jwt.SigningMethodHS256, claims).SignedString([]byte(ownAppSecret))
	if err != nil {
		panic(err)
	}
	return token
}

// gatewayMint is the gateway's internal mint endpoint, signing every app's tokens with
// ownAppSecret.
func (s *RouterSuite) gatewayMint() *httptest.Server {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer "+suiteMintBearer {
			http.Error(w, "no", http.StatusUnauthorized)
			return
		}
		var body struct {
			APIKey     string `json:"api_key"`
			UserID     string `json:"user_id"`
			TTLSeconds int    `json:"ttl_seconds"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		lasts, claims := 24*time.Hour, jwt.MapClaims{}
		if strings.HasSuffix(r.URL.Path, "/user-token") {
			lasts, claims["user_id"] = time.Duration(body.TTLSeconds)*time.Second, body.UserID
		}
		expires := time.Now().Add(lasts)
		claims["exp"] = expires.Unix()
		_ = json.NewEncoder(w).Encode(map[string]any{"api_key": body.APIKey, "token": appToken(claims), "expires_at": expires})
	}))
	s.T().Cleanup(server.Close)
	return server
}
