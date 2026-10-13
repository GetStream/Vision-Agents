//go:build integration

package streamapp

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"time"

	"github.com/golang-jwt/jwt/v5"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

const (
	forwardedKey    = "forwarded-key"
	forwardedSecret = "the-apps-own-secret"
	gatewayBearer   = "the-routers-bearer"
)

// gateway is the mint endpoint of the gateway in front of the router, signing with the
// app's secret, which only it holds.
type gateway struct {
	*httptest.Server
	clock *clock
	mu    sync.Mutex
	asked []map[string]any
}

func (s *StoredSuite) gateway() *gateway {
	g := &gateway{clock: s.clock}
	g.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer "+gatewayBearer {
			http.Error(w, "no", http.StatusUnauthorized)
			return
		}
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		body["path"] = r.URL.Path
		g.mu.Lock()
		g.asked = append(g.asked, body)
		g.mu.Unlock()
		lasts := 24 * time.Hour
		claims := jwt.MapClaims{}
		if ttl, ok := body["ttl_seconds"].(float64); ok {
			lasts = time.Duration(ttl) * time.Second
			claims["user_id"] = body["user_id"]
		}
		expires := g.clock.Now().Add(lasts)
		claims["exp"] = expires.Unix()
		_ = json.NewEncoder(w).Encode(map[string]any{
			"api_key": body["api_key"], "token": signed(claims), "expires_at": expires,
		})
	}))
	s.T().Cleanup(g.Close)
	return g
}

func (g *gateway) requests() []map[string]any {
	g.mu.Lock()
	defer g.mu.Unlock()
	return append([]map[string]any(nil), g.asked...)
}

func signed(claims jwt.MapClaims) string {
	token, err := jwt.NewWithClaims(jwt.SigningMethodHS256, claims).SignedString([]byte(forwardedSecret))
	if err != nil {
		panic(err)
	}
	return token
}

// serverToken is a server token the gateway minted, lasting the time given from now.
func (s *StoredSuite) serverToken(lasts time.Duration) string {
	return signed(jwt.MapClaims{"exp": s.clock.Now().Add(lasts).Unix()})
}

func (s *StoredSuite) forwarded(mint *gateway) *Forwarded {
	options := ForwardedOptions{Store: s.store, Sealer: s.sealer, BaseURL: s.stream.URL, Now: s.clock.Now}
	if mint != nil {
		var err error
		options.Mint, err = NewMint(mint.URL, gatewayBearer)
		s.Require().NoError(err)
	}
	forwarded, err := NewForwarded(options)
	s.Require().NoError(err)
	return forwarded
}

func (s *StoredSuite) forwardedSource(forwarded *Forwarded) *Stored {
	source, err := NewStored(StoredOptions{
		Store: s.store, Deployment: s.deployment, Forwarded: forwarded, Now: s.clock.Now,
	})
	s.Require().NoError(err)
	return source
}

func (s *StoredSuite) TestAnAppIsActedInWithTheTokenTheGatewayForwardedForIt() {
	forwarded := s.forwarded(nil)
	token := s.serverToken(24 * time.Hour)
	_, err := forwarded.Keep(s.ctx, s.customer, forwardedKey, token)
	s.Require().NoError(err)

	identity, err := s.forwardedSource(forwarded).For(s.ctx, s.customer)

	s.Require().NoError(err)
	s.Equal(s.app, identity.StreamApp)
	s.Equal(forwardedKey, identity.APIKey)
	s.Equal(token, identity.Token.Reveal())
	s.True(identity.Secret.Empty(), "the router holds none of the app's secrets")
}

func (s *StoredSuite) TestAnotherReplicaActsWithATokenOneReplicaKept() {
	token := s.serverToken(24 * time.Hour)
	_, err := s.forwarded(nil).Keep(s.ctx, s.customer, forwardedKey, token)
	s.Require().NoError(err)

	identity, err := s.forwardedSource(s.forwarded(nil)).For(s.ctx, s.customer)

	s.Require().NoError(err)
	s.Equal(token, identity.Token.Reveal())
}

func (s *StoredSuite) TestAnOlderTokenDoesNotReplaceANewerOne() {
	forwarded := s.forwarded(nil)
	newer := s.serverToken(24 * time.Hour)
	_, err := forwarded.Keep(s.ctx, s.customer, forwardedKey, newer)
	s.Require().NoError(err)

	kept, err := s.forwarded(nil).Keep(s.ctx, s.customer, forwardedKey, s.serverToken(23*time.Hour))

	s.Require().NoError(err)
	s.False(kept)
	identity, err := s.forwardedSource(s.forwarded(nil)).For(s.ctx, s.customer)
	s.Require().NoError(err)
	s.Equal(newer, identity.Token.Reveal())
}

func (s *StoredSuite) TestATokenThatNeverExpiresIsNotKept() {
	_, err := s.forwarded(nil).Keep(s.ctx, s.customer, forwardedKey, signed(jwt.MapClaims{}))

	s.Error(err)
}

func (s *StoredSuite) TestATokenAboutToExpireIsRefreshedAtTheGateway() {
	mint := s.gateway()
	forwarded := s.forwarded(mint)
	_, err := forwarded.Keep(s.ctx, s.customer, forwardedKey, s.serverToken(30*time.Minute))
	s.Require().NoError(err)

	identity, err := s.forwardedSource(forwarded).For(s.ctx, s.customer)

	s.Require().NoError(err)
	s.Equal(s.clock.Now().Add(24*time.Hour).Unix(), identity.TokenExpires.Unix())
	asked := mint.requests()
	s.Require().Len(asked, 1)
	s.Equal("/internal/v1/apps/"+s.customer+"/server-token", asked[0]["path"])
	s.Equal(forwardedKey, asked[0]["api_key"])
}

func (s *StoredSuite) TestAnExpiredTokenTheGatewayWillNotReplaceActsNowhere() {
	forwarded := s.forwarded(nil)
	_, err := forwarded.Keep(s.ctx, s.customer, forwardedKey, s.serverToken(time.Hour+time.Minute))
	s.Require().NoError(err)
	s.clock.Advance(2 * time.Hour)

	_, err = s.forwardedSource(forwarded).For(s.ctx, s.customer)

	s.ErrorIs(err, ErrNoIdentity)
}

func (s *StoredSuite) TestAnAppNoTokenWasForwardedForActsNowhere() {
	_, err := s.forwardedSource(s.forwarded(nil)).For(s.ctx, s.customer)

	s.ErrorIs(err, ErrNoIdentity)
}

// placeholder is the row the migration seeds for an app that had registered its keys: no
// token, already expired, naming the key to mint with.
func (s *StoredSuite) placeholder() {
	_, err := s.store.KeepStreamAppToken(s.ctx, store.StreamAppToken{
		StreamAppPK: s.app, CustomerID: s.customer, APIKey: forwardedKey, Sealed: []byte{}, ExpiresAt: time.Unix(0, 0),
	})
	s.Require().NoError(err)
}

func (s *StoredSuite) TestASeededPlaceholderIsRefreshedAtTheGateway() {
	s.placeholder()
	mint := s.gateway()

	identity, err := s.forwardedSource(s.forwarded(mint)).For(s.ctx, s.customer)

	s.Require().NoError(err)
	s.Equal(forwardedKey, identity.APIKey)
	s.Equal(s.clock.Now().Add(24*time.Hour).Unix(), identity.TokenExpires.Unix())
	asked := mint.requests()
	s.Require().Len(asked, 1)
	s.Equal("/internal/v1/apps/"+s.customer+"/server-token", asked[0]["path"])
	s.Equal(forwardedKey, asked[0]["api_key"])
	reread, err := s.forwardedSource(s.forwarded(nil)).For(s.ctx, s.customer)
	s.Require().NoError(err)
	s.Equal(identity.Token.Reveal(), reread.Token.Reveal(), "the minted token replaced the placeholder")
}

func (s *StoredSuite) TestASeededPlaceholderWithNoGatewayActsNowhere() {
	s.placeholder()

	_, err := s.forwardedSource(s.forwarded(nil)).For(s.ctx, s.customer)

	s.ErrorIs(err, ErrNoIdentity)
}

func (s *StoredSuite) TestWorkPinnedToAnAppIsFinishedWithItsForwardedToken() {
	forwarded := s.forwarded(nil)
	token := s.serverToken(24 * time.Hour)
	_, err := forwarded.Keep(s.ctx, s.customer, forwardedKey, token)
	s.Require().NoError(err)

	identity, err := s.forwardedSource(forwarded).ForApp(s.ctx, s.customer, s.app)

	s.Require().NoError(err)
	s.Equal(token, identity.Token.Reveal())
}

func (s *StoredSuite) TestWorkPinnedToAnotherAppIsNotFinishedWithAForwardedToken() {
	forwarded := s.forwarded(nil)
	_, err := forwarded.Keep(s.ctx, s.customer, forwardedKey, s.serverToken(24*time.Hour))
	s.Require().NoError(err)

	_, err = s.forwardedSource(forwarded).ForApp(s.ctx, s.customer, s.app+1)

	s.ErrorIs(err, ErrStreamAppMoved)
}

func (s *StoredSuite) TestAClientBuiltFromAForwardedTokenSendsIt() {
	var authorization string
	recording := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		authorization = r.Header.Get("Authorization")
		_, _ = w.Write([]byte(`{}`))
	}))
	s.T().Cleanup(recording.Close)
	forwarded, err := NewForwarded(ForwardedOptions{Store: s.store, Sealer: s.sealer, BaseURL: recording.URL, Now: s.clock.Now})
	s.Require().NoError(err)
	token := s.serverToken(24 * time.Hour)
	_, err = forwarded.Keep(s.ctx, s.customer, forwardedKey, token)
	s.Require().NoError(err)
	clients := NewClients(s.forwardedSource(forwarded), ClientsOptions{Now: s.clock.Now})

	bound, err := clients.For(s.ctx, s.customer)
	s.Require().NoError(err)
	_, _ = bound.Client.GetApp(s.ctx, nil)

	s.Equal(token, authorization)
}

func (s *StoredSuite) TestAUserTokenForAForwardedAppIsSignedAtTheGateway() {
	mint := s.gateway()
	forwarded := s.forwarded(mint)
	_, err := forwarded.Keep(s.ctx, s.customer, forwardedKey, s.serverToken(24*time.Hour))
	s.Require().NoError(err)
	clients := NewClients(s.forwardedSource(forwarded), ClientsOptions{Now: s.clock.Now})
	bound, err := clients.For(s.ctx, s.customer)
	s.Require().NoError(err)

	token, err := clients.MintUserToken(s.ctx, bound, "agent-1", time.Hour)

	s.Require().NoError(err)
	claims := jwt.MapClaims{}
	_, err = jwt.ParseWithClaims(token, claims, func(*jwt.Token) (any, error) { return []byte(forwardedSecret), nil },
		jwt.WithTimeFunc(s.clock.Now))
	s.Require().NoError(err)
	s.Equal("agent-1", claims["user_id"])
	s.Equal("/internal/v1/apps/"+s.customer+"/user-token", mint.requests()[0]["path"])
}

func (s *StoredSuite) TestAKeptTokenIsUsedForACustomerThatHadNoneAMomentAgo() {
	forwarded := s.forwarded(nil)
	clients := NewClients(s.forwardedSource(forwarded), ClientsOptions{Now: s.clock.Now})
	_, err := clients.For(s.ctx, s.customer)
	s.Require().ErrorIs(err, ErrNoIdentity)

	s.Require().NoError(clients.KeepToken(s.ctx, s.customer, forwardedKey, s.serverToken(24*time.Hour)))

	_, err = clients.For(s.ctx, s.customer)
	s.NoError(err)
}
