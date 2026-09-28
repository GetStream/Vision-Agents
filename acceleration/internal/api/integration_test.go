//go:build integration

package api

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/hibiken/asynq"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/blob"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors"
	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge"
	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge/urls"
	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/mcp"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/search"
	"github.com/GetStream/Vision-Agents/acceleration/internal/searchrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sttrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts/voices"
	"github.com/GetStream/Vision-Agents/acceleration/internal/ttsrouter"
)

type APIIntegrationSuite struct {
	suite.Suite
	ctx        context.Context
	store      *store.Store
	api        *Server
	live       *live.Client
	server     *httptest.Server
	sealer     *auth.Sealer
	customerID string
	base       time.Time
	knowledge  *base
}

func TestAPIIntegrationSuite(t *testing.T) {
	suite.Run(t, new(APIIntegrationSuite))
}

func (s *APIIntegrationSuite) SetupSuite() {
	dsn := os.Getenv("ROUTER_POSTGRES_DSN")
	address := os.Getenv("ROUTER_REDIS_ADDR")
	if dsn == "" || address == "" {
		s.T().Skip("ROUTER_POSTGRES_DSN and ROUTER_REDIS_ADDR must be set")
	}

	s.ctx = context.Background()

	pgStore, err := store.Open(dsn)
	s.Require().NoError(err)
	s.Require().NoError(pgStore.Migrate(s.ctx))
	s.store = pgStore
	sealer, err := auth.NewSealer("connector-integration-test-key")
	s.Require().NoError(err)
	s.sealer = sealer

	liveClient, err := live.New(live.Options{Address: address})
	s.Require().NoError(err)
	s.live = liveClient

	config, err := routing.DefaultConfig()
	s.Require().NoError(err)

	speech, err := sttrouter.New(sttrouter.Options{
		Config:   config[routing.STT],
		Registry: sttrouter.DefaultRegistry(),
		Store:    pgStore,
		Live:     liveClient,
	})
	s.Require().NoError(err)
	s.T().Cleanup(speech.Close)

	voice, err := ttsrouter.New(ttsrouter.Options{
		Config:   config[routing.TTS],
		Registry: ttsrouter.DefaultRegistry(),
		Store:    pgStore,
		Live:     liveClient,
	})
	s.Require().NoError(err)
	s.T().Cleanup(voice.Close)

	finding, err := searchrouter.New(searchrouter.Options{
		Config:   config[routing.Search],
		Registry: searchrouter.DefaultRegistry(),
	})
	s.Require().NoError(err)
	s.T().Cleanup(finding.Close)

	// The recording paths are wired against providers that answer in process: what a real
	// batch endpoint makes of a real file is the provider package's own suite, and what is
	// under test here is the job - a row, a result and a callback.
	transcribers := sttrouter.NewTranscriberRegistry()
	transcribers.Register("deepgram", func(spec routing.Spec) (stt.Transcriber, error) {
		return &recordedTranscriber{model: spec.Model}, nil
	})
	transcriptions, err := sttrouter.NewRecordings(sttrouter.Options{
		Config:       config[routing.STT],
		Transcribers: transcribers,
		Store:        pgStore,
		Live:         liveClient,
	})
	s.Require().NoError(err)
	s.T().Cleanup(transcriptions.Close)

	recorders := ttsrouter.NewRecorderRegistry()
	recorders.Register("elevenlabs", func(spec routing.Spec) (tts.Recorder, error) {
		return &recordedVoice{model: spec.Model}, nil
	})
	recordings, err := ttsrouter.NewRecordings(ttsrouter.Options{
		Config:    config[routing.TTS],
		Recorders: recorders,
		Store:     pgStore,
		Live:      liveClient,
	})
	s.Require().NoError(err)
	s.T().Cleanup(recordings.Close)

	s.knowledge = newBase()
	server, err := NewServer(Options{
		Routers: map[routing.Modality]routing.Inspector{
			routing.STT:    speech,
			routing.TTS:    voice,
			routing.Search: finding,
		},
		Streams: &Streams{
			STT:            speech,
			TTS:            voice,
			Transcriptions: transcriptions,
			Speech:         recordings,
		},
		Store:            pgStore,
		CredentialSealer: sealer,
		Live:             liveClient,
		PublicURL:        "https://router.test",
		DashboardURL:     "https://dashboard.test",
		Voices:           s.voiceService(pgStore),
		Knowledge:        s.knowledge,
		KnowledgeURLs:    s.knowledgeURLs(pgStore, address),
		// Minting a token signs one rather than fetching it, so a made-up app is enough
		// to exercise the join path without a real Stream account behind it.
		StreamKey:    testStreamKey,
		StreamSecret: testStreamSecret,
	})
	s.Require().NoError(err)
	s.api = server
	s.server = httptest.NewServer(server.Handler())

	s.base = time.Date(2026, 4, 1, 9, 0, 0, 0, time.UTC)
}

// voiceService wires the voice paths against a directory and a provider that always takes
// the recordings, so the HTTP surface can be exercised without cloning anything for real.
func (s *APIIntegrationSuite) voiceService(pgStore *store.Store) *voices.Service {
	provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/text-to-speech/el-cloned" {
			_, _ = w.Write([]byte("spoken"))
			return
		}
		_, _ = w.Write([]byte(`{"voice_id":"el-cloned"}`))
	}))
	s.T().Cleanup(provider.Close)

	bucket, err := blob.Open(s.ctx, "file://"+s.T().TempDir())
	s.Require().NoError(err)
	s.T().Cleanup(func() { s.Require().NoError(bucket.Close()) })

	cloner, err := voices.NewElevenLabs(voices.ElevenLabsOptions{APIKey: "secret", BaseURL: provider.URL})
	s.Require().NoError(err)
	cloners := voices.NewRegistry()
	cloners.Register("elevenlabs", cloner)

	service, err := voices.NewService(voices.Options{Store: pgStore, Bucket: bucket, Cloners: cloners})
	s.Require().NoError(err)
	return service
}

// knowledgeURLs wires the url paths against a crawler that always answers and a knowledge
// base in memory, so the HTTP surface can be exercised without fetching anything for real.
//
// Its queue is in a Redis database of its own, so a router running against the same Redis
// does not take the reads for itself.
func (s *APIIntegrationSuite) knowledgeURLs(pgStore *store.Store, address string) *urls.Service {
	service, err := urls.New(urls.Options{
		Store:         pgStore,
		Redis:         asynq.RedisClientOpt{Addr: address, DB: 14},
		Reader:        pageReader{},
		Writer:        newBase(),
		CheckInterval: 10 * time.Millisecond,
	})
	s.Require().NoError(err)
	s.Require().NoError(service.Start())
	s.T().Cleanup(func() { s.Require().NoError(service.Close()) })
	return service
}

// read waits for the worker to have read a page, and returns it as the API then describes it.
func (s *APIIntegrationSuite) read(id string, after *time.Time) KnowledgeUrl {
	var page KnowledgeUrl
	s.Require().Eventually(func() bool {
		response, payload := s.do(http.MethodGet, "/v1/agents/knowledge/urls/"+id, "")
		s.Require().Equal(http.StatusOK, response.StatusCode, string(payload))
		s.Require().NoError(json.Unmarshal(payload, &page))
		return page.LastIndexedAt != nil && (after == nil || page.LastIndexedAt.After(*after))
	}, 10*time.Second, 10*time.Millisecond)
	return page
}

// pageReader answers every url with the same page, which is enough for the endpoints to be
// exercised: what a real crawler makes of a real page is the provider's own suite.
type pageReader struct{}

func (pageReader) Read(_ context.Context, address string) (search.Page, error) {
	return search.Page{
		URL:   address,
		Title: "Pricing",
		Text:  "# Pricing\n\nA call costs a penny.\n",
	}, nil
}

// recordedTranscriber transcribes whatever it is handed into the same transcript, so the
// job path can be followed from the row it creates to the result a caller reads back.
type recordedTranscriber struct{ model string }

func (t *recordedTranscriber) Transcribe(_ context.Context, recording stt.Recording) (stt.Transcription, error) {
	transcription := stt.Transcription{
		Text:            "a call costs a penny",
		Language:        "en",
		AudioDurationMs: 4000,
	}
	if recording.Words {
		transcription.Words = []stt.Word{
			{Text: "a", StartMs: 0, EndMs: 200, Confidence: 0.9},
			{Text: "call", StartMs: 200, EndMs: 600, Confidence: 0.9},
		}
	}
	if recording.Diarize {
		transcription.Speakers = []string{"speaker_0"}
	}
	return transcription, nil
}

func (t *recordedTranscriber) Start(context.Context) error { return nil }
func (t *recordedTranscriber) Close() error                { return nil }
func (t *recordedTranscriber) Provider() string            { return "deepgram" }
func (t *recordedTranscriber) Model() string               { return t.model }

// recordedVoice speaks whatever it is handed into the same audio.
type recordedVoice struct{ model string }

func (v *recordedVoice) Record(_ context.Context, recording tts.Recording) (tts.Recorded, error) {
	format := recording.Format
	if format == "" {
		format = "mp3_44100_128"
	}
	return tts.Recorded{
		Audio:           []byte{0xff, 0xfb, 0x90},
		Format:          format,
		AudioDurationMs: 1200,
		Characters:      int64(len(recording.Text)),
	}, nil
}

func (v *recordedVoice) Start(context.Context) error { return nil }
func (v *recordedVoice) Close() error                { return nil }
func (v *recordedVoice) Provider() string            { return "elevenlabs" }
func (v *recordedVoice) Model() string               { return v.model }

func (s *APIIntegrationSuite) TearDownSuite() {
	if s.server != nil {
		s.server.Close()
	}
	if s.store != nil {
		s.Require().NoError(s.store.Close())
	}
	if s.live != nil {
		s.live.Close()
	}
}

func (s *APIIntegrationSuite) SetupTest() {
	s.customerID = "customer-" + time.Now().Format("150405.000000000")
	s.knowledge.mu.Lock()
	s.knowledge.passages = map[string]knowledge.Document{}
	s.knowledge.mu.Unlock()
}

func (s *APIIntegrationSuite) TestAnonymousConnectorCanBeActivatedWithoutAKeyring() {
	connection := store.ConnectorConnection{
		CustomerID:  s.customerID,
		ConnectorID: "custom_public_catalog",
		OwnerType:   "app",
		Endpoint:    "https://example.com/mcp",
		AuthType:    "none",
	}
	s.Require().NoError(s.store.CreateConnectorConnection(s.ctx, &connection))
	s.T().Cleanup(func() {
		s.Require().NoError(s.store.DeleteConnectorConnection(s.ctx, s.customerID, connection.ID))
	})

	configuredSealer := s.api.credentialSealer
	s.api.credentialSealer = nil
	defer func() { s.api.credentialSealer = configuredSealer }()
	response, payload := s.do(http.MethodPut,
		"/v1/agents/connections/"+connection.ID+"/credentials", `{"expected_revision":1}`)
	s.Require().Equal(http.StatusOK, response.StatusCode, string(payload))

	stored, err := s.store.ConnectorConnection(s.ctx, s.customerID, connection.ID)
	s.Require().NoError(err)
	s.Equal(store.ConnectorConnected, stored.Status)
	s.Equal("none", stored.AuthType)
	s.Empty(stored.CredentialSealed)
}

func (s *APIIntegrationSuite) TestCustomAPIKeyConnectorKeepsCredentialWriteOnlyAndUsesItAtRuntime() {
	response, payload := s.do(http.MethodPost, "/v1/agents/connectors", `{
		"id":"custom_crm","name":"Custom CRM","endpoint":"https://8.8.8.8/mcp",
		"auth_mode":"api_key","api_key_header":"X-Api-Key"
	}`)
	s.Require().Equal(http.StatusCreated, response.StatusCode, string(payload))

	response, payload = s.do(http.MethodPost, "/v1/agents/connections", `{
		"connector_id":"custom_crm","owner":{"type":"app"},"label":"Sales CRM"
	}`)
	s.Require().Equal(http.StatusCreated, response.StatusCode, string(payload))
	var connection ConnectorConnection
	s.Require().NoError(json.Unmarshal(payload, &connection))
	s.Equal(ConnectorConnectionAuthTypeApiKey, connection.AuthType)
	s.Equal(ConnectorConnectionStatusPending, connection.Status)
	s.Equal(1, connection.Revision)

	const secret = "customer-api-key-742"
	response, payload = s.do(http.MethodPut, "/v1/agents/connections/"+connection.Id+"/credentials", `{
		"expected_revision":1,"api_key":"`+secret+`"
	}`)
	s.Require().Equal(http.StatusOK, response.StatusCode, string(payload))
	s.NotContains(string(payload), secret)
	var connected ConnectorConnection
	s.Require().NoError(json.Unmarshal(payload, &connected))
	s.Equal(ConnectorConnectionStatusConnected, connected.Status)
	s.Equal(2, connected.Revision)

	stored, err := s.store.ConnectorConnection(s.ctx, s.customerID, connection.Id)
	s.Require().NoError(err)
	s.NotContains(string(stored.CredentialSealed), secret)
	request := httptest.NewRequest(http.MethodPost, stored.Endpoint, nil)
	s.Require().NoError(connectors.AuthorizeRequest(s.ctx, s.store, s.sealer, s.customerID, connection.Id, nil, request))
	s.Equal(secret, request.Header.Get("X-Api-Key"))
}

func (s *APIIntegrationSuite) TestConnectorCredentialIsRewrappedToTheCurrentKeyOnUse() {
	oldSealer, err := auth.NewSealer("old-connector-kek")
	s.Require().NoError(err)
	rotatedSealer, err := auth.NewSealerWithKeyring(2, map[int]string{
		1: "old-connector-kek",
		2: "current-connector-kek",
	})
	s.Require().NoError(err)

	connection := store.ConnectorConnection{
		CustomerID:  s.customerID,
		ConnectorID: "github",
		OwnerType:   "app",
		Endpoint:    "https://api.githubcopilot.com/mcp/",
		AuthType:    connectors.AuthOAuth2,
	}
	s.Require().NoError(s.store.CreateConnectorConnection(s.ctx, &connection))
	connection, err = s.store.ConnectorConnection(s.ctx, s.customerID, connection.ID)
	s.Require().NoError(err)
	connection.Revision = 2
	connection.CredentialKEKVersion = 1
	connection.Status = store.ConnectorConnected
	expiresAt := time.Now().UTC().Add(time.Hour)
	connection.ExpiresAt = &expiresAt
	connection.CredentialSealed, err = connectors.SealCredentials(oldSealer, s.customerID, connection.ID, connection.Revision,
		connectors.Credentials{AuthType: connectors.AuthOAuth2, AccessToken: "pre-rotation-access-token"})
	s.Require().NoError(err)
	s.Require().NoError(s.store.SaveConnectorConnectionAtRevision(s.ctx, &connection, 1))

	resolved, err := connectors.ResolveCredentials(s.ctx, s.store, rotatedSealer, s.customerID, connection.ID, nil)
	s.Require().NoError(err)
	s.Equal("pre-rotation-access-token", resolved.AccessToken)
	stored, err := s.store.ConnectorConnection(s.ctx, s.customerID, connection.ID)
	s.Require().NoError(err)
	s.Equal(2, stored.Revision, "wrapping-key rotation does not change the grant revision")
	s.Equal(2, stored.CredentialKEKVersion)
	opened, err := connectors.OpenCredentials(rotatedSealer, s.customerID, connection.ID, stored.Revision,
		stored.CredentialKEKVersion, stored.CredentialSealed)
	s.Require().NoError(err)
	s.Equal("pre-rotation-access-token", opened.AccessToken)
}

func (s *APIIntegrationSuite) TestAUserCannotReadValidateOrDisconnectAnotherUsersConnector() {
	connection := store.ConnectorConnection{
		CustomerID:       s.customerID,
		ConnectorID:      "gong",
		OwnerType:        "user",
		OwnerID:          "alice",
		Endpoint:         "https://mcp.gong.io/mcp",
		AuthType:         connectors.AuthOAuth2,
		Status:           store.ConnectorConnected,
		CredentialSealed: []byte("sealed-user-grant"),
	}
	s.Require().NoError(s.store.CreateConnectorConnection(s.ctx, &connection))
	connection, err := s.store.ConnectorConnection(s.ctx, s.customerID, connection.ID)
	s.Require().NoError(err)
	connection.Status = store.ConnectorConnected
	connection.CredentialSealed = []byte("sealed-user-grant")
	s.Require().NoError(s.store.SaveConnectorConnectionAtRevision(s.ctx, &connection, connection.Revision))

	response, payload := s.do(http.MethodGet, "/v1/agents/connections", "")
	s.Require().Equal(http.StatusOK, response.StatusCode, string(payload))
	var listed []ConnectorConnection
	s.Require().NoError(json.Unmarshal(payload, &listed))
	s.Empty(listed)

	for _, request := range []struct {
		method string
		path   string
	}{
		{method: http.MethodGet, path: "/v1/agents/connections/" + connection.ID},
		{method: http.MethodGet, path: "/v1/agents/connections/" + connection.ID + "/tools"},
		{method: http.MethodPost, path: "/v1/agents/connections/" + connection.ID + "/validate"},
		{method: http.MethodDelete, path: "/v1/agents/connections/" + connection.ID},
	} {
		response, payload = s.do(request.method, request.path, "")
		s.Equal(http.StatusNotFound, response.StatusCode, string(payload))
	}

	stored, err := s.store.ConnectorConnection(s.ctx, s.customerID, connection.ID)
	s.Require().NoError(err)
	s.Equal(store.ConnectorConnected, stored.Status)
	s.Equal([]byte("sealed-user-grant"), stored.CredentialSealed)
}

func (s *APIIntegrationSuite) TestOnlyABackendMayCreateAConnectionForItsVerifiedUser() {
	definition := store.ConnectorDefinition{
		CustomerID: s.customerID,
		ID:         "custom_identity",
		Name:       "Identity test connector",
		Endpoint:   "https://8.8.8.8/mcp",
		AuthType:   connectors.AuthNone,
	}
	s.Require().NoError(s.store.CreateConnectorDefinition(s.ctx, &definition))

	ctxFor := func(serverSide bool, userID string) context.Context {
		ctx := context.WithValue(s.ctx, customerContextKey{}, s.customerID)
		ctx = context.WithValue(ctx, serverSideContextKey{}, serverSide)
		ctx = context.WithValue(ctx, callerContextKey{}, routing.Caller{UserID: userID})
		return context.WithValue(ctx, kindContextKey{}, auth.KindServer)
	}
	create := func(ctx context.Context, ownerID string) CreateConnectorConnectionResponseObject {
		response, err := s.api.CreateConnectorConnection(ctx, CreateConnectorConnectionRequestObject{
			Body: &CreateConnectorConnectionJSONRequestBody{
				ConnectorId: definition.ID,
				Owner: ConnectorOwner{
					Type:   ConnectorOwnerTypeUser,
					UserId: &ownerID,
				},
			},
		})
		s.Require().NoError(err)
		return response
	}

	response := create(ctxFor(true, "alice"), "alice")
	created, ok := response.(CreateConnectorConnection201JSONResponse)
	s.Require().True(ok, "the backend may create a connection for its verified user")
	s.Equal(ConnectorConnectionOwnerTypeUser, created.OwnerType)
	s.Equal("alice", value(created.OwnerId))
	s.T().Cleanup(func() {
		s.Require().NoError(s.store.DeleteConnectorConnection(s.ctx, s.customerID, created.Id))
	})

	response = create(ctxFor(true, "alice"), "bob")
	denied, ok := response.(CreateConnectorConnection400JSONResponse)
	s.Require().True(ok, "a backend cannot choose another user as the connection owner")
	s.Contains(denied.BadRequestJSONResponse.Error, "must match the verified user")

	response = create(ctxFor(false, "alice"), "alice")
	denied, ok = response.(CreateConnectorConnection400JSONResponse)
	s.Require().True(ok, "a client request cannot create a user-owned connection")
	s.Contains(denied.BadRequestJSONResponse.Error, "authenticated backend")
}

func (s *APIIntegrationSuite) TestOAuthGrantImportStoresOnlyAnEncryptedProviderBoundGrant() {
	var provider *httptest.Server
	mux := http.NewServeMux()
	mux.HandleFunc("/.well-known/oauth-protected-resource/mcp", func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"resource": provider.URL + "/resource", "authorization_servers": []string{provider.URL},
		})
	})
	mux.HandleFunc("/.well-known/oauth-authorization-server", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"issuer":                                provider.URL,
			"authorization_endpoint":                provider.URL + "/authorize",
			"token_endpoint":                        provider.URL + "/token",
			"jwks_uri":                              provider.URL + "/jwks",
			"response_types_supported":              []string{"code"},
			"code_challenge_methods_supported":      []string{"S256"},
			"token_endpoint_auth_methods_supported": []string{"client_secret_basic"},
		})
	})
	mux.HandleFunc("/token", func(w http.ResponseWriter, r *http.Request) {
		clientID, clientSecret, ok := r.BasicAuth()
		if !ok || clientID != "imported-public-client" || clientSecret != "imported-client-secret" ||
			r.FormValue("grant_type") != "refresh_token" || r.FormValue("refresh_token") != "imported-refresh-token" ||
			r.FormValue("resource") != provider.URL+"/resource" {
			http.Error(w, "unexpected token refresh request", http.StatusBadRequest)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"access_token": "rotated-access-token", "refresh_token": "rotated-refresh-token", "expires_in": 3600,
		})
	})
	provider = httptest.NewServer(mux)
	defer provider.Close()

	previousHTTP := s.api.oauth.HTTP
	s.api.oauth.HTTP = provider.Client()
	s.T().Cleanup(func() { s.api.oauth.HTTP = previousHTTP })

	definition := store.ConnectorDefinition{
		CustomerID: s.customerID,
		ID:         "custom_import",
		Name:       "Import Test",
		Endpoint:   provider.URL + "/mcp",
		AuthType:   connectors.AuthOAuth2,
	}
	s.Require().NoError(s.store.CreateConnectorDefinition(s.ctx, &definition))
	connection := store.ConnectorConnection{
		CustomerID:  s.customerID,
		ConnectorID: definition.ID,
		OwnerType:   "app",
		Endpoint:    definition.Endpoint,
		AuthType:    connectors.AuthOAuth2,
	}
	s.Require().NoError(s.store.CreateConnectorConnection(s.ctx, &connection))

	const accessToken = "imported-access-token"
	const refreshToken = "imported-refresh-token"
	const clientSecret = "imported-client-secret"
	body, err := json.Marshal(map[string]any{
		"expected_revision":   1,
		"access_token":        accessToken,
		"refresh_token":       refreshToken,
		"expires_at":          time.Now().UTC().Add(-time.Minute),
		"granted_scopes":      []string{},
		"oauth_client_id":     "imported-public-client",
		"oauth_client_secret": clientSecret,
	})
	s.Require().NoError(err)
	response, payload := s.do(http.MethodPut, "/v1/agents/connections/"+connection.ID+"/credentials", string(body))
	s.Require().Equal(http.StatusOK, response.StatusCode, string(payload))
	s.NotContains(string(payload), accessToken)
	s.NotContains(string(payload), refreshToken)
	s.NotContains(string(payload), clientSecret)

	stored, err := s.store.ConnectorConnection(s.ctx, s.customerID, connection.ID)
	s.Require().NoError(err)
	s.Equal(store.ConnectorConnected, stored.Status)
	s.Equal(2, stored.Revision)
	s.NotContains(string(stored.CredentialSealed), accessToken)
	s.NotContains(string(stored.CredentialSealed), refreshToken)
	s.NotContains(string(stored.CredentialSealed), clientSecret)
	credentials, err := connectors.OpenCredentials(s.sealer, s.customerID, connection.ID, stored.Revision,
		stored.CredentialKEKVersion, stored.CredentialSealed)
	s.Require().NoError(err)
	s.Equal(accessToken, credentials.AccessToken)
	s.Equal(refreshToken, credentials.RefreshToken)
	s.Equal("imported-public-client", credentials.OAuthClientID)
	s.Equal(clientSecret, credentials.OAuthClientSecret)
	s.Equal("client_secret_basic", credentials.ClientAuthMethod)
	s.Equal(provider.URL, credentials.OAuthIssuer)
	s.Equal(provider.URL+"/resource", credentials.Resource)
	s.Equal(provider.URL+"/token", credentials.TokenEndpoint)

	request := httptest.NewRequest(http.MethodPost, stored.Endpoint, nil)
	s.Require().NoError(connectors.AuthorizeRequest(s.ctx, s.store, s.sealer, s.customerID, connection.ID, s.api.oauth, request))
	s.Equal("Bearer rotated-access-token", request.Header.Get("Authorization"))
	stored, err = s.store.ConnectorConnection(s.ctx, s.customerID, connection.ID)
	s.Require().NoError(err)
	s.Equal(3, stored.Revision)
	credentials, err = connectors.OpenCredentials(s.sealer, s.customerID, connection.ID, stored.Revision,
		stored.CredentialKEKVersion, stored.CredentialSealed)
	s.Require().NoError(err)
	s.Equal("rotated-access-token", credentials.AccessToken)
	s.Equal("rotated-refresh-token", credentials.RefreshToken)
}

func (s *APIIntegrationSuite) TestOAuthCallbackExchangesPKCEAndStoresTheGrantAfterBrowserHandoff() {
	var provider *httptest.Server
	var validRegistration atomic.Bool
	var validTokenRequest atomic.Bool
	var tokenRequests atomic.Int32
	expectedPKCEChallenge := ""
	mux := http.NewServeMux()
	mux.HandleFunc("/.well-known/oauth-protected-resource/mcp", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"resource": provider.URL + "/resource", "authorization_servers": []string{provider.URL},
		})
	})
	mux.HandleFunc("/.well-known/oauth-authorization-server", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"issuer":                                         provider.URL,
			"authorization_endpoint":                         provider.URL + "/authorize",
			"token_endpoint":                                 provider.URL + "/token",
			"registration_endpoint":                          provider.URL + "/register",
			"response_types_supported":                       []string{"code"},
			"code_challenge_methods_supported":               []string{"S256"},
			"token_endpoint_auth_methods_supported":          []string{"none"},
			"authorization_response_iss_parameter_supported": true,
		})
	})
	mux.HandleFunc("/register", func(w http.ResponseWriter, r *http.Request) {
		var request map[string]any
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			http.Error(w, "invalid registration request", http.StatusBadRequest)
			return
		}
		validRegistration.Store(r.Method == http.MethodPost && request["token_endpoint_auth_method"] == "none")
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"client_id": "connector-e2e-client", "token_endpoint_auth_method": "none",
		})
	})
	mux.HandleFunc("/token", func(w http.ResponseWriter, r *http.Request) {
		tokenRequests.Add(1)
		_ = r.ParseForm()
		verifier := r.FormValue("code_verifier")
		digest := sha256.Sum256([]byte(verifier))
		challenge := base64.RawURLEncoding.EncodeToString(digest[:])
		validTokenRequest.Store(r.Method == http.MethodPost && r.FormValue("grant_type") == "authorization_code" &&
			r.FormValue("code") == "accepted-code" && r.FormValue("client_id") == "connector-e2e-client" &&
			r.FormValue("redirect_uri") == s.server.URL+mcp.CallbackPath &&
			challenge == expectedPKCEChallenge && r.FormValue("resource") == provider.URL+"/resource")
		if !validTokenRequest.Load() {
			http.Error(w, "invalid token request", http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"access_token": "callback-access-token", "refresh_token": "callback-refresh-token",
			"expires_in": 3600, "scope": "read:workspace",
		})
	})
	provider = httptest.NewServer(mux)
	defer provider.Close()

	definition := store.ConnectorDefinition{
		CustomerID: s.customerID,
		ID:         "custom_oauth_callback",
		Name:       "OAuth callback test",
		Endpoint:   provider.URL + "/mcp",
		AuthType:   connectors.AuthOAuth2,
	}
	s.Require().NoError(s.store.CreateConnectorDefinition(s.ctx, &definition))
	connection := store.ConnectorConnection{
		CustomerID:  s.customerID,
		ConnectorID: definition.ID,
		OwnerType:   "app",
		Endpoint:    definition.Endpoint,
		AuthType:    connectors.AuthOAuth2,
	}
	s.Require().NoError(s.store.CreateConnectorConnection(s.ctx, &connection))
	connection, err := s.store.ConnectorConnection(s.ctx, s.customerID, connection.ID)
	s.Require().NoError(err)

	previousOAuth := s.api.oauth
	previousPublicURL := s.api.publicURL
	s.api.publicURL = s.server.URL
	s.api.oauth = &mcp.OAuthClient{HTTP: provider.Client(), PublicURL: s.server.URL}
	s.T().Cleanup(func() {
		s.api.oauth = previousOAuth
		s.api.publicURL = previousPublicURL
	})
	response, payload := s.do(http.MethodPost, "/v1/agents/connections/"+connection.ID+"/authorizations", `{}`)
	s.Require().Equal(http.StatusOK, response.StatusCode, string(payload))
	var authorization ConnectorAuthorization
	s.Require().NoError(json.Unmarshal(payload, &authorization))
	parsedLaunch, err := url.Parse(authorization.AuthorizationUrl)
	s.Require().NoError(err)
	s.Equal(s.server.URL, parsedLaunch.Scheme+"://"+parsedLaunch.Host)

	request, err := http.NewRequestWithContext(s.ctx, http.MethodPost, authorization.AuthorizationUrl,
		strings.NewReader(`{"handoff_token":"`+authorization.HandoffToken+`"}`))
	s.Require().NoError(err)
	request.Header.Set("Origin", s.server.URL)
	request.Header.Set("Content-Type", "application/json")
	handoffClient := &http.Client{CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
	response, err = handoffClient.Do(request)
	s.Require().NoError(err)
	var handoff struct {
		AuthorizationURL string `json:"authorization_url"`
	}
	s.Require().NoError(json.NewDecoder(response.Body).Decode(&handoff))
	response.Body.Close()
	s.Equal(http.StatusOK, response.StatusCode)
	cookies := response.Cookies()
	s.Require().Len(cookies, 1)

	authorizeURL, err := url.Parse(handoff.AuthorizationURL)
	s.Require().NoError(err)
	state := authorizeURL.Query().Get("state")
	s.NotEmpty(state)
	s.Equal("S256", authorizeURL.Query().Get("code_challenge_method"))
	s.Equal("connector-e2e-client", authorizeURL.Query().Get("client_id"))
	expectedPKCEChallenge = authorizeURL.Query().Get("code_challenge")
	s.NotEmpty(expectedPKCEChallenge)

	callbackQuery := url.Values{
		"state": {state}, "code": {"accepted-code"}, "iss": {provider.URL},
	}
	callback, err := http.NewRequestWithContext(s.ctx, http.MethodGet,
		s.server.URL+mcp.CallbackPath+"?"+callbackQuery.Encode(), nil)
	s.Require().NoError(err)
	callback.AddCookie(cookies[0])
	response, err = handoffClient.Do(callback)
	s.Require().NoError(err)
	response.Body.Close()
	s.Equal(http.StatusFound, response.StatusCode)
	s.Contains(response.Header.Get("Location"), "connection_id="+connection.ID)
	s.Contains(response.Header.Get("Location"), "status=connected")
	s.True(validRegistration.Load())
	s.True(validTokenRequest.Load(), "the authorization code must be exchanged with the PKCE verifier and expected resource")
	s.Equal(int32(1), tokenRequests.Load())

	replayedCallback, err := http.NewRequestWithContext(s.ctx, http.MethodGet,
		s.server.URL+mcp.CallbackPath+"?"+callbackQuery.Encode(), nil)
	s.Require().NoError(err)
	replayedCallback.AddCookie(cookies[0])
	replayedResponse, err := handoffClient.Do(replayedCallback)
	s.Require().NoError(err)
	replayedResponse.Body.Close()
	s.Equal(http.StatusBadRequest, replayedResponse.StatusCode, "a successful authorization state cannot be consumed twice")
	s.Equal(int32(1), tokenRequests.Load(), "replay must not redeem the provider code again")

	stored, err := s.store.ConnectorConnection(s.ctx, s.customerID, connection.ID)
	s.Require().NoError(err)
	s.Equal(store.ConnectorConnected, stored.Status)
	s.Equal(2, stored.Revision)
	s.NotContains(string(stored.CredentialSealed), "callback-access-token")
	credentials, err := connectors.OpenCredentials(s.sealer, s.customerID, stored.ID, stored.Revision,
		stored.CredentialKEKVersion, stored.CredentialSealed)
	s.Require().NoError(err)
	s.Equal("callback-access-token", credentials.AccessToken)
	s.Equal("callback-refresh-token", credentials.RefreshToken)
	s.Equal("connector-e2e-client", credentials.OAuthClientID)
	s.Equal(provider.URL, credentials.OAuthIssuer)
}

func (s *APIIntegrationSuite) TestConcurrentCredentialResolutionCommitsOneRotatedRefreshToken() {
	var refreshRequests atomic.Int32
	refreshServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseForm(); err != nil || r.Method != http.MethodPost ||
			r.FormValue("grant_type") != "refresh_token" || r.FormValue("refresh_token") != "old-refresh-token" {
			http.Error(w, "unexpected refresh request", http.StatusBadRequest)
			return
		}
		refreshRequests.Add(1)
		time.Sleep(20 * time.Millisecond)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"access_token": "rotated-access-token", "refresh_token": "rotated-refresh-token", "expires_in": 3600,
		})
	}))
	defer refreshServer.Close()

	connection := store.ConnectorConnection{
		CustomerID:  s.customerID,
		ConnectorID: "github",
		OwnerType:   "app",
		Endpoint:    "https://api.githubcopilot.com/mcp/",
		AuthType:    connectors.AuthOAuth2,
	}
	s.Require().NoError(s.store.CreateConnectorConnection(s.ctx, &connection))
	s.T().Cleanup(func() {
		s.Require().NoError(s.store.DeleteConnectorConnection(s.ctx, s.customerID, connection.ID))
	})
	connection, err := s.store.ConnectorConnection(s.ctx, s.customerID, connection.ID)
	s.Require().NoError(err)
	connection.Revision++
	connection.Status = store.ConnectorConnected
	expiredAt := time.Now().UTC().Add(-time.Minute)
	connection.ExpiresAt = &expiredAt
	connection.CredentialKEKVersion = s.sealer.CurrentVersion()
	connection.CredentialSealed, err = connectors.SealCredentials(s.sealer, s.customerID, connection.ID, connection.Revision,
		connectors.Credentials{
			AuthType:        connectors.AuthOAuth2,
			AccessToken:     "expired-access-token",
			RefreshToken:    "old-refresh-token",
			OAuthClientID:   "github-test-client",
			TokenEndpoint:   refreshServer.URL + "/token",
			RefreshEndpoint: refreshServer.URL + "/token",
		})
	s.Require().NoError(err)
	s.Require().NoError(s.store.SaveConnectorConnectionAtRevision(s.ctx, &connection, connection.Revision-1))

	const callers = 8
	type result struct {
		credentials connectors.Credentials
		err         error
	}
	start := make(chan struct{})
	results := make(chan result, callers)
	var wait sync.WaitGroup
	wait.Add(callers)
	authenticator := &mcp.OAuthClient{HTTP: refreshServer.Client()}
	for range callers {
		go func() {
			defer wait.Done()
			<-start
			credentials, err := connectors.ResolveCredentials(s.ctx, s.store, s.sealer, s.customerID, connection.ID, authenticator)
			results <- result{credentials: credentials, err: err}
		}()
	}
	close(start)
	wait.Wait()
	close(results)
	for got := range results {
		s.Require().NoError(got.err)
		s.Equal("rotated-access-token", got.credentials.AccessToken)
		s.Equal("rotated-refresh-token", got.credentials.RefreshToken)
	}
	s.Equal(int32(1), refreshRequests.Load(), "the rotating refresh token must be redeemed once")

	stored, err := s.store.ConnectorConnection(s.ctx, s.customerID, connection.ID)
	s.Require().NoError(err)
	s.Equal(connection.Revision+1, stored.Revision)
	s.Equal(store.ConnectorConnected, stored.Status)
	storedCredentials, err := connectors.OpenCredentials(s.sealer, s.customerID, stored.ID, stored.Revision,
		stored.CredentialKEKVersion, stored.CredentialSealed)
	s.Require().NoError(err)
	s.Equal("rotated-access-token", storedCredentials.AccessToken)
	s.Equal("rotated-refresh-token", storedCredentials.RefreshToken)
}

func (s *APIIntegrationSuite) TestRefreshOutcomeSurvivesLostResponsesAndCanceledWorkers() {
	for _, outcome := range []string{"lost response", "canceled worker", "temporary failure before expiry", "temporary failure after expiry"} {
		s.Run(outcome, func() {
			ctx, cancel := context.WithCancel(s.ctx)
			defer cancel()
			var requests atomic.Int32
			connection := store.ConnectorConnection{
				CustomerID: s.customerID, ConnectorID: "github", OwnerType: "app",
				Endpoint: "https://api.githubcopilot.com/mcp/", AuthType: connectors.AuthOAuth2,
			}
			s.Require().NoError(s.store.CreateConnectorConnection(s.ctx, &connection))
			s.T().Cleanup(func() {
				s.Require().NoError(s.store.DeleteConnectorConnection(s.ctx, s.customerID, connection.ID))
			})
			checkpointed := make(chan bool, 2)
			provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				requests.Add(1)
				stored, err := s.store.ConnectorConnection(s.ctx, s.customerID, connection.ID)
				checkpointed <- err == nil && stored.Status == store.ConnectorNeedsReauth
				if outcome == "canceled worker" {
					cancel()
					return
				}
				if outcome == "lost response" {
					_, _ = w.Write([]byte(`{"access_token":`))
					return
				}
				w.WriteHeader(http.StatusServiceUnavailable)
				_, _ = w.Write([]byte(`{"error":"temporarily_unavailable"}`))
			}))
			defer provider.Close()

			connection.Status = store.ConnectorConnected
			expires := time.Now().Add(30 * time.Second)
			if outcome == "temporary failure after expiry" {
				expires = time.Now().Add(-time.Minute)
			}
			connection.ExpiresAt = &expires
			credentials := connectors.Credentials{
				AuthType: connectors.AuthOAuth2, AccessToken: "old-access", RefreshToken: "old-refresh",
				OAuthClientID: "test-client", TokenEndpoint: provider.URL,
			}
			var err error
			connection.CredentialSealed, err = connectors.SealCredentials(s.sealer, s.customerID, connection.ID, connection.Revision, credentials)
			s.Require().NoError(err)
			s.Require().NoError(s.store.SaveConnectorConnectionAtRevision(s.ctx, &connection, connection.Revision))
			authenticator := &mcp.OAuthClient{HTTP: provider.Client()}
			got, err := connectors.ResolveCredentials(ctx, s.store, s.sealer, s.customerID, connection.ID, authenticator)
			s.True(<-checkpointed, "the refresh checkpoint must be durable before the provider receives the token")
			stored, readErr := s.store.ConnectorConnection(s.ctx, s.customerID, connection.ID)
			s.Require().NoError(readErr)
			if strings.HasPrefix(outcome, "temporary") {
				if outcome == "temporary failure before expiry" {
					s.NoError(err)
					s.Equal("old-access", got.AccessToken)
				} else {
					s.ErrorIs(err, connectors.ErrCredentialTemporarilyUnavailable)
				}
				s.Equal(store.ConnectorConnected, stored.Status)
				s.Contains(stored.LastError, "temporarily unavailable")
				return
			}
			s.ErrorIs(err, connectors.ErrReauthorizationRequired)
			s.Empty(got.AccessToken)
			s.Equal(store.ConnectorNeedsReauth, stored.Status)
			s.Contains(stored.LastError, "did not finish durably")
			_, err = connectors.ResolveCredentials(s.ctx, s.store, s.sealer, s.customerID, connection.ID, authenticator)
			s.ErrorIs(err, connectors.ErrReauthorizationRequired)
			s.Equal(int32(1), requests.Load(), "an uncertain rotating token must not be retried")
		})
	}
}

func (s *APIIntegrationSuite) TestOAuthDenialPreservesTheExistingGrant() {
	connection := store.ConnectorConnection{
		CustomerID:  s.customerID,
		ConnectorID: "slack",
		OwnerType:   "app",
		Endpoint:    "https://mcp.slack.com/mcp",
	}
	s.Require().NoError(s.store.CreateConnectorConnection(s.ctx, &connection))
	s.T().Cleanup(func() {
		s.Require().NoError(s.store.DeleteConnectorConnection(s.ctx, s.customerID, connection.ID))
	})
	connection.Revision++
	connection.Status = store.ConnectorConnected
	connection.CredentialKEKVersion = s.sealer.CurrentVersion()
	sealedCredentials, err := connectors.SealCredentials(s.sealer, s.customerID, connection.ID, connection.Revision,
		connectors.Credentials{
			AuthType:     connectors.AuthOAuth2,
			AccessToken:  "existing-slack-access-token",
			RefreshToken: "existing-slack-refresh-token",
		})
	s.Require().NoError(err)
	connection.CredentialSealed = sealedCredentials
	s.Require().NoError(s.store.SaveConnectorConnectionAtRevision(s.ctx, &connection, connection.Revision-1))

	const state = "oauth-state-for-denial"
	const attemptID = "oauth-denial-attempt"
	const browserBinding = "oauth-denial-browser-binding"
	sealed, err := connectors.SealAuthorizationAttempt(s.sealer, attemptID, connectors.AuthorizationAttempt{
		ConnectionID:   connection.ID,
		Revision:       connection.Revision,
		ConnectorID:    connection.ConnectorID,
		BrowserBinding: browserBinding,
		Pending:        mcp.PendingAuthorization{State: state},
	})
	s.Require().NoError(err)
	s.Require().NoError(s.store.CreateConnectorAuthorizationAttempt(s.ctx, &store.ConnectorAuthorizationAttempt{
		ID:            attemptID,
		CustomerID:    s.customerID,
		ConnectionID:  connection.ID,
		StateHash:     store.OAuthStateHash(state),
		AttemptSealed: sealed,
		KEKVersion:    s.sealer.CurrentVersion(),
		ExpiresAt:     time.Now().Add(time.Minute),
	}))

	request := httptest.NewRequest(http.MethodGet,
		mcp.CallbackPath+"?state="+state+"&error=access_denied", nil)
	request.AddCookie(&http.Cookie{Name: connectorAuthorizationCookieName(attemptID), Value: browserBinding})
	recorder := httptest.NewRecorder()
	s.api.Handler().ServeHTTP(recorder, request)

	s.Equal(http.StatusFound, recorder.Code)
	s.Contains(recorder.Header().Get("Location"), "status=failed")
	_, err = s.store.ConnectorAuthorizationAttemptByState(s.ctx, state)
	s.Error(err, "a provider denial consumes the authorization attempt")
	stored, err := s.store.ConnectorConnection(s.ctx, s.customerID, connection.ID)
	s.Require().NoError(err)
	s.Equal(store.ConnectorConnected, stored.Status)
	s.Equal(connection.Revision, stored.Revision)
	s.Equal(connection.CredentialSealed, stored.CredentialSealed)
	credentials, err := connectors.OpenCredentials(s.sealer, s.customerID, stored.ID, stored.Revision,
		stored.CredentialKEKVersion, stored.CredentialSealed)
	s.Require().NoError(err)
	s.Equal("existing-slack-access-token", credentials.AccessToken)
	s.Equal("existing-slack-refresh-token", credentials.RefreshToken)
}

func (s *APIIntegrationSuite) TestOAuthCallbackPreservesTheConnectionWhenAuthorizationSelectsAnotherAccount() {
	suffix := time.Now().UTC().Format("150405.000000000")
	teamID, userID := "T-old", "U-old"
	var tokenRequests atomic.Int32
	provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		tokenRequests.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"access_token":  "new-account-access-token",
			"refresh_token": "new-account-refresh-token",
			"team":          map[string]string{"id": teamID},
			"authed_user":   map[string]string{"id": userID},
		})
	}))
	defer provider.Close()

	previousHTTP := s.api.oauth.HTTP
	s.api.oauth.HTTP = provider.Client()
	s.T().Cleanup(func() { s.api.oauth.HTTP = previousHTTP })

	connection := store.ConnectorConnection{
		CustomerID:  s.customerID,
		ConnectorID: "slack",
		OwnerType:   "app",
		Endpoint:    "https://mcp.slack.com/mcp",
	}
	s.Require().NoError(s.store.CreateConnectorConnection(s.ctx, &connection))
	s.T().Cleanup(func() {
		s.Require().NoError(s.store.DeleteConnectorConnection(s.ctx, s.customerID, connection.ID))
	})
	createAttempt := func(state, attemptID, browserBinding string, revision int) {
		sealedAttempt, err := connectors.SealAuthorizationAttempt(s.sealer, attemptID, connectors.AuthorizationAttempt{
			ConnectionID:   connection.ID,
			Revision:       revision,
			ConnectorID:    connection.ConnectorID,
			BrowserBinding: browserBinding,
			Pending: mcp.PendingAuthorization{
				ConnectorID:      connection.ConnectorID,
				State:            state,
				ClientID:         "slack-test-client",
				ClientAuthMethod: "none",
				TokenEndpoint:    provider.URL,
			},
		})
		s.Require().NoError(err)
		s.Require().NoError(s.store.CreateConnectorAuthorizationAttempt(s.ctx, &store.ConnectorAuthorizationAttempt{
			ID:            attemptID,
			CustomerID:    s.customerID,
			ConnectionID:  connection.ID,
			StateHash:     store.OAuthStateHash(state),
			AttemptSealed: sealedAttempt,
			KEKVersion:    s.sealer.CurrentVersion(),
			ExpiresAt:     time.Now().Add(time.Minute),
		}))
	}
	finishAttempt := func(state, attemptID, browserBinding string) *httptest.ResponseRecorder {
		query := url.Values{"state": {state}, "code": {"provider-code"}}
		request := httptest.NewRequest(http.MethodGet, mcp.CallbackPath+"?"+query.Encode(), nil)
		request.AddCookie(&http.Cookie{Name: connectorAuthorizationCookieName(attemptID), Value: browserBinding})
		recorder := httptest.NewRecorder()
		s.api.Handler().ServeHTTP(recorder, request)
		return recorder
	}

	state := "oauth-state-for-first-account-" + suffix
	attemptID := "first-account-attempt-" + suffix
	browserBinding := "first-account-browser-" + suffix
	createAttempt(state, attemptID, browserBinding, connection.Revision)
	firstResponse := finishAttempt(state, attemptID, browserBinding)
	s.Equal(http.StatusFound, firstResponse.Code)
	s.Contains(firstResponse.Header().Get("Location"), "status=connected")
	stored, err := s.store.ConnectorConnection(s.ctx, s.customerID, connection.ID)
	s.Require().NoError(err)
	s.Equal(store.ConnectorConnected, stored.Status)
	s.Equal("slack:T-old:U-old", stored.AccountID, "the callback must persist stable provider identity")
	connection = stored

	teamID, userID = "T-new", "U-new"
	state = "oauth-state-for-account-switch-" + suffix
	attemptID = "account-switch-attempt-" + suffix
	browserBinding = "account-switch-browser-" + suffix
	createAttempt(state, attemptID, browserBinding, connection.Revision)
	secondResponse := finishAttempt(state, attemptID, browserBinding)
	s.Equal(http.StatusFound, secondResponse.Code)
	s.Contains(secondResponse.Header().Get("Location"), "status=failed")
	s.Equal(int32(2), tokenRequests.Load())
	stored, err = s.store.ConnectorConnection(s.ctx, s.customerID, connection.ID)
	s.Require().NoError(err)
	s.Equal(store.ConnectorConnected, stored.Status)
	s.Equal(connection.Revision, stored.Revision)
	s.Equal("slack:T-old:U-old", stored.AccountID)
	s.Contains(stored.LastError, "different or unverified provider account")
	credentials, err := connectors.OpenCredentials(s.sealer, s.customerID, stored.ID, stored.Revision,
		stored.CredentialKEKVersion, stored.CredentialSealed)
	s.Require().NoError(err)
	s.Equal("new-account-access-token", credentials.AccessToken)
	s.Equal("new-account-refresh-token", credentials.RefreshToken)
}

func (s *APIIntegrationSuite) TestOAuthCallbackRequiresTheInitiatingBrowserBeforeConsumingState() {
	connection := store.ConnectorConnection{
		CustomerID:  s.customerID,
		ConnectorID: "slack",
		OwnerType:   "app",
		Endpoint:    "https://mcp.slack.com/mcp",
	}
	s.Require().NoError(s.store.CreateConnectorConnection(s.ctx, &connection))

	const state = "oauth-state-for-browser-binding"
	const browserBinding = "browser-session-secret"
	const attemptID = "browser-binding-attempt"
	sealed, err := connectors.SealAuthorizationAttempt(s.sealer, attemptID, connectors.AuthorizationAttempt{
		ConnectionID:   connection.ID,
		Revision:       connection.Revision,
		ConnectorID:    connection.ConnectorID,
		BrowserBinding: browserBinding,
		Pending: mcp.PendingAuthorization{
			State:        state,
			CodeVerifier: "pkce-verifier",
		},
	})
	s.Require().NoError(err)
	s.Require().NoError(s.store.CreateConnectorAuthorizationAttempt(s.ctx, &store.ConnectorAuthorizationAttempt{
		ID:            attemptID,
		CustomerID:    s.customerID,
		ConnectionID:  connection.ID,
		StateHash:     store.OAuthStateHash(state),
		AttemptSealed: sealed,
		KEKVersion:    s.sealer.CurrentVersion(),
		ExpiresAt:     time.Now().Add(time.Minute),
	}))

	callbackURL := s.server.URL + mcp.CallbackPath + "?state=" + state
	request, err := http.NewRequestWithContext(s.ctx, http.MethodGet, callbackURL, nil)
	s.Require().NoError(err)
	response, err := http.DefaultClient.Do(request)
	s.Require().NoError(err)
	response.Body.Close()
	s.Equal(http.StatusForbidden, response.StatusCode)

	request, err = http.NewRequestWithContext(s.ctx, http.MethodGet, callbackURL, nil)
	s.Require().NoError(err)
	request.AddCookie(&http.Cookie{Name: connectorAuthorizationCookieName(attemptID), Value: "wrong-browser"})
	response, err = http.DefaultClient.Do(request)
	s.Require().NoError(err)
	response.Body.Close()
	s.Equal(http.StatusForbidden, response.StatusCode)

	_, err = s.store.ConnectorAuthorizationAttemptByState(s.ctx, state)
	s.Require().NoError(err, "a rejected browser must not consume the real user's one-time state")

	request, err = http.NewRequestWithContext(s.ctx, http.MethodGet, callbackURL, nil)
	s.Require().NoError(err)
	request.AddCookie(&http.Cookie{Name: connectorAuthorizationCookieName(attemptID), Value: browserBinding})
	response, err = http.DefaultClient.Do(request)
	s.Require().NoError(err)
	response.Body.Close()
	s.Equal(http.StatusBadRequest, response.StatusCode, "the valid browser reaches code validation")
	_, err = s.store.ConnectorAuthorizationAttemptByState(s.ctx, state)
	s.Error(err, "the valid browser claims the state exactly once")
}

func (s *APIIntegrationSuite) TestOAuthCallbackRejectsAnUnexpectedAuthorizationServerIssuer() {
	suffix := time.Now().Format("150405.000000000")
	state := "oauth-state-for-issuer-check-" + suffix
	attemptID := "issuer-check-attempt-" + suffix
	browserBinding := "issuer-check-browser-" + suffix
	tokenRequests := 0
	provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		tokenRequests++
		_ = json.NewEncoder(w).Encode(map[string]any{"access_token": "unexpected"})
	}))
	defer provider.Close()

	previousHTTP := s.api.oauth.HTTP
	s.api.oauth.HTTP = provider.Client()
	s.T().Cleanup(func() { s.api.oauth.HTTP = previousHTTP })
	connection := store.ConnectorConnection{
		CustomerID:  s.customerID,
		ConnectorID: "slack",
		OwnerType:   "app",
		Endpoint:    "https://mcp.slack.com/mcp",
	}
	s.Require().NoError(s.store.CreateConnectorConnection(s.ctx, &connection))
	sealed, err := connectors.SealAuthorizationAttempt(s.sealer, attemptID, connectors.AuthorizationAttempt{
		ConnectionID:   connection.ID,
		Revision:       connection.Revision,
		ConnectorID:    connection.ConnectorID,
		BrowserBinding: browserBinding,
		Pending: mcp.PendingAuthorization{
			State:         state,
			CodeVerifier:  "pkce-verifier",
			ClientID:      "oauth-client",
			Issuer:        "https://expected.example",
			RequireIssuer: true,
			TokenEndpoint: provider.URL,
		},
	})
	s.Require().NoError(err)
	s.Require().NoError(s.store.CreateConnectorAuthorizationAttempt(s.ctx, &store.ConnectorAuthorizationAttempt{
		ID:            attemptID,
		CustomerID:    s.customerID,
		ConnectionID:  connection.ID,
		StateHash:     store.OAuthStateHash(state),
		AttemptSealed: sealed,
		KEKVersion:    s.sealer.CurrentVersion(),
		ExpiresAt:     time.Now().Add(time.Minute),
	}))

	request := httptest.NewRequest(http.MethodGet, mcp.CallbackPath+"?state="+state+"&code=provider-code&iss=https%3A%2F%2Fattacker.example", nil)
	request.AddCookie(&http.Cookie{Name: connectorAuthorizationCookieName(attemptID), Value: browserBinding})
	recorder := httptest.NewRecorder()
	s.api.Handler().ServeHTTP(recorder, request)

	s.Equal(http.StatusFound, recorder.Code)
	s.Contains(recorder.Header().Get("Location"), "status=failed")
	s.Zero(tokenRequests, "the router must reject a mismatched issuer before redeeming the code")
	stored, err := s.store.ConnectorConnection(s.ctx, s.customerID, connection.ID)
	s.Require().NoError(err)
	s.Equal(store.ConnectorFailed, stored.Status)
	s.Equal("authorization server identity could not be verified", stored.LastError)
	s.Empty(stored.CredentialSealed)
}

func (s *APIIntegrationSuite) TestOAuthCallbackRejectsAnAttemptForAReplacedConnectionRevision() {
	suffix := time.Now().Format("150405.000000000")
	state := "oauth-state-for-replaced-connection-" + suffix
	attemptID := "replaced-connection-attempt-" + suffix
	browserBinding := "replaced-connection-browser-" + suffix
	tokenRequests := 0
	provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		tokenRequests++
		_ = json.NewEncoder(w).Encode(map[string]any{
			"access_token":  "stale-authorization-token",
			"refresh_token": "stale-refresh-token",
		})
	}))
	defer provider.Close()

	previousHTTP := s.api.oauth.HTTP
	s.api.oauth.HTTP = provider.Client()
	s.T().Cleanup(func() { s.api.oauth.HTTP = previousHTTP })
	connection := store.ConnectorConnection{
		CustomerID:  s.customerID,
		ConnectorID: "slack",
		OwnerType:   "app",
		Endpoint:    "https://mcp.slack.com/mcp",
	}
	s.Require().NoError(s.store.CreateConnectorConnection(s.ctx, &connection))
	sealed, err := connectors.SealAuthorizationAttempt(s.sealer, attemptID, connectors.AuthorizationAttempt{
		ConnectionID:   connection.ID,
		Revision:       connection.Revision,
		ConnectorID:    connection.ConnectorID,
		BrowserBinding: browserBinding,
		Pending: mcp.PendingAuthorization{
			State:            state,
			CodeVerifier:     "stale-pkce-verifier",
			ClientID:         "oauth-client",
			ClientAuthMethod: "none",
			TokenEndpoint:    provider.URL,
		},
	})
	s.Require().NoError(err)
	s.Require().NoError(s.store.CreateConnectorAuthorizationAttempt(s.ctx, &store.ConnectorAuthorizationAttempt{
		ID:            attemptID,
		CustomerID:    s.customerID,
		ConnectionID:  connection.ID,
		StateHash:     store.OAuthStateHash(state),
		AttemptSealed: sealed,
		KEKVersion:    s.sealer.CurrentVersion(),
		ExpiresAt:     time.Now().Add(time.Minute),
	}))

	replacement := connection
	replacement.Status = store.ConnectorConnected
	replacement.Revision++
	replacement.CredentialKEKVersion = s.sealer.CurrentVersion()
	replacement.CredentialSealed, err = connectors.SealCredentials(s.sealer, s.customerID, connection.ID,
		replacement.Revision, connectors.Credentials{
			AuthType:    connectors.AuthOAuth2,
			AccessToken: "replacement-account-token",
		})
	s.Require().NoError(err)
	s.Require().NoError(s.store.SaveConnectorConnectionAtRevision(s.ctx, &replacement, connection.Revision))

	request := httptest.NewRequest(http.MethodGet,
		mcp.CallbackPath+"?state="+state+"&code=provider-code", nil)
	request.AddCookie(&http.Cookie{Name: connectorAuthorizationCookieName(attemptID), Value: browserBinding})
	recorder := httptest.NewRecorder()
	s.api.Handler().ServeHTTP(recorder, request)

	s.Equal(http.StatusConflict, recorder.Code)
	s.Zero(tokenRequests, "a callback for the replaced revision must not redeem its code")
	_, err = s.store.ConnectorAuthorizationAttemptByState(s.ctx, state)
	s.Error(err, "the rejected callback consumes its one-time state")
	stored, err := s.store.ConnectorConnection(s.ctx, s.customerID, connection.ID)
	s.Require().NoError(err)
	s.Equal(store.ConnectorConnected, stored.Status)
	s.Equal(replacement.Revision, stored.Revision)
	credentials, err := connectors.OpenCredentials(s.sealer, s.customerID, connection.ID, stored.Revision,
		stored.CredentialKEKVersion, stored.CredentialSealed)
	s.Require().NoError(err)
	s.Equal("replacement-account-token", credentials.AccessToken)
}

func (s *APIIntegrationSuite) TestOAuthBrowserHandoffSetsTheRouterOriginCookie() {
	connection := store.ConnectorConnection{
		CustomerID:  s.customerID,
		ConnectorID: "slack",
		OwnerType:   "app",
		Endpoint:    "https://mcp.slack.com/mcp",
	}
	s.Require().NoError(s.store.CreateConnectorConnection(s.ctx, &connection))

	suffix := time.Now().Format("150405.000000000")
	state := "oauth-state-for-router-origin-handoff-" + suffix
	browserBinding := "handoff-secret-only-in-the-initiating-tab-" + suffix
	attemptID := "router-origin-handoff-" + suffix
	sealed, err := connectors.SealAuthorizationAttempt(s.sealer, attemptID, connectors.AuthorizationAttempt{
		ConnectionID:   connection.ID,
		Revision:       connection.Revision,
		ConnectorID:    connection.ConnectorID,
		BrowserBinding: browserBinding,
		Pending: mcp.PendingAuthorization{
			State:        state,
			CodeVerifier: "pkce-verifier",
			AuthorizeURL: "https://slack.example/authorize?state=" + state,
		},
	})
	s.Require().NoError(err)
	s.Require().NoError(s.store.CreateConnectorAuthorizationAttempt(s.ctx, &store.ConnectorAuthorizationAttempt{
		ID:            attemptID,
		CustomerID:    s.customerID,
		ConnectionID:  connection.ID,
		StateHash:     store.OAuthStateHash(state),
		AttemptSealed: sealed,
		KEKVersion:    s.sealer.CurrentVersion(),
		ExpiresAt:     time.Now().Add(time.Minute),
	}))

	url := s.server.URL + connectorOAuthLaunchPath + attemptID
	request, err := http.NewRequestWithContext(s.ctx, http.MethodPost, url, strings.NewReader(`{"handoff_token":"wrong"}`))
	s.Require().NoError(err)
	request.Header.Set("Origin", "https://attacker.example")
	request.Header.Set("Content-Type", "application/json")
	response, err := http.DefaultClient.Do(request)
	s.Require().NoError(err)
	response.Body.Close()
	s.Equal(http.StatusForbidden, response.StatusCode, "a different origin cannot plant the callback cookie")

	request, err = http.NewRequestWithContext(s.ctx, http.MethodPost, url, strings.NewReader(`{"handoff_token":"`+browserBinding+`"}`))
	s.Require().NoError(err)
	request.Header.Set("Origin", "https://router.test")
	request.Header.Set("Content-Type", "application/json")
	response, err = http.DefaultClient.Do(request)
	s.Require().NoError(err)
	defer response.Body.Close()
	s.Equal(http.StatusOK, response.StatusCode)
	var launchResponse struct {
		AuthorizationURL string `json:"authorization_url"`
	}
	s.Require().NoError(json.NewDecoder(response.Body).Decode(&launchResponse))
	s.Equal("https://slack.example/authorize?state="+state, launchResponse.AuthorizationURL)
	cookie := response.Cookies()
	s.Require().Len(cookie, 1)
	s.Equal(connectorAuthorizationCookieName(attemptID), cookie[0].Name)
	s.Equal(browserBinding, cookie[0].Value)
	s.Equal(mcp.CallbackPath, cookie[0].Path)
}

func (s *APIIntegrationSuite) TestOldConfigWritesCannotOverwriteConnectorBindings() {
	config := store.AgentConfig{
		CustomerID: s.customerID,
		Name:       "connector-config-" + time.Now().Format("150405.000000000"),
		Connectors: []store.ConnectorBinding{{
			Name:        "slack",
			ConnectorID: "slack",
			Connection:  store.ConnectionBinding{Type: "fixed", ConnectionID: "conn-retained"},
			Tools:       []store.ToolGrant{},
		}},
	}
	s.Require().NoError(s.store.CreateAgentConfig(s.ctx, &config))

	response, payload := s.do(http.MethodPut, "/v1/agents/configs/"+config.ID, `{"name":"connector-config"}`)
	s.Equal(http.StatusConflict, response.StatusCode, string(payload))
	s.Contains(string(payload), "include connectors when updating it")

	response, payload = s.do(http.MethodPost, "/v1/agents/sync", `{"name":"`+config.Name+`","hash":"legacy-sync-hash"}`)
	s.Equal(http.StatusConflict, response.StatusCode, string(payload))
	s.Contains(string(payload), "include connectors when syncing it")

	stored, err := s.store.AgentConfig(s.ctx, s.customerID, config.ID)
	s.Require().NoError(err)
	s.Equal(config.Connectors, stored.Connectors, "a legacy-shaped update must preserve the binding")
}

// do issues a request against the live test server with the customer header set.
func (s *APIIntegrationSuite) do(method, path, body string) (*http.Response, []byte) {
	var reader *strings.Reader
	if body == "" {
		reader = strings.NewReader("")
	} else {
		reader = strings.NewReader(body)
	}

	request, err := http.NewRequestWithContext(s.ctx, method, s.server.URL+path, reader)
	s.Require().NoError(err)
	request.Header.Set(CustomerHeader, s.customerID)
	if body != "" {
		request.Header.Set("Content-Type", "application/json")
	}

	response, err := http.DefaultClient.Do(request)
	s.Require().NoError(err)
	defer response.Body.Close()

	payload := make([]byte, 0)
	buffer := make([]byte, 4096)
	for {
		n, err := response.Body.Read(buffer)
		payload = append(payload, buffer[:n]...)
		if err != nil {
			break
		}
	}
	return response, payload
}

// recordTurn stores a completed speech-to-text turn for the current customer.
func (s *APIIntegrationSuite) recordTurn(at time.Time, audioMs int64, latencyMs float64, success bool) {
	request := &store.Request{
		Modality:   "stt",
		CustomerID: s.customerID,
		Provider:   "deepgram",
		Model:      "flux-general-en",
		StartedAt:  at,
		AudioMs:    audioMs,
		LatencyMs:  &latencyMs,
		Success:    success,
	}
	if !success {
		request.ErrorCode = "provider_fatal"
	}
	s.Require().NoError(s.store.RecordRequest(s.ctx, request))
}

// recordSynthesis stores a completed text-to-speech synthesis for the current customer.
func (s *APIIntegrationSuite) recordSynthesis(at time.Time, characters, costMicros int64) {
	latencyMs := 180.0
	s.Require().NoError(s.store.RecordRequest(s.ctx, &store.Request{
		Modality:   "tts",
		CustomerID: s.customerID,
		Provider:   "elevenlabs",
		Model:      "eleven_flash_v2_5",
		StartedAt:  at,
		AudioMs:    2500,
		Characters: characters,
		CostMicros: costMicros,
		LatencyMs:  &latencyMs,
		Success:    true,
	}))
}

// recordLabelled stores a completed LLM completion carrying the customer's own cost labels.
func (s *APIIntegrationSuite) recordLabelled(at time.Time, costMicros int64, tags map[string]string) {
	latencyMs := 320.0
	s.Require().NoError(s.store.RecordRequest(s.ctx, &store.Request{
		Modality:     "llm",
		CustomerID:   s.customerID,
		Provider:     "openai",
		Model:        "gpt-5.6-sol",
		Tags:         tags,
		StartedAt:    at,
		InputTokens:  400,
		OutputTokens: 120,
		CostMicros:   costMicros,
		LatencyMs:    &latencyMs,
		Success:      true,
	}))
}

func (s *APIIntegrationSuite) TestHealthReportsBothDependenciesAsOk() {
	response, payload := s.do(http.MethodGet, "/health", "")

	s.Equal(http.StatusOK, response.StatusCode)

	var status HealthStatus
	s.Require().NoError(json.Unmarshal(payload, &status))
	s.Equal(Ok, status.Status)
	s.Equal("ok", status.Dependencies["postgres"])
	s.Equal("ok", status.Dependencies["redis"])
	s.Equal("ok", status.Dependencies["stt"])
	s.Equal("ok", status.Dependencies["tts"])
}

func (s *APIIntegrationSuite) TestRollupThenStatsReportsTheCustomersUsage() {
	s.recordTurn(s.base.Add(5*time.Minute), 3000, 120, true)
	s.recordTurn(s.base.Add(6*time.Minute), 2000, 240, true)
	s.recordTurn(s.base.Add(7*time.Minute), 1000, 360, false)

	window := fmt.Sprintf(
		`{"granularity":"hourly","from":%q,"to":%q}`,
		s.base.Format(time.RFC3339), s.base.Add(time.Hour).Format(time.RFC3339),
	)
	response, payload := s.do(http.MethodPost, "/v1/stats/rollup", window)
	s.Require().Equal(http.StatusOK, response.StatusCode, string(payload))

	var rollup RollupResult
	s.Require().NoError(json.Unmarshal(payload, &rollup))
	s.Equal(GranularityHourly, rollup.Granularity)
	s.Positive(rollup.BucketsWritten)

	path := fmt.Sprintf(
		"/v1/stt/stats?granularity=hourly&from=%s&to=%s",
		s.base.Format(time.RFC3339), s.base.Add(time.Hour).Format(time.RFC3339),
	)
	response, payload = s.do(http.MethodGet, path, "")
	s.Require().Equal(http.StatusOK, response.StatusCode, string(payload))

	var buckets []StatsBucket
	s.Require().NoError(json.Unmarshal(payload, &buckets))
	s.Require().Len(buckets, 1)

	bucket := buckets[0]
	s.Equal("deepgram", bucket.Provider)
	s.Equal("flux-general-en", bucket.Model)
	s.EqualValues(6000, bucket.AudioMsTotal, "audio duration is what providers bill for")
	s.EqualValues(3, bucket.RequestCount)
	s.EqualValues(1, bucket.ErrorCount)
	s.Require().NotNil(bucket.LatencyP50Ms)
	s.InDelta(240.0, *bucket.LatencyP50Ms, 0.001)
	s.Require().NotNil(bucket.Uptime)
	s.InDelta(2.0/3.0, *bucket.Uptime, 0.001)
}

func (s *APIIntegrationSuite) TestStatsAreReportedPerModality() {
	s.recordTurn(s.base.Add(5*time.Minute), 3000, 100, true)
	s.recordSynthesis(s.base.Add(6*time.Minute), 128, 6400)

	window := fmt.Sprintf(
		`{"from":%q,"to":%q}`,
		s.base.Format(time.RFC3339), s.base.Add(time.Hour).Format(time.RFC3339),
	)
	response, _ := s.do(http.MethodPost, "/v1/stats/rollup", window)
	s.Require().Equal(http.StatusOK, response.StatusCode)

	from, to := s.base.Format(time.RFC3339), s.base.Add(time.Hour).Format(time.RFC3339)

	response, payload := s.do(http.MethodGet, fmt.Sprintf("/v1/stt/stats?from=%s&to=%s", from, to), "")
	s.Require().Equal(http.StatusOK, response.StatusCode)

	var transcription []StatsBucket
	s.Require().NoError(json.Unmarshal(payload, &transcription))
	s.Require().Len(transcription, 1, "the synthesis belongs to the other modality")
	s.EqualValues(3000, transcription[0].AudioMsTotal)

	response, payload = s.do(http.MethodGet, fmt.Sprintf("/v1/tts/stats?from=%s&to=%s", from, to), "")
	s.Require().Equal(http.StatusOK, response.StatusCode)

	var synthesis []StatsBucket
	s.Require().NoError(json.Unmarshal(payload, &synthesis))
	s.Require().Len(synthesis, 1)
	s.Equal("elevenlabs", synthesis[0].Provider)
	s.EqualValues(128, synthesis[0].CharactersTotal)
	s.EqualValues(6400, synthesis[0].CostMicrosTotal, "cost is aggregated alongside usage")
}

func (s *APIIntegrationSuite) TestStatsAreScopedToTheCallingCustomer() {
	s.recordTurn(s.base.Add(5*time.Minute), 3000, 100, true)

	// Another customer's traffic in the same bucket must not show up.
	other := &store.Request{
		Modality:   "stt",
		CustomerID: s.customerID + "-other",
		Provider:   "deepgram",
		Model:      "flux-general-en",
		StartedAt:  s.base.Add(5 * time.Minute),
		AudioMs:    99000,
		Success:    true,
	}
	s.Require().NoError(s.store.RecordRequest(s.ctx, other))

	window := fmt.Sprintf(
		`{"from":%q,"to":%q}`,
		s.base.Format(time.RFC3339), s.base.Add(time.Hour).Format(time.RFC3339),
	)
	response, _ := s.do(http.MethodPost, "/v1/stats/rollup", window)
	s.Require().Equal(http.StatusOK, response.StatusCode)

	path := fmt.Sprintf(
		"/v1/stt/stats?from=%s&to=%s",
		s.base.Format(time.RFC3339), s.base.Add(time.Hour).Format(time.RFC3339),
	)
	response, payload := s.do(http.MethodGet, path, "")
	s.Require().Equal(http.StatusOK, response.StatusCode)

	var buckets []StatsBucket
	s.Require().NoError(json.Unmarshal(payload, &buckets))
	s.Require().Len(buckets, 1)
	s.EqualValues(3000, buckets[0].AudioMsTotal)
}

func (s *APIIntegrationSuite) TestDailyGranularityCollapsesTheHours() {
	s.recordTurn(s.base.Add(1*time.Hour), 1000, 100, true)
	s.recordTurn(s.base.Add(6*time.Hour), 2000, 100, true)

	window := fmt.Sprintf(
		`{"granularity":"daily","from":%q,"to":%q}`,
		s.base.Format(time.RFC3339), s.base.Add(24*time.Hour).Format(time.RFC3339),
	)
	response, payload := s.do(http.MethodPost, "/v1/stats/rollup", window)
	s.Require().Equal(http.StatusOK, response.StatusCode, string(payload))

	var rollup RollupResult
	s.Require().NoError(json.Unmarshal(payload, &rollup))
	s.Equal(GranularityDaily, rollup.Granularity)

	path := fmt.Sprintf(
		"/v1/stt/stats?granularity=daily&from=%s&to=%s",
		s.base.Add(-24*time.Hour).Format(time.RFC3339), s.base.Add(24*time.Hour).Format(time.RFC3339),
	)
	response, payload = s.do(http.MethodGet, path, "")
	s.Require().Equal(http.StatusOK, response.StatusCode)

	var buckets []StatsBucket
	s.Require().NoError(json.Unmarshal(payload, &buckets))
	s.Require().Len(buckets, 1, "both hours belong to the same day")
	s.EqualValues(3000, buckets[0].AudioMsTotal)
}

func (s *APIIntegrationSuite) TestSpendCoversEveryModalityWithoutARollup() {
	s.recordTurn(s.base.Add(5*time.Minute), 3000, 120, true)
	s.recordSynthesis(s.base.Add(6*time.Minute), 128, 6400)

	from, to := s.base.Format(time.RFC3339), s.base.Add(24*time.Hour).Format(time.RFC3339)
	response, payload := s.do(http.MethodGet,
		"/v1/stats/spend?granularity=daily&from="+from+"&to="+to, "")
	s.Require().Equal(http.StatusOK, response.StatusCode, string(payload))

	var buckets []SpendBucket
	s.Require().NoError(json.Unmarshal(payload, &buckets))
	s.Require().Len(buckets, 2, "one group per modality, with no rollup having run")
	s.Equal("tts", buckets[0].Value, "the synthesis is what cost money")
	s.EqualValues(6400, buckets[0].CostMicrosTotal)
	s.Equal("stt", buckets[1].Value)
	s.EqualValues(1, buckets[1].RequestCount)
}

func (s *APIIntegrationSuite) TestSpendCanBeGroupedByACostLabel() {
	s.recordLabelled(s.base.Add(5*time.Minute), 6000, map[string]string{"product": "support"})
	s.recordLabelled(s.base.Add(6*time.Minute), 2000, map[string]string{"product": "sales"})
	s.recordSynthesis(s.base.Add(7*time.Minute), 64, 100)

	from, to := s.base.Format(time.RFC3339), s.base.Add(24*time.Hour).Format(time.RFC3339)
	response, payload := s.do(http.MethodGet,
		"/v1/stats/spend?group_by=product&granularity=daily&from="+from+"&to="+to, "")
	s.Require().Equal(http.StatusOK, response.StatusCode, string(payload))

	var buckets []SpendBucket
	s.Require().NoError(json.Unmarshal(payload, &buckets))
	s.Require().Len(buckets, 3)
	s.Equal("support", buckets[0].Value)
	s.EqualValues(6000, buckets[0].CostMicrosTotal)
	s.Equal("sales", buckets[1].Value)
	s.Equal("", buckets[2].Value, "the synthesis carries no product, and is still part of the bill")
	s.EqualValues(100, buckets[2].CostMicrosTotal)
}

func (s *APIIntegrationSuite) TestTagKeysReportWhichLabelIsWorthABreakdown() {
	s.recordLabelled(s.base.Add(5*time.Minute), 6000,
		map[string]string{"product": "support", "environment": "production"})
	s.recordLabelled(s.base.Add(6*time.Minute), 2000,
		map[string]string{"product": "sales", "environment": "production"})

	from, to := s.base.Format(time.RFC3339), s.base.Add(24*time.Hour).Format(time.RFC3339)
	response, payload := s.do(http.MethodGet, "/v1/stats/tags/keys?from="+from+"&to="+to, "")
	s.Require().Equal(http.StatusOK, response.StatusCode, string(payload))

	var keys []TagKeySummary
	s.Require().NoError(json.Unmarshal(payload, &keys))
	s.Require().Len(keys, 2)

	byKey := map[string]TagKeySummary{}
	for _, key := range keys {
		byKey[key.Key] = key
	}
	s.EqualValues(2, byKey["product"].ValueCount)
	s.InDelta(1.0, byKey["product"].Coverage, 0.001, "every request carries it")
	s.Require().Len(byKey["product"].TopValues, 2)
	s.Equal("support", byKey["product"].TopValues[0].Value, "biggest spend first")
	s.InDelta(0.75, byKey["product"].TopValues[0].Share, 0.001)
	s.EqualValues(1, byKey["environment"].ValueCount, "one value is context, not a breakdown")
}

func (s *APIIntegrationSuite) TestActivityReportsWhoUsedTheAgentsAndHowMuch() {
	session := &store.AgentSession{
		ID:         "session-" + s.customerID,
		CustomerID: s.customerID,
		AgentName:  "docs",
		UserID:     "randy",
		CallerKind: "authenticated",
		State:      store.SessionRunning,
		CreatedAt:  s.base.Add(time.Minute),
	}
	s.Require().NoError(s.store.SaveSession(s.ctx, session))
	s.Require().NoError(s.store.StartResponse(s.ctx, &store.AgentResponse{
		ID:         "response-" + s.customerID,
		SessionID:  session.ID,
		CustomerID: s.customerID,
		Said:       "how much does it cost",
		CreatedAt:  s.base.Add(2 * time.Minute),
	}))

	from, to := s.base.Format(time.RFC3339), s.base.Add(24*time.Hour).Format(time.RFC3339)
	response, payload := s.do(http.MethodGet, "/v1/stats/activity?from="+from+"&to="+to, "")
	s.Require().Equal(http.StatusOK, response.StatusCode, string(payload))

	var buckets []ActivityBucket
	s.Require().NoError(json.Unmarshal(payload, &buckets))
	s.Require().Len(buckets, 1)
	s.EqualValues(1, buckets[0].Sessions)
	s.EqualValues(1, buckets[0].Messages)
	s.EqualValues(1, buckets[0].ActiveUsers, "one person asked one thing")
	s.Zero(buckets[0].Calls, "nobody rang anybody")
}

func (s *APIIntegrationSuite) TestProvidersReportLiveHealth() {
	s.Require().NoError(s.live.RecordRequest(s.ctx, live.Usage{
		Modality: "stt", CustomerID: s.customerID,
		Provider: "deepgram", Model: "flux-general-en",
		LatencyMs: 150, AudioMs: 1000, Success: true,
	}))

	response, payload := s.do(http.MethodGet, "/v1/stt/providers", "")
	s.Require().Equal(http.StatusOK, response.StatusCode)

	var providers []Provider
	s.Require().NoError(json.Unmarshal(payload, &providers))

	var english *Provider
	for i := range providers {
		if providers[i].Model == "flux-general-en" {
			english = &providers[i]
		}
	}
	s.Require().NotNil(english)
	s.Positive(english.Health.Requests, "health should come from the live counters")
	s.True(english.Health.Available)
}

func (s *APIIntegrationSuite) TestAModelThatServedRequestsHasAShareOfThem() {
	s.Require().NoError(s.store.RecordRequest(s.ctx, &store.Request{
		Modality: "stt", CustomerID: "someone-else",
		Provider: "parakeet", Model: "parakeet-tdt-0.6b-v3",
		StartedAt: time.Now(), Success: true,
	}))

	config, err := routing.DefaultConfig()
	s.Require().NoError(err)
	speech, err := sttrouter.New(sttrouter.Options{Config: config[routing.STT], Registry: sttrouter.DefaultRegistry()})
	s.Require().NoError(err)
	s.T().Cleanup(speech.Close)
	// A server of its own, so the popularity it reports was counted after the request above.
	server, err := NewServer(Options{
		Routers: map[routing.Modality]routing.Inspector{routing.STT: speech},
		Store:   s.store,
	})
	s.Require().NoError(err)
	fresh := httptest.NewServer(server.Handler())
	s.T().Cleanup(fresh.Close)

	request, err := http.NewRequestWithContext(s.ctx, http.MethodGet, fresh.URL+"/v1/stt/providers", nil)
	s.Require().NoError(err)
	request.Header.Set(CustomerHeader, s.customerID)
	response, err := http.DefaultClient.Do(request)
	s.Require().NoError(err)
	defer response.Body.Close()
	s.Require().Equal(http.StatusOK, response.StatusCode)

	var providers []Provider
	s.Require().NoError(json.NewDecoder(response.Body).Decode(&providers))
	for _, provider := range providers {
		s.Require().NotNil(provider.UsageShare)
		s.GreaterOrEqual(*provider.UsageShare, 0.0)
		s.LessOrEqual(*provider.UsageShare, 1.0)
		if provider.Model == "parakeet-tdt-0.6b-v3" && provider.Provider == "parakeet" {
			s.Positive(*provider.UsageShare, "another customer's requests count towards popularity")
		}
	}
}

func (s *APIIntegrationSuite) TestAnAgentConfigSurvivesBeingStoredAndReadBack() {
	response, payload := s.do(http.MethodPost, "/v1/agents/configs", `{
		"name":"support","llm":"llm-fast","tts":"en-low-latency","voice":"aurora",
		"subagent":"llm-best","instructions":"be brief","skills":["think","refund"],
		"keyterms":["Vision Agents","Stream"],
		"knowledge_namespace":"handbook","sandbox":"daytona","tags":{"project":"support"}
	}`)
	s.Require().Equal(http.StatusCreated, response.StatusCode, string(payload))

	var created AgentConfig
	s.Require().NoError(json.Unmarshal(payload, &created))
	s.Require().NotEmpty(created.Id)
	s.Equal("support", created.Name)

	response, payload = s.do(http.MethodGet, "/v1/agents/configs/"+created.Id, "")
	s.Require().Equal(http.StatusOK, response.StatusCode, string(payload))

	var read AgentConfig
	s.Require().NoError(json.Unmarshal(payload, &read))
	s.Require().NotNil(read.Llm)
	s.Equal("llm-fast", *read.Llm)
	s.Require().NotNil(read.Voice)
	s.Equal("aurora", *read.Voice)
	s.Require().NotNil(read.Skills)
	s.Equal([]string{"think", "refund"}, *read.Skills)
	s.Require().NotNil(read.Keyterms)
	s.Equal([]string{"Vision Agents", "Stream"}, *read.Keyterms)
	s.Require().NotNil(read.KnowledgeNamespace)
	s.Equal("handbook", *read.KnowledgeNamespace)
	s.Require().NotNil(read.Sandbox)
	s.Equal(Daytona, *read.Sandbox)
	s.Require().NotNil(read.Tags)
	s.Equal("support", (*read.Tags)["project"])
}

func (s *APIIntegrationSuite) TestAConfigNamingASandboxNobodyRunsIsRefused() {
	response, payload := s.do(http.MethodPost, "/v1/agents/configs",
		`{"name":"support","sandbox":"docker"}`)

	s.Equal(http.StatusBadRequest, response.StatusCode, string(payload))
	s.Contains(string(payload), "docker")
}

func (s *APIIntegrationSuite) TestAConfigNamingMoreKeytermsThanAnyProviderTakesIsRefused() {
	terms := make([]string, stt.MaxKeyterms+1)
	for i := range terms {
		terms[i] = fmt.Sprintf("%q", fmt.Sprintf("term-%d", i))
	}
	body := fmt.Sprintf(`{"name":"support","keyterms":[%s]}`, strings.Join(terms, ","))

	response, payload := s.do(http.MethodPost, "/v1/agents/configs", body)

	s.Equal(http.StatusBadRequest, response.StatusCode, string(payload))
	s.Contains(string(payload), "keyterms")
}

func (s *APIIntegrationSuite) TestUpdatingAConfigReplacesWhatItWas() {
	_, payload := s.do(http.MethodPost, "/v1/agents/configs",
		`{"name":"support","llm":"llm-fast","instructions":"be brief"}`)
	var created AgentConfig
	s.Require().NoError(json.Unmarshal(payload, &created))

	response, payload := s.do(http.MethodPut, "/v1/agents/configs/"+created.Id,
		`{"name":"support","llm":"llm-best"}`)
	s.Require().Equal(http.StatusOK, response.StatusCode, string(payload))

	var updated AgentConfig
	s.Require().NoError(json.Unmarshal(payload, &updated))
	s.Equal(created.Id, updated.Id, "an update keeps the id callers already hold")
	s.Require().NotNil(updated.Llm)
	s.Equal("llm-best", *updated.Llm)
	s.Nil(updated.Instructions, "a field left out of a replacement is gone from it")
}

func (s *APIIntegrationSuite) TestADeletedConfigCannotBeUsedAgain() {
	_, payload := s.do(http.MethodPost, "/v1/agents/configs", `{"name":"support"}`)
	var created AgentConfig
	s.Require().NoError(json.Unmarshal(payload, &created))

	response, _ := s.do(http.MethodDelete, "/v1/agents/configs/"+created.Id, "")
	s.Require().Equal(http.StatusNoContent, response.StatusCode)

	response, _ = s.do(http.MethodGet, "/v1/agents/configs/"+created.Id, "")
	s.Equal(http.StatusNotFound, response.StatusCode)

	// The name is free again, which is what makes deleting one usable rather than final.
	response, payload = s.do(http.MethodPost, "/v1/agents/configs", `{"name":"support"}`)
	s.Equal(http.StatusCreated, response.StatusCode, string(payload))
}

func (s *APIIntegrationSuite) TestAnotherCustomersConfigIsNotFound() {
	_, payload := s.do(http.MethodPost, "/v1/agents/configs", `{"name":"support"}`)
	var created AgentConfig
	s.Require().NoError(json.Unmarshal(payload, &created))

	request, err := http.NewRequestWithContext(s.ctx, http.MethodGet,
		s.server.URL+"/v1/agents/configs/"+created.Id, strings.NewReader(""))
	s.Require().NoError(err)
	request.Header.Set(CustomerHeader, "somebody-else")

	response, err := http.DefaultClient.Do(request)
	s.Require().NoError(err)
	defer response.Body.Close()

	s.Equal(http.StatusNotFound, response.StatusCode)
}

func (s *APIIntegrationSuite) TestASkillIsStoredAndListed() {
	response, payload := s.do(http.MethodPost, "/v1/agents/skills", `{
		"name":"refund","description":"work out what a caller is owed",
		"instructions":"Read the order and the policy, then say what to refund.",
		"deadline_ms":20000
	}`)
	s.Require().Equal(http.StatusCreated, response.StatusCode, string(payload))

	var created Skill
	s.Require().NoError(json.Unmarshal(payload, &created))
	s.Require().NotNil(created.DeadlineMs)
	s.EqualValues(20000, *created.DeadlineMs)

	response, payload = s.do(http.MethodGet, "/v1/agents/skills", "")
	s.Require().Equal(http.StatusOK, response.StatusCode)

	var listed []Skill
	s.Require().NoError(json.Unmarshal(payload, &listed))
	s.Require().Len(listed, 1)
	s.Equal("refund", listed[0].Name)
}

func (s *APIIntegrationSuite) TestAKnowledgeUrlIsReadStoredAndListed() {
	response, payload := s.do(http.MethodPost, "/v1/agents/knowledge/urls",
		`{"namespace":"docs","url":"https://example.com/pricing"}`)
	s.Require().Equal(http.StatusCreated, response.StatusCode, string(payload))

	var created KnowledgeUrl
	s.Require().NoError(json.Unmarshal(payload, &created))
	s.Equal(KnowledgeUrlStatePending, created.State, "the page is read after the request, not during it")

	read := s.read(created.Id, nil)
	s.Equal(KnowledgeUrlStateIndexed, read.State)
	s.Equal(1, read.Passages)
	s.Require().NotNil(read.Title)
	s.Equal("Pricing", *read.Title)

	response, payload = s.do(http.MethodGet, "/v1/agents/knowledge/urls?namespace=docs", "")
	s.Require().Equal(http.StatusOK, response.StatusCode, string(payload))

	var listed []KnowledgeUrl
	s.Require().NoError(json.Unmarshal(payload, &listed))
	s.Require().Len(listed, 1)
	s.Equal("https://example.com/pricing", listed[0].Url)
}

func (s *APIIntegrationSuite) TestAPageIsNamedByWhatSubscribedToItRatherThanWhatItCallsItself() {
	response, payload := s.do(http.MethodPost, "/v1/agents/knowledge/urls",
		`{"namespace":"docs","url":"https://example.com/pricing","title":"What a call costs","description":"The page sales points at."}`)
	s.Require().Equal(http.StatusCreated, response.StatusCode, string(payload))

	var created KnowledgeUrl
	s.Require().NoError(json.Unmarshal(payload, &created))
	s.Require().NotNil(created.Title)
	s.Equal("What a call costs", *created.Title, "the crawler called it Pricing")
	s.Require().NotNil(created.Description)
	s.Equal("The page sales points at.", *created.Description)

	// The same declaration applied again is a re-read, so a caller with a file of pages
	// does not have to work out which of them the base already has.
	response, payload = s.do(http.MethodPost, "/v1/agents/knowledge/urls",
		`{"namespace":"docs","url":"https://example.com/pricing","title":"What a call costs"}`)
	s.Require().Equal(http.StatusCreated, response.StatusCode, string(payload))

	var again KnowledgeUrl
	s.Require().NoError(json.Unmarshal(payload, &again))
	s.Equal(created.Id, again.Id)
	s.Nil(again.Description)
}

func (s *APIIntegrationSuite) TestADeletedKnowledgeUrlIsNoLongerSubscribedTo() {
	_, payload := s.do(http.MethodPost, "/v1/agents/knowledge/urls",
		`{"namespace":"docs","url":"https://example.com/pricing"}`)
	var created KnowledgeUrl
	s.Require().NoError(json.Unmarshal(payload, &created))

	response, _ := s.do(http.MethodDelete, "/v1/agents/knowledge/urls/"+created.Id, "")
	s.Require().Equal(http.StatusNoContent, response.StatusCode)

	response, _ = s.do(http.MethodGet, "/v1/agents/knowledge/urls/"+created.Id, "")
	s.Equal(http.StatusNotFound, response.StatusCode)

	// The url is free again, which is what makes removing one usable rather than final.
	response, payload = s.do(http.MethodPost, "/v1/agents/knowledge/urls",
		`{"namespace":"docs","url":"https://example.com/pricing"}`)
	s.Equal(http.StatusCreated, response.StatusCode, string(payload))
}

func (s *APIIntegrationSuite) TestReadingAPageAgainMovesWhenItWasLastIndexed() {
	_, payload := s.do(http.MethodPost, "/v1/agents/knowledge/urls",
		`{"namespace":"docs","url":"https://example.com/pricing"}`)
	var created KnowledgeUrl
	s.Require().NoError(json.Unmarshal(payload, &created))
	first := s.read(created.Id, nil)

	response, payload := s.do(http.MethodPost,
		"/v1/agents/knowledge/urls/"+created.Id+"/index", "")
	s.Require().Equal(http.StatusOK, response.StatusCode, string(payload))

	reindexed := s.read(created.Id, first.LastIndexedAt)
	s.Equal(created.Id, reindexed.Id)
}

func (s *APIIntegrationSuite) TestSomethingThatIsNotAFetchablePageIsRefused() {
	response, payload := s.do(http.MethodPost, "/v1/agents/knowledge/urls",
		`{"namespace":"docs","url":"mailto:sales@example.com"}`)

	s.Require().Equal(http.StatusBadRequest, response.StatusCode)

	var failure Error
	s.Require().NoError(json.Unmarshal(payload, &failure))
	s.Contains(failure.Error, "mailto:sales@example.com")
}

func (s *APIIntegrationSuite) TestAnotherCustomersKnowledgeUrlIsNotFound() {
	_, payload := s.do(http.MethodPost, "/v1/agents/knowledge/urls",
		`{"namespace":"docs","url":"https://example.com/pricing"}`)
	var created KnowledgeUrl
	s.Require().NoError(json.Unmarshal(payload, &created))

	request, err := http.NewRequestWithContext(s.ctx, http.MethodGet,
		s.server.URL+"/v1/agents/knowledge/urls/"+created.Id, strings.NewReader(""))
	s.Require().NoError(err)
	request.Header.Set(CustomerHeader, "somebody-else")

	response, err := http.DefaultClient.Do(request)
	s.Require().NoError(err)
	defer response.Body.Close()

	s.Equal(http.StatusNotFound, response.StatusCode)
}

// documents lists the calling customer's documents in one knowledge base.
func (s *APIIntegrationSuite) documents(namespace string) []IndexedKnowledgeDocument {
	response, payload := s.do(http.MethodGet, "/v1/agents/knowledge/documents?namespace="+namespace, "")
	s.Require().Equal(http.StatusOK, response.StatusCode, string(payload))

	var listed []IndexedKnowledgeDocument
	s.Require().NoError(json.Unmarshal(payload, &listed))
	return listed
}

func (s *APIIntegrationSuite) TestAPostedDocumentIsListed() {
	response, payload := s.do(http.MethodPost, "/v1/agents/knowledge",
		`{"namespace":"`+s.customerID+`","documents":[{"source":"pricing.md","text":"# Pricing\n\nA penny.\n\n# Support\n\nA day."}]}`)
	s.Require().Equal(http.StatusOK, response.StatusCode, string(payload))

	listed := s.documents(s.customerID)
	s.Require().Len(listed, 1)
	s.Equal("pricing.md", listed[0].Source)
	s.Equal(2, listed[0].Passages)
}

func (s *APIIntegrationSuite) TestADocumentReadsBackAsItWasLastPosted() {
	for _, text := range []string{`# Pricing\n\nA penny.`, `# Pricing\n\nTwo pennies.`} {
		response, payload := s.do(http.MethodPost, "/v1/agents/knowledge",
			`{"namespace":"`+s.customerID+`","documents":[{"source":"pricing.md","text":"`+text+`"}]}`)
		s.Require().Equal(http.StatusOK, response.StatusCode, string(payload))
	}
	listed := s.documents(s.customerID)
	s.Require().Len(listed, 1)
	s.Nil(listed[0].Text, "a listing leaves the text out")

	response, payload := s.do(http.MethodGet, "/v1/agents/knowledge/documents/"+listed[0].Id, "")
	s.Require().Equal(http.StatusOK, response.StatusCode, string(payload))
	var read IndexedKnowledgeDocument
	s.Require().NoError(json.Unmarshal(payload, &read))
	s.Require().NotNil(read.Text)
	s.Equal("# Pricing\n\nTwo pennies.", *read.Text)
}

func (s *APIIntegrationSuite) TestADocumentReadsBackAsThePassagesItWasCutInto() {
	s.do(http.MethodPost, "/v1/agents/knowledge",
		`{"namespace":"`+s.customerID+`","documents":[{"source":"pricing.md","text":"# Pricing\n\nA penny.\n\n# Support\n\nA day."}]}`)
	listed := s.documents(s.customerID)
	s.Require().Len(listed, 1)

	response, payload := s.do(http.MethodGet, "/v1/agents/knowledge/documents/"+listed[0].Id+"/passages", "")
	s.Require().Equal(http.StatusOK, response.StatusCode, string(payload))
	var passages []KnowledgePassage
	s.Require().NoError(json.Unmarshal(payload, &passages))
	s.Require().Len(passages, 2)
	s.Contains(passages[0].Text, "A penny.")
	s.Contains(passages[1].Text, "A day.")

	s.customerID = "somebody-else"
	response, _ = s.do(http.MethodGet, "/v1/agents/knowledge/documents/"+listed[0].Id+"/passages", "")
	s.Equal(http.StatusNotFound, response.StatusCode)
}

func (s *APIIntegrationSuite) TestADocumentPostedShorterLeavesNoOldTailBehind() {
	namespace := s.customerID
	s.do(http.MethodPost, "/v1/agents/knowledge",
		`{"namespace":"`+namespace+`","documents":[{"source":"pricing.md","text":"# Pricing\n\nA penny.\n\n# Support\n\nA day."}]}`)
	response, payload := s.do(http.MethodPost, "/v1/agents/knowledge",
		`{"namespace":"`+namespace+`","documents":[{"source":"pricing.md","text":"# Pricing\n\nTuppence."}]}`)
	s.Require().Equal(http.StatusOK, response.StatusCode, string(payload))

	listed := s.documents(namespace)
	s.Require().Len(listed, 1, "posting the same source again is an edit")
	s.Equal(1, listed[0].Passages)

	_, passages := s.knowledge.stored()
	s.Contains(passages, "pricing.md#0")
	s.NotContains(passages, "pricing.md#1")
}

func (s *APIIntegrationSuite) TestADeletedDocumentIsNoLongerListedOrFound() {
	namespace := s.customerID
	s.do(http.MethodPost, "/v1/agents/knowledge",
		`{"namespace":"`+namespace+`","documents":[{"source":"refunds.md","text":"# Refunds\n\nThirty days."}]}`)
	listed := s.documents(namespace)
	s.Require().Len(listed, 1)

	response, _ := s.do(http.MethodDelete, "/v1/agents/knowledge/documents/"+listed[0].Id, "")
	s.Require().Equal(http.StatusNoContent, response.StatusCode)

	s.Empty(s.documents(namespace))
	_, passages := s.knowledge.stored()
	s.NotContains(passages, "refunds.md#0")

	response, _ = s.do(http.MethodDelete, "/v1/agents/knowledge/documents/"+listed[0].Id, "")
	s.Equal(http.StatusNotFound, response.StatusCode)
}

func (s *APIIntegrationSuite) TestAnotherCustomersDocumentCannotBeDeleted() {
	s.do(http.MethodPost, "/v1/agents/knowledge",
		`{"namespace":"docs","documents":[{"source":"refunds.md","text":"# Refunds\n\nThirty days."}]}`)
	listed := s.documents("docs")
	s.Require().Len(listed, 1)

	s.customerID = "somebody-else"
	response, _ := s.do(http.MethodDelete, "/v1/agents/knowledge/documents/"+listed[0].Id, "")

	s.Equal(http.StatusNotFound, response.StatusCode)
}

func (s *APIIntegrationSuite) TestASyncedDirectoryListsItsFilesAndForgetsTheOnesTakenOut() {
	response, payload := s.do(http.MethodPost, "/v1/agents/sync", `{
		"name":"librarian","hash":"v1",
		"knowledge":[
			{"source":"pricing.md","text":"# Pricing\n\nA penny."},
			{"source":"refunds.md","text":"# Refunds\n\nThirty days."}
		]
	}`)
	s.Require().Equal(http.StatusOK, response.StatusCode, string(payload))

	var synced SyncAgentResult
	s.Require().NoError(json.Unmarshal(payload, &synced))
	s.Require().NotNil(synced.Config.KnowledgeNamespace)
	s.Len(s.documents(*synced.Config.KnowledgeNamespace), 2)

	response, payload = s.do(http.MethodPost, "/v1/agents/sync", `{
		"name":"librarian","hash":"v2",
		"knowledge":[{"source":"pricing.md","text":"# Pricing\n\nA penny."}]
	}`)
	s.Require().Equal(http.StatusOK, response.StatusCode, string(payload))

	listed := s.documents("librarian")
	s.Require().Len(listed, 1)
	s.Equal("pricing.md", listed[0].Source)

	response, payload = s.do(http.MethodPost, "/v1/agents/sync", `{"name":"librarian","hash":"v3"}`)
	s.Require().Equal(http.StatusOK, response.StatusCode, string(payload))
	s.Empty(s.documents("librarian"), "a directory with no knowledge left holds none")
}

func (s *APIIntegrationSuite) TestASyncedDirectorysPagesAreReadIntoItsKnowledge() {
	response, payload := s.do(http.MethodPost, "/v1/agents/sync", `{
		"name":"librarian","hash":"v1",
		"knowledge_urls":[{"url":"https://example.com/pricing","title":"What a call costs"}]
	}`)
	s.Require().Equal(http.StatusOK, response.StatusCode, string(payload))

	var synced SyncAgentResult
	s.Require().NoError(json.Unmarshal(payload, &synced))
	s.Require().NotNil(synced.Config.KnowledgeNamespace, "pages alone are still a knowledge base")
	s.Equal("librarian", *synced.Config.KnowledgeNamespace)

	response, payload = s.do(http.MethodGet, "/v1/agents/knowledge/urls?namespace=librarian", "")
	s.Require().Equal(http.StatusOK, response.StatusCode, string(payload))

	var listed []KnowledgeUrl
	s.Require().NoError(json.Unmarshal(payload, &listed))
	s.Require().Len(listed, 1)
	s.Equal("https://example.com/pricing", listed[0].Url)
	s.Require().NotNil(listed[0].Title)
	s.Equal("What a call costs", *listed[0].Title)
}

func (s *APIIntegrationSuite) TestAConfigRemembersWhichSearchItRoutesTo() {
	response, payload := s.do(http.MethodPost, "/v1/agents/configs",
		`{"name":"support","search":"en-high-accuracy"}`)
	s.Require().Equal(http.StatusCreated, response.StatusCode, string(payload))

	var created AgentConfig
	s.Require().NoError(json.Unmarshal(payload, &created))
	s.Require().NotNil(created.Search)
	s.Equal("en-high-accuracy", *created.Search)
}

func (s *APIIntegrationSuite) TestARouterConfigSurvivesBeingStoredAndReadBack() {
	response, payload := s.do(http.MethodPost, "/v1/router/configs", `{
		"name":"healthcare",
		"stt":{"target":"en-recorded","providers":["deepgram/nova-3","en-recorded"],"diarize":true,"keyterms":["perioperative"]},
		"tts":{"target":"en-low-latency","voice":"aurora","speed":1.1},
		"llm":{"providers":["gemini/gemini-3.8-flash","llm-fast"],"temperature":0.2},
		"search":{"providers":["tavily/advanced","exa"],"depth":"standard","include_domains":["nice.org.uk"]}
	}`)
	s.Require().Equal(http.StatusCreated, response.StatusCode, string(payload))

	var created RouterConfig
	s.Require().NoError(json.Unmarshal(payload, &created))
	s.Require().NotEmpty(created.Id)

	response, payload = s.do(http.MethodGet, "/v1/router/configs/"+created.Id, "")
	s.Require().Equal(http.StatusOK, response.StatusCode, string(payload))

	var read RouterConfig
	s.Require().NoError(json.Unmarshal(payload, &read))
	s.Require().NotNil(read.Stt)
	s.Equal("en-recorded", *read.Stt.Target)
	s.Require().NotNil(read.Stt.Providers)
	s.Equal([]string{"deepgram/nova-3", "en-recorded"}, *read.Stt.Providers,
		"a priority list is the order it was written in, which is the point of writing one")
	s.Require().NotNil(read.Stt.Keyterms)
	s.Equal([]string{"perioperative"}, *read.Stt.Keyterms)
	s.Require().NotNil(read.Tts)
	s.InDelta(1.1, *read.Tts.Speed, 0.001)
	s.Require().NotNil(read.Llm)
	s.InDelta(0.2, *read.Llm.Temperature, 0.001)
	s.Require().NotNil(read.Llm.Providers)
	s.Equal([]string{"gemini/gemini-3.8-flash", "llm-fast"}, *read.Llm.Providers)
	s.Require().NotNil(read.Search)
	s.Equal([]string{"nice.org.uk"}, *read.Search.IncludeDomains)
	s.Require().NotNil(read.Search.Providers)
	s.Equal([]string{"tavily/advanced", "exa"}, *read.Search.Providers)
}

func (s *APIIntegrationSuite) TestAConfigFallingBackToASearchNobodyOffersIsRefused() {
	response, payload := s.do(http.MethodPost, "/v1/router/configs",
		`{"name":"clinic","search":{"providers":["exa","altavista"]}}`)

	s.Require().Equal(http.StatusBadRequest, response.StatusCode, string(payload))

	var failure Error
	s.Require().NoError(json.Unmarshal(payload, &failure))
	s.Contains(failure.Error, "altavista")
}

func (s *APIIntegrationSuite) TestARouterConfigIsFoundByNameAsWellAsById() {
	// Naming a config is what a caller writes in their own code, so the id they never saw
	// cannot be the only way back to it.
	_, payload := s.do(http.MethodPost, "/v1/router/configs",
		`{"name":"clinic","stt":{"target":"en-recorded"}}`)
	var created RouterConfig
	s.Require().NoError(json.Unmarshal(payload, &created))

	response, payload := s.do(http.MethodPost, "/v1/stt/recordings",
		`{"config_id":"clinic","source":{"url":"https://example.test/call.mp3"}}`)
	s.Require().Equal(http.StatusAccepted, response.StatusCode, string(payload))
}

func (s *APIIntegrationSuite) TestUpdatingARouterConfigReplacesWhatItWas() {
	_, payload := s.do(http.MethodPost, "/v1/router/configs",
		`{"name":"clinic","stt":{"target":"en-recorded","diarize":true}}`)
	var created RouterConfig
	s.Require().NoError(json.Unmarshal(payload, &created))

	response, payload := s.do(http.MethodPut, "/v1/router/configs/"+created.Id,
		`{"name":"clinic","stt":{"target":"multilingual-recorded"}}`)
	s.Require().Equal(http.StatusOK, response.StatusCode, string(payload))

	var updated RouterConfig
	s.Require().NoError(json.Unmarshal(payload, &updated))
	s.Equal(created.Id, updated.Id, "an update keeps the id callers already hold")
	s.Equal("multilingual-recorded", *updated.Stt.Target)
	s.Nil(updated.Stt.Diarize, "a field left out of a replacement is gone from it")
}

func (s *APIIntegrationSuite) TestAConfigNamingAVoiceThisDeploymentHasNeverHeardOfIsRefused() {
	// Storing it would leave a config that fails every call made under it, which is worth
	// hearing about while it is being written rather than once a socket is open.
	response, payload := s.do(http.MethodPost, "/v1/router/configs",
		`{"name":"clinic","tts":{"providers":["vocalizer"]}}`)

	s.Require().Equal(http.StatusBadRequest, response.StatusCode, string(payload))

	var failure Error
	s.Require().NoError(json.Unmarshal(payload, &failure))
	s.Contains(failure.Error, "vocalizer")
}

func (s *APIIntegrationSuite) TestAConfigWithOverwritesForAVoiceNobodyOffersIsRefused() {
	response, payload := s.do(http.MethodPost, "/v1/router/configs",
		`{"name":"clinic","tts":{"overwrites":{"vocalizer":{"voice_id":"v-1"}}}}`)

	s.Require().Equal(http.StatusBadRequest, response.StatusCode, string(payload))

	var failure Error
	s.Require().NoError(json.Unmarshal(payload, &failure))
	s.Contains(failure.Error, "no voice for")
}

func (s *APIIntegrationSuite) TestAConfigAskingAVoiceForARetentionNothingCanBeComparedAgainstIsRefused() {
	response, payload := s.do(http.MethodPost, "/v1/router/configs",
		`{"name":"clinic","tts":{"data_policy":{"retention":"ages"}}}`)

	s.Require().Equal(http.StatusBadRequest, response.StatusCode, string(payload))

	var failure Error
	s.Require().NoError(json.Unmarshal(payload, &failure))
	s.Contains(failure.Error, "ages")
}

func (s *APIIntegrationSuite) TestAConfigHoldingASystemPromptForAConversationIsRefused() {
	// The agent that holds the conversation has instructions of its own and sends them
	// when it opens the session. A config that also carried some would overwrite them
	// from somewhere nobody thought to look.
	response, payload := s.do(http.MethodPost, "/v1/router/configs",
		`{"name":"clinic","sts":{"target":"sts-fast","instructions":"Be brief."}}`)

	s.Require().Equal(http.StatusBadRequest, response.StatusCode, string(payload))

	var failure Error
	s.Require().NoError(json.Unmarshal(payload, &failure))
	s.Contains(failure.Error, "instructions")
}

func (s *APIIntegrationSuite) TestARouterConfigNobodyHasIsRefusedRatherThanIgnored() {
	// A caller that named a config meant it: transcribing at whatever the fallback happens
	// to be is not what they asked for.
	response, payload := s.do(http.MethodPost, "/v1/stt/recordings",
		`{"config_id":"nope","source":{"url":"https://example.test/call.mp3"}}`)

	s.Require().Equal(http.StatusBadRequest, response.StatusCode)

	var failure Error
	s.Require().NoError(json.Unmarshal(payload, &failure))
	s.Contains(failure.Error, "no such router config")
}

func (s *APIIntegrationSuite) TestARecordingIsTranscribedAndReadBack() {
	response, payload := s.do(http.MethodPost, "/v1/stt/recordings", `{
		"source":{"url":"https://example.test/call.mp3"},
		"options":{"diarize":true,"words":true,"language":"en"}
	}`)
	s.Require().Equal(http.StatusAccepted, response.StatusCode, string(payload))

	var accepted Transcription
	s.Require().NoError(json.Unmarshal(payload, &accepted))
	s.Require().NotEmpty(accepted.Id)
	s.Equal(RecordingStatusQueued, accepted.Status, "a job is a row before it is work")

	finished := s.finishedTranscription(accepted.Id)
	s.Require().NotNil(finished.Text)
	s.Equal("a call costs a penny", *finished.Text)
	s.Require().NotNil(finished.Words)
	s.Len(*finished.Words, 2, "word timings were asked for")
	s.Require().NotNil(finished.Speakers)
	s.Equal([]string{"speaker_0"}, *finished.Speakers)
	s.Require().NotNil(finished.Provider)
	s.Equal("deepgram", *finished.Provider)
}

func (s *APIIntegrationSuite) TestSubtitlesAreRenderedFromTheTimingsWhoeverServedThem() {
	// Vendors offer subtitles inconsistently, and every one of them returns what it takes
	// to render them, so a caller asking for srt gets it from whichever one answered.
	response, payload := s.do(http.MethodPost, "/v1/stt/recordings", `{
		"source":{"url":"https://example.test/call.mp3"},
		"options":{"output":"srt"}
	}`)
	s.Require().Equal(http.StatusAccepted, response.StatusCode, string(payload))

	var accepted Transcription
	s.Require().NoError(json.Unmarshal(payload, &accepted))

	finished := s.finishedTranscription(accepted.Id)
	s.Require().NotNil(finished.Subtitles)
	s.Contains(*finished.Subtitles, "00:00:00,000 --> 00:00:00,600")
	s.Contains(*finished.Subtitles, "a call")
}

func (s *APIIntegrationSuite) TestARecordingWithNothingToTranscribeIsRefused() {
	response, payload := s.do(http.MethodPost, "/v1/stt/recordings", `{"source":{}}`)

	s.Require().Equal(http.StatusBadRequest, response.StatusCode)

	var failure Error
	s.Require().NoError(json.Unmarshal(payload, &failure))
	s.Contains(failure.Error, "url or the audio")
}

func (s *APIIntegrationSuite) TestAnOutputFormatNobodyRendersIsRefusedBeforeTheJobRuns() {
	response, payload := s.do(http.MethodPost, "/v1/stt/recordings", `{
		"source":{"url":"https://example.test/call.mp3"},
		"options":{"output":"ass"}
	}`)

	s.Require().Equal(http.StatusBadRequest, response.StatusCode, string(payload))
	s.Contains(string(payload), "srt or vtt")
}

func (s *APIIntegrationSuite) TestATextIsSpokenIntoOneFileAndReadBack() {
	response, payload := s.do(http.MethodPost, "/v1/tts/recordings", `{
		"text":"Chapter one. A call costs a penny.",
		"options":{"format":"mp3_44100_128"}
	}`)
	s.Require().Equal(http.StatusAccepted, response.StatusCode, string(payload))

	var accepted Speech
	s.Require().NoError(json.Unmarshal(payload, &accepted))
	s.Require().NotEmpty(accepted.Id)

	var finished Speech
	s.Require().Eventually(func() bool {
		_, payload := s.do(http.MethodGet, "/v1/tts/recordings/"+accepted.Id, "")
		s.Require().NoError(json.Unmarshal(payload, &finished))
		return finished.Status != RecordingStatusQueued && finished.Status != RecordingStatusRunning
	}, 10*time.Second, 25*time.Millisecond, "the job never finished")

	s.Require().Equal(RecordingStatusCompleted, finished.Status, value(finished.Error))
	s.Require().NotNil(finished.Audio)
	s.NotEmpty(*finished.Audio)
	s.Require().NotNil(finished.Format)
	s.Equal("mp3_44100_128", *finished.Format)
	s.Require().NotNil(finished.Characters)
	s.EqualValues(34, *finished.Characters)
}

func (s *APIIntegrationSuite) TestASpeechJobWithNothingToSayIsRefused() {
	response, payload := s.do(http.MethodPost, "/v1/tts/recordings", `{"text":"   "}`)

	s.Require().Equal(http.StatusBadRequest, response.StatusCode)
	s.Contains(string(payload), "nothing to say")
}

func (s *APIIntegrationSuite) TestAFinishedRecordingCallsBackWhoeverAskedToBeTold() {
	told := make(chan Transcription, 1)
	listener := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
		var finished Transcription
		s.Require().NoError(json.NewDecoder(r.Body).Decode(&finished))
		told <- finished
	}))
	defer listener.Close()

	body := fmt.Sprintf(`{"source":{"url":"https://example.test/call.mp3"},"callback":%q}`,
		listener.URL+"/done")
	response, payload := s.do(http.MethodPost, "/v1/stt/recordings", body)
	s.Require().Equal(http.StatusAccepted, response.StatusCode, string(payload))

	select {
	case finished := <-told:
		s.Equal(RecordingStatusCompleted, finished.Status)
		s.Require().NotNil(finished.Text)
		s.Equal("a call costs a penny", *finished.Text)
	case <-time.After(10 * time.Second):
		s.Fail("nobody was told the job had finished")
	}
}

func (s *APIIntegrationSuite) TestAnotherCustomersRecordingIsNotFound() {
	_, payload := s.do(http.MethodPost, "/v1/stt/recordings",
		`{"source":{"url":"https://example.test/call.mp3"}}`)
	var accepted Transcription
	s.Require().NoError(json.Unmarshal(payload, &accepted))

	request, err := http.NewRequestWithContext(s.ctx, http.MethodGet,
		s.server.URL+"/v1/stt/recordings/"+accepted.Id, strings.NewReader(""))
	s.Require().NoError(err)
	request.Header.Set(CustomerHeader, "somebody-else")

	response, err := http.DefaultClient.Do(request)
	s.Require().NoError(err)
	defer response.Body.Close()

	s.Equal(http.StatusNotFound, response.StatusCode)
}

// finishedTranscription polls a job until it is neither queued nor running.
func (s *APIIntegrationSuite) finishedTranscription(id string) Transcription {
	var finished Transcription
	s.Require().Eventually(func() bool {
		_, payload := s.do(http.MethodGet, "/v1/stt/recordings/"+id, "")
		s.Require().NoError(json.Unmarshal(payload, &finished))
		return finished.Status != RecordingStatusQueued && finished.Status != RecordingStatusRunning
	}, 10*time.Second, 25*time.Millisecond, "the job never finished")

	s.Require().Equal(RecordingStatusCompleted, finished.Status, value(finished.Error))
	return finished
}

func (s *APIIntegrationSuite) TestASkillWithoutADescriptionIsRefused() {
	// The description is the whole of how the fast model decides when to hand work over,
	// so a skill without one would never be reached for.
	response, payload := s.do(http.MethodPost, "/v1/agents/skills",
		`{"name":"refund","description":"","instructions":"work it out"}`)

	s.Require().Equal(http.StatusBadRequest, response.StatusCode)

	var failure Error
	s.Require().NoError(json.Unmarshal(payload, &failure))
	s.Contains(failure.Error, "description")
}

func (s *APIIntegrationSuite) TestSyncingAnAgentStoresItsInstructionsAndSkills() {
	response, payload := s.do(http.MethodPost, "/v1/agents/sync", `{
		"name":"support","hash":"v1",
		"instructions":"Be brief.",
		"skills":[{"name":"refund","description":"work out a refund","instructions":"Read the policy."}]
	}`)
	s.Require().Equal(http.StatusOK, response.StatusCode, string(payload))

	var first SyncAgentResult
	s.Require().NoError(json.Unmarshal(payload, &first))
	s.False(first.Unchanged)
	s.Equal("support", first.Config.Name)
	s.Require().NotNil(first.Config.Instructions)
	s.Equal("Be brief.", *first.Config.Instructions)
	s.Require().NotNil(first.Config.Skills)
	s.Equal([]string{"refund"}, *first.Config.Skills)
	s.Require().NotNil(first.Config.SyncHash)
	s.Equal("v1", *first.Config.SyncHash)

	again, payload := s.do(http.MethodPost, "/v1/agents/sync", `{
		"name":"support","hash":"v1",
		"instructions":"Be brief.",
		"skills":[{"name":"refund","description":"work out a refund","instructions":"Read the policy."}]
	}`)
	s.Require().Equal(http.StatusOK, again.StatusCode, string(payload))

	var second SyncAgentResult
	s.Require().NoError(json.Unmarshal(payload, &second))
	s.True(second.Unchanged, "the same hash means nothing was written")
	s.Equal(first.Config.Id, second.Config.Id)

	changed, payload := s.do(http.MethodPost, "/v1/agents/sync", `{
		"name":"support","hash":"v2",
		"instructions":"Be even briefer."
	}`)
	s.Require().Equal(http.StatusOK, changed.StatusCode, string(payload))

	var third SyncAgentResult
	s.Require().NoError(json.Unmarshal(payload, &third))
	s.False(third.Unchanged)
	s.Equal(first.Config.Id, third.Config.Id)
	s.Require().NotNil(third.Config.Instructions)
	s.Equal("Be even briefer.", *third.Config.Instructions)
}

func (s *APIIntegrationSuite) TestSyncingAnAgentStoresWhatItsDeclarationRunsItOn() {
	response, payload := s.do(http.MethodPost, "/v1/agents/sync", `{
		"name":"analyst","hash":"v1","mode":"text",
		"llm":"llm-fast","subagent":"llm-best","tts":"en-low-latency","voice":"aurora",
		"greeting":"Hello.","keyterms":["Vision Agents"],"sandbox":"daytona",
		"tags":{"project":"analyst"}
	}`)
	s.Require().Equal(http.StatusOK, response.StatusCode, string(payload))

	var result SyncAgentResult
	s.Require().NoError(json.Unmarshal(payload, &result))
	s.Equal(AgentModeText, result.Config.Mode)
	s.Require().NotNil(result.Config.Llm)
	s.Equal("llm-fast", *result.Config.Llm)
	s.Require().NotNil(result.Config.Subagent)
	s.Equal("llm-best", *result.Config.Subagent)
	s.Require().NotNil(result.Config.Voice)
	s.Equal("aurora", *result.Config.Voice)
	s.Require().NotNil(result.Config.Greeting)
	s.Equal("Hello.", *result.Config.Greeting)
	s.Require().NotNil(result.Config.Keyterms)
	s.Equal([]string{"Vision Agents"}, *result.Config.Keyterms)
	s.Require().NotNil(result.Config.Sandbox)
	s.Equal(Daytona, *result.Config.Sandbox)
	s.Require().NotNil(result.Config.Tags)
	s.Equal("analyst", (*result.Config.Tags)["project"])
}

func (s *APIIntegrationSuite) TestASyncThatNamesNoModelLeavesTheOneStored() {
	_, payload := s.do(http.MethodPost, "/v1/agents/sync",
		`{"name":"switchboard","hash":"v1","llm":"llm-fast","sandbox":"daytona"}`)
	var first SyncAgentResult
	s.Require().NoError(json.Unmarshal(payload, &first))

	// A directory that says nothing about a model should not blank the one the dashboard
	// chose, so only what the declaration named is written.
	response, payload := s.do(http.MethodPost, "/v1/agents/sync",
		`{"name":"switchboard","hash":"v2","instructions":"Be brief."}`)
	s.Require().Equal(http.StatusOK, response.StatusCode, string(payload))

	var second SyncAgentResult
	s.Require().NoError(json.Unmarshal(payload, &second))
	s.Equal(first.Config.Id, second.Config.Id)
	s.Require().NotNil(second.Config.Llm)
	s.Equal("llm-fast", *second.Config.Llm)
	s.Require().NotNil(second.Config.Sandbox)
	s.Equal(Daytona, *second.Config.Sandbox)
}

func (s *APIIntegrationSuite) TestASyncNamingASandboxNobodyRunsIsRefused() {
	response, payload := s.do(http.MethodPost, "/v1/agents/sync",
		`{"name":"analyst","hash":"v1","sandbox":"docker"}`)

	s.Equal(http.StatusBadRequest, response.StatusCode, string(payload))
	s.Contains(string(payload), "docker")
}

func (s *APIIntegrationSuite) TestASyncNamingAModeNobodyRunsIsRefused() {
	response, payload := s.do(http.MethodPost, "/v1/agents/sync",
		`{"name":"analyst","hash":"v1","mode":"txt"}`)

	s.Equal(http.StatusBadRequest, response.StatusCode, string(payload))
	s.Contains(string(payload), "text")
}

// campaign creates a campaign over a stored config and returns it.
func (s *APIIntegrationSuite) campaign(concurrency int) Campaign {
	_, payload := s.do(http.MethodPost, "/v1/agents/configs", `{"name":"winback","llm":"llm-fast"}`)
	var config AgentConfig
	s.Require().NoError(json.Unmarshal(payload, &config))

	body := fmt.Sprintf(
		`{"name":"may","config_id":%q,"from_number":"+15550100","concurrency":%d}`,
		config.Id, concurrency)
	response, payload := s.do(http.MethodPost, "/v1/agents/campaigns", body)
	s.Require().Equal(http.StatusCreated, response.StatusCode, string(payload))

	var created Campaign
	s.Require().NoError(json.Unmarshal(payload, &created))
	return created
}

func (s *APIIntegrationSuite) TestACampaignIsCreatedStoppedWithNobodyToRing() {
	created := s.campaign(3)

	s.Equal(CampaignStateDraft, created.State, "a campaign that started itself would ring people nobody added yet")
	s.Equal(3, created.Concurrency)

	response, payload := s.do(http.MethodGet, "/v1/agents/campaigns/"+created.Id+"/contacts", "")
	s.Require().Equal(http.StatusOK, response.StatusCode, string(payload))

	var contacts []Contact
	s.Require().NoError(json.Unmarshal(payload, &contacts))
	s.Empty(contacts)
}

func (s *APIIntegrationSuite) TestACampaignNamingAConfigNobodyHasIsRefused() {
	// A campaign that named a config it could not use would fail one call at a time, at
	// whatever hour somebody started it.
	response, payload := s.do(http.MethodPost, "/v1/agents/campaigns",
		`{"name":"may","config_id":"nope","from_number":"+15550100"}`)

	s.Require().Equal(http.StatusBadRequest, response.StatusCode)

	var failure Error
	s.Require().NoError(json.Unmarshal(payload, &failure))
	s.Contains(failure.Error, "config")
}

func (s *APIIntegrationSuite) TestContactsAreRungInTheOrderTheyWereAdded() {
	created := s.campaign(1)

	response, payload := s.do(http.MethodPost, "/v1/agents/campaigns/"+created.Id+"/contacts", `{
		"contacts":[
			{"to_number":"+15550111","instructions":"ask about the trial"},
			{"to_number":"+15550222"}
		]
	}`)
	s.Require().Equal(http.StatusCreated, response.StatusCode, string(payload))

	claimed, found, err := s.store.ClaimContact(s.ctx, created.Id)
	s.Require().NoError(err)
	s.Require().True(found)
	s.Equal("+15550111", claimed.ToNumber)
	s.Equal("ask about the trial", claimed.Instructions)
	s.Equal(store.Calling, claimed.State, "a claimed contact is nobody else's to ring")
	s.Equal(1, claimed.Attempts)

	next, found, err := s.store.ClaimContact(s.ctx, created.Id)
	s.Require().NoError(err)
	s.Require().True(found)
	s.Equal("+15550222", next.ToNumber, "the same person was taken twice")

	_, found, err = s.store.ClaimContact(s.ctx, created.Id)
	s.Require().NoError(err)
	s.False(found, "there was nobody left to ring")
}

func (s *APIIntegrationSuite) TestACallThatNeverFinishedIsRungAgainRatherThanLost() {
	// A process that stopped mid-call leaves a contact claimed by nobody. Ringing them
	// again is better than a campaign that quietly skips them.
	created := s.campaign(1)
	_, _ = s.do(http.MethodPost, "/v1/agents/campaigns/"+created.Id+"/contacts",
		`{"contacts":[{"to_number":"+15550111"}]}`)

	_, found, err := s.store.ClaimContact(s.ctx, created.Id)
	s.Require().NoError(err)
	s.Require().True(found)

	s.Require().NoError(s.store.ReleaseContacts(s.ctx, created.Id))

	again, found, err := s.store.ClaimContact(s.ctx, created.Id)
	s.Require().NoError(err)
	s.Require().True(found)
	s.Equal(2, again.Attempts, "the second attempt is counted as one")
}

func (s *APIIntegrationSuite) TestWhatBecameOfAContactIsShownAgainstIt() {
	created := s.campaign(1)
	_, _ = s.do(http.MethodPost, "/v1/agents/campaigns/"+created.Id+"/contacts",
		`{"contacts":[{"to_number":"+15550111"},{"to_number":"+15550222"}]}`)

	first, _, err := s.store.ClaimContact(s.ctx, created.Id)
	s.Require().NoError(err)
	s.Require().NoError(s.store.FinishContact(s.ctx, store.Contact{
		ID: first.ID, State: store.Done, CallID: "session-1", VendorCallID: "CA1",
	}))

	second, _, err := s.store.ClaimContact(s.ctx, created.Id)
	s.Require().NoError(err)
	s.Require().NoError(s.store.FinishContact(s.ctx, store.Contact{
		ID: second.ID, State: store.Failed, Error: "the number is not in service",
	}))

	response, payload := s.do(http.MethodGet, "/v1/agents/campaigns/"+created.Id+"/contacts", "")
	s.Require().Equal(http.StatusOK, response.StatusCode)

	var contacts []Contact
	s.Require().NoError(json.Unmarshal(payload, &contacts))
	s.Require().Len(contacts, 2)
	s.Equal(ContactStateDone, contacts[0].State)
	s.Require().NotNil(contacts[0].CallId)
	s.Equal("session-1", *contacts[0].CallId)
	s.Equal(ContactStateFailed, contacts[1].State)
	s.Require().NotNil(contacts[1].Error)
	s.Contains(*contacts[1].Error, "not in service")
}

func (s *APIIntegrationSuite) TestACampaignCannotBeStartedWithoutAnythingToRingWith() {
	// This deployment has no telephony, so starting one has to say so rather than
	// reporting a campaign that is running and will never call anybody.
	created := s.campaign(1)

	response, payload := s.do(http.MethodPost, "/v1/agents/campaigns/"+created.Id+"/start", "")

	s.Require().Equal(http.StatusBadRequest, response.StatusCode, string(payload))
}

func (s *APIIntegrationSuite) TestAnotherCustomersCampaignIsNotFound() {
	created := s.campaign(1)

	request, err := http.NewRequestWithContext(s.ctx, http.MethodGet,
		s.server.URL+"/v1/agents/campaigns/"+created.Id, strings.NewReader(""))
	s.Require().NoError(err)
	request.Header.Set(CustomerHeader, "somebody-else")

	response, err := http.DefaultClient.Do(request)
	s.Require().NoError(err)
	defer response.Body.Close()

	s.Equal(http.StatusNotFound, response.StatusCode)
}

func (s *APIIntegrationSuite) TestACallIsFoundAfterTheSessionRunningItIsGone() {
	// A session lives in a map in memory. This is the whole point of the row: the call
	// is still there once the process that held it is not.
	call := store.Call{
		ID:         "session-" + s.customerID,
		CustomerID: s.customerID,
		CallID:     "call-1",
		AgentID:    "agent-1",
		Direction:  store.Outbound,
		ToNumber:   "+15550101",
		StartedAt:  s.base,
	}
	s.Require().NoError(s.store.StartCall(s.ctx, &call))

	response, payload := s.do(http.MethodGet, "/v1/agents/calls/"+call.ID, "")
	s.Require().Equal(http.StatusOK, response.StatusCode, string(payload))

	var read Call
	s.Require().NoError(json.Unmarshal(payload, &read))
	s.Equal("call-1", read.CallId)
	s.Equal(Outbound, read.Direction)
	s.Require().NotNil(read.ToNumber)
	s.Equal("+15550101", *read.ToNumber)
	s.Nil(read.EndedAt, "a call nobody has ended is still running")
}

// testStreamKey and testStreamSecret stand in for a Stream app. The secret signs the join
// tokens, which is what lets a test read one back and see who it was minted for.
const (
	testStreamKey    = "test-app"
	testStreamSecret = "test-secret"
)

func (s *APIIntegrationSuite) TestAJoinTokenSaysWhichCallToJoinAndWhoAsIt() {
	// The browser is handed a token and a call, never the secret: whoever holds this can
	// join one call as one user until it expires, and can sign nothing of their own.
	call := store.Call{
		ID: "session-" + s.customerID, CustomerID: s.customerID,
		CallID: "call-1", AgentID: "agent-1", StartedAt: s.base,
	}
	s.Require().NoError(s.store.StartCall(s.ctx, &call))

	response, payload := s.do(http.MethodPost, "/v1/agents/calls/"+call.ID+"/token", `{}`)
	s.Require().Equal(http.StatusOK, response.StatusCode, string(payload))

	var minted CallToken
	s.Require().NoError(json.Unmarshal(payload, &minted))
	s.Equal(testStreamKey, minted.ApiKey)
	s.Equal("call-1", minted.CallId, "the Stream call, not the id we hold it by")
	s.Equal("default", minted.CallType)
	s.NotEqual("agent-1", minted.UserId, "a listener is not the agent")
	s.True(minted.ExpiresAt.After(time.Now()), "a token that has expired is no use")

	claimed := jwt.MapClaims{}
	_, err := jwt.ParseWithClaims(minted.Token, claimed, func(*jwt.Token) (any, error) {
		return []byte(testStreamSecret), nil
	})
	s.Require().NoError(err, "the token is signed with the app secret")
	s.Equal(minted.UserId, claimed["user_id"], "and it is signed for the user it names")
}

func (s *APIIntegrationSuite) TestAJoinTokenIsMintedForTheUserTheCallerAsksFor() {
	call := store.Call{
		ID: "session-" + s.customerID, CustomerID: s.customerID,
		CallID: "call-1", AgentID: "agent-1", StartedAt: s.base,
	}
	s.Require().NoError(s.store.StartCall(s.ctx, &call))

	_, payload := s.do(http.MethodPost, "/v1/agents/calls/"+call.ID+"/token",
		`{"user_id":"thierry","user_name":"Thierry"}`)

	var minted CallToken
	s.Require().NoError(json.Unmarshal(payload, &minted))
	s.Equal("thierry", minted.UserId)
	s.Equal("Thierry", minted.UserName)

	claimed := jwt.MapClaims{}
	_, err := jwt.ParseWithClaims(minted.Token, claimed, func(*jwt.Token) (any, error) {
		return []byte(testStreamSecret), nil
	})
	s.Require().NoError(err)
	s.Equal("thierry", claimed["user_id"])
}

func (s *APIIntegrationSuite) TestACallAnotherCustomerHoldsCannotBeJoined() {
	// Handing out a token for somebody else's call would be handing out their call.
	call := store.Call{
		ID: "session-" + s.customerID, CustomerID: s.customerID,
		CallID: "call-1", AgentID: "agent-1", StartedAt: s.base,
	}
	s.Require().NoError(s.store.StartCall(s.ctx, &call))

	request, err := http.NewRequestWithContext(s.ctx, http.MethodPost,
		s.server.URL+"/v1/agents/calls/"+call.ID+"/token", strings.NewReader(`{}`))
	s.Require().NoError(err)
	request.Header.Set(CustomerHeader, "somebody-else")

	response, err := http.DefaultClient.Do(request)
	s.Require().NoError(err)
	defer response.Body.Close()

	s.Equal(http.StatusNotFound, response.StatusCode)
}

func (s *APIIntegrationSuite) TestTheRunningCallsAreTheOnesThatHaveNotEnded() {
	running := store.Call{
		ID: "running-" + s.customerID, CustomerID: s.customerID,
		CallID: "call-1", AgentID: "agent-1", StartedAt: s.base,
	}
	finished := store.Call{
		ID: "finished-" + s.customerID, CustomerID: s.customerID,
		CallID: "call-2", AgentID: "agent-2", StartedAt: s.base.Add(-time.Hour),
	}
	s.Require().NoError(s.store.StartCall(s.ctx, &running))
	s.Require().NoError(s.store.StartCall(s.ctx, &finished))
	s.Require().NoError(s.store.FinishCall(s.ctx, finished.ID, s.base))

	response, payload := s.do(http.MethodGet, "/v1/agents/calls?running=true", "")
	s.Require().Equal(http.StatusOK, response.StatusCode, string(payload))

	var listed []Call
	s.Require().NoError(json.Unmarshal(payload, &listed))
	s.Require().Len(listed, 1)
	s.Equal(running.ID, listed[0].Id)

	response, payload = s.do(http.MethodGet, "/v1/agents/calls", "")
	s.Require().Equal(http.StatusOK, response.StatusCode)
	s.Require().NoError(json.Unmarshal(payload, &listed))
	s.Require().Len(listed, 2, "both calls happened")
	s.Equal(running.ID, listed[0].Id, "newest first")
	s.Require().NotNil(listed[1].EndedAt)
}

func (s *APIIntegrationSuite) TestAnAgentIdSaysWhoseChannelItIsAndWhatRanOnIt() {
	// A message arriving on a channel names the agent and nothing else. Without this row
	// there is no way to tell whose message it is, and a message that cannot be billed to
	// anybody cannot be answered.
	//
	// The agent id belongs to this test rather than being shared: a channel is looked up by
	// it alone, so two tests naming the same one are asking about each other's calls.
	agentID := "agent-" + s.customerID
	call := store.Call{
		ID: "session-" + s.customerID, CustomerID: s.customerID,
		CallID: "call-1", AgentID: agentID, ConfigID: "chat_support", StartedAt: s.base,
	}
	s.Require().NoError(s.store.StartCall(s.ctx, &call))

	found, err := s.store.CallByAgent(s.ctx, agentID)

	s.Require().NoError(err)
	s.Equal(s.customerID, found.CustomerID)
	s.Equal("chat_support", found.ConfigID, "which agent is being written to")
}

func (s *APIIntegrationSuite) TestWritingToAChannelReachesTheLastConversationOnIt() {
	// An agent id outlives the call that made it, so the same channel can hold several. The
	// one somebody writing there is continuing is the last one.
	agentID := "agent-" + s.customerID
	older := store.Call{
		ID: "older-" + s.customerID, CustomerID: s.customerID,
		CallID: "call-1", AgentID: agentID, ConfigID: "old_config",
		StartedAt: s.base.Add(-time.Hour),
	}
	newer := store.Call{
		ID: "newer-" + s.customerID, CustomerID: s.customerID,
		CallID: "call-2", AgentID: agentID, ConfigID: "chat_support",
		StartedAt: s.base,
	}
	s.Require().NoError(s.store.StartCall(s.ctx, &older))
	s.Require().NoError(s.store.StartCall(s.ctx, &newer))

	found, err := s.store.CallByAgent(s.ctx, agentID)

	s.Require().NoError(err)
	s.Equal(newer.ID, found.ID)
	s.Equal("chat_support", found.ConfigID)
}

func (s *APIIntegrationSuite) TestNoConversationIsFoundForAChannelNoAgentHasBeenOn() {
	// This is a message in a channel that merely looks like an agent's. There is nobody to
	// bill it to and nothing to start, and saying so is the whole answer.
	_, err := s.store.CallByAgent(s.ctx, "a-channel-nobody-ran-on")

	s.Require().Error(err)
}

func (s *APIIntegrationSuite) TestACallKeepsTheTimeItFirstEnded() {
	// An agent leaves once. A second close is the same leaving reported again, and must
	// not stretch the call to cover it.
	call := store.Call{
		ID: "session-" + s.customerID, CustomerID: s.customerID,
		CallID: "call-1", AgentID: "agent-1", StartedAt: s.base,
	}
	s.Require().NoError(s.store.StartCall(s.ctx, &call))
	s.Require().NoError(s.store.FinishCall(s.ctx, call.ID, s.base.Add(time.Minute)))
	s.Require().NoError(s.store.FinishCall(s.ctx, call.ID, s.base.Add(time.Hour)))

	read, err := s.store.Call(s.ctx, s.customerID, call.ID)
	s.Require().NoError(err)
	s.Require().NotNil(read.EndedAt)
	s.WithinDuration(s.base.Add(time.Minute), *read.EndedAt, time.Second)
}

func (s *APIIntegrationSuite) TestWhatACallDecidedIsFoundByTheRowRecordingIt() {
	// A dashboard holds the row's id, and the agent wrote its reasoning against the call
	// it joined. Reading one by the other is what puts the decision log on the page.
	call := store.Call{
		ID: "session-" + s.customerID, CustomerID: s.customerID,
		CallID: "call-1", AgentID: "agent-1", StartedAt: s.base,
	}
	s.Require().NoError(s.store.StartCall(s.ctx, &call))
	s.Require().NoError(s.store.RecordCallEvents(s.ctx, []store.CallEvent{{
		CustomerID: s.customerID, CallID: call.CallID, AgentID: call.AgentID,
		At: s.base.Add(time.Second), Kind: "answer", Reason: "a complete thought",
		Said: "how is the weather",
	}}))

	response, payload := s.do(http.MethodGet, "/v1/agents/calls/"+call.ID+"/events", "")
	s.Require().Equal(http.StatusOK, response.StatusCode, string(payload))

	var decided []CallEvent
	s.Require().NoError(json.Unmarshal(payload, &decided))
	s.Require().Len(decided, 1)
	s.Equal(DecisionKind("answer"), decided[0].Kind)
	s.Require().NotNil(decided[0].Said)
	s.Equal("how is the weather", *decided[0].Said)
}

func (s *APIIntegrationSuite) TestFinishedNativeCallIncludesConversationAndSubagentModels() {
	call := store.Call{
		ID: "session-" + s.customerID, CustomerID: s.customerID,
		CallID: "call-1", AgentID: "agent-1", StartedAt: s.base,
		STS: "openai/gpt-realtime-2", Subagent: "openai/gpt-5.6-sol",
	}
	s.Require().NoError(s.store.StartCall(s.ctx, &call))
	for modality, model := range map[string]string{"sts": "gpt-realtime-2", "llm": "gpt-5.6-sol"} {
		s.Require().NoError(s.store.RecordRequest(s.ctx, &store.Request{
			CustomerID: s.customerID, AgentID: call.AgentID,
			Modality: modality, Provider: "openai", Model: model,
			StartedAt: s.base.Add(time.Second), Success: true,
		}))
	}
	s.Require().NoError(s.store.FinishCall(s.ctx, call.ID, s.base.Add(time.Minute)))

	response, payload := s.do(http.MethodGet, "/v1/agents/calls/"+call.ID, "")
	s.Require().Equal(http.StatusOK, response.StatusCode, string(payload))
	var rendered Call
	s.Require().NoError(json.Unmarshal(payload, &rendered))
	s.Equal("openai/gpt-realtime-2", value(rendered.StsUsed))
	s.Equal("openai/gpt-5.6-sol", value(rendered.SubagentUsed))
	s.Nil(rendered.SttUsed)
	s.Nil(rendered.LlmUsed)
	s.Nil(rendered.TtsUsed)
}

func (s *APIIntegrationSuite) TestAFinishedCallReportsWhatItSpentAndWhoItSpokeTo() {
	call := store.Call{
		ID: "session-" + s.customerID, CustomerID: s.customerID,
		CallID: "call-1", AgentID: "agent-1", UserID: "ada", StartedAt: s.base,
	}
	s.Require().NoError(s.store.StartCall(s.ctx, &call))
	s.Require().NoError(s.store.RecordRequest(s.ctx, &store.Request{
		CustomerID: s.customerID, AgentID: call.AgentID,
		Modality: "llm", Provider: "openai", Model: "gpt-5.6-sol",
		StartedAt:   s.base.Add(time.Second),
		InputTokens: 900, CachedInputTokens: 400, OutputTokens: 150,
		CostMicros: 2500, Success: true,
	}))
	s.Require().NoError(s.store.FinishCall(s.ctx, call.ID, s.base.Add(time.Minute)))

	response, payload := s.do(http.MethodGet, "/v1/agents/calls/"+call.ID, "")

	s.Require().Equal(http.StatusOK, response.StatusCode, string(payload))
	var rendered Call
	s.Require().NoError(json.Unmarshal(payload, &rendered))
	s.Equal("ada", value(rendered.UserId))
	s.Require().NotNil(rendered.Usage)
	s.Equal(int64(900), rendered.Usage.InputTokens)
	s.Equal(int64(400), rendered.Usage.CachedInputTokens)
	s.Equal(int64(150), rendered.Usage.OutputTokens)
	s.Equal(int64(2500), rendered.Usage.CostMicros)
	s.Equal(int64(1), rendered.Usage.Requests)
}

func (s *APIIntegrationSuite) TestACallStillRunningIsNotToldWhatItHasSpentSoFar() {
	call := store.Call{
		ID: "session-" + s.customerID, CustomerID: s.customerID,
		CallID: "call-1", AgentID: "agent-1", StartedAt: s.base,
	}
	s.Require().NoError(s.store.StartCall(s.ctx, &call))
	s.Require().NoError(s.store.RecordRequest(s.ctx, &store.Request{
		CustomerID: s.customerID, AgentID: call.AgentID,
		Modality: "llm", Provider: "openai", Model: "gpt-5.6-sol",
		StartedAt: s.base.Add(time.Second), InputTokens: 900, Success: true,
	}))

	response, payload := s.do(http.MethodGet, "/v1/agents/calls/"+call.ID, "")

	s.Require().Equal(http.StatusOK, response.StatusCode, string(payload))
	var rendered Call
	s.Require().NoError(json.Unmarshal(payload, &rendered))
	s.Nil(rendered.Usage, "what a conversation cost is a question asked after it")
}

func (s *APIIntegrationSuite) TestAnotherCustomersCallIsNotFound() {
	call := store.Call{
		ID: "session-" + s.customerID, CustomerID: "somebody-else",
		CallID: "call-1", AgentID: "agent-1", StartedAt: s.base,
	}
	s.Require().NoError(s.store.StartCall(s.ctx, &call))

	response, _ := s.do(http.MethodGet, "/v1/agents/calls/"+call.ID, "")
	s.Equal(http.StatusNotFound, response.StatusCode)
}

func (s *APIIntegrationSuite) TestVoiceProvidersAreRankedSeparately() {
	// A speech-to-text failure must not make the text-to-speech provider look unhealthy.
	s.Require().NoError(s.live.RecordRequest(s.ctx, live.Usage{
		Modality: "stt", CustomerID: s.customerID,
		Provider: "elevenlabs", Model: "eleven_flash_v2_5",
		LatencyMs: 150, Success: false,
	}))

	response, payload := s.do(http.MethodGet, "/v1/tts/providers", "")
	s.Require().Equal(http.StatusOK, response.StatusCode)

	var providers []Provider
	s.Require().NoError(json.Unmarshal(payload, &providers))
	s.Require().NotEmpty(providers)

	for _, provider := range providers {
		if provider.Model == "eleven_flash_v2_5" {
			s.Zero(provider.Health.Errors, "health is keyed by modality")
		}
	}
}

func (s *APIIntegrationSuite) TestAVoiceIsRecordedPreparedAndReadBack() {
	response, payload := s.do(http.MethodPost, "/v1/agents/voices",
		`{"name":"founder","description":"the one from the ad"}`)
	s.Require().Equal(http.StatusCreated, response.StatusCode, string(payload))

	var created Voice
	s.Require().NoError(json.Unmarshal(payload, &created))
	s.Require().NotEmpty(created.Id)
	s.Equal("founder", created.Name)
	s.Empty(*created.Samples, "a voice starts with nothing recorded")

	audio := base64.StdEncoding.EncodeToString([]byte("pretend this is speech"))
	response, payload = s.do(http.MethodPost, "/v1/agents/voices/"+created.Id+"/samples",
		fmt.Sprintf(`{"audio":%q,"filename":"clip.wav","content_type":"audio/wav","transcript":"hello"}`, audio))
	s.Require().Equal(http.StatusCreated, response.StatusCode, string(payload))

	var recorded Voice
	s.Require().NoError(json.Unmarshal(payload, &recorded))
	s.Require().Len(*recorded.Samples, 1)
	s.EqualValues(22, *(*recorded.Samples)[0].Bytes)

	response, payload = s.do(http.MethodPost, "/v1/agents/voices/"+created.Id+"/prepare", `{}`)
	s.Require().Equal(http.StatusOK, response.StatusCode, string(payload))

	var prepared Voice
	s.Require().NoError(json.Unmarshal(payload, &prepared))
	s.Require().Len(*prepared.Bindings, 1)
	s.Equal(VoiceBindingStateReady, (*prepared.Bindings)[0].State)
	s.Equal("el-cloned", *(*prepared.Bindings)[0].ExternalId,
		"a session names the voice, and the provider is asked for its own id")
}

func (s *APIIntegrationSuite) TestVoiceProvidersAreTheOnesThisDeploymentCanCloneWith() {
	response, payload := s.do(http.MethodGet, "/v1/agents/voices/providers", "")
	s.Require().Equal(http.StatusOK, response.StatusCode, string(payload))

	var listed VoiceProviders
	s.Require().NoError(json.Unmarshal(payload, &listed))
	s.Equal([]string{"elevenlabs"}, listed.Providers)
}

func (s *APIIntegrationSuite) TestAPreparedVoiceCanBeHeardThroughItsProvider() {
	_, payload := s.do(http.MethodPost, "/v1/agents/voices", `{"name":"founder"}`)
	var created Voice
	s.Require().NoError(json.Unmarshal(payload, &created))
	audio := base64.StdEncoding.EncodeToString([]byte("pretend this is speech"))
	s.do(http.MethodPost, "/v1/agents/voices/"+created.Id+"/samples",
		fmt.Sprintf(`{"audio":%q,"filename":"clip.wav"}`, audio))
	s.do(http.MethodPost, "/v1/agents/voices/"+created.Id+"/prepare", `{}`)

	response, payload := s.do(http.MethodPost, "/v1/agents/voices/"+created.Id+"/preview",
		`{"provider":"elevenlabs"}`)
	s.Require().Equal(http.StatusOK, response.StatusCode, string(payload))

	var preview VoicePreview
	s.Require().NoError(json.Unmarshal(payload, &preview))
	s.Equal("elevenlabs", preview.Provider)
	s.Equal("audio/mpeg", preview.ContentType)
	s.Equal([]byte("spoken"), preview.Audio)
}

func (s *APIIntegrationSuite) TestAVoiceCannotBeHeardThroughAProviderThatDoesNotHaveIt() {
	_, payload := s.do(http.MethodPost, "/v1/agents/voices", `{"name":"founder"}`)
	var created Voice
	s.Require().NoError(json.Unmarshal(payload, &created))

	response, payload := s.do(http.MethodPost, "/v1/agents/voices/"+created.Id+"/preview",
		`{"provider":"elevenlabs"}`)

	s.Equal(http.StatusBadRequest, response.StatusCode, string(payload))
	s.Contains(string(payload), "not ready with elevenlabs")
}

func (s *APIIntegrationSuite) TestAVoiceWithNothingRecordedCannotBePrepared() {
	_, payload := s.do(http.MethodPost, "/v1/agents/voices", `{"name":"founder"}`)
	var created Voice
	s.Require().NoError(json.Unmarshal(payload, &created))

	response, payload := s.do(http.MethodPost, "/v1/agents/voices/"+created.Id+"/prepare", `{}`)

	s.Equal(http.StatusBadRequest, response.StatusCode, string(payload))
	s.Contains(string(payload), "add a recording")
}

func (s *APIIntegrationSuite) TestAVoiceBelongingToSomebodyElseIsNotThere() {
	_, payload := s.do(http.MethodPost, "/v1/agents/voices", `{"name":"founder"}`)
	var created Voice
	s.Require().NoError(json.Unmarshal(payload, &created))

	s.customerID = "somebody-else"
	response, _ := s.do(http.MethodGet, "/v1/agents/voices/"+created.Id, "")

	s.Equal(http.StatusNotFound, response.StatusCode)
}

func (s *APIIntegrationSuite) TestAVoiceNeedsAName() {
	response, payload := s.do(http.MethodPost, "/v1/agents/voices", `{"name":"  "}`)

	s.Equal(http.StatusBadRequest, response.StatusCode, string(payload))
	s.Contains(string(payload), "needs a name")
}

func (s *APIIntegrationSuite) TestADeletedVoiceStopsBeingListed() {
	_, payload := s.do(http.MethodPost, "/v1/agents/voices", `{"name":"founder"}`)
	var created Voice
	s.Require().NoError(json.Unmarshal(payload, &created))

	response, _ := s.do(http.MethodDelete, "/v1/agents/voices/"+created.Id, "")
	s.Require().Equal(http.StatusNoContent, response.StatusCode)

	response, payload = s.do(http.MethodGet, "/v1/agents/voices", "")
	s.Require().Equal(http.StatusOK, response.StatusCode)

	var listed []Voice
	s.Require().NoError(json.Unmarshal(payload, &listed))
	s.Empty(listed)
}
