// Command router serves the model router's HTTP API.
package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/agent/streamedge"
	"github.com/GetStream/Vision-Agents/acceleration/internal/api"
	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/blob"
	"github.com/GetStream/Vision-Agents/acceleration/internal/campaign"
	"github.com/GetStream/Vision-Agents/acceleration/internal/chatlog"
	"github.com/GetStream/Vision-Agents/acceleration/internal/dispatch"
	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge"
	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge/turbopuffer"
	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge/urls"
	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/memory"
	"github.com/GetStream/Vision-Agents/acceleration/internal/memory/mem0"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone/vendors"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/search/exa"
	"github.com/GetStream/Vision-Agents/acceleration/internal/searchrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/simulation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sttrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts/cartesia"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts/elevenlabs"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts/fish"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts/voices"
	"github.com/GetStream/Vision-Agents/acceleration/internal/ttsrouter"
)

const (
	addressEnvVar     = "ROUTER_ADDR"
	postgresEnvVar    = "ROUTER_POSTGRES_DSN"
	redisEnvVar       = "ROUTER_REDIS_ADDR"
	configEnvVar      = "ROUTER_CONFIG"
	phoneConfigEnvVar = "ROUTER_PHONE_CONFIG"
	// publicURLEnvVar is where this service is reachable from the internet, which the
	// telephony vendors that fetch a call plan on answer need in order to fetch it. It is
	// not ROUTER_ADDR: that is where to listen, which behind a load balancer is not where
	// anyone reaches it.
	publicURLEnvVar = "ROUTER_PUBLIC_URL"
	// streamSecretEnvVar signs the call events Stream sends to the inbound hook. It is the
	// app secret rather than a webhook-specific one, and it also signs the tokens a
	// browser joins a call with.
	streamSecretEnvVar = "STREAM_API_SECRET"
	// streamKeyEnvVar names the Stream app those tokens are for, which a browser needs in
	// order to join with one.
	streamKeyEnvVar = "STREAM_API_KEY"
	// corsOriginsEnvVar names the browser origins allowed to call this API directly,
	// comma separated. It exists for the dashboard, which talks to the router rather than
	// through a server of its own. Unset means no browser may.
	corsOriginsEnvVar = "ROUTER_CORS_ORIGINS"
	// authModeEnvVar decides who the router believes a caller is. "noauth" trusts the
	// headers a proxy in front of it sets, and is only safe when nothing else can reach
	// it; "api_key" verifies a key and the token signed with its secret.
	authModeEnvVar = "ROUTER_AUTH_MODE"
	// authKEKEnvVar unseals the stored key secrets. It lives outside the database on
	// purpose: it is what makes a leaked backup ciphertext rather than credentials.
	authKEKEnvVar     = "ROUTER_AUTH_KEK"
	logLevelEnvVar    = "ROUTER_LOG_LEVEL"
	defaultAddress    = ":8080"
	shutdownGrace     = 10 * time.Second
	readHeaderTimeout = 10 * time.Second
	// crawlTimeout bounds reading one page into a knowledge base. It is generous compared
	// to a search because nobody is on the phone waiting for it: a page that has to be
	// crawled live rather than served from an index takes seconds, and giving up on it
	// leaves a subscription that never works.
	crawlTimeout = 60 * time.Second
	// lastUsedInterval throttles how often a key's use is recorded. Writing on every
	// request would double the writes of a busy key, and recording nothing means nobody
	// can answer whether a key is still in use, so nobody ever revokes one.
	lastUsedInterval = time.Minute
)

func main() {
	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: logLevel()}))
	slog.SetDefault(logger)

	if err := run(logger); err != nil {
		logger.Error("router stopped", "error", err)
		os.Exit(1)
	}
}

// logLevel reads ROUTER_LOG_LEVEL. Debug is where the turn-taking decisions are: what was
// heard, what the flow controller made of it, and why the agent did or did not speak.
func logLevel() slog.Level {
	var level slog.Level
	text := os.Getenv(logLevelEnvVar)
	if text == "" {
		return slog.LevelInfo
	}
	if err := level.UnmarshalText([]byte(text)); err != nil {
		return slog.LevelInfo
	}
	return level
}

func dashboardBaseURL() string {
	if value := os.Getenv("DASHBOARD_BASE_URL"); value != "" {
		return value
	}
	return "http://localhost:3000"
}

// newAuthenticator builds the authenticator the deployment's mode asks for.
//
// api_key needs both a store to look keys up in and the key that unseals their secrets, and
// says which is missing rather than starting and refusing every request for a reason only
// visible in a 401.
func newAuthenticator(pgStore *store.Store, logger *slog.Logger) (auth.Authenticator, error) {
	mode, err := auth.ParseMode(os.Getenv(authModeEnvVar))
	if err != nil {
		return nil, err
	}

	if mode == auth.NoAuth {
		logger.Warn("running without authentication: anyone who can reach this router can "+
			"read and spend any customer's account, so only a trusted proxy should be able to",
			"mode", auth.NoAuth, "set", authModeEnvVar)
		return auth.New(mode, nil)
	}

	if pgStore == nil {
		return nil, fmt.Errorf("%s=%s needs %s, because that is where the keys are",
			authModeEnvVar, auth.APIKey, postgresEnvVar)
	}
	sealer, err := auth.NewSealer(os.Getenv(authKEKEnvVar))
	if err != nil {
		return nil, fmt.Errorf("%s=%s needs %s: %w", authModeEnvVar, auth.APIKey, authKEKEnvVar, err)
	}

	return auth.New(mode, func(ctx context.Context, key string) (auth.App, error) {
		// The shape of the key is checked before the database is, so a truncated paste
		// costs nothing to reject.
		if !auth.ValidKey(key) {
			return auth.App{}, auth.ErrUnauthenticated
		}
		owner, err := pgStore.LiveAPIKey(ctx, key)
		if err != nil {
			return auth.App{}, auth.ErrUnauthenticated
		}
		secret, err := sealer.Open(owner.Sealed)
		if err != nil {
			return auth.App{}, fmt.Errorf("unseal key %s: %w", key, err)
		}
		if err := pgStore.TouchAPIKey(ctx, key, lastUsedInterval); err != nil {
			logger.Debug("could not record key use", "key", key, "error", err)
		}
		return auth.App{
			OrganizationID: owner.OrganizationID,
			AppID:          owner.AppID,
			Secret:         secret,
		}, nil
	})
}

func run(logger *slog.Logger) error {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	config, err := routing.LoadConfig(os.Getenv(configEnvVar))
	if err != nil {
		return err
	}

	// Postgres and Redis are optional so the API can be brought up for inspection before
	// the data stores exist. /health reports what is missing.
	var pgStore *store.Store
	if dsn := os.Getenv(postgresEnvVar); dsn != "" {
		pgStore, err = store.Open(dsn)
		if err != nil {
			return err
		}
		defer pgStore.Close()

		if err := pgStore.Migrate(ctx); err != nil {
			return err
		}
	} else {
		logger.Warn("no database configured, statistics will not be recorded", "env", postgresEnvVar)
	}

	var liveClient *live.Client
	if address := os.Getenv(redisEnvVar); address != "" {
		liveClient, err = live.New(live.Options{Address: address})
		if err != nil {
			return err
		}
		defer liveClient.Close()
	} else {
		logger.Warn("no redis configured, routing will not use live health", "env", redisEnvVar)
	}

	// Voices a customer brought with them live in an object bucket and a few tables. The
	// resolver only reads the tables, so a deployment with a database but no bucket can
	// still speak in voices another one prepared.
	var resolver routing.VoiceResolver
	if pgStore != nil {
		resolver = voices.NewResolver(pgStore)
	}

	bucket, err := blob.Open(ctx, os.Getenv(blob.EnvURL))
	if err != nil {
		return err
	}
	if bucket != nil {
		defer bucket.Close()
	}

	// A modality the config says nothing about is simply not served, and its paths 404.
	routers := map[routing.Modality]routing.Inspector{}
	streams := &api.Streams{}

	if section, ok := config[routing.STT]; ok {
		speech, err := sttrouter.New(sttrouter.Options{
			Config:   section,
			Registry: sttrouter.DefaultRegistry(),
			Store:    pgStore,
			Live:     liveClient,
			Logger:   logger,
		})
		if err != nil {
			return err
		}
		defer speech.Close()
		routers[routing.STT] = speech
		streams.STT = speech

		// The recording half of the same section. It is a second router rather than a
		// second method because it routes to different models: the batch endpoints,
		// which are the ones declared realtime: false.
		recorded, err := sttrouter.NewRecordings(sttrouter.Options{
			Config:       section,
			Transcribers: sttrouter.DefaultTranscriberRegistry(),
			Store:        pgStore,
			Live:         liveClient,
			Logger:       logger,
		})
		if err != nil {
			return err
		}
		defer recorded.Close()
		streams.Transcriptions = recorded
	}

	if section, ok := config[routing.TTS]; ok {
		voice, err := ttsrouter.New(ttsrouter.Options{
			Config:   section,
			Registry: ttsrouter.DefaultRegistry(),
			Store:    pgStore,
			Live:     liveClient,
			Voices:   resolver,
			Logger:   logger,
		})
		if err != nil {
			return err
		}
		defer voice.Close()
		routers[routing.TTS] = voice
		streams.TTS = voice

		recorded, err := ttsrouter.NewRecordings(ttsrouter.Options{
			Config:    section,
			Recorders: ttsrouter.DefaultRecorderRegistry(),
			Store:     pgStore,
			Live:      liveClient,
			Voices:    resolver,
			Logger:    logger,
		})
		if err != nil {
			return err
		}
		defer recorded.Close()
		streams.Speech = recorded
	}

	if section, ok := config[routing.LLM]; ok {
		chat, err := llmrouter.New(llmrouter.Options{
			Config:   section,
			Registry: llmrouter.DefaultRegistry(),
			Store:    pgStore,
			Live:     liveClient,
			Logger:   logger,
		})
		if err != nil {
			return err
		}
		defer chat.Close()
		routers[routing.LLM] = chat
		streams.LLM = chat
	}

	// Search is routed like the three above, so a deployment with no key for any of the
	// providers still inspects and reports on them: what stops a session searching is a
	// candidate refusing to be built, not the section being absent.
	var finding *searchrouter.Router
	if section, ok := config[routing.Search]; ok {
		finding, err = searchrouter.New(searchrouter.Options{
			Config:   section,
			Registry: searchrouter.DefaultRegistry(),
			Store:    pgStore,
			Live:     liveClient,
			Logger:   logger,
		})
		if err != nil {
			return err
		}
		defer finding.Close()
		routers[routing.Search] = finding
		streams.Search = finding
	}

	telephony, err := buildPhone(pgStore, liveClient, logger)
	if err != nil {
		return err
	}

	// Without a turbopuffer key an agent knows only what its instructions say: the lookup
	// tool is offered to no session, and there is nothing to fill either.
	var base *turbopuffer.Store
	if search, err := turbopuffer.New(turbopuffer.Options{Logger: logger}); err != nil {
		logger.Debug("nothing will be looked up or written down", "error", err)
	} else {
		base = search
		defer base.Close()
	}

	// Conversations need all three modalities, so a deployment configured for only one
	// still inspects routing and reports statistics while the session paths say there
	// are none.
	sessions, err := buildSessions(streams, pgStore, liveClient, telephony, base, finding, logger)
	if err != nil {
		return err
	}
	if sessions != nil {
		defer sessions.Shutdown()
	}

	// A campaign is a phone call, a conversation and a row, so it runs only where all
	// three are configured. Elsewhere the campaign paths say so.
	var campaigns *campaign.Runner
	if pgStore != nil && telephony != nil && sessions != nil {
		campaigns, err = campaign.New(campaign.Options{
			Store:    pgStore,
			Phone:    telephony,
			Sessions: sessions,
			Logger:   logger,
		})
		if err != nil {
			return err
		}
		defer campaigns.Close()
	}

	// A simulation is a conversation, a model to judge it and a row, so it too runs only
	// where all three are configured. Elsewhere a simulation can be written down but the
	// path that runs it says why it cannot.
	var simulations *simulation.Runner
	if pgStore != nil && sessions != nil && streams.LLM != nil {
		simulations, err = simulation.New(simulation.Options{
			Store:    pgStore,
			Sessions: sessions,
			LLM:      streams.LLM,
			// Speech is what an audio simulation needs and a text one does not, so it is
			// passed where it exists rather than required.
			TTS:    streams.TTS,
			STT:    streams.STT,
			Logger: logger,
		})
		if err != nil {
			return err
		}
		defer simulations.Close()

		// Runs an older process left going are nobody's to finish: the conversations were
		// held in it, and it is gone.
		if err := simulations.Abandon(ctx); err != nil {
			logger.Error("could not write off the runs an older router left going", "error", err)
		}
	}

	// Reading a transcript back needs the same credentials writing one does. Without
	// them the calls are still listed; only what was said on them is missing.
	var transcripts *chatlog.Reader
	if reader, err := chatlog.NewReader(chatlog.ReaderOptions{}); err != nil {
		logger.Debug("transcripts will not be readable", "error", err)
	} else {
		transcripts = reader
	}

	// Bringing a voice needs somewhere to keep the recordings, a place to record them and
	// at least one provider willing to be taught. Missing any of those, the voice paths
	// say so rather than half-working.
	voiceService, err := buildVoices(pgStore, bucket, logger)
	if err != nil {
		return err
	}

	// Keeping a knowledge base filled from a url needs a row, a base and a crawler.
	// Missing any of those, the url paths say so rather than storing a subscription
	// nothing would ever honour.
	pages, err := buildKnowledgeURLs(pgStore, base, logger)
	if err != nil {
		return err
	}

	// Inbound calls are answered by whoever is connected to the dispatch socket, so the
	// pool exists whether or not anybody is: an empty pool is a call nobody answers, which
	// is a different thing from a deployment that does not dispatch at all.
	workers := dispatch.NewPool()

	authenticator, err := newAuthenticator(pgStore, logger)
	if err != nil {
		return err
	}

	options := api.Options{
		Routers:       routers,
		Voices:        voiceService,
		KnowledgeURLs: pages,
		Store:         pgStore,
		Live:          liveClient,
		Phone:         telephony,
		Sessions:      sessions,
		Streams:       streams,
		Transcripts:   transcripts,
		Campaigns:     campaigns,
		Simulations:   simulations,
		Dispatch:      workers,
		StreamSecret:  os.Getenv(streamSecretEnvVar),
		StreamKey:     os.Getenv(streamKeyEnvVar),
		CORSOrigins:   splitList(os.Getenv(corsOriginsEnvVar)),
		PublicURL:     os.Getenv(publicURLEnvVar),
		DashboardURL:  dashboardBaseURL(),
		Auth:          authenticator,
		Logger:        logger,
	}
	if options.StreamSecret == "" {
		logger.Warn("no "+streamSecretEnvVar+" set, so inbound calls cannot be dispatched: "+
			"the call events Stream sends cannot be told apart from anyone who found the url",
			"hook", "POST /v1/phone/hooks/stream")
	}
	if options.StreamKey == "" {
		logger.Warn("no "+streamKeyEnvVar+" set, so nobody can join a call from a browser",
			"endpoint", "POST /v1/agents/calls/{id}/token")
	}
	// A nil *turbopuffer.Store in an interface is not a nil interface, so the absence has
	// to stay absent rather than becoming a value that says it is there.
	if base != nil {
		options.Knowledge = base
	}

	server, err := api.NewServer(options)
	if err != nil {
		return err
	}

	address := os.Getenv(addressEnvVar)
	if address == "" {
		address = defaultAddress
	}

	httpServer := &http.Server{
		Addr:              address,
		Handler:           server.Handler(),
		ReadHeaderTimeout: readHeaderTimeout,
	}

	listening := make(chan error, 1)
	go func() {
		logger.Info("listening", "address", address, "modalities", len(routers))
		if err := httpServer.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			listening <- err
			return
		}
		listening <- nil
	}()

	select {
	case err := <-listening:
		return err
	case <-ctx.Done():
		logger.Info("shutting down")
		shutdownCtx, cancel := context.WithTimeout(context.Background(), shutdownGrace)
		defer cancel()
		return httpServer.Shutdown(shutdownCtx)
	}
}

// splitList reads a comma-separated environment variable, dropping the empty entries a
// trailing comma leaves behind.
func splitList(raw string) []string {
	var entries []string
	for _, entry := range strings.Split(raw, ",") {
		if trimmed := strings.TrimSpace(entry); trimmed != "" {
			entries = append(entries, trimmed)
		}
	}
	return entries
}

// buildSessions wires the part of the router that holds conversations rather than
// describing them.
//
// It returns nil when a modality is missing, because a conversation needs all three and a
// manager that could not start one is worse than a path that says there are none. The
// factories live here rather than in the session package so the Stream edge, whose Opus
// path is cgo, stays out of everything that only needs to be tested.
func buildSessions(
	streams *api.Streams,
	pgStore *store.Store,
	liveClient *live.Client,
	telephony *phone.Service,
	base *turbopuffer.Store,
	finding *searchrouter.Router,
	logger *slog.Logger,
) (*session.Manager, error) {
	if streams.STT == nil || streams.TTS == nil || streams.LLM == nil {
		logger.Warn("not serving sessions, which need all three modalities configured")
		return nil, nil
	}

	// Without a mem0 key a session starts every call knowing nothing but its
	// instructions, which is the behaviour before memory existed.
	var remembering memory.Store
	if recall, err := mem0.New(mem0.Options{Logger: logger}); err != nil {
		logger.Debug("sessions will not remember anything between calls", "error", err)
	} else {
		remembering = recall
	}

	var reading knowledge.Store
	if base != nil {
		reading = base
	}

	return session.NewManager(session.ManagerOptions{
		LLM:       streams.LLM,
		STT:       streams.STT,
		TTS:       streams.TTS,
		Memory:    remembering,
		Knowledge: reading,
		Search:    finding,
		Phone:     telephony,
		Store:     pgStore,
		Live:      liveClient,
		Logger:    logger,
		Edge: func(spec session.Spec, logger *slog.Logger) (agent.Edge, error) {
			return streamedge.New(streamedge.Options{
				CallID:   spec.CallID,
				CallType: spec.CallType,
				User:     streamedge.User{ID: spec.UserID, Name: spec.UserName},
				Logger:   logger,
			})
		},
		Transcript: func(spec session.Spec, logger *slog.Logger) (session.Transcript, error) {
			// A voice call leaves nothing behind, so what was said is stored in a chat
			// channel named after the agent.
			return chatlog.New(chatlog.Options{
				AgentID: spec.AgentID,
				Agent:   chatlog.User{ID: spec.UserID, Name: spec.UserName},
				Logger:  logger,
			})
		},
	})
}

// buildKnowledgeURLs wires the control plane for pages a knowledge base is kept filled
// from.
//
// It returns nil unless there is a database to remember a subscription, a knowledge base to
// write the passages into and a key for something that can read a page, since a url that is
// recorded and never fetched is a promise nothing keeps. The url paths report the absence.
//
// Exa is built here rather than taken from the search router because the two want opposite
// timeouts: a search happens while somebody waits on the phone, and a live crawl of a page
// nobody is listening to can take as long as it takes.
func buildKnowledgeURLs(
	pgStore *store.Store,
	base *turbopuffer.Store,
	logger *slog.Logger,
) (*urls.Service, error) {
	if pgStore == nil || base == nil {
		logger.Debug("not serving knowledge urls",
			"database", pgStore != nil, "knowledge", base != nil)
		return nil, nil
	}

	reader, err := exa.New(exa.Options{Timeout: crawlTimeout, Logger: logger})
	if err != nil {
		logger.Debug("not serving knowledge urls: nothing can read a page", "error", err)
		return nil, nil
	}

	return urls.New(urls.Options{
		Store:  pgStore,
		Reader: reader,
		Writer: base,
		Logger: logger,
	})
}

// buildVoices wires the control plane for voices a customer brought with them.
//
// It returns nil when there is no database, no bucket, or no provider this deployment has
// a key for, since a voice needs a row, somewhere to keep the recordings and somebody to
// teach them to. The voice paths report the absence rather than failing halfway through an
// upload.
func buildVoices(
	pgStore *store.Store,
	bucket *blob.Bucket,
	logger *slog.Logger,
) (*voices.Service, error) {
	if pgStore == nil || bucket == nil {
		logger.Debug("not serving voices of your own", "database", pgStore != nil, "bucket", bucket != nil)
		return nil, nil
	}

	cloners := voices.NewRegistry()
	if cloner, err := voices.NewElevenLabs(voices.ElevenLabsOptions{}); err == nil {
		cloners.Register(elevenlabs.ProviderName, cloner)
	}
	if cloner, err := voices.NewCartesia(voices.CartesiaOptions{}); err == nil {
		cloners.Register(cartesia.ProviderName, cloner)
	}
	if cloner, err := voices.NewFish(voices.FishOptions{}); err == nil {
		cloners.Register(fish.ProviderName, cloner)
	}
	if len(cloners.Providers()) == 0 {
		logger.Warn("not serving voices of your own: no provider this deployment has a key for can be taught one")
		return nil, nil
	}

	return voices.NewService(voices.Options{
		Store:   pgStore,
		Bucket:  bucket,
		Cloners: cloners,
		Logger:  logger,
	})
}

// buildPhone wires the telephony service. Stream credentials are only needed to attach a
// number, so a deployment without them still lists vendors and searches for numbers, and
// the operations that need them say so.
func buildPhone(
	pgStore *store.Store,
	liveClient *live.Client,
	logger *slog.Logger,
) (*phone.Service, error) {
	config, err := phone.LoadConfig(os.Getenv(phoneConfigEnvVar))
	if err != nil {
		return nil, err
	}

	var stream *phone.Stream
	if streaming, err := phone.NewStream(phone.StreamOptions{}); err == nil {
		stream = streaming
	} else {
		logger.Warn("no stream credentials, numbers cannot be attached to a call", "error", err)
	}

	var recorder *routing.Recorder
	if pgStore != nil || liveClient != nil {
		recorder = routing.NewRecorder(routing.Phone, pgStore, liveClient, logger)
	}

	return phone.NewService(phone.ServiceOptions{
		Registry:  vendors.Registry(config),
		Store:     pgStore,
		Stream:    stream,
		Recorder:  recorder,
		PublicURL: os.Getenv(publicURLEnvVar),
		Logger:    logger,
	})
}
