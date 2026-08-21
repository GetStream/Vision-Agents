// Command router serves the model router's HTTP API.
package main

import (
	"context"
	"errors"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/agent/streamedge"
	"github.com/GetStream/Vision-Agents/acceleration/internal/api"
	"github.com/GetStream/Vision-Agents/acceleration/internal/campaign"
	"github.com/GetStream/Vision-Agents/acceleration/internal/chatlog"
	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge"
	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge/turbopuffer"
	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/memory"
	"github.com/GetStream/Vision-Agents/acceleration/internal/memory/mem0"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone/vendors"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sttrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/ttsrouter"
)

const (
	addressEnvVar     = "ROUTER_ADDR"
	postgresEnvVar    = "ROUTER_POSTGRES_DSN"
	redisEnvVar       = "ROUTER_REDIS_ADDR"
	configEnvVar      = "ROUTER_CONFIG"
	phoneConfigEnvVar = "ROUTER_PHONE_CONFIG"
	logLevelEnvVar    = "ROUTER_LOG_LEVEL"
	defaultAddress    = ":8080"
	shutdownGrace     = 10 * time.Second
	readHeaderTimeout = 10 * time.Second
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
	}

	if section, ok := config[routing.TTS]; ok {
		voice, err := ttsrouter.New(ttsrouter.Options{
			Config:   section,
			Registry: ttsrouter.DefaultRegistry(),
			Store:    pgStore,
			Live:     liveClient,
			Logger:   logger,
		})
		if err != nil {
			return err
		}
		defer voice.Close()
		routers[routing.TTS] = voice
		streams.TTS = voice
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
	sessions, err := buildSessions(streams, pgStore, liveClient, telephony, base, logger)
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

	// Reading a transcript back needs the same credentials writing one does. Without
	// them the calls are still listed; only what was said on them is missing.
	var transcripts *chatlog.Reader
	if reader, err := chatlog.NewReader(chatlog.ReaderOptions{}); err != nil {
		logger.Debug("transcripts will not be readable", "error", err)
	} else {
		transcripts = reader
	}

	options := api.Options{
		Routers:     routers,
		Store:       pgStore,
		Live:        liveClient,
		Phone:       telephony,
		Sessions:    sessions,
		Streams:     streams,
		Transcripts: transcripts,
		Campaigns:   campaigns,
		Logger:      logger,
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
		Registry: vendors.Registry(config),
		Store:    pgStore,
		Stream:   stream,
		Recorder: recorder,
		Logger:   logger,
	})
}
