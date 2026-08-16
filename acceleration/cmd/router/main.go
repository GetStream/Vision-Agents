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

	"github.com/GetStream/Vision-Agents/acceleration/internal/api"
	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone/vendors"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
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
	defaultAddress    = ":8080"
	shutdownGrace     = 10 * time.Second
	readHeaderTimeout = 10 * time.Second
)

func main() {
	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelInfo}))
	slog.SetDefault(logger)

	if err := run(logger); err != nil {
		logger.Error("router stopped", "error", err)
		os.Exit(1)
	}
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
	}

	telephony, err := buildPhone(pgStore, liveClient, logger)
	if err != nil {
		return err
	}

	server, err := api.NewServer(api.Options{
		Routers: routers,
		Store:   pgStore,
		Live:    liveClient,
		Phone:   telephony,
		Logger:  logger,
	})
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
