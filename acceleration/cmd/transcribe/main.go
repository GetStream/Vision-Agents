// Command transcribe joins a LiveKit room as a bot, feeds every subscribed audio track
// through the STT router, and prints the transcripts to the terminal.
//
// The LiveKit SDK handles Opus decoding, resampling and jitter, so what reaches the
// router is already the 16 kHz mono PCM16 the providers want. That path is cgo, so
// pkg-config, libopus, libopusfile and libsoxr must be installed to build this.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/livekit/media-sdk"
	lksdk "github.com/livekit/server-sdk-go/v2"
	lkmedia "github.com/livekit/server-sdk-go/v2/pkg/media"
	"github.com/pion/webrtc/v4"

	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sttrouter"
)

const (
	urlEnvVar       = "LIVEKIT_URL"
	apiKeyEnvVar    = "LIVEKIT_API_KEY"
	apiSecretEnvVar = "LIVEKIT_API_SECRET"
	postgresEnvVar  = "ROUTER_POSTGRES_DSN"
	redisEnvVar     = "ROUTER_REDIS_ADDR"
	configEnvVar    = "ROUTER_CONFIG"
)

func main() {
	room := flag.String("room", "", "LiveKit room to join (required)")
	target := flag.String("target", "en-low-latency", "provider/model or capability shortcut")
	customerID := flag.String("customer", "demo", "customer the usage is billed to")
	identity := flag.String("identity", "stt-router-bot", "participant identity to join as")
	languages := flag.String("languages", "", "comma-separated language hints, e.g. en,es")
	verbose := flag.Bool("verbose", false, "log partial transcripts and lifecycle events")
	var tags routing.TagsFlag
	flag.Var(&tags, "tag", "cost label as key=value, repeat for several")
	flag.Parse()

	level := slog.LevelWarn
	if *verbose {
		level = slog.LevelDebug
	}
	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: level}))
	slog.SetDefault(logger)

	if err := run(*room, *target, *customerID, *identity, splitLanguages(*languages), tags.Tags, logger); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}

func run(
	roomName, target, customerID, identity string,
	languages []string,
	tags routing.Tags,
	logger *slog.Logger,
) error {
	if roomName == "" {
		return errors.New("-room is required")
	}

	url, apiKey, apiSecret := os.Getenv(urlEnvVar), os.Getenv(apiKeyEnvVar), os.Getenv(apiSecretEnvVar)
	if url == "" || apiKey == "" || apiSecret == "" {
		return fmt.Errorf("%s, %s and %s must be set", urlEnvVar, apiKeyEnvVar, apiSecretEnvVar)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	sttRouter, cleanup, err := buildRouter(ctx, logger)
	if err != nil {
		return err
	}
	defer cleanup()

	tracks := &trackSet{
		ctx:        ctx,
		router:     sttRouter,
		target:     target,
		customerID: customerID,
		tags:       tags,
		languages:  languages,
		logger:     logger,
		handle:     printEvents,
	}
	defer tracks.closeAll()

	callback := &lksdk.RoomCallback{
		ParticipantCallback: lksdk.ParticipantCallback{
			OnTrackSubscribed: func(track *webrtc.TrackRemote, _ *lksdk.RemoteTrackPublication, participant *lksdk.RemoteParticipant) {
				if track.Kind() != webrtc.RTPCodecTypeAudio {
					return
				}
				if err := tracks.add(track, participant); err != nil {
					fmt.Fprintf(os.Stderr, "could not transcribe %s: %v\n", participant.Identity(), err)
				}
			},
			OnTrackUnsubscribed: func(track *webrtc.TrackRemote, _ *lksdk.RemoteTrackPublication, _ *lksdk.RemoteParticipant) {
				tracks.remove(track.ID())
			},
		},
		OnDisconnected: stop,
	}

	room, err := lksdk.ConnectToRoom(url, lksdk.ConnectInfo{
		APIKey:              apiKey,
		APISecret:           apiSecret,
		RoomName:            roomName,
		ParticipantIdentity: identity,
	}, callback)
	if err != nil {
		return fmt.Errorf("join room %s: %w", roomName, err)
	}
	defer room.Disconnect()

	fmt.Printf("joined %s as %s, routing to %s. Ctrl-C to stop.\n", roomName, identity, target)

	<-ctx.Done()
	fmt.Println("\nleaving")
	return nil
}

// buildRouter wires the router, using Postgres and Redis when they are configured. The
// demo is useful without them: it just stops recording usage.
func buildRouter(ctx context.Context, logger *slog.Logger) (*sttrouter.Router, func(), error) {
	config, err := routing.LoadConfig(os.Getenv(configEnvVar))
	if err != nil {
		return nil, nil, err
	}
	section, ok := config[routing.STT]
	if !ok {
		return nil, nil, errors.New("the config declares no stt providers")
	}

	var closers []func()
	cleanup := func() {
		for i := len(closers) - 1; i >= 0; i-- {
			closers[i]()
		}
	}

	var pgStore *store.Store
	if dsn := os.Getenv(postgresEnvVar); dsn != "" {
		pgStore, err = store.Open(dsn)
		if err != nil {
			cleanup()
			return nil, nil, err
		}
		closers = append(closers, func() { pgStore.Close() })

		if err := pgStore.Migrate(ctx); err != nil {
			cleanup()
			return nil, nil, err
		}
	}

	var liveClient *live.Client
	if address := os.Getenv(redisEnvVar); address != "" {
		liveClient, err = live.New(live.Options{Address: address})
		if err != nil {
			cleanup()
			return nil, nil, err
		}
		closers = append(closers, liveClient.Close)
	}

	sttRouter, err := sttrouter.New(sttrouter.Options{
		Config:   section,
		Registry: sttrouter.DefaultRegistry(),
		Store:    pgStore,
		Live:     liveClient,
		Logger:   logger,
	})
	if err != nil {
		cleanup()
		return nil, nil, err
	}
	closers = append(closers, sttRouter.Close)

	return sttRouter, cleanup, nil
}

// trackSet keeps one router session per subscribed audio track.
type trackSet struct {
	ctx        context.Context
	router     *sttrouter.Router
	target     string
	customerID string
	tags       routing.Tags
	languages  []string
	logger     *slog.Logger
	// handle consumes a session's events for the lifetime of the track.
	handle func(*sttrouter.Session, stt.Participant)

	mu     sync.Mutex
	active map[string]*trackSession
}

type trackSession struct {
	session *sttrouter.Session
	track   *lkmedia.PCMRemoteTrack
}

func (t *trackSet) add(track *webrtc.TrackRemote, participant *lksdk.RemoteParticipant) error {
	session, err := t.router.Start(t.ctx, sttrouter.Request{
		CustomerID:    t.customerID,
		Tags:          t.tags,
		Target:        t.target,
		LanguageHints: t.languages,
	})
	if err != nil {
		return err
	}

	speaker := stt.Participant{
		ID:     participant.SID(),
		UserID: participant.Identity(),
		Name:   participant.Name(),
	}

	writer := &pcmWriter{session: session, participant: speaker}
	pcmTrack, err := lkmedia.NewPCMRemoteTrack(track, writer,
		lkmedia.WithTargetSampleRate(stt.SampleRate),
		lkmedia.WithTargetChannels(1),
	)
	if err != nil {
		session.Close()
		return fmt.Errorf("decode audio: %w", err)
	}

	go t.handle(session, speaker)

	t.mu.Lock()
	if t.active == nil {
		t.active = map[string]*trackSession{}
	}
	t.active[track.ID()] = &trackSession{session: session, track: pcmTrack}
	t.mu.Unlock()

	fmt.Printf("transcribing %s via %s/%s\n", speaker.UserID, session.Provider(), session.Model())
	return nil
}

func (t *trackSet) remove(trackID string) {
	t.mu.Lock()
	entry, ok := t.active[trackID]
	delete(t.active, trackID)
	t.mu.Unlock()

	if ok {
		entry.track.Close()
		entry.session.Close()
	}
}

func (t *trackSet) closeAll() {
	t.mu.Lock()
	entries := t.active
	t.active = nil
	t.mu.Unlock()

	for _, entry := range entries {
		entry.track.Close()
		entry.session.Close()
	}
}

// pcmWriter hands decoded audio to the router. The SDK has already resampled and
// downmixed it by this point.
type pcmWriter struct {
	session     *sttrouter.Session
	participant stt.Participant
}

func (w *pcmWriter) WriteSample(sample media.PCM16Sample) error {
	return w.session.ProcessAudio(stt.PcmData{
		Samples:    sample,
		SampleRate: stt.SampleRate,
		Channels:   1,
	}, w.participant)
}

func (w *pcmWriter) Close() error { return nil }

// printEvents prints transcripts as they arrive, marking partials so they are easy to tell
// apart from settled text.
func printEvents(session *sttrouter.Session, speaker stt.Participant) {
	for event := range session.Events() {
		switch typed := event.(type) {
		case stt.Transcript:
			timestamp := time.Now().Format("15:04:05")
			if typed.Final() {
				fmt.Printf("[%s] %s: %s\n", timestamp, speaker.UserID, typed.Text)
			} else {
				fmt.Printf("[%s] %s (partial): %s\n", timestamp, speaker.UserID, typed.Text)
			}
		case stt.Error:
			fmt.Fprintf(os.Stderr, "provider error (%s): %v\n", typed.Provider, typed.Err)
		}
	}
}

func splitLanguages(value string) []string {
	if strings.TrimSpace(value) == "" {
		return nil
	}

	parts := strings.Split(value, ",")
	languages := make([]string, 0, len(parts))
	for _, part := range parts {
		if trimmed := strings.TrimSpace(part); trimmed != "" {
			languages = append(languages, trimmed)
		}
	}
	return languages
}
