package stream

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"
	"github.com/gorilla/websocket"
)

// routed is a stand-in for the modality half of the acceleration backend: the sockets and
// the recording jobs. It is a real HTTP server with a real WebSocket upgrader, so what is
// under test is the exchange rather than a description of it.
type routed struct {
	*httptest.Server

	mu sync.Mutex
	// opened is the start frame each modality socket was opened with.
	opened map[string]Frame
	// asked are the recording jobs the router was sent.
	asked []map[string]any
	// searched are the questions it was asked.
	searched []acceleration.SearchJSONRequestBody
	// failing makes a job fail, which is the other half of what a job does.
	failing bool
}

func newRouted(t *testing.T) *routed {
	t.Helper()

	backend := &routed{opened: map[string]Frame{}}
	mux := http.NewServeMux()

	mux.HandleFunc("GET /v1/{modality}/stream", func(w http.ResponseWriter, r *http.Request) {
		connection, err := (&websocket.Upgrader{}).Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer connection.Close()

		for {
			var frame Frame
			if err := connection.ReadJSON(&frame); err != nil {
				return
			}
			switch frame.Type() {
			case "start":
				backend.mu.Lock()
				backend.opened[r.PathValue("modality")] = frame
				backend.mu.Unlock()
			case "speak":
				_ = connection.WriteJSON(Frame{"type": "synthesis_complete"})
			case "respond":
				_ = connection.WriteJSON(Frame{"type": "delta", "text": "it "})
				_ = connection.WriteJSON(Frame{"type": "complete", "text": "it depends"})
			}
		}
	})

	mux.HandleFunc("POST /v1/stt/recordings", backend.accept)
	mux.HandleFunc("POST /v1/tts/recordings", backend.accept)

	mux.HandleFunc("GET /v1/stt/recordings/{id}", func(w http.ResponseWriter, _ *http.Request) {
		backend.mu.Lock()
		failing := backend.failing
		backend.mu.Unlock()

		reply(w, http.StatusOK, backend.transcription(failing))
	})

	mux.HandleFunc("GET /v1/tts/recordings/{id}", func(w http.ResponseWriter, _ *http.Request) {
		reply(w, http.StatusOK, map[string]any{
			"id": "recording-1", "status": "completed", "format": "mp3_44100_128",
			"audio": []byte{0xff, 0xfb, 0x90}, "characters": 12,
			"created_at": time.Now(), "updated_at": time.Now(),
		})
	})

	mux.HandleFunc("POST /v1/search", func(w http.ResponseWriter, r *http.Request) {
		var request acceleration.SearchJSONRequestBody
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		backend.mu.Lock()
		backend.searched = append(backend.searched, request)
		backend.mu.Unlock()

		reply(w, http.StatusOK, map[string]any{
			"provider": "exa", "model": "fast", "answer": "within the hour",
			"results": []map[string]any{{"url": "https://nice.org.uk/1"}},
		})
	})

	backend.Server = httptest.NewServer(mux)
	t.Cleanup(backend.Close)
	return backend
}

// accept takes a job and answers with the queued row, which is what makes it a job.
func (r *routed) accept(w http.ResponseWriter, request *http.Request) {
	var body map[string]any
	if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	r.mu.Lock()
	r.asked = append(r.asked, body)
	r.mu.Unlock()

	reply(w, http.StatusAccepted, map[string]any{
		"id": "recording-1", "status": "queued",
		"created_at": time.Now(), "updated_at": time.Now(),
	})
}

func (r *routed) transcription(failing bool) map[string]any {
	if failing {
		return map[string]any{
			"id": "recording-1", "status": "failed", "error": "no provider took the recording",
			"created_at": time.Now(), "updated_at": time.Now(),
		}
	}
	return map[string]any{
		"id": "recording-1", "status": "completed", "text": "a call costs a penny",
		"language": "en", "speakers": []string{"speaker_0"}, "provider": "deepgram",
		"created_at": time.Now(), "updated_at": time.Now(),
	}
}

// opening is the start frame a modality socket was opened with.
func (r *routed) opening(t *testing.T, modality string) Frame {
	t.Helper()

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		r.mu.Lock()
		frame, ok := r.opened[modality]
		r.mu.Unlock()
		if ok {
			return frame
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("no %s socket was opened", modality)
	return nil
}

func reply(w http.ResponseWriter, status int, body any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(body)
}

func routerFor(backend *routed) Router {
	return Router{
		Config:  "healthcare",
		Tags:    map[string]string{"project": "clinic"},
		Backend: Backend{URL: backend.URL, CustomerID: "acme"},
	}
}

func TestATranscriptionSocketOpensFromTheNamedConfig(t *testing.T) {
	backend := newRouted(t)
	yes := true

	transcriber, err := routerFor(backend).STT().Realtime(t.Context(),
		&acceleration.SttOptions{Diarize: &yes})
	if err != nil {
		t.Fatal(err)
	}
	defer transcriber.Close()

	opening := backend.opening(t, "stt")
	if opening.String("config_id") != "healthcare" {
		t.Errorf("the socket was opened with %+v", opening)
	}
	if !opening.Frame("stt").Bool("diarize") {
		t.Errorf("the override did not travel: %+v", opening["stt"])
	}
	if opening.Frame("tags").String("project") != "clinic" {
		t.Errorf("the cost labels did not travel: %+v", opening["tags"])
	}
}

func TestAVoiceSpeaksWhatItIsSentAndSaysWhenItIsDone(t *testing.T) {
	backend := newRouted(t)

	voice, err := routerFor(backend).TTS().Realtime(t.Context(), nil)
	if err != nil {
		t.Fatal(err)
	}
	defer voice.Close()

	if err := voice.Speak("hello there"); err != nil {
		t.Fatal(err)
	}
	if spoken := <-voice.Audio(); !spoken.Done {
		t.Errorf("the utterance did not finish: %+v", spoken)
	}
}

func TestAModelAnswersAsItIsWritten(t *testing.T) {
	backend := newRouted(t)

	model, err := routerFor(backend).LLM().Realtime(t.Context(), nil)
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()

	if err := model.Ask(Question{Messages: []Said{{Role: "user", Content: "how long?"}}}); err != nil {
		t.Fatal(err)
	}

	var written string
	for answer := range model.Answers() {
		written += answer.Delta
		if answer.Done {
			if answer.Text != "it depends" {
				t.Errorf("the answer is %q", answer.Text)
			}
			break
		}
	}
	if written != "it " {
		t.Errorf("the deltas arrived as %q", written)
	}
}

func TestARoutedSocketNeedsATargetSomewhere(t *testing.T) {
	backend := newRouted(t)
	router := Router{Backend: Backend{URL: backend.URL, CustomerID: "acme"}}

	if _, err := router.STT().Realtime(t.Context(), nil); err == nil {
		t.Fatal("a socket that names neither a config nor a target routed anyway")
	}
}

func TestARecordingIsWaitedForAndHandedBackWhole(t *testing.T) {
	backend := newRouted(t)
	yes := true

	transcript, err := routerFor(backend).STT().Recording(t.Context(),
		Recorded{URL: "https://example.test/call.mp3"},
		&acceleration.SttOptions{Words: &yes})
	if err != nil {
		t.Fatal(err)
	}

	if transcript.Text == nil || *transcript.Text != "a call costs a penny" {
		t.Errorf("the transcript is %+v", transcript)
	}
	if *transcript.Provider != "deepgram" {
		t.Errorf("nobody said who transcribed it: %+v", transcript)
	}

	asked := backend.asked[0]
	if asked["config_id"] != "healthcare" {
		t.Errorf("the job was sent as %+v", asked)
	}
	if source, _ := asked["source"].(map[string]any); source["url"] != "https://example.test/call.mp3" {
		t.Errorf("the recording was sent as %+v", asked["source"])
	}
}

func TestARecordingWithACallbackIsNotWaitedFor(t *testing.T) {
	backend := newRouted(t)

	job, err := routerFor(backend).STT().Recording(t.Context(),
		Recorded{URL: "https://example.test/call.mp3", Callback: "https://example.test/done"}, nil)
	if err != nil {
		t.Fatal(err)
	}

	if job.Status != acceleration.RecordingStatusQueued {
		t.Errorf("a caller being called back polled anyway: %s", job.Status)
	}
	if backend.asked[0]["callback"] != "https://example.test/done" {
		t.Errorf("the callback did not travel: %+v", backend.asked[0])
	}
}

func TestAFailedRecordingSaysWhatWentWrong(t *testing.T) {
	backend := newRouted(t)
	backend.failing = true

	_, err := routerFor(backend).STT().Recording(t.Context(),
		Recorded{URL: "https://example.test/call.mp3"}, nil)

	if err == nil {
		t.Fatal("a failed job was handed back as a transcript")
	}
}

func TestARecordingIsEitherAUrlOrTheAudioItself(t *testing.T) {
	backend := newRouted(t)

	if _, err := routerFor(backend).STT().Recording(t.Context(), Recorded{}, nil); err == nil {
		t.Fatal("a recording with nothing to transcribe was accepted")
	}
	_, err := routerFor(backend).STT().Recording(t.Context(),
		Recorded{URL: "https://example.test/call.mp3", Audio: []byte("clip")}, nil)
	if err == nil {
		t.Fatal("a recording that was both was accepted")
	}
}

func TestAWholeTextIsSpokenIntoOneFile(t *testing.T) {
	backend := newRouted(t)
	format := "mp3_44100_128"

	audiobook, err := routerFor(backend).TTS().Recording(t.Context(), "Chapter one.",
		&acceleration.TtsOptions{Format: &format})
	if err != nil {
		t.Fatal(err)
	}

	if *audiobook.Format != format {
		t.Errorf("the audio came back as %+v", audiobook)
	}
	if len(*audiobook.Audio) == 0 {
		t.Error("the file is empty")
	}
}

func TestASearchAnswersOutOfWhatIsTrueNow(t *testing.T) {
	backend := newRouted(t)
	results := 5

	found, err := routerFor(backend).Search(t.Context(), "perioperative antibiotic guidance",
		&acceleration.SearchOptions{Results: &results})
	if err != nil {
		t.Fatal(err)
	}

	if *found.Answer != "within the hour" || found.Results[0].Url != "https://nice.org.uk/1" {
		t.Errorf("the search answered %+v", found)
	}
	asked := backend.searched[0]
	if *asked.ConfigId != "healthcare" || *asked.Options.Results != results {
		t.Errorf("the question was asked as %+v", asked)
	}
}
