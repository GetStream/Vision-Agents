package stream

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"
)

// configured is a stand-in for the config half of the acceleration backend: list, create
// and update, keyed by name the way the real one is. It is a real HTTP server, so what a
// test sees stored is what the router would have received.
type configured struct {
	*httptest.Server

	mu sync.Mutex
	// stored are the configs written so far, in the order they were created.
	stored []acceleration.RouterConfig
}

func newConfigured(t *testing.T) *configured {
	t.Helper()

	backend := &configured{}
	mux := http.NewServeMux()

	mux.HandleFunc("GET /v1/router/configs", func(w http.ResponseWriter, _ *http.Request) {
		backend.mu.Lock()
		listed := append([]acceleration.RouterConfig{}, backend.stored...)
		backend.mu.Unlock()

		reply(w, http.StatusOK, listed)
	})

	mux.HandleFunc("POST /v1/router/configs", func(w http.ResponseWriter, r *http.Request) {
		wanted, ok := decodeConfig(w, r)
		if !ok {
			return
		}

		backend.mu.Lock()
		config := storedFrom(wanted, "config-"+strings.TrimSpace(wanted.Name))
		backend.stored = append(backend.stored, config)
		backend.mu.Unlock()

		reply(w, http.StatusCreated, config)
	})

	mux.HandleFunc("PUT /v1/router/configs/{id}", func(w http.ResponseWriter, r *http.Request) {
		wanted, ok := decodeConfig(w, r)
		if !ok {
			return
		}
		held := r.PathValue("id")

		backend.mu.Lock()
		config := storedFrom(wanted, held)
		for i, one := range backend.stored {
			if one.Id == held {
				backend.stored[i] = config
			}
		}
		backend.mu.Unlock()

		reply(w, http.StatusOK, config)
	})

	backend.Server = httptest.NewServer(mux)
	t.Cleanup(backend.Close)
	return backend
}

func decodeConfig(w http.ResponseWriter, r *http.Request) (acceleration.RouterConfigRequest, bool) {
	var wanted acceleration.RouterConfigRequest
	if err := json.NewDecoder(r.Body).Decode(&wanted); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return wanted, false
	}
	return wanted, true
}

func storedFrom(wanted acceleration.RouterConfigRequest, id string) acceleration.RouterConfig {
	return acceleration.RouterConfig{
		Id: id, Name: wanted.Name,
		Stt: wanted.Stt, Tts: wanted.Tts, Llm: wanted.Llm, Search: wanted.Search,
		Tags:      wanted.Tags,
		CreatedAt: time.Now(), UpdatedAt: time.Now(),
	}
}

// count is how many configs the backend is holding.
func (c *configured) count() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.stored)
}

func backendFor(server *configured) Backend {
	return Backend{URL: server.URL, CustomerID: "acme"}
}

func TestAConfigIsStoredUnderItsNameAndEditedNextTime(t *testing.T) {
	server := newConfigured(t)
	no := false

	stored, err := DefineRouter(t.Context(), backendFor(server), RouterOptions{
		Name: "healthcare",
		STT: &acceleration.SttOptions{
			Providers:  &[]string{"deepgram", "parakeet"},
			DataPolicy: &acceleration.DataPolicy{AllowTraining: &no},
		},
	})
	if err != nil {
		t.Fatalf("defining the config: %v", err)
	}
	if got := *stored.Stt.Providers; got[0] != "deepgram" || got[1] != "parakeet" {
		t.Fatalf("the priority list came back as %v", got)
	}

	again, err := DefineRouter(t.Context(), backendFor(server), RouterOptions{
		Name: "healthcare",
		STT:  &acceleration.SttOptions{Providers: &[]string{"grok"}},
	})
	if err != nil {
		t.Fatalf("redefining the config: %v", err)
	}

	if again.Id != stored.Id {
		t.Errorf("naming a config twice should edit the one that is there, got %s then %s",
			stored.Id, again.Id)
	}
	if server.count() != 1 {
		t.Errorf("the backend is holding %d configs, not 1", server.count())
	}
	if got := *again.Stt.Providers; len(got) != 1 || got[0] != "grok" {
		t.Errorf("the priority list came back as %v", got)
	}
}

func TestAConfigNeedsAName(t *testing.T) {
	server := newConfigured(t)

	_, err := DefineRouter(t.Context(), backendFor(server), RouterOptions{})

	if err == nil || !strings.Contains(err.Error(), "needs a name") {
		t.Fatalf("an unnamed config should be refused, got %v", err)
	}
}

func TestConfigureSTTCarriesTheOtherModalitiesForward(t *testing.T) {
	server := newConfigured(t)
	voice := "sonic"

	if _, err := DefineRouter(t.Context(), backendFor(server), RouterOptions{
		Name: "healthcare",
		TTS:  &acceleration.TtsOptions{Voice: &voice},
	}); err != nil {
		t.Fatalf("defining the config: %v", err)
	}

	router := Router{Config: "healthcare", Backend: backendFor(server)}
	stored, err := router.ConfigureSTT(t.Context(),
		&acceleration.SttOptions{Providers: &[]string{"deepgram"}})
	if err != nil {
		t.Fatalf("configuring transcription: %v", err)
	}

	if got := *stored.Stt.Providers; len(got) != 1 || got[0] != "deepgram" {
		t.Errorf("the priority list came back as %v", got)
	}
	if stored.Tts == nil || stored.Tts.Voice == nil || *stored.Tts.Voice != "sonic" {
		t.Error("writing how a config hears should not drop how it speaks")
	}
	if server.count() != 1 {
		t.Errorf("the backend is holding %d configs, not 1", server.count())
	}
}

func TestConfigureSTTNeedsANamedRouter(t *testing.T) {
	server := newConfigured(t)
	router := Router{Backend: backendFor(server)}

	_, err := router.ConfigureSTT(t.Context(), &acceleration.SttOptions{})

	if err == nil || !strings.Contains(err.Error(), "needs one") {
		t.Fatalf("an unnamed router should be refused, got %v", err)
	}
}

func TestADirectoryOfYamlBecomesOneConfigEach(t *testing.T) {
	server := newConfigured(t)
	directory := t.TempDir()

	write(t, directory, "healthcare.yaml", `
tags:
  team: clinical
stt:
  providers: [deepgram, parakeet]
  data_policy:
    allow_training: false
    retention: none
`)
	write(t, directory, "support.yaml", "name: support-desk\nstt:\n  providers: [grok]\n")

	stored, err := SyncRouters(t.Context(), backendFor(server), directory)
	if err != nil {
		t.Fatalf("syncing: %v", err)
	}

	if len(stored) != 2 {
		t.Fatalf("stored %d configs, not 2", len(stored))
	}
	if stored[0].Name != "healthcare" || stored[1].Name != "support-desk" {
		t.Errorf("a file names its config and its filename does when it does not, got %s and %s",
			stored[0].Name, stored[1].Name)
	}
	if got := *stored[0].Stt.Providers; got[0] != "deepgram" || got[1] != "parakeet" {
		t.Errorf("the priority list came back as %v", got)
	}
	if policy := stored[0].Stt.DataPolicy; policy == nil ||
		policy.AllowTraining == nil || *policy.AllowTraining ||
		policy.Retention == nil || *policy.Retention != "none" {
		t.Errorf("the data policy came back as %+v", stored[0].Stt.DataPolicy)
	}
	if tags := stored[0].Tags; tags == nil || (*tags)["team"] != "clinical" {
		t.Errorf("the tags came back as %v", stored[0].Tags)
	}
}

func TestSyncingTheSameDirectoryTwiceEditsWhatIsStored(t *testing.T) {
	server := newConfigured(t)
	directory := t.TempDir()

	write(t, directory, "healthcare.yaml", "stt:\n  providers: [deepgram]\n")
	first, err := SyncRouters(t.Context(), backendFor(server), directory)
	if err != nil {
		t.Fatalf("syncing: %v", err)
	}

	write(t, directory, "healthcare.yaml", "stt:\n  providers: [grok]\n")
	again, err := SyncRouters(t.Context(), backendFor(server), directory)
	if err != nil {
		t.Fatalf("syncing again: %v", err)
	}

	if again[0].Id != first[0].Id {
		t.Errorf("a second sync should edit rather than duplicate, got %s then %s",
			first[0].Id, again[0].Id)
	}
	if server.count() != 1 {
		t.Errorf("the backend is holding %d configs, not 1", server.count())
	}
	if got := *again[0].Stt.Providers; got[0] != "grok" {
		t.Errorf("the priority list came back as %v", got)
	}
}

func TestAMisspeltOptionInAFileIsRefused(t *testing.T) {
	server := newConfigured(t)
	directory := t.TempDir()
	write(t, directory, "healthcare.yaml", "stt:\n  diarise: true\n")

	_, err := SyncRouters(t.Context(), backendFor(server), directory)

	if err == nil || !strings.Contains(err.Error(), "diarise") {
		t.Fatalf("a misspelt option should be reported rather than stored, got %v", err)
	}
	if server.count() != 0 {
		t.Error("nothing should have been stored")
	}
}

func TestADirectoryWithNoYamlInItIsRefused(t *testing.T) {
	server := newConfigured(t)
	directory := t.TempDir()
	write(t, directory, "notes.txt", "nothing to route")

	_, err := SyncRouters(t.Context(), backendFor(server), directory)

	if err == nil || !strings.Contains(err.Error(), "no .yaml") {
		t.Fatalf("a directory with nothing to sync should say so, got %v", err)
	}
}

func write(t *testing.T, directory, name, contents string) {
	t.Helper()

	if err := os.WriteFile(filepath.Join(directory, name), []byte(contents), 0o600); err != nil {
		t.Fatalf("writing %s: %v", name, err)
	}
}
