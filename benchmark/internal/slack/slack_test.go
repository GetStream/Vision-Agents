package slack

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
)

// fakeSlack is the three upload methods of Slack's Web API, keeping what each was sent.
type fakeSlack struct {
	mu        sync.Mutex
	server    *httptest.Server
	tokens    []string
	uploads   map[string][]byte
	completed map[string]string
	// refuse is the error every method answers with, as Slack does for a missing scope.
	refuse string
}

func newFakeSlack(t *testing.T) *fakeSlack {
	f := &fakeSlack{uploads: map[string][]byte{}}
	mux := http.NewServeMux()
	mux.HandleFunc("/api/files.getUploadURLExternal", func(w http.ResponseWriter, r *http.Request) {
		if f.refused(w, r) {
			return
		}
		id := "F" + r.FormValue("filename")
		fmt.Fprintf(w, `{"ok":true,"upload_url":%q,"file_id":%q}`, f.server.URL+"/upload/"+id, id)
	})
	mux.HandleFunc("/upload/", func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		f.mu.Lock()
		f.uploads[strings.TrimPrefix(r.URL.Path, "/upload/")] = body
		f.mu.Unlock()
	})
	mux.HandleFunc("/api/files.completeUploadExternal", func(w http.ResponseWriter, r *http.Request) {
		if f.refused(w, r) {
			return
		}
		f.mu.Lock()
		f.completed = map[string]string{
			"files":           r.FormValue("files"),
			"channel_id":      r.FormValue("channel_id"),
			"initial_comment": r.FormValue("initial_comment"),
		}
		f.mu.Unlock()
		fmt.Fprint(w, `{"ok":true,"files":[]}`)
	})
	f.server = httptest.NewServer(mux)
	t.Cleanup(f.server.Close)
	return f
}

func (f *fakeSlack) refused(w http.ResponseWriter, r *http.Request) bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.tokens = append(f.tokens, r.Header.Get("Authorization"))
	if f.refuse == "" {
		return false
	}
	fmt.Fprintf(w, `{"ok":false,"error":%q,"needed":"files:write"}`, f.refuse)
	return true
}

func TestPostSendsTheFilesInOneMessageWithTheText(t *testing.T) {
	fake := newFakeSlack(t)
	client := Client{Token: "xoxb-test", BaseURL: fake.server.URL + "/api"}

	err := client.Post(t.Context(), "C123", "*Voicebench*", []File{
		{Name: "card.png", Title: "Scorecard", Data: []byte("png bytes")},
		{Name: "report.html", Title: "Full report", Data: []byte("<html>")},
	})
	if err != nil {
		t.Fatal(err)
	}

	if string(fake.uploads["Fcard.png"]) != "png bytes" || string(fake.uploads["Freport.html"]) != "<html>" {
		t.Fatalf("uploads: %q", fake.uploads)
	}
	if fake.completed["channel_id"] != "C123" || fake.completed["initial_comment"] != "*Voicebench*" {
		t.Fatalf("completed: %v", fake.completed)
	}
	var files []map[string]string
	if err := json.Unmarshal([]byte(fake.completed["files"]), &files); err != nil {
		t.Fatal(err)
	}
	if len(files) != 2 || files[0]["id"] != "Fcard.png" || files[0]["title"] != "Scorecard" || files[1]["id"] != "Freport.html" {
		t.Fatalf("files: %v", files)
	}
	for _, token := range fake.tokens {
		if token != "Bearer xoxb-test" {
			t.Fatalf("authorization %q", token)
		}
	}
}

func TestPostNamesTheMissingScope(t *testing.T) {
	fake := newFakeSlack(t)
	fake.refuse = "missing_scope"
	client := Client{Token: "xoxb-test", BaseURL: fake.server.URL + "/api"}

	err := client.Post(t.Context(), "C123", "hi", []File{{Name: "card.png", Data: []byte("png")}})

	if err == nil || !strings.Contains(err.Error(), "missing_scope (the app needs the files:write scope)") {
		t.Fatalf("error: %v", err)
	}
	if fake.completed != nil {
		t.Fatal("a message was posted after the upload failed")
	}
}

func TestPostSendsNothingWithoutAToken(t *testing.T) {
	fake := newFakeSlack(t)

	err := Client{BaseURL: fake.server.URL + "/api"}.Post(t.Context(), "C123", "hi", nil)

	if err == nil || !strings.Contains(err.Error(), "bot token") {
		t.Fatalf("error: %v", err)
	}
	if len(fake.tokens) != 0 {
		t.Fatal("Slack was called without a token")
	}
}
