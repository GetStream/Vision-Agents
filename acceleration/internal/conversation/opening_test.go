package conversation

import (
	"net/http"
	"net/http/httptest"
	"net/http/httputil"
	"net/url"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation/chattest"
	getstream "github.com/GetStream/getstream-go/v5"
	"github.com/stretchr/testify/require"
)

// heldChat is Chat with one conversation's requests stopped in flight, which is what an
// open looks like from the service while it waits for an answer.
type heldChat struct {
	URL string
	// entered is closed once the held conversation's first request has arrived.
	entered chan struct{}
	release chan struct{}
	once    sync.Once
}

// holdingChat serves Chat from memory and holds every request naming cid until the test
// lets it go.
func holdingChat(t *testing.T, cid string) *heldChat {
	t.Helper()
	target, err := url.Parse(chattest.NewServer(t).URL)
	require.NoError(t, err)
	chat := &heldChat{entered: make(chan struct{}), release: make(chan struct{})}
	proxy := httputil.NewSingleHostReverseProxy(target)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.Contains(r.URL.Path, cid) {
			chat.once.Do(func() { close(chat.entered) })
			<-chat.release
		}
		proxy.ServeHTTP(w, r)
	}))
	t.Cleanup(server.Close)
	chat.URL = server.URL
	return chat
}

// TestOpeningOneConversationDoesNotWaitOnAnother: opening a conversation asks Chat several
// times, and a conversation whose answers never come must not stop anybody else's from
// opening. Every open used to take one service-wide lock across those requests, which made
// every conversation opened anywhere queue behind every other.
func TestOpeningOneConversationDoesNotWaitOnAnother(t *testing.T) {
	chat := holdingChat(t, "held-conversation")
	client, err := getstream.NewClient("test", "secret", getstream.WithBaseUrl(chat.URL))
	require.NoError(t, err)
	service := NewForChat(client)
	t.Cleanup(func() { close(chat.release) })

	stuck := make(chan error, 1)
	go func() {
		_, _, _, err := service.CreateInApp(t.Context(), 0, "customer", "agent", "agent:held-conversation", "", "", nil)
		stuck <- err
	}()
	select {
	case <-chat.entered:
	case err := <-stuck:
		t.Fatalf("the held conversation finished instead of waiting on Chat: %v", err)
	case <-time.After(10 * time.Second):
		t.Fatal("the held conversation never reached Chat")
	}

	opened := make(chan error, 1)
	go func() {
		_, _, _, err := service.CreateInApp(t.Context(), 0, "customer", "agent", "agent:other-conversation", "", "", nil)
		opened <- err
	}()
	select {
	case err := <-opened:
		require.NoError(t, err)
	case <-time.After(10 * time.Second):
		t.Fatal("a second conversation waited on the held one")
	}
}

// TestTwoOpensOfOneConversationAreTakenInTurn: the one that gets there first holds it and
// the second is told so, rather than both reading the history and both believing they have
// it.
func TestTwoOpensOfOneConversationAreTakenInTurn(t *testing.T) {
	service := NewForChat(chattest.NewServer(t).Client)

	var wg sync.WaitGroup
	errs := make([]error, 2)
	for i := range errs {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, _, _, errs[i] = service.CreateInApp(t.Context(), 0, "customer", "agent", "agent:one-conversation", "", "", nil)
		}()
	}
	wg.Wait()

	require.Condition(t, func() bool {
		return (errs[0] == nil) != (errs[1] == nil)
	}, "exactly one open holds the conversation, got %v and %v", errs[0], errs[1])
	for _, err := range errs {
		if err != nil {
			require.ErrorContains(t, err, "already open")
		}
	}
}
