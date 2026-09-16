// Command backend is the part of the Swift demo that holds the credential.
//
// The router serves a device five operations: opening a conversation, listing and closing
// one, the session events socket, and search. Everything else is server-side only, and a
// token for joining the agent's Stream call is one of the things it will not mint for a
// phone. So the phone asks this, and this asks the router.
//
// That is the shape a real app has anyway. A token is an authorisation, and deciding whether
// the person holding the phone may join a particular call is the application's decision, not
// the router's. Here the decision is "yes", because a demo has nobody to sign in; the comment
// where that happens says what a real one would do instead.
//
//	STREAM_ACCELERATION_URL=http://localhost:8080 \
//	STREAM_ACCELERATION_CUSTOMER_ID=examples \
//	go run ./backend
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"log"
	"net/http"
	"os"
	"os/signal"
	"time"

	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"
	"github.com/GetStream/Vision-Agents/sdks/go/stream"
)

func main() {
	addr := flag.String("addr", ":8099", "where to listen")
	flag.Parse()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	if err := run(ctx, *addr); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Fatal(err)
	}
}

func run(ctx context.Context, addr string) error {
	client, err := stream.Backend{}.Client()
	if err != nil {
		return err
	}

	mux := http.NewServeMux()
	mux.HandleFunc("POST /call-token", callToken(client))

	server := &http.Server{Addr: addr, Handler: mux, ReadHeaderTimeout: 5 * time.Second}
	go func() {
		<-ctx.Done()
		closing, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = server.Shutdown(closing)
	}()

	log.Printf("minting call tokens on %s", addr)
	return server.ListenAndServe()
}

// callToken answers the phone with credentials for the call a session is holding.
func callToken(client *acceleration.ClientWithResponses) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var asked struct {
			SessionID string `json:"session_id"`
		}
		if err := json.NewDecoder(r.Body).Decode(&asked); err != nil || asked.SessionID == "" {
			http.Error(w, "a session_id is required", http.StatusBadRequest)
			return
		}

		// Where a real application would authenticate the caller and decide whether they
		// may join this particular call. A demo has nobody signed in, so it decides yes.
		minted, err := client.CreateCallTokenWithResponse(
			r.Context(), asked.SessionID, acceleration.CallTokenRequest{})
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}
		if minted.JSON200 == nil {
			http.Error(w, minted.Status(), http.StatusBadGateway)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(minted.JSON200); err != nil {
			log.Printf("could not answer with a token: %v", err)
		}
	}
}
