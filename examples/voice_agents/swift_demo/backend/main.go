// Command backend is the part of the Swift demo that holds the credential.
//
// The phone is one Stream user for everything it does: the router, and the agent's call it
// joins over Stream Video. Proving that is a Stream user token, signed with the app's secret,
// and a phone holds no secret. So the phone asks this for one, and hands it to setUser.
//
// That is the shape a real app has anyway. A token is an authorisation, and deciding who the
// person holding the phone is belongs to the application, not the router. Here the answer is
// "whoever they say", because a demo has nobody to sign in; the comment where that happens
// says what a real one would do instead.
//
//	STREAM_API_SECRET=<the secret of the Stream app the router runs in> go run ./backend
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

	"github.com/golang-jwt/jwt/v5"
)

// tokenValidity is how long a token lasts. The SDK asks for a new one when Stream says the
// one it has expired, so an hour-long call outlives it.
const tokenValidity = time.Hour

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
	secret := os.Getenv("STREAM_API_SECRET")
	if secret == "" {
		return errors.New("STREAM_API_SECRET is needed: the secret of the Stream app the router runs in")
	}

	mux := http.NewServeMux()
	mux.HandleFunc("POST /stream-token", streamToken(secret))

	server := &http.Server{Addr: addr, Handler: mux, ReadHeaderTimeout: 5 * time.Second}
	go func() {
		<-ctx.Done()
		closing, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = server.Shutdown(closing)
	}()

	log.Printf("minting Stream user tokens on %s", addr)
	return server.ListenAndServe()
}

// streamToken answers the phone with a Stream user token for the user it names.
func streamToken(secret string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var asked struct {
			UserID string `json:"user_id"`
		}
		if err := json.NewDecoder(r.Body).Decode(&asked); err != nil || asked.UserID == "" {
			http.Error(w, "a user_id is required", http.StatusBadRequest)
			return
		}

		// Where a real application would authenticate the caller and sign for the user they
		// signed in as. A demo has nobody signed in, so it signs for whoever is asked for.
		now := time.Now()
		token, err := jwt.NewWithClaims(jwt.SigningMethodHS256, jwt.MapClaims{
			"user_id": asked.UserID,
			"iat":     jwt.NewNumericDate(now),
			"exp":     jwt.NewNumericDate(now.Add(tokenValidity)),
		}).SignedString([]byte(secret))
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(map[string]string{"token": token}); err != nil {
			log.Printf("could not answer with a token: %v", err)
		}
	}
}
