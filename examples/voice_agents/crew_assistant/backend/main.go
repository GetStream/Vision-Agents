// Command backend is the half of the crew assistant that stays up on a server.
//
// It signs the Stream token the phone is its user with, as the Swift assistant's does, and it
// stands in for the restaurant's own systems: the schedule, the team, the handbook, the
// equipment guides, the repair tickets and the leaders' inbox. The phone's tools read and
// write them here, and the app's leader view reads the inbox. Everything lives in memory and
// starts from the same restaurant each time.
//
// It also makes the agent speak first. A clock-in, a clock-out or a return from time off is an
// event the restaurant's systems would send: here the app sends it, and this creates a
// scheduled task for the person's conversation and runs it at once, so the agent's message is
// the first thing in it.
//
// A demo has nobody signed in, so it acts for whichever user the phone names. A real one
// would act for the user its own sign-in vouched for.
//
//	STREAM_API_SECRET=<the secret of the Stream app the router runs in> \
//	STREAM_ACCELERATION_URL=http://localhost:8080 \
//	STREAM_ACCELERATION_CUSTOMER_ID=examples \
//	GOWORK=off go run ./backend
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"time"

	"github.com/golang-jwt/jwt/v5"

	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"
	"github.com/GetStream/Vision-Agents/sdks/go/stream"
)

// tokenValidity is how long a Stream token lasts. The SDK asks for a new one when it expires.
const tokenValidity = time.Hour

// agentName is the agent configure stored from this directory.
const agentName = "crew_assistant"

func main() {
	addr := flag.String("addr", ":8097", "where to listen")
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
	restaurant := newRestaurant(time.Now())

	mux := http.NewServeMux()
	mux.HandleFunc("POST /stream-token", streamToken(secret))
	mux.HandleFunc("POST /events", agentSpeaksFirst)
	restaurant.routes(mux)

	server := &http.Server{Addr: addr, Handler: logged(mux), ReadHeaderTimeout: 5 * time.Second}
	go func() {
		<-ctx.Done()
		closing, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = server.Shutdown(closing)
	}()

	log.Printf("minting Stream user tokens and serving Harbor Grill for %s on %s", agentName, addr)
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
		answer(w, map[string]string{"token": token})
	}
}

// events are what the restaurant's systems tell the agent about, by kind, and what the agent
// is told to do when one comes in.
var events = map[string]struct{ title, instructions string }{
	"clock_in": {
		title: "Pre-shift fit check",
		instructions: "The person is about to clock in. Greet them by first name, look up their next " +
			"shift with get_my_schedule, and give them the pre-shift fit check as interactive UI: " +
			"the uniform checklist and the focus areas for that shift.",
	},
	"clock_out": {
		title: "End of shift",
		instructions: "The person just clocked out. Thank them for the shift and give them the " +
			"end-of-shift reflection as interactive UI: a win, a challenge, a guest moment and " +
			"their energy.",
	},
	"back_from_time_off": {
		title: "While you were out",
		instructions: "The person is back from time off that started six days ago. Welcome them " +
			"back, call get_local_context, then get_updates_since with the day their time off " +
			"started, and summarize what they missed: what changes for them first, then team news.",
	},
	"onboarding": {
		title: "Your first 30 days",
		instructions: "The person started at Harbor Grill this week. Welcome them warmly, introduce " +
			"the leadership team and their onboarding buddy with get_team, say what their first " +
			"week looks like in three short points, and offer a quick menu quiz or booking a " +
			"check-in with their trainer.",
	},
}

// agentSpeaksFirst has the agent open the person's conversation about an event, by creating a
// scheduled task in it and running it now.
func agentSpeaksFirst(w http.ResponseWriter, r *http.Request) {
	var asked struct {
		UserID    string `json:"user_id"`
		SessionID string `json:"session_id"`
		Kind      string `json:"kind"`
	}
	if err := json.NewDecoder(r.Body).Decode(&asked); err != nil || asked.UserID == "" || asked.SessionID == "" {
		http.Error(w, "a user_id and a session_id are required", http.StatusBadRequest)
		return
	}
	event, known := events[asked.Kind]
	if !known {
		http.Error(w, "kind is one of clock_in, clock_out, back_from_time_off and onboarding", http.StatusBadRequest)
		return
	}
	ctx := r.Context()
	client, err := stream.Backend{ActingFor: asked.UserID}.Client()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	in := "P1D"
	created, err := client.CreateScheduledTaskWithResponse(ctx,
		&acceleration.CreateScheduledTaskParams{XStreamUserId: &asked.UserID},
		acceleration.CreateScheduledTaskJSONRequestBody{
			Title:        event.title,
			Instructions: event.instructions,
			Schedule:     acceleration.ScheduledTaskSchedule{In: &in},
			SessionId:    &asked.SessionID,
		})
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	task := created.JSON201
	if task == nil {
		http.Error(w, refused("creating the task", created.StatusCode(), created.Body), http.StatusBadGateway)
		return
	}
	ran, err := client.RunScheduledTaskWithResponse(ctx, task.Id)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	if ran.StatusCode() >= http.StatusMultipleChoices {
		http.Error(w, refused("running the task", ran.StatusCode(), ran.Body), http.StatusBadGateway)
		return
	}
	// The task was only a way to run once now: it goes once its run is well under way.
	go func(id string) {
		time.Sleep(2 * time.Minute)
		if deleted, err := client.DeleteScheduledTaskWithResponse(context.Background(), id); err != nil || deleted.StatusCode() >= http.StatusMultipleChoices {
			log.Printf("the %s task %s stays scheduled: %v", asked.Kind, id, err)
		}
	}(task.Id)
	answer(w, map[string]string{"task_id": task.Id})
}

// refused says what the router answered instead of the success it was asked for.
func refused(doing string, status int, body []byte) string {
	said := strings.TrimSpace(string(body))
	var envelope struct {
		Error struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if json.Unmarshal(body, &envelope) == nil && envelope.Error.Message != "" {
		said = envelope.Error.Message
	}
	return fmt.Sprintf("%s (%d): %s", doing, status, said)
}

func answer(w http.ResponseWriter, value any) {
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(value); err != nil {
		log.Printf("could not answer: %v", err)
	}
}

func logged(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		log.Printf("%s %s", r.Method, r.URL.Path)
		next.ServeHTTP(w, r)
	})
}
