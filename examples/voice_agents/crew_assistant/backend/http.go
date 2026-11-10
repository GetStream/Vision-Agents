package main

import (
	"context"
	"encoding/json"
	"errors"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/golang-jwt/jwt/v5"
)

// tokenValidity is how long a Stream token lasts. The SDK asks for a new one when it expires.
const tokenValidity = time.Hour

// server is what the phone reaches over HTTP: its Stream token, the agent speaking first, the
// three tools the phone asks the person about before running, and the leaders' inbox. Every
// route but the token answers only to a Stream token this backend signed, and acts for its user.
type server struct {
	secret []byte
	crew   *restaurant
	roles  roles
	// speak has the agent open the user's conversation about an event, and answers the task id.
	speak func(ctx context.Context, userID, sessionID, kind, timezone string) (string, error)
}

func (s *server) routes(mux *http.ServeMux) {
	mux.HandleFunc("POST /stream-token", s.streamToken)
	mux.HandleFunc("POST /events", s.signedIn(s.event))
	mux.HandleFunc("POST /crew/swaps", s.signedIn(s.swap))
	mux.HandleFunc("POST /crew/tickets", s.signedIn(s.ticket))
	mux.HandleFunc("POST /crew/check-ins", s.signedIn(s.checkIn))
	mux.HandleFunc("GET /crew/leader", s.signedIn(s.leaderView))
}

// streamToken stands in for the restaurant's own sign-in, which a real app has: it signs a
// Stream token for whichever user the phone names. Someone the restaurant hasn't met is enrolled
// as a new hire.
func (s *server) streamToken(w http.ResponseWriter, r *http.Request) {
	var asked struct {
		UserID string `json:"user_id"`
	}
	if err := json.NewDecoder(r.Body).Decode(&asked); err != nil || asked.UserID == "" {
		http.Error(w, "a user_id is required", http.StatusBadRequest)
		return
	}
	if err := s.join(r.Context(), asked.UserID); err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	now := time.Now()
	token, err := jwt.NewWithClaims(jwt.SigningMethodHS256, jwt.MapClaims{
		"user_id": asked.UserID,
		"iat":     jwt.NewNumericDate(now),
		"exp":     jwt.NewNumericDate(now.Add(tokenValidity)),
	}).SignedString(s.secret)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	answer(w, map[string]string{"token": token})
}

// join enrols userID as a new hire if the restaurant hasn't met them, and keeps their crew role
// on their Stream user.
func (s *server) join(ctx context.Context, userID string) error {
	if _, err := s.crew.member(userID); err == nil {
		return nil
	}
	name, _, _, err := s.roles.lookup(ctx, userID)
	if err != nil {
		return err
	}
	person, enrolled := s.crew.enrol(userID, name)
	if !enrolled {
		return nil
	}
	log.Printf("enrolled %s as a new hire", userID)
	return s.roles.publish(ctx, []Person{person})
}

// signedIn answers only a request carrying a Stream token this backend signed, for its user.
func (s *server) signedIn(next func(http.ResponseWriter, *http.Request, string)) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		userID, err := s.userOf(r)
		if err != nil {
			http.Error(w, err.Error(), http.StatusUnauthorized)
			return
		}
		if err := s.join(r.Context(), userID); err != nil {
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}
		next(w, r, userID)
	}
}

// userOf is the user of the request's bearer token.
func (s *server) userOf(r *http.Request) (string, error) {
	raw, found := strings.CutPrefix(r.Header.Get("Authorization"), "Bearer ")
	if !found || raw == "" {
		return "", errors.New("a Stream token is required, as Authorization: Bearer <token>")
	}
	claims := jwt.MapClaims{}
	if _, err := jwt.ParseWithClaims(raw, claims, func(*jwt.Token) (any, error) { return s.secret, nil },
		jwt.WithValidMethods([]string{jwt.SigningMethodHS256.Alg()})); err != nil {
		return "", errors.New("the Stream token isn't valid: " + err.Error())
	}
	userID, _ := claims["user_id"].(string)
	if userID == "" {
		return "", errors.New("the Stream token names no user")
	}
	return userID, nil
}

func (s *server) event(w http.ResponseWriter, r *http.Request, userID string) {
	var asked struct {
		SessionID string `json:"session_id"`
		Kind      string `json:"kind"`
		Timezone  string `json:"timezone"`
	}
	if err := json.NewDecoder(r.Body).Decode(&asked); err != nil || asked.SessionID == "" {
		http.Error(w, "a session_id is required", http.StatusBadRequest)
		return
	}
	if _, known := events[asked.Kind]; !known {
		http.Error(w, "kind is one of clock_in, clock_out, back_from_time_off and onboarding", http.StatusBadRequest)
		return
	}
	taskID, err := s.speak(r.Context(), userID, asked.SessionID, asked.Kind, asked.Timezone)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	answer(w, map[string]string{"task_id": taskID})
}

func (s *server) swap(w http.ResponseWriter, r *http.Request, userID string) {
	var asked struct {
		ShiftID    string `json:"shift_id"`
		TeammateID string `json:"teammate_id"`
		Note       string `json:"note"`
	}
	if !decode(w, r, &asked) {
		return
	}
	respond(w)(s.crew.swap(userID, asked.ShiftID, asked.TeammateID, asked.Note))
}

func (s *server) ticket(w http.ResponseWriter, r *http.Request, userID string) {
	var asked struct {
		Asset    string `json:"asset"`
		Summary  string `json:"summary"`
		Details  string `json:"details"`
		Priority string `json:"priority"`
	}
	if !decode(w, r, &asked) {
		return
	}
	respond(w)(s.crew.ticket(userID, asked.Asset, asked.Summary, asked.Details, asked.Priority))
}

func (s *server) checkIn(w http.ResponseWriter, r *http.Request, userID string) {
	var asked struct {
		With  string `json:"with"`
		When  string `json:"when"`
		Topic string `json:"topic"`
	}
	if !decode(w, r, &asked) {
		return
	}
	respond(w)(s.crew.checkIn(userID, asked.With, asked.When, asked.Topic))
}

// leaderView is the leaders' inbox, for a user whose Stream user says they lead.
func (s *server) leaderView(w http.ResponseWriter, r *http.Request, userID string) {
	_, role, _, err := s.roles.lookup(r.Context(), userID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	if role != roleLeader {
		http.Error(w, "only leaders read the leaders' inbox", http.StatusForbidden)
		return
	}
	answer(w, s.crew.leaderView())
}

func decode(w http.ResponseWriter, r *http.Request, into any) bool {
	if err := json.NewDecoder(r.Body).Decode(into); err != nil {
		http.Error(w, "the body isn't JSON: "+err.Error(), http.StatusBadRequest)
		return false
	}
	return true
}

// respond answers with what the restaurant answered, or the status its refusal says.
func respond(w http.ResponseWriter) func(any, error) {
	return func(value any, err error) {
		if err != nil {
			http.Error(w, err.Error(), statusOf(err))
			return
		}
		answer(w, value)
	}
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
