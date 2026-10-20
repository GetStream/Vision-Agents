// Command backend is the half of the Swift assistant that stays up on a server.
//
// It holds what a phone may not. The Stream secret signs the token the phone is its user
// with, as in the Swift demo. And only a server-side caller may reach a person's connections
// on the router, so the phone asks this to start its person's Gmail consent, to say whether
// Gmail is connected, to disconnect it, and to send a mail they approved. The send goes
// through the router's connection proxy: Gmail's MCP server writes drafts but cannot send,
// and the router adds the person's Google credential, which neither this nor the phone sees.
//
// A demo has nobody signed in, so it acts for whichever user the phone names. A real one
// would act for the user its own sign-in vouched for.
//
//	STREAM_API_SECRET=<the secret of the Stream app the router runs in> \
//	STREAM_ACCELERATION_URL=http://localhost:8080 \
//	STREAM_ACCELERATION_CUSTOMER_ID=examples \
//	go run ./backend
package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"mime"
	"net/http"
	"net/mail"
	"os"
	"os/signal"
	"strings"
	"time"

	"github.com/golang-jwt/jwt/v5"

	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"
	"github.com/GetStream/Vision-Agents/sdks/go/stream"
)

// tokenValidity is how long a Stream token lasts. The SDK asks for a new one when Stream says
// the one it has expired, so an hour-long call outlives it.
const tokenValidity = time.Hour

// gmail is the built-in connector the agent binds as each user's own.
const gmail = "gmail"

func main() {
	addr := flag.String("addr", ":8098", "where to listen")
	returnURL := flag.String("return-url", "io.getstream.visionagents.swiftassistant:/gmail",
		"where the browser goes once a Gmail consent ends: the app's own scheme, in reverse domain form, so its sign-in sheet closes")
	flag.Parse()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	if err := run(ctx, *addr, *returnURL); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Fatal(err)
	}
}

func run(ctx context.Context, addr, returnURL string) error {
	secret := os.Getenv("STREAM_API_SECRET")
	if secret == "" {
		return errors.New("STREAM_API_SECRET is needed: the secret of the Stream app the router runs in")
	}

	mux := http.NewServeMux()
	mux.HandleFunc("POST /stream-token", streamToken(secret))
	mux.HandleFunc("GET /gmail", gmailStatus)
	mux.HandleFunc("POST /gmail/connect", gmailConnect(returnURL))
	mux.HandleFunc("DELETE /gmail", gmailDisconnect)
	mux.HandleFunc("POST /gmail/send", gmailSend)

	server := &http.Server{Addr: addr, Handler: mux, ReadHeaderTimeout: 5 * time.Second}
	go func() {
		<-ctx.Done()
		closing, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = server.Shutdown(closing)
	}()

	log.Printf("minting Stream user tokens and reaching Gmail on %s", addr)
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

// gmailStatus says whether the user has Gmail connected.
func gmailStatus(w http.ResponseWriter, r *http.Request) {
	user := r.URL.Query().Get("user_id")
	if user == "" {
		http.Error(w, "a user_id is required", http.StatusBadRequest)
		return
	}
	connections, err := gmailConnections(r.Context(), user)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	connected := connectedOne(connections)
	answer(w, map[string]bool{"connected": connected != nil})
}

// gmailConnect starts a Gmail consent for the user: on their newest Gmail connection, or a
// new one. The phone opens the launch page with the handoff token in its fragment, and the
// router sends the browser back to returnURL once the person has answered Google.
func gmailConnect(returnURL string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var asked struct {
			UserID string `json:"user_id"`
		}
		if err := json.NewDecoder(r.Body).Decode(&asked); err != nil || asked.UserID == "" {
			http.Error(w, "a user_id is required", http.StatusBadRequest)
			return
		}
		ctx := r.Context()
		client, err := stream.Backend{ActingFor: asked.UserID}.Client()
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		connections, err := gmailConnections(ctx, asked.UserID)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}
		id := ""
		if len(connections) > 0 && connections[0].Id != nil {
			id = *connections[0].Id
		} else {
			created, err := client.CreateConnectionWithResponse(ctx, acceleration.ConnectionRequest{
				ConnectorId: gmail,
				Owner:       acceleration.ConnectionOwner{Type: acceleration.ConnectionOwnerTypeUser, UserId: &asked.UserID},
			})
			if err != nil {
				http.Error(w, err.Error(), http.StatusBadGateway)
				return
			}
			if created.JSON201 == nil || created.JSON201.Id == nil {
				http.Error(w, refused("creating a Gmail connection", created.StatusCode(), created.Body), http.StatusBadGateway)
				return
			}
			id = *created.JSON201.Id
		}
		started, err := client.CreateAuthorizationWithResponse(ctx, id, acceleration.AuthorizationRequest{ReturnUrl: &returnURL})
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}
		if started.JSON201 == nil {
			http.Error(w, refused("starting the Gmail consent", started.StatusCode(), started.Body), http.StatusBadGateway)
			return
		}
		answer(w, map[string]string{
			"launch_url":    started.JSON201.LaunchUrl,
			"handoff_token": started.JSON201.HandoffToken,
		})
	}
}

// gmailDisconnect deletes the user's Gmail connections, which revokes what Google granted.
func gmailDisconnect(w http.ResponseWriter, r *http.Request) {
	user := r.URL.Query().Get("user_id")
	if user == "" {
		http.Error(w, "a user_id is required", http.StatusBadRequest)
		return
	}
	ctx := r.Context()
	client, err := stream.Backend{ActingFor: user}.Client()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	connections, err := gmailConnections(ctx, user)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	for _, connection := range connections {
		if connection.Id == nil {
			continue
		}
		deleted, err := client.DeleteConnectionWithResponse(ctx, *connection.Id, nil)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}
		if deleted.StatusCode() >= http.StatusMultipleChoices {
			http.Error(w, refused("deleting a Gmail connection", deleted.StatusCode(), deleted.Body), http.StatusBadGateway)
			return
		}
	}
	w.WriteHeader(http.StatusNoContent)
}

// gmailSend sends a plain-text mail from the user's Gmail, through the router's proxy for
// their connection, which adds their Google credential on the way.
func gmailSend(w http.ResponseWriter, r *http.Request) {
	var asked struct {
		UserID  string `json:"user_id"`
		To      string `json:"to"`
		Subject string `json:"subject"`
		Body    string `json:"body"`
	}
	if err := json.NewDecoder(r.Body).Decode(&asked); err != nil || asked.UserID == "" {
		http.Error(w, "a user_id is required", http.StatusBadRequest)
		return
	}
	message, err := plainMail(asked.To, asked.Subject, asked.Body)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	ctx := r.Context()
	connections, err := gmailConnections(ctx, asked.UserID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	connected := connectedOne(connections)
	if connected == nil {
		http.Error(w, "Gmail is not connected: connect it from the app first", http.StatusConflict)
		return
	}
	sent, err := proxy(ctx, asked.UserID, *connected.Id, "gmail/v1/users/me/messages/send",
		map[string]string{"raw": base64.URLEncoding.EncodeToString(message)})
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	var gmailMessage struct {
		ID string `json:"id"`
	}
	if err := json.Unmarshal(sent, &gmailMessage); err != nil {
		http.Error(w, "gmail: the answer to a send did not read: "+err.Error(), http.StatusBadGateway)
		return
	}
	answer(w, map[string]string{"id": gmailMessage.ID})
}

// gmailConnections are the user's Gmail connections, newest first.
func gmailConnections(ctx context.Context, user string) ([]acceleration.Connection, error) {
	client, err := stream.Backend{ActingFor: user}.Client()
	if err != nil {
		return nil, err
	}
	owner, connector := acceleration.ConnectionOwnerTypeUser, gmail
	listed, err := client.ListConnectionsWithResponse(ctx, &acceleration.ListConnectionsParams{
		OwnerType: &owner, ConnectorId: &connector,
	})
	if err != nil {
		return nil, err
	}
	if listed.JSON200 == nil {
		return nil, errors.New(refused("listing Gmail connections", listed.StatusCode(), listed.Body))
	}
	if listed.JSON200.Items == nil {
		return nil, nil
	}
	return *listed.JSON200.Items, nil
}

// connectedOne is the first of connections that is connected, or nil.
func connectedOne(connections []acceleration.Connection) *acceleration.Connection {
	for _, connection := range connections {
		if connection.Status == acceleration.ConnectionStatusConnected && connection.Id != nil {
			return &connection
		}
	}
	return nil
}

// proxy posts body to path under the connection's api_base, as the user's backend.
func proxy(ctx context.Context, user, connection, path string, body any) ([]byte, error) {
	resolved, err := stream.Backend{ActingFor: user}.Resolve()
	if err != nil {
		return nil, err
	}
	credentials, err := resolved.Credentials()
	if err != nil {
		return nil, err
	}
	encoded, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost,
		resolved.URL+"/v1/agents/connections/"+connection+"/proxy/"+path, bytes.NewReader(encoded))
	if err != nil {
		return nil, err
	}
	request.Header = credentials
	request.Header.Set("Content-Type", "application/json")
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	read, err := io.ReadAll(io.LimitReader(response.Body, 1<<20))
	if err != nil {
		return nil, err
	}
	if response.StatusCode >= http.StatusMultipleChoices {
		return nil, errors.New(refused("sending through Gmail", response.StatusCode, read))
	}
	return read, nil
}

// plainMail is an RFC 5322 message of one plain-text part. The recipients are parsed and
// written back out, and the subject encoded, so neither can add a header of its own.
func plainMail(to, subject, body string) ([]byte, error) {
	recipients, err := mail.ParseAddressList(to)
	if err != nil {
		return nil, fmt.Errorf("gmail: who it is to does not read as addresses: %w", err)
	}
	written := make([]string, 0, len(recipients))
	for _, recipient := range recipients {
		written = append(written, recipient.String())
	}
	subject = strings.Join(strings.Fields(subject), " ")
	var message bytes.Buffer
	fmt.Fprintf(&message, "To: %s\r\n", strings.Join(written, ", "))
	fmt.Fprintf(&message, "Subject: %s\r\n", mime.QEncoding.Encode("utf-8", subject))
	message.WriteString("MIME-Version: 1.0\r\n")
	message.WriteString("Content-Type: text/plain; charset=\"UTF-8\"\r\n")
	message.WriteString("Content-Transfer-Encoding: 8bit\r\n\r\n")
	message.WriteString(strings.ReplaceAll(body, "\n", "\r\n"))
	return message.Bytes(), nil
}

// refused says what the router or Google answered instead of the success it was asked for:
// the message of the router's error envelope, which says what to fix, or else the body.
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
