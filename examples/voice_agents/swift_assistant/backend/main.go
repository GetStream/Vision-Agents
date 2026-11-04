// Command backend is the half of the Swift assistant that stays up on a server.
//
// It holds what a phone may not. The Stream secret signs the token the phone is its user
// with, as in the Swift demo. And only a server-side caller may reach a person's connections
// on the router, so the phone asks this to send a mail they approved. The send goes through
// the router's connection proxy: Gmail's MCP server writes drafts but cannot send, and the
// router adds the person's Google credential, which neither this nor the phone sees. The
// person connects Gmail in the chat, where the router asks for the sign-in itself.
//
// It also says how the agent answers, which the app's settings choose: in Markdown, or with
// interactive UI in A2UI or OpenUI. Writing an agent's config is server-side only, so this
// rewrites the agent's instructions: instructions.md, and for a UI format what teaches the
// model to write it with the app's components, a2ui/instructions.md or openui/instructions.md.
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
	"path/filepath"
	"strings"
	"time"

	"github.com/golang-jwt/jwt/v5"

	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"
	"github.com/GetStream/Vision-Agents/sdks/go/agents"
	"github.com/GetStream/Vision-Agents/sdks/go/stream"
)

// tokenValidity is how long a Stream token lasts. The SDK asks for a new one when Stream says
// the one it has expired, so an hour-long call outlives it.
const tokenValidity = time.Hour

// gmail is the built-in connector the agent binds as each user's own.
const gmail = "gmail"

// markdown is the reply format with no UI, which every client shows.
const markdown = "markdown"

// uiFormats are the formats the agent can write a reply's UI in, by what each is called in the
// heading that adds what teaches it to the agent's instructions.
var uiFormats = map[string]string{"a2ui": "A2UI", "openui": "OpenUI"}

func main() {
	addr := flag.String("addr", ":8098", "where to listen")
	dir := flag.String("dir", ".", "the agent directory configure stored the agent from")
	flag.Parse()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	if err := run(ctx, *addr, *dir); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Fatal(err)
	}
}

func run(ctx context.Context, addr, dir string) error {
	secret := os.Getenv("STREAM_API_SECRET")
	if secret == "" {
		return errors.New("STREAM_API_SECRET is needed: the secret of the Stream app the router runs in")
	}
	agent, err := agents.New(agents.Options{Dir: dir})
	if err != nil {
		return err
	}

	mux := http.NewServeMux()
	mux.HandleFunc("POST /stream-token", streamToken(secret))
	mux.HandleFunc("POST /gmail/send", gmailSend)
	mux.HandleFunc("GET /reply-format", replyFormat(agent))
	mux.HandleFunc("PUT /reply-format", setReplyFormat(agent, dir))

	server := &http.Server{Addr: addr, Handler: mux, ReadHeaderTimeout: 5 * time.Second}
	go func() {
		<-ctx.Done()
		closing, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = server.Shutdown(closing)
	}()

	log.Printf("minting Stream user tokens, reaching Gmail and setting how %s answers on %s", agent.Name(), addr)
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
		http.Error(w, "Gmail is not connected yet. Call gmail__list_tools, which asks them to connect it, and then send again.", http.StatusConflict)
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

// replyFormat answers with how the agent answers: the UI format its instructions teach it, or
// markdown when they teach it none.
func replyFormat(agent *agents.Agent) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		_, config, err := agentConfig(r.Context(), agent)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}
		format := markdown
		for id, name := range uiFormats {
			if config.Instructions != nil && strings.Contains(*config.Instructions, "\n"+uiHeading(name)+"\n") {
				format = id
			}
		}
		answer(w, map[string]string{"format": format})
	}
}

// setReplyFormat has the agent answer in the format asked for. A conversation keeps the
// instructions it started with, so the next one is the first to answer this way.
func setReplyFormat(agent *agents.Agent, dir string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var asked struct {
			Format string `json:"format"`
		}
		if err := json.NewDecoder(r.Body).Decode(&asked); err != nil || (asked.Format != markdown && uiFormats[asked.Format] == "") {
			http.Error(w, "a format is required: markdown, a2ui or openui", http.StatusBadRequest)
			return
		}
		instructions, err := instructionsFor(dir, asked.Format)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		ctx := r.Context()
		client, config, err := agentConfig(ctx, agent)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}
		patched, err := client.PatchAgentConfigWithResponse(ctx, config.Id,
			acceleration.PatchAgentConfigJSONRequestBody{Instructions: &instructions})
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}
		if patched.JSON200 == nil {
			http.Error(w, refused("changing the agent's instructions", patched.StatusCode(), patched.Body), http.StatusBadGateway)
			return
		}
		answer(w, map[string]string{"format": asked.Format})
	}
}

// instructionsFor are what the agent is told to answer in format: instructions.md, and for a UI
// format what teaches the model to write it, generated from the app's components by the
// format's prompt writer (a2ui-prompt or openui-prompt).
func instructionsFor(dir, format string) (string, error) {
	told, err := os.ReadFile(filepath.Join(dir, "instructions.md"))
	if err != nil {
		return "", err
	}
	if format == markdown {
		return string(told), nil
	}
	ui, err := os.ReadFile(filepath.Join(dir, format, "instructions.md"))
	if err != nil {
		return "", err
	}
	return string(told) + "\n\n" + uiHeading(uiFormats[format]) + "\n\n" + string(ui), nil
}

// uiHeading introduces what teaches the agent a UI format, and is how its instructions say which.
func uiHeading(name string) string {
	return "## Interactive UI in " + name
}

// agentConfig is the router's config of the agent configure stored from the directory.
func agentConfig(ctx context.Context, agent *agents.Agent) (*acceleration.ClientWithResponses, *acceleration.AgentConfig, error) {
	client, err := agent.Client()
	if err != nil {
		return nil, nil, err
	}
	name := agent.Name()
	listed, err := client.ListAgentConfigsWithResponse(ctx, &acceleration.ListAgentConfigsParams{Name: &name})
	if err != nil {
		return nil, nil, err
	}
	if listed.JSON200 == nil {
		return nil, nil, errors.New(refused("finding the agent", listed.StatusCode(), listed.Body))
	}
	if len(*listed.JSON200) == 0 {
		return nil, nil, fmt.Errorf("the router has no agent named %s: run go run ./configure", name)
	}
	return client, &(*listed.JSON200)[0], nil
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
