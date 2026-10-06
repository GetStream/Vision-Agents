package api

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/channels"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/danielgtaylor/huma/v2"
)

// noChannels is what the channel paths say on a deployment that cannot hold the credentials.
// Sealing them needs a key encryption key, and without one there is nowhere safe to put a
// WhatsApp token, so the line is refused rather than stored in the clear.
var noChannels = notConfigured("channels are not available: no key encryption key configured")

var unknownAccount = APIError{
	Type: ErrorTypeNotFound, Code: codeChannelAccountNotFound,
	Message: "no such channel account",
}

// connectChannel stores an app's credentials for a line and says where its provider should
// deliver.
func (s *Server) connectChannel(ctx context.Context, request *connectChannelRequest) (*channelAccountResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, missingCustomer
	}
	if s.store == nil {
		return nil, noConfigs
	}
	if s.secrets == nil {
		return nil, noChannels
	}
	if request.Body == nil {
		return nil, invalidRequest("a request body is required")
	}

	body := *request.Body
	kind := channels.Kind(body.Kind)
	number := strings.TrimSpace(body.Number)
	if !kind.Valid() {
		return nil, invalidRequest("no channel called " + body.Kind)
	}
	if message, ok := channelCredentialsComplaint(kind, body); !ok {
		return nil, invalidRequest(message)
	}
	// A number a vendor sells is the one the router can configure and bill, so an SMS line
	// has to be one the app actually holds rather than one it typed.
	if kind == channels.SMS {
		if message, ok := s.ownsNumber(ctx, customerID, number); !ok {
			return nil, invalidRequest(message)
		}
	}

	token, err := channelToken()
	if err != nil {
		return nil, err
	}
	account := store.ChannelAccount{
		CustomerID: customerID,
		Kind:       string(kind),
		E164:       number,
		AccountID:  strings.TrimSpace(value(body.AccountId)),
		Token:      token,
	}
	sealed, err := s.sealChannelSecrets(customerID, channels.Secrets{
		Token:     strings.TrimSpace(value(body.Token)),
		Signing:   strings.TrimSpace(value(body.Signing)),
		Challenge: strings.TrimSpace(value(body.Challenge)),
	})
	if err != nil {
		return nil, err
	}
	account.SecretsSealed = sealed
	account.SecretsKEKVersion = s.secrets.CurrentVersion()
	if err := s.store.SaveChannelAccount(ctx, &account); err != nil {
		return nil, invalidRequest(err.Error())
	}

	rendered := s.channelAccountOf(account)
	// A number bought here can be pointed at this router without anybody visiting a
	// dashboard, so it is, and the reply says the line is ready. WhatsApp and iMessage have
	// no such call: their webhook is set where the app lives, which is why the URL is shown.
	if telnyx, ok := channels.Telnyx(nil); ok && kind == channels.SMS {
		line, err := channels.Open(s.secrets, account)
		if err != nil {
			return nil, err
		}
		if err := telnyx.ConfigureMessaging(ctx, line, rendered.WebhookUrl); err != nil {
			return nil, invalidRequest("stored the credentials, but Telnyx refused to " +
				"deliver to " + rendered.WebhookUrl + ": " + err.Error())
		}
		rendered.Delivering = true
	}
	return &channelAccountResponse{Body: rendered}, nil
}

// receiveChannelMessage is the unauthenticated webhook a channel's provider delivers to. The
// token in the path names the line, and each delivery is checked against the signing secret
// that line was connected with, so it is that provider or nobody.
func (s *Server) receiveChannelMessage(w http.ResponseWriter, r *http.Request) {
	if s.channels == nil {
		writeError(w, gone("this deployment answers on no channels"))
		return
	}
	token := r.PathValue("token")
	// A GET is a provider checking the URL before it will deliver to it, which is WhatsApp's
	// way round: it asks for the verify token back along with a challenge to echo.
	if r.Method == http.MethodGet {
		writeChannelAnswer(w, s.channels.Challenge(r.Context(), token, r.URL.Query()))
		return
	}
	body, err := io.ReadAll(io.LimitReader(r.Body, channels.MaxDeliveryBytes+1))
	if err != nil {
		writeError(w, invalidRequest(err.Error()))
		return
	}
	if len(body) > channels.MaxDeliveryBytes {
		writeError(w, payloadTooLarge("a delivery is at most 256 KiB"))
		return
	}
	writeChannelAnswer(w, s.channels.Receive(r.Context(), token, r.Header, body))
}

// writeChannelAnswer answers a delivery. A challenge is echoed as itself rather than as JSON,
// because that is what the provider compares against what it sent.
func writeChannelAnswer(w http.ResponseWriter, answer channels.Answer) {
	if answer.Text != "" {
		w.Header().Set("Content-Type", "text/plain")
		w.WriteHeader(answer.Status)
		_, _ = w.Write([]byte(answer.Text))
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(answer.Status)
	_ = json.NewEncoder(w).Encode(answer.Body)
}

// linkChannelNumber mints a code somebody signed in texts from their phone to claim it. It is
// how an agent whose channels say identity: link learns which end user a number belongs to.
func (s *Server) linkChannelNumber(ctx context.Context, request *linkChannelRequest) (*channelLinkResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, missingCustomer
	}
	if s.channels == nil {
		return nil, noChannels
	}
	if request.Body == nil {
		return nil, invalidRequest("a request body is required")
	}

	config, err := s.configs.AgentConfig(ctx, customerID, request.Body.ConfigId)
	if err != nil {
		return nil, unknownConfig
	}
	if config.Channels.Identity != store.ChannelIdentityLink {
		return nil, invalidRequest(config.Name + " identifies a sender by their number, " +
			"so there is nothing to link: set channels.identity to link to need a code")
	}

	link, err := s.channels.Link(ctx, customerID, config.ID, request.Body.UserId)
	if err != nil {
		return nil, invalidRequest(err.Error())
	}
	return &channelLinkResponse{Body: ChannelLink{Code: link.Code, ExpiresAt: link.ExpiresAt}}, nil
}

// listChannelAccounts returns the lines the app has connected. The credentials are not among
// them: they are sent once and never read back, the way a password is.
func (s *Server) listChannelAccounts(ctx context.Context, _ *struct{}) (*listChannelAccountsResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, missingCustomer
	}
	if s.store == nil {
		return nil, noConfigs
	}

	found, err := s.store.ChannelAccounts(ctx, customerID)
	if err != nil {
		return nil, err
	}
	listed := make([]ChannelAccount, 0, len(found))
	for _, account := range found {
		listed = append(listed, s.channelAccountOf(account))
	}
	return &listChannelAccountsResponse{Body: listed}, nil
}

// disconnectChannel drops a line's credentials, so nothing is delivered or sent on it again.
func (s *Server) disconnectChannel(ctx context.Context, request *disconnectChannelRequest) (*struct{}, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, missingCustomer
	}
	if s.store == nil {
		return nil, noConfigs
	}

	found, err := s.store.ChannelAccounts(ctx, customerID)
	if err != nil {
		return nil, err
	}
	for _, account := range found {
		if account.ID != request.Id {
			continue
		}
		if err := s.store.DeleteChannelAccount(ctx, customerID, account.Kind, account.E164); err != nil {
			return nil, unknownAccount
		}
		return nil, nil
	}
	return nil, unknownAccount
}

// ownsNumber reports whether a number is one this app bought here.
func (s *Server) ownsNumber(ctx context.Context, customerID, number string) (string, bool) {
	held, err := s.store.Number(ctx, customerID, number)
	if err != nil {
		return "this app holds no number " + number + ": buy one with POST /v1/phone/numbers", false
	}
	for _, capability := range held.Capabilities {
		if capability == string(phone.SMS) {
			return "", true
		}
	}
	return number + " cannot carry text messages", false
}

// channelCredentialsComplaint reports what a line is missing, if anything. What each channel
// needs is the provider's, and saying so here is what keeps a half-connected line from being
// stored and then failing on the first message.
func channelCredentialsComplaint(kind channels.Kind, body ConnectChannelRequest) (string, bool) {
	switch kind {
	case channels.WhatsApp:
		switch {
		case value(body.AccountId) == "":
			return "whatsapp needs account_id: the phone number id from Meta", false
		case value(body.Token) == "":
			return "whatsapp needs token: a Meta access token for the WhatsApp Business account", false
		case value(body.Signing) == "":
			return "whatsapp needs signing: the app secret Meta signs deliveries with", false
		case value(body.Challenge) == "":
			return "whatsapp needs challenge: the verify token Meta's webhook setup echoes back", false
		}
	case channels.IMessage:
		switch {
		case value(body.Token) == "":
			return "imessage needs token: a Linq API key", false
		case value(body.Signing) == "":
			return "imessage needs signing: the whsec_ secret Linq signs deliveries with", false
		}
	case channels.SMS:
		switch {
		case value(body.Token) == "":
			return "sms needs token: a Telnyx API key", false
		case value(body.Signing) == "":
			return "sms needs signing: the Telnyx public key deliveries are signed with", false
		}
	}
	return "", true
}

// sealChannelSecrets wraps a line's credentials under this deployment's key, bound to the
// customer so ciphertext moved to another app's row does not open.
func (s *Server) sealChannelSecrets(customerID string, secrets channels.Secrets) ([]byte, error) {
	raw, err := json.Marshal(secrets)
	if err != nil {
		return nil, stack.Wrap(err)
	}
	return s.secrets.SealWithAAD(string(raw), []byte(customerID))
}

// channelAccountOf renders a line for the wire, with the URL its provider delivers to. That
// URL is the point of connecting: it is what is pasted into Meta's or Linq's setup.
func (s *Server) channelAccountOf(account store.ChannelAccount) ChannelAccount {
	rendered := ChannelAccount{
		Id:         account.ID,
		Kind:       ChannelKind(account.Kind),
		Number:     account.E164,
		WebhookUrl: strings.TrimSuffix(s.publicURL, "/") + channels.HookPath + account.Token,
		CreatedAt:  account.CreatedAt,
		UpdatedAt:  account.UpdatedAt,
	}
	rendered.AccountId = optional(account.AccountID)
	return rendered
}

// channelsComplaint reports what is wrong with the lines a config names, if anything.
//
// A line has to be one the app connected, because the credentials are what makes it
// answerable, and it has to be a line no other agent has claimed: two agents on one number
// would both answer every message to it.
func (s *Server) channelsComplaint(ctx context.Context, config store.AgentConfig) (string, bool) {
	unconnected, message, ok := s.channelsWarnings(ctx, config)
	if !ok {
		return message, false
	}
	if len(unconnected) > 0 {
		return unconnected[0], false
	}
	return "", true
}

// channelsWarnings is channelsComplaint for a sync, which stores a line the app has not
// connected yet and says so: a provider can take days to approve a number, and the repo
// declaring it should not stop syncing meanwhile. Nothing answers on such a line until
// it is connected, since a message arrives through the connected account.
func (s *Server) channelsWarnings(ctx context.Context, config store.AgentConfig) ([]string, string, bool) {
	named := config.Channels.Lines()
	if len(named) == 0 {
		return nil, "", true
	}
	if identity := config.Channels.Identity; identity != store.ChannelIdentityPhone && identity != store.ChannelIdentityLink {
		return nil, "channels.identity is phone or link, not " + identity, false
	}
	var unconnected []string
	for _, kind := range channels.Kinds {
		number, ok := named[string(kind)]
		if !ok {
			continue
		}
		if _, err := s.store.ChannelAccount(ctx, config.CustomerID, string(kind), number); errors.Is(err, store.ErrUnknownChannelAccount) {
			unconnected = append(unconnected, "channels."+string(kind)+": this app has not connected "+number+
				": connect it with POST /v1/agents/channels")
		} else if err != nil {
			return nil, err.Error(), false
		}
		claimed, found, err := s.store.ConfigOnChannel(ctx, config.CustomerID, string(kind), number)
		if err != nil {
			return nil, err.Error(), false
		}
		if found && claimed.ID != config.ID {
			return nil, "channels." + string(kind) + ": " + number + " already answers as " + claimed.Name, false
		}
	}
	return unconnected, "", true
}

// channelsOf reads the lines a caller named, defaulting how a sender is identified.
func channelsOf(declared *AgentChannels) store.AgentChannels {
	if declared == nil {
		return store.AgentChannels{}
	}
	named := store.AgentChannels{
		WhatsApp: channelLineOf(declared.Whatsapp),
		SMS:      channelLineOf(declared.Sms),
		IMessage: channelLineOf(declared.Imessage),
	}
	if len(named.Lines()) == 0 {
		return store.AgentChannels{}
	}
	// A number nobody has tied to an end user is a person of their own, which is what an
	// agent that keeps nothing personal wants and the safe thing for one that does.
	named.Identity = store.ChannelIdentityPhone
	if declared.Identity != nil {
		named.Identity = string(*declared.Identity)
	}
	return named
}

func channelLineOf(line *ChannelLineRequest) *store.ChannelLine {
	if line == nil || strings.TrimSpace(line.Number) == "" {
		return nil
	}
	return &store.ChannelLine{Number: strings.TrimSpace(line.Number)}
}

// renderedChannels is a config's lines as the API shows them, or nothing for none.
func renderedChannels(named store.AgentChannels) *AgentChannels {
	if len(named.Lines()) == 0 {
		return nil
	}
	identity := ChannelIdentity(named.Identity)
	return &AgentChannels{
		Whatsapp: renderedChannelLine(named.WhatsApp),
		Sms:      renderedChannelLine(named.SMS),
		Imessage: renderedChannelLine(named.IMessage),
		Identity: &identity,
	}
}

func renderedChannelLine(line *store.ChannelLine) *ChannelLineRequest {
	if line == nil {
		return nil
	}
	return &ChannelLineRequest{Number: line.Number}
}

// AgentChannels are the lines an agent answers on outside a Stream Chat channel.
type AgentChannels struct {
	Whatsapp *ChannelLineRequest `json:"whatsapp,omitempty"`
	Sms      *ChannelLineRequest `json:"sms,omitempty"`
	Imessage *ChannelLineRequest `json:"imessage,omitempty"`
	Identity *ChannelIdentity    `json:"identity,omitempty"`
}

func (*AgentChannels) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "The lines this agent answers on besides its Stream Chat channel. " +
		"Each names a number the app connected with POST /v1/agents/channels, and only one " +
		"agent may answer on a number. A message that arrives is answered in the sender's own " +
		"conversation, so what they say is kept and shown wherever the rest of it is."
	return schema
}

// ChannelLineRequest is one line an agent is reachable on.
type ChannelLineRequest struct {
	Number string `json:"number" minLength:"2" maxLength:"20" doc:"The number people write to, in E.164. It must be a line this app connected."`
}

// ChannelIdentity is how a sender on a channel becomes an end user.
type ChannelIdentity string

func (ChannelIdentity) Schema(registry huma.Registry) *huma.Schema {
	return namedEnum(registry, "ChannelIdentity",
		"How a sender becomes an end user. phone makes each number an end user of its own, "+
			"phone:+15551234567, so anybody who writes is answered. link answers only a number "+
			"somebody tied to an end user with a code from POST /v1/agents/channels/links, which "+
			"is what an agent reading a person's own calendar or orders needs. Omitted is phone.",
		store.ChannelIdentityPhone, store.ChannelIdentityLink)
}

// ChannelKind is which channel a line carries.
type ChannelKind string

func (ChannelKind) Schema(registry huma.Registry) *huma.Schema {
	return namedEnum(registry, "ChannelKind",
		"A channel a conversation can be carried over.",
		string(channels.WhatsApp), string(channels.SMS), string(channels.IMessage))
}

// ChannelAccount is one line the app has connected.
type ChannelAccount struct {
	Id     string      `json:"id"`
	Kind   ChannelKind `json:"kind"`
	Number string      `json:"number" doc:"The number people write to, in E.164."`
	// AccountId is shown because it is the provider's own name for the line, which is what
	// an operator matches against their dashboard. The credentials are not.
	AccountId  *string `json:"account_id,omitempty" doc:"The provider's own id for the line, such as WhatsApp's phone number id."`
	WebhookUrl string  `json:"webhook_url" doc:"Where the provider should deliver. Paste it into Meta's or Linq's webhook setup; a Telnyx number is pointed at it for you."`
	// Delivering says the router finished the setup itself, which only Telnyx allows.
	Delivering bool      `json:"delivering" doc:"True when the provider was pointed at the webhook URL for you, so there is nothing left to paste."`
	CreatedAt  time.Time `json:"created_at"`
	UpdatedAt  time.Time `json:"updated_at"`
}

func (*ChannelAccount) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "One line the app answers on: the number people write to and where " +
		"its provider delivers. The credentials are write-only, so they are never shown here."
	return schema
}

// ConnectChannelRequest is the credentials for one line.
type ConnectChannelRequest struct {
	Kind   string `json:"kind" minLength:"1" doc:"whatsapp, sms or imessage."`
	Number string `json:"number" minLength:"2" maxLength:"20" doc:"The number people write to, in E.164. For sms it must be a number this app bought with POST /v1/phone/numbers."`
	// The credentials are named for what they do, because the three providers need the same
	// things under their own names.
	AccountId *string `json:"account_id,omitempty" doc:"The provider's own id for the line. WhatsApp's phone number id; not needed by the others."`
	Token     *string `json:"token,omitempty" doc:"What authenticates a send: a Meta access token, a Telnyx API key, a Linq API key."`
	Signing   *string `json:"signing,omitempty" doc:"What the provider signs deliveries with: Meta's app secret, Telnyx's public key, Linq's whsec_ secret."`
	Challenge *string `json:"challenge,omitempty" doc:"The verify token Meta's webhook setup echoes back. WhatsApp only."`
}

func (*ConnectChannelRequest) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "Credentials for one line, connected once for the app and named by " +
		"any number of agents under channels in agent.yaml. Sending a line already connected " +
		"replaces its credentials and keeps its webhook URL."
	return schema
}

type connectChannelRequest struct {
	Body *ConnectChannelRequest
}

type channelAccountResponse struct {
	Body ChannelAccount
}

type listChannelAccountsResponse struct {
	Body []ChannelAccount
}

type disconnectChannelRequest struct {
	Id string `path:"id" doc:"The account, as the connect answered with."`
}

// registerChannels declares the channel account operations. They are server-side only: they
// carry a provider's credentials, which an end user's device has no business holding.
func (s *Server) registerChannels(api huma.API) {
	huma.Register(api, huma.Operation{
		OperationID: "connectChannel",
		Method:      http.MethodPost,
		Path:        "/v1/agents/channels",
		Summary:     "Connect a channel line",
		Description: "Stores an app's credentials for a WhatsApp, text or iMessage line and " +
			"answers with the URL its provider should deliver to. An agent is then reachable " +
			"there by naming the number under `channels` in its `agent.yaml`.\n\n" +
			"Sending a line that is already connected replaces its credentials and keeps its " +
			"webhook URL, so rotating a token does not mean setting the webhook up again.\n\n" +
			"Server-side only: it carries provider credentials.",
		Responses: map[string]*huma.Response{"200": {Description: "The line, with where to deliver"}},
		Errors:    []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden},
	}, s.connectChannel)
	huma.Register(api, huma.Operation{
		OperationID: "listChannelAccounts",
		Method:      http.MethodGet,
		Path:        "/v1/agents/channels",
		Summary:     "List channel lines",
		Description: "The lines this app has connected, oldest first. Credentials are never " +
			"read back.\n\nServer-side only.",
		Responses: map[string]*huma.Response{"200": {Description: "The lines"}},
		Errors:    []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden},
	}, s.listChannelAccounts)
	huma.Register(api, huma.Operation{
		OperationID:   "disconnectChannel",
		Method:        http.MethodDelete,
		Path:          "/v1/agents/channels/{id}",
		Summary:       "Disconnect a channel line",
		Description:   "Drops the line's credentials, so nothing is delivered or sent on it again.\n\nServer-side only.",
		DefaultStatus: http.StatusNoContent,
		Responses:     map[string]*huma.Response{"204": {Description: "Disconnected"}},
		Errors:        []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.disconnectChannel)
	huma.Register(api, huma.Operation{
		OperationID: "linkChannelNumber",
		Method:      http.MethodPost,
		Path:        "/v1/agents/channels/links",
		Summary:     "Mint a code to claim a number",
		Description: "Answers with a code to show somebody already signed in. The number that texts " +
			"it to the agent belongs to that end user from then on, which is what an agent " +
			"reading a person's own calendar or orders needs before it says a word. Only an " +
			"agent whose `channels.identity` is `link` has anything to link.\n\nServer-side only.",
		Responses: map[string]*huma.Response{"200": {Description: "The code and when it stops working"}},
		Errors:    []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.linkChannelNumber)
}

// ChannelLink is a code somebody texts to claim their number.
type ChannelLink struct {
	Code      string    `json:"code" doc:"The code to show the end user. They text it to the agent from the number they want answered."`
	ExpiresAt time.Time `json:"expires_at" doc:"When the code stops working."`
}

func (*ChannelLink) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "A code that ties the number texting it to one end user, for one agent."
	return schema
}

type linkChannelRequest struct {
	Body *LinkChannelRequest
}

// LinkChannelRequest asks for a code on behalf of somebody signed in.
type LinkChannelRequest struct {
	ConfigId string `json:"config_id" minLength:"1" doc:"The agent whose channels the number is being claimed on."`
	UserId   string `json:"user_id" minLength:"1" doc:"The end user the number will belong to."`
}

type channelLinkResponse struct {
	Body ChannelLink
}

// channelToken is a webhook's path segment.
func channelToken() (string, error) {
	raw := make([]byte, 24)
	if _, err := rand.Read(raw); err != nil {
		return "", stack.Wrap(err)
	}
	return hex.EncodeToString(raw), nil
}
