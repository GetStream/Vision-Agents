package api

import (
	"bytes"
	"encoding/json"
	"errors"
	"net/http"
	"reflect"
	"slices"
	"strconv"
	"strings"

	"github.com/danielgtaylor/huma/v2"
	"github.com/danielgtaylor/huma/v2/adapters/humachi"
	"github.com/go-chi/chi/v5"
	"gopkg.in/yaml.v3"
)

const specDescription = `Routes speech-to-text and text-to-speech traffic across providers and reports what it cost. Every path is scoped by modality, so the same provider serving two modalities is reported on separately. Who the caller is depends on ROUTER_AUTH_MODE: ` + "`api_key`" + `, the default, wants an API key and a token signed with its secret; ` + "`proxy`" + ` believes the X-Stream- headers something in front of the router set; ` + "`noauth`" + ` reads a trusted X-Customer-Id header and takes every caller for that customer's own backend.

Every operation is server-side only unless it is marked ` + "`x-client-accessible`" + `, and eight are: ` + "`createSession`, `querySessions`, `getSession`, `updateSession`, `stopSession`, `deleteSession`" + `, the session events socket and ` + "`search`" + `. Those are the whole of holding a conversation and looking something up, which is all an end user's device has any business doing. Everything else is refused with a 403 unless the caller is a process the customer runs, because a device holding a token its own backend minted may hold a conversation and may not rewrite the agent holding it. A server-side caller sends ` + "`Stream-Auth-Type: server`" + ` and a token carrying ` + "`server: true`" + ` and no ` + "`user_id`" + ` claim. Both are required, and the token is what proves it, since nothing signs the header.

The default is that way round because the cost of forgetting is asymmetric. An operation left unmarked is one nobody decided to open, and refusing it is a bug report; opening it silently is a breach.

An end user comes at one of three levels — anonymous, guest or authenticated — and an app may turn the first two away, in which case their requests are refused with a 403 whatever they ask for.

A session belongs to whoever opened it, and an end user reaches only their own. The session records the customer, the end user and which kind of caller that was, and one that does not match is told the session does not exist rather than that it may not have it. A caller presenting a verified token is a different owner from an anonymous one claiming the same name, so naming somebody else's user id gets an anonymous caller nowhere. The exception is a session a backend opened in a named user's name, which that user's own device reaches: a server-side caller may send X-Stream-User-Id to say who it is acting for, and is then held only to the daily quotas the customer set for that user, never to the router's default.

An end user's device is capped at a number of model responses and a number of tokens per UTC day, counted against both the ` + "`user_id`" + ` its token names and the address it came from. A caller with nothing left is answered 429 with a ` + "`Retry-After`" + ` naming the seconds until the limit resets; on a socket the same refusal arrives as an ` + "`error`" + ` frame, because the socket was already open when the day ran out. A server-side caller is held only to quotas the customer set: one naming a user counts toward that user's quota, and is refused only work that asks a model for something (a session, a turn, a run, a socket), never a read. The router's default allowances, which count devices only, are ROUTER_RATE_LIMIT_MESSAGES_PER_DAY and ROUTER_RATE_LIMIT_TOKENS_PER_DAY, and a deployment with no Redis to count in caps nothing. In noauth mode the end user is named by ` + "`X-Stream-User-Id`" + `, or by a ` + "`user_id`" + ` query parameter on a socket.
`

// Either the customer header, or an API key and the token signed with its secret. Which of
// the two a deployment accepts is ROUTER_AUTH_MODE, not a per-request choice: noauth reads
// the header and ignores keys, api_key verifies the pair and ignores the header.
var specSecurity = []map[string][]string{
	{"CustomerId": {}},
	{"ApiKey": {}, "AppToken": {}, "AuthType": {}},
}

var securitySchemes = map[string]*huma.SecurityScheme{
	"OpsKey": {
		Type: "apiKey",
		In:   "header",
		Name: opsKeyHeader,
		Description: "Stream staff's own key, for reviewing 10DLC use cases across apps. It is " +
			"held by Stream's internal tools and never by an app or a browser.",
	},
	"ManageKey": {
		Type: "apiKey",
		In:   "header",
		Name: manageKeyHeader,
		Description: "Stream staff's billing key, for price lists. Held by Stream's proxy, which " +
			"adds it to a staff member's call, or by a self-hosted router's operator, and never " +
			"by an app or a browser. A deployment without one serves no management operation.",
	},
	"CustomerId": {
		Type: "apiKey",
		In:   "header",
		Name: "X-Customer-Id",
		Description: "The customer, taken at face value. Read in noauth mode, where there is " +
			"nothing else to go on, and in proxy mode, where something in front has already " +
			"decided who the caller is. X-Stream-App-Id and X-Stream-Organization-Id say the " +
			"same thing with the organization around it. It is ignored entirely in api_key " +
			"mode, where the key decides and a header would only be a way around it.",
	},
	"ApiKey": {
		Type: "apiKey",
		In:   "header",
		Name: "X-Api-Key",
		Description: "The public half of a credential, which names the app and the secret to " +
			"verify with. A socket sends it as the api_key query parameter, because a browser " +
			"WebSocket cannot set a header.",
	},
	"AppToken": {
		Type:         "http",
		Scheme:       "bearer",
		BearerFormat: "JWT",
		Description: "A token signed HS256 with the secret belonging to the API key. A socket " +
			"sends it as the token query parameter. A token minted for an end user to hold " +
			"names them in a user_id claim; one a backend mints for itself carries server: " +
			"true instead, and is what every operation not marked x-client-accessible " +
			"requires. The user_id claim is also what owns the sessions that end user opens, " +
			"so a token is what stops one person reading another's conversation. A backend, " +
			"which names no user in its token, says which of its users it is acting for in " +
			"X-Stream-User-Id instead; that header is unsigned and needs to be, since a caller " +
			"holding the secret could mint a token for anyone. Beside it, X-Stream-User-Email says " +
			"the email that user signs in with, which the router keeps, the latest one per user, " +
			"for a connector that checks who consents (nessy). It is read from a server-side caller " +
			"only: an end user's device cannot set it.",
	},
	"AuthType": {
		Type: "apiKey",
		In:   "header",
		Name: "Stream-Auth-Type",
		Description: "Which kind of credential is presented: server for a process the " +
			"customer runs, jwt for a request made on an end user's behalf, which is what a " +
			"caller that names neither is taken to be. A socket in proxy mode, with no proxy " +
			"that declares kinds, may send jwt or anonymous as the stream-auth-type query " +
			"parameter instead, but never server, so a browser socket cannot claim to be a " +
			"backend. The header is the caller's " +
			"declaration and AppToken is the proof, so both have to say the same thing.",
	},
}

// sharedResponses are the error responses every operation describes in the same words, by
// status code and the name each is declared under.
var sharedResponses = map[string]string{
	"400": "BadRequest",
	"401": "Unauthorized",
	"402": "PaymentRequired",
	"403": "Forbidden",
	"404": "NotFound",
	"500": "InternalError",
	"503": "Unavailable",
}

var responseDescriptions = map[string]string{
	"BadRequest":   "The request was malformed",
	"Unauthorized": "The customer header is missing",
	"PaymentRequired": "The organization may not spend money now, so work that costs money is " +
		"refused: its credit is spent (insufficient_credits), its spend hard limit for the day " +
		"or the month is reached whatever its credit (system_limit_reached), or a budget is " +
		"spent (spend_limit_reached). The message says which, and for a limit when it resets. " +
		"Work on a model that costs nothing is still served.",
	"Forbidden": "The caller is known and may not do this. Usually the operation is " +
		"server-side only: it needs Stream-Auth-Type: server and a token carrying server: true, " +
		"which means it cannot be reached from an end user's device. With code " +
		"account_suspended, the organization is suspended, and every request is refused until " +
		"it is restored; the message says to contact support.",
	"NotFound": "No such modality, provider or shortcut",
	"InternalError": "Something went wrong that is not the caller's doing. The body says " +
		"only that: quote the response's X-Request-Id to find out more.",
	"Unavailable": "The request cannot be served now and may be retried. With code " +
		"spend_check_unavailable, the router could not share what the organization is spending " +
		"with the other routers for a while, so work that costs money is refused until it can; " +
		"Retry-After says when to try again, and work on a model that costs nothing is still served.",
}

// answerHeaders are what an answer to a customer's backend says of money, declared once under
// components/headers and referred to from every answer that can carry them.
var answerHeaders = map[string]*huma.Header{
	headerCostMicros: {
		Description: "What the request cost, in millionths of a dollar: the sum of the usage it recorded. " +
			"Sent on a success to a server-side caller when the request recorded any, and never to an end user's " +
			"device. An operation that answers before its work runs, such as createResponse, sends none.",
		Schema: &huma.Schema{Type: huma.TypeInteger, Format: "int64"},
	},
	headerCostEstimated: {
		Description: "true when some of X-Stream-Cost-Micros is the router's own count, because the " +
			"vendor never said what the work used. Absent otherwise.",
		Schema: &huma.Schema{Type: huma.TypeString, Enum: []any{"true"}},
	},
	headerBalanceMicros: {
		Description: "The organization's credit left, in millionths of a dollar, less what its work under " +
			"way holds: the figure admission refuses at. Sent on a success and on a refusal for money to " +
			"a server-side caller once the router has read the organization's account, and never to " +
			"an end user's device. A router that " +
			"bills nobody sends none.",
		Schema: &huma.Schema{Type: huma.TypeInteger, Format: "int64"},
	},
	headerBalanceWarning: {
		Description: "low when X-Stream-Balance-Micros is at or below the organization's warning " +
			"threshold (its policy's balance_warning_micros, else its tier's), exhausted at or below " +
			"0. Absent while the balance is above the threshold.",
		Schema: &huma.Schema{Type: huma.TypeString, Enum: []any{balanceLow, balanceExhausted}},
	},
}

// costHeaderNames are the headers an answer that recorded work carries, and balanceHeaderNames
// the ones every answer to a backend carries, a refusal for money included.
var (
	costHeaderNames    = []string{headerCostMicros, headerCostEstimated}
	balanceHeaderNames = []string{headerBalanceMicros, headerBalanceWarning}
)

// declareHeaders refers the response to the named headers of answerHeaders.
func declareHeaders(response *huma.Response, names ...string) {
	if response.Headers == nil {
		response.Headers = map[string]*huma.Header{}
	}
	for _, name := range names {
		response.Headers[name] = &huma.Header{Ref: "#/components/headers/" + name}
	}
}

// There is one Huma API per process, so its error constructor is set once for all of it.
func init() {
	huma.NewError = func(status int, message string, errs ...error) huma.StatusError {
		details := make([]string, 0, len(errs))
		for _, err := range withoutRefusedValues(errs) {
			details = append(details, err.Error())
		}
		if len(details) > 0 {
			message += ": " + strings.Join(details, "; ")
		}
		// The spec promises a 400 for a request that does not validate, where Huma's own
		// answer is a 422.
		if status == http.StatusUnprocessableEntity {
			failure := invalidRequest(message)
			failure.Code = codeValidationFailed
			return failure
		}
		return statusError(status, message)
	}
	huma.NewErrorWithContext = answerFailure
}

// withoutRefusedValues is errs as an answer or the request log may repeat them. A detail names
// the value it refused. The value of a secret is the secret, and for a field missing or
// unexpected the value is the whole object around it, whatever else it carries. Only a single
// value at a field that is not a secret, and short enough to read, is repeated back.
func withoutRefusedValues(errs []error) []error {
	kept := make([]error, 0, len(errs))
	for _, err := range errs {
		var detail *huma.ErrorDetail
		if errors.As(err, &detail) && (strings.Contains(strings.ToLower(detail.Location), "secret") || !scalar(detail.Value) || long(detail.Value)) {
			kept = append(kept, errors.New(detail.Message+" ("+detail.Location+")"))
			continue
		}
		kept = append(kept, err)
	}
	return kept
}

// refusedValueLimit is the longest refused value an error repeats. A body that is not JSON
// at all is refused as one string, and an upload's is megabytes of a customer's document
// that would go back in the answer and into the request log.
const refusedValueLimit = 200

// long reports whether a refused value is text too long to repeat.
func long(value any) bool {
	text, ok := value.(string)
	return ok && len(text) > refusedValueLimit
}

// scalar reports whether a value is one thing rather than an object or a list of them.
func scalar(value any) bool {
	if value == nil {
		return true
	}
	switch reflect.ValueOf(value).Kind() {
	case reflect.Map, reflect.Slice, reflect.Array, reflect.Struct, reflect.Pointer, reflect.Interface:
		return false
	}
	return true
}

// namedEnum declares a string enum as a schema of its own, where a Huma enum tag would
// repeat the values inline on every field using it, and returns the reference to it.
func namedEnum(registry huma.Registry, name, description string, values ...string) *huma.Schema {
	enum := make([]any, 0, len(values))
	for _, value := range values {
		enum = append(enum, value)
	}
	schema := &huma.Schema{Type: huma.TypeString, Description: description, Enum: enum}
	schema.PrecomputeMessages()
	registry.Map()[name] = schema
	return &huma.Schema{Ref: "#/components/schemas/" + name}
}

// optionalParam is a query parameter a request may leave out. Huma takes no pointer for one,
// and a handler still needs to tell absent from the zero value.
type optionalParam[T any] struct {
	Value T
	Set   bool
}

func (o optionalParam[T]) Schema(registry huma.Registry) *huma.Schema {
	schema := huma.SchemaFromType(registry, reflect.TypeFor[T]())
	schema.Nullable = false
	// A Go int is documented as a plain integer, which clients have always been generated
	// from; int64 says so itself.
	if reflect.TypeFor[T]().Kind() == reflect.Int {
		schema.Format = ""
	}
	return schema
}

func (o *optionalParam[T]) Receiver() reflect.Value {
	return reflect.ValueOf(o).Elem().Field(0)
}

func (o *optionalParam[T]) OnParamSet(set bool, _ any) {
	o.Set = set
}

// ptr is the value when the request named one, and nil when it left it out.
func (o optionalParam[T]) ptr() *T {
	if !o.Set {
		return nil
	}
	value := o.Value
	return &value
}

// itemLimit is a length bound on the items of an array, which a struct tag can only put on
// the array itself.
func itemLimit(length int) *int {
	return &length
}

// errorResponse is a failure an operation describes in words of its own, answered in the
// Error shape every other failure is.
func errorResponse(description string) *huma.Response {
	return &huma.Response{
		Description: description,
		Content: map[string]*huma.MediaType{
			"application/json": {Schema: &huma.Schema{Ref: "#/components/schemas/ErrorResponse"}},
		},
	}
}

// newAPI registers every operation declared in Go on router, and returns the API whose
// OpenAPI document describes them.
func (s *Server) newAPI(router chi.Router) huma.API {
	responses := make(map[string]*huma.Response, len(responseDescriptions))
	for name, description := range responseDescriptions {
		responses[name] = &huma.Response{
			Description: description,
			Content: map[string]*huma.MediaType{
				"application/json": {Schema: &huma.Schema{Ref: "#/components/schemas/ErrorResponse"}},
			},
		}
	}
	declareHeaders(responses["PaymentRequired"], balanceHeaderNames...)
	config := huma.Config{
		OpenAPI: &huma.OpenAPI{
			OpenAPI: "3.1.0",
			Info: &huma.Info{
				Title:       "Model Router",
				Version:     "0.2.0",
				Description: specDescription,
			},
			Servers:  []*huma.Server{{URL: "http://localhost:8080"}},
			Security: specSecurity,
			Components: &huma.Components{
				Schemas:         huma.NewMapRegistry("#/components/schemas/", huma.DefaultSchemaNamer),
				Responses:       responses,
				Headers:         answerHeaders,
				SecuritySchemes: securitySchemes,
			},
			OnAddOperation: []huma.AddOpFunc{shareErrorResponses},
		},
		Formats:       huma.DefaultFormats,
		DefaultFormat: "application/json",
	}
	// The generated operations ignore a field they do not know, and a client written
	// against a newer spec sends them.
	config.AllowAdditionalPropertiesByDefault = true

	api := humachi.New(router, config)
	api.UseMiddleware(s.requireStaffKey(api))
	api.UseMiddleware(requireCustomer(api))
	s.registerHealth(api)
	s.registerPolicies(api)
	s.registerUserPolicies(api)
	s.registerLimitGrants(api)
	s.registerLimits(api)
	s.registerAccount(api)
	s.registerPrices(api)
	s.registerSettings(api)
	s.registerSessionQuery(api)
	s.registerSessionUpdate(api)
	s.registerSessionStop(api)
	s.registerSessionVoice(api)
	s.registerSessionDelete(api)
	s.registerMemories(api)
	s.registerSessionTools(api)
	s.registerConfigs(api)
	s.registerServer(api)
	s.registerRouterconfigs(api)
	s.registerVoices(api)
	s.registerCustomModels(api)
	s.registerProviderKeys(api)
	s.registerScheduledTasks(api)
	s.registerKnowledge(api)
	s.registerKnowledgeurls(api)
	s.registerPhone(api)
	s.registerConversations(api)
	s.registerCampaigns(api)
	s.registerSimulations(api)
	s.registerCalls(api)
	s.registerReviews(api)
	s.registerSessionCreate(api)
	s.registerRuns(api)
	s.registerGuests(api)
	s.registerSessions(api)
	s.registerResponses(api)
	s.registerSearch(api)
	s.registerClassify(api)
	s.registerRecordings(api)
	s.registerImages(api)
	s.registerDetection(api)
	documentHandWritten(api)
	s.registerPlugins(api)
	s.registerConfigPatch(api)
	s.registerSync(api)
	s.registerAudit(api)
	s.registerConnectors(api)
	s.registerOAuthClients(api)
	s.registerConnectorProviderApps(api)
	s.registerEventDestinations(api)
	s.registerConnections(api)
	s.registerAuthorizations(api)
	s.registerConnectionTools(api)
	s.registerConnectionRecords(api)
	s.registerChannels(api)
	s.registerDLC(api)
	s.registerSuspensions(api)
	s.registerManage(api)
	s.registerPriceLists(api)
	return api
}

// requireCustomer answers a request nobody authenticated with a 401 before its input is
// read, so a caller with no credential is told that rather than what is wrong with its body.
// An operation declaring no security at all is reached before there is a caller to ask for.
func requireCustomer(api huma.API) func(huma.Context, func(huma.Context)) {
	return func(ctx huma.Context, next func(huma.Context)) {
		// A staff operation is answered for no customer: requireStaffKey let it through.
		public := ctx.Operation().Security != nil && len(ctx.Operation().Security) == 0 || staffOperation(ctx.Operation())
		if _, known := CustomerFrom(ctx.Context()); !known && !public {
			writeOperationError(ctx, errMissingCustomer)
			return
		}
		next(ctx)
	}
}

// shareErrorResponses points an operation's error responses at the shared ones, and folds
// Huma's 422 into the 400 its error constructor answers with instead.
func shareErrorResponses(_ *huma.OpenAPI, operation *huma.Operation) {
	if _, ok := operation.Responses["422"]; ok {
		delete(operation.Responses, "422")
		if _, declared := operation.Responses["400"]; !declared {
			operation.Responses["400"] = &huma.Response{}
		}
	}
	// Any operation can fail in a way that is not the caller's.
	if _, ok := operation.Responses["500"]; !ok {
		operation.Responses["500"] = &huma.Response{}
	}
	// Any operation a customer calls can be refused for money. A socket is refused in a frame,
	// after the upgrade.
	_, socket := operation.Responses["101"]
	public := operation.Security != nil && len(operation.Security) == 0
	if _, ok := operation.Responses["402"]; !ok && !socket && !public && !staffOperation(operation) {
		operation.Responses["402"] = &huma.Response{}
	}
	// Whatever can be refused for money can be refused while the spend check is unavailable, and
	// can record work: its answers to a backend say what it cost and what credit is left.
	if _, ok := operation.Responses["402"]; ok {
		if _, declared := operation.Responses["503"]; !declared {
			operation.Responses["503"] = &huma.Response{}
		}
		for code, response := range operation.Responses {
			if strings.HasPrefix(code, "2") && response.Ref == "" {
				declareHeaders(response, append(slices.Clone(costHeaderNames), balanceHeaderNames...)...)
			}
		}
	}
	for code, name := range sharedResponses {
		// A failure the operation describes in words of its own keeps them.
		response, ok := operation.Responses[code]
		if !ok {
			continue
		}
		status, _ := strconv.Atoi(code)
		if response.Description == "" || response.Description == http.StatusText(status) {
			operation.Responses[code] = &huma.Response{Ref: "#/components/responses/" + name}
		}
	}
}

// operationSummary is what the access checks need to know about one operation in the
// spec, whichever half of it declares the operation.
type operationSummary struct {
	method, path string
	// public is an operation declaring no security at all, reached before there is a
	// caller to classify.
	public bool
	// open is an operation marked x-client-accessible.
	open bool
}

// specifiedOperations lists every operation document declares.
func specifiedOperations(document *huma.OpenAPI) []operationSummary {
	var operations []operationSummary
	for path, item := range document.Paths {
		for method, operation := range map[string]*huma.Operation{
			http.MethodGet: item.Get, http.MethodPost: item.Post, http.MethodPut: item.Put,
			http.MethodPatch: item.Patch, http.MethodDelete: item.Delete,
		} {
			if operation == nil {
				continue
			}
			open, _ := operation.Extensions[clientAccessibleExtension].(bool)
			operations = append(operations, operationSummary{
				method: method,
				path:   path,
				public: operation.Security != nil && len(operation.Security) == 0,
				open:   open,
			})
		}
	}
	return operations
}

// Spec renders the router's whole OpenAPI document, which is what api/openapi.yaml holds.
func Spec() ([]byte, error) {
	rendered, err := (&Server{}).newAPI(chi.NewRouter()).OpenAPI().Downgrade()
	if err != nil {
		return nil, err
	}
	var document map[string]any
	decoder := json.NewDecoder(bytes.NewReader(rendered))
	// Numbers stay as written rather than turning into floats yaml prints as 1e+08.
	decoder.UseNumber()
	if err := decoder.Decode(&document); err != nil {
		return nil, err
	}
	tidy(document)

	var out bytes.Buffer
	out.WriteString("# Generated by cmd/openapi from the operations in internal/api. Do not edit.\n")
	encoder := yaml.NewEncoder(&out)
	encoder.SetIndent(2)
	if err := encoder.Encode(document); err != nil {
		return nil, err
	}
	if err := encoder.Close(); err != nil {
		return nil, err
	}
	return out.Bytes(), nil
}

// tidy rewrites what Huma rendered into what clients are generated from: a number as a
// number rather than the json.Number yaml would quote, and no additionalProperties: true.
// Huma writes that on every object, and client generators turn it into a catch-all map
// on the type; a schema that means it says so with a schema of its own.
func tidy(value any) any {
	switch value := value.(type) {
	case map[string]any:
		if allowed, ok := value["additionalProperties"].(bool); ok && allowed {
			delete(value, "additionalProperties")
		}
		for key, child := range value {
			value[key] = tidy(child)
		}
	case []any:
		for i, child := range value {
			value[i] = tidy(child)
		}
	case string:
		// Huma writes a base64 string's encoding as the 3.1 contentEncoding, which the
		// downgrade spells base64. OpenAPI 3.0, which clients are generated from, says byte.
		if value == "base64" {
			return "byte"
		}
	case json.Number:
		if integer, err := value.Int64(); err == nil {
			return integer
		}
		float, _ := value.Float64()
		return float
	}
	return value
}

// object returns the mapping under key, adding an empty one when there is none.
func object(parent map[string]any, key string) map[string]any {
	if child, ok := parent[key].(map[string]any); ok {
		return child
	}
	child := map[string]any{}
	parent[key] = child
	return child
}
