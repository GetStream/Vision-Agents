package api

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"reflect"
	"strings"

	"github.com/danielgtaylor/huma/v2"
	"github.com/danielgtaylor/huma/v2/adapters/humachi"
	"github.com/go-chi/chi/v5"
	"gopkg.in/yaml.v3"

	specs "github.com/GetStream/Vision-Agents/acceleration/api"
)

const specDescription = `Routes speech-to-text and text-to-speech traffic across providers and reports what it cost. Every path is scoped by modality, so the same provider serving two modalities is reported on separately. Who the caller is depends on ROUTER_AUTH_MODE: ` + "`api_key`" + `, the default, wants an API key and a token signed with its secret; ` + "`proxy`" + ` believes the X-Stream- headers something in front of the router set; ` + "`noauth`" + ` reads a trusted X-Customer-Id header and takes every caller for that customer's own backend.

Every operation is server-side only unless it is marked ` + "`x-client-accessible`" + `, and six are: ` + "`createSession`, `listSessions`, `getSession`, `closeSession`" + `, the session events socket and ` + "`search`" + `. Those are the whole of holding a conversation and looking something up, which is all an end user's device has any business doing. Everything else is refused with a 403 unless the caller is a process the customer runs, because a device holding a token its own backend minted may hold a conversation and may not rewrite the agent holding it. A server-side caller sends ` + "`Stream-Auth-Type: server`" + ` and a token carrying ` + "`server: true`" + ` and no ` + "`user_id`" + ` claim. Both are required, and the token is what proves it, since nothing signs the header.

The default is that way round because the cost of forgetting is asymmetric. An operation left unmarked is one nobody decided to open, and refusing it is a bug report; opening it silently is a breach.

An end user comes at one of three levels — anonymous, guest or authenticated — and an app may turn the first two away, in which case their requests are refused with a 403 whatever they ask for.

A session belongs to whoever opened it, and an end user reaches only their own. The session records the customer, the end user and which kind of caller that was, and one that does not match is told the session does not exist rather than that it may not have it. A caller presenting a verified token is a different owner from an anonymous one claiming the same name, so naming somebody else's user id gets an anonymous caller nowhere. The exception is a session a backend opened in a named user's name, which that user's own device reaches: a server-side caller may send X-Stream-User-Id to say who it is acting for, and is charged no daily limit for doing so.

An end user's device is capped at a number of model responses and a number of tokens per UTC day, counted against both the ` + "`user_id`" + ` its token names and the address it came from. A caller with nothing left is answered 429 with a ` + "`Retry-After`" + ` naming the seconds until the limit resets; on a socket the same refusal arrives as an ` + "`error`" + ` frame, because the socket was already open when the day ran out. A server-side caller is not capped: a process the customer runs is trusted with the spend it was given. The two allowances are ROUTER_RATE_LIMIT_MESSAGES_PER_DAY and ROUTER_RATE_LIMIT_TOKENS_PER_DAY, and a deployment with no Redis to count in caps nothing. In noauth mode the end user is named by ` + "`X-Stream-User-Id`" + `, or by a ` + "`user_id`" + ` query parameter on a socket.
`

// Either the customer header, or an API key and the token signed with its secret. Which of
// the two a deployment accepts is ROUTER_AUTH_MODE, not a per-request choice: noauth reads
// the header and ignores keys, api_key verifies the pair and ignores the header.
var specSecurity = []map[string][]string{
	{"CustomerId": {}},
	{"ApiKey": {}, "AppToken": {}, "AuthType": {}},
}

var securitySchemes = map[string]*huma.SecurityScheme{
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
			"holding the secret could mint a token for anyone.",
	},
	"AuthType": {
		Type: "apiKey",
		In:   "header",
		Name: "Stream-Auth-Type",
		Description: "Which kind of credential is presented: server for a process the " +
			"customer runs, jwt for a request made on an end user's behalf, which is what a " +
			"caller that names neither is taken to be. It has no query parameter counterpart, " +
			"so a browser socket cannot claim to be a backend. The header is the caller's " +
			"declaration and AppToken is the proof, so both have to say the same thing.",
	},
}

// sharedResponses are the error responses every operation describes in the same words, by
// status code and the name each is declared under.
var sharedResponses = map[string]string{
	"400": "BadRequest",
	"401": "Unauthorized",
	"403": "Forbidden",
	"404": "NotFound",
}

var responseDescriptions = map[string]string{
	"BadRequest":   "The request was malformed",
	"Unauthorized": "The customer header is missing",
	"Forbidden": "The caller is known and this operation is server-side only. It needs " +
		"Stream-Auth-Type: server and a token carrying server: true, which means it cannot " +
		"be reached from an end user's device.",
	"NotFound": "No such modality, provider or shortcut",
}

// There is one Huma API per process, so its error constructor is set once for all of it.
func init() {
	huma.NewError = func(status int, message string, errs ...error) huma.StatusError {
		// The spec promises a 400 for a request that does not validate, where Huma's own
		// answer is a 422.
		if status == http.StatusUnprocessableEntity {
			status = http.StatusBadRequest
		}
		details := make([]string, 0, len(errs))
		for _, err := range errs {
			details = append(details, err.Error())
		}
		if len(details) > 0 {
			message += ": " + strings.Join(details, "; ")
		}
		return &apiError{status: status, Message: message}
	}
}

// apiError is how a Huma operation reports a failure, in the {"error": "..."} shape the
// generated operations and the sockets answer with.
type apiError struct {
	status  int
	Message string `json:"error"`
}

func (e *apiError) Error() string  { return e.Message }
func (e *apiError) GetStatus() int { return e.status }

// Schema documents the error as the Error schema rather than one of its own.
func (*apiError) Schema(registry huma.Registry) *huma.Schema {
	return registry.Schema(reflect.TypeFor[Error](), true, "")
}

// namedEnum declares a string enum as a schema of its own, where a Huma enum tag would
// repeat the values inline on every field using it, and returns the reference to it.
func namedEnum(registry huma.Registry, name, description string, values ...string) *huma.Schema {
	enum := make([]any, 0, len(values))
	for _, value := range values {
		enum = append(enum, value)
	}
	registry.Map()[name] = &huma.Schema{Type: huma.TypeString, Description: description, Enum: enum}
	return &huma.Schema{Ref: "#/components/schemas/" + name}
}

// newAPI registers every operation declared in Go on router, and returns the API whose
// OpenAPI document describes them.
func (s *Server) newAPI(router chi.Router) huma.API {
	responses := make(map[string]*huma.Response, len(responseDescriptions))
	for name, description := range responseDescriptions {
		responses[name] = &huma.Response{
			Description: description,
			Content: map[string]*huma.MediaType{
				"application/json": {Schema: &huma.Schema{Ref: "#/components/schemas/Error"}},
			},
		}
	}
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
	s.registerHealth(api)
	s.registerPolicies(api)
	return api
}

// shareErrorResponses points an operation's error responses at the shared ones, and folds
// Huma's 422 into the 400 its error constructor answers with instead.
func shareErrorResponses(_ *huma.OpenAPI, operation *huma.Operation) {
	if _, ok := operation.Responses["422"]; ok {
		delete(operation.Responses, "422")
		operation.Responses["400"] = &huma.Response{}
	}
	for code, name := range sharedResponses {
		if _, ok := operation.Responses[code]; ok {
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

// specifiedOperations lists every operation document declares in Go and every one
// generated from api/legacy.yaml, leaving out what oapi-codegen was told to skip.
func specifiedOperations(document *huma.OpenAPI) ([]operationSummary, error) {
	legacy, err := GetSpec()
	if err != nil {
		return nil, fmt.Errorf("api: could not read the embedded spec: %w", err)
	}
	var operations []operationSummary
	for path, item := range legacy.Paths.Map() {
		for method, operation := range item.Operations() {
			open, _ := operation.Extensions[clientAccessibleExtension].(bool)
			operations = append(operations, operationSummary{
				method: method,
				path:   path,
				public: operation.Security != nil && len(*operation.Security) == 0,
				open:   open,
			})
		}
	}
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
	return operations, nil
}

// Spec renders the router's whole OpenAPI document, which is what api/openapi.yaml holds:
// the operations declared in Go, and the ones api/legacy.yaml still describes by hand.
//
// A component both declare is taken from api/legacy.yaml. The Go type behind it is the one
// oapi-codegen generated from there, which renders the same schema without its words.
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
	var legacy map[string]any
	if err := yaml.Unmarshal(specs.Legacy, &legacy); err != nil {
		return nil, fmt.Errorf("api: could not read api/legacy.yaml: %w", err)
	}

	paths := object(document, "paths")
	for path, item := range object(legacy, "paths") {
		declared := object(paths, path)
		for method, operation := range item.(map[string]any) {
			if _, taken := declared[method]; taken {
				return nil, fmt.Errorf("api: %s %s is declared in Go and in api/legacy.yaml",
					strings.ToUpper(method), path)
			}
			declared[method] = operation
		}
	}
	components := object(document, "components")
	for section, entries := range object(legacy, "components") {
		declared := object(components, section)
		for name, entry := range entries.(map[string]any) {
			declared[name] = entry
		}
	}

	var out bytes.Buffer
	out.WriteString("# Generated by cmd/openapi from the operations in internal/api and api/legacy.yaml. Do not edit.\n")
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

// tidy rewrites what Huma rendered into what the hand-written half says: a number as a
// number rather than the json.Number yaml would quote, and no additionalProperties: true.
// Huma writes that on every object, and client generators turn it into a catch-all map
// on the type; a schema that means it says so in api/legacy.yaml.
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
