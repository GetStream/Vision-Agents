// Package stream is the client half of the acceleration backend: a whole voice or text
// pipeline that runs there rather than here, and the phone numbers it answers on.
package stream

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"
	"regexp"
	"slices"
	"strings"
	"time"

	"github.com/golang-jwt/jwt/v5"

	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"
)

const (
	// URLEnv points this SDK at another router, such as one running locally.
	URLEnv = "STREAM_ACCELERATION_URL"
	// RegionEnv holds this SDK to the hosted router in one region, such as us-east4.
	RegionEnv = "STREAM_ACCELERATION_REGION"
	// CustomerEnv names who the work is billed to on a router with no proxy in front of it,
	// which is one run locally. Setting it is what turns authentication off.
	CustomerEnv = "STREAM_ACCELERATION_CUSTOMER_ID"
	// APIKeyEnv and APISecretEnv are the Stream app a hosted router is reached as. The
	// proxy in front of one authenticates the app rather than trusting the header below.
	APIKeyEnv    = "STREAM_API_KEY"
	APISecretEnv = "STREAM_API_SECRET"
	// CustomerHeader carries the identity every request and every cost row is keyed by.
	CustomerHeader = "X-Customer-Id"
	// APIKeyHeader names the Stream app a request is made for.
	APIKeyHeader = "api_key"
	// AuthTypeHeader says how the token is meant to be read. AuthTypeJWT is the only answer
	// the proxy accepts; anonymous access is refused outright.
	//
	// It is only ever sent to the proxy, which verifies the token and then strips this
	// header before the router sees it. Sent to a router directly it would mean the
	// opposite of what is wanted: with nothing in front to have checked it, a router reads
	// this as a caller acting for an end user's device, and a device may hold a
	// conversation but not rewrite the agent holding it.
	AuthTypeHeader = "stream-auth-type"
	AuthTypeJWT    = "jwt"
	// AuthorizationHeader carries the token signed with the app's secret.
	AuthorizationHeader = "authorization"
	// UserHeader names the end user a server-side caller is acting for.
	UserHeader = "X-Stream-User-Id"
	// ClientHeader says which client a request came from, which is what the router's audit
	// log records a configuration change as having been made with. An SDK syncing an agent
	// directory is this one; the dashboard and the CLI name themselves.
	ClientHeader = "X-Stream-Client"
	ClientSDK    = "sdk"
	// UsageCategoriesHeader is where a caller the proxy is not in front of names its usage
	// categories, its team among them as team:<name>, joined with commas. Behind the proxy
	// they are claims on the token instead, and the proxy deletes this header.
	UsageCategoriesHeader = "X-Stream-Usage-Categories"
	// MaxUsageCategories is the most usage categories a caller may name, its team included.
	MaxUsageCategories = 10
	// DefaultURL is the hosted router in us-east4. STREAM_ACCELERATION_URL or
	// STREAM_ACCELERATION_REGION moves it.
	DefaultURL = "https://accelerate-us-east4.gcp.stream-io-api.com"
)

// tokenValidity is how long a minted token lasts. Short, because it is minted per request
// and a stolen one should stop working sooner than the credential behind it.
const tokenValidity = time.Hour

// ErrNoCredential is returned when there is no app credential to authenticate with. Falling
// back to the customer header instead would send a request the proxy refuses, and report it
// as whatever the proxy says rather than as what it is.
// region is the shape of a region name, such as us-east4 or europe-west4. Only the shape
// is checked: which regions are served is the hosted router's to say, so offering another
// needs no release of this SDK.
var region = regexp.MustCompile(`^[a-z]+-[a-z]+[0-9]+$`)

// A team is a usage category of its own, team:<name>, which is why its name is shorter and
// has no ':'.
const teamPrefix = "team:"

var (
	teamName     = regexp.MustCompile(`^[A-Za-z0-9_.-]{1,59}$`)
	categoryName = regexp.MustCompile(`^[A-Za-z0-9_.:-]{1,64}$`)
)

// RegionURL is the hosted router held to one region: requests to it are served there and
// nowhere else, even when that region is full or down.
func RegionURL(name string) (string, error) {
	if !region.MatchString(name) {
		return "", fmt.Errorf("stream: %q is not a region, such as us-east4", name)
	}
	return "https://accelerate-" + name + ".gcp.stream-io-api.com", nil
}

// hosted says whether url is the hosted router in some region, which is always behind
// the proxy.
func hosted(url string) bool {
	name, ok := strings.CutPrefix(url, "https://accelerate-")
	if !ok {
		return false
	}
	name, ok = strings.CutSuffix(name, ".gcp.stream-io-api.com")
	return ok && region.MatchString(name)
}

var ErrNoCredential = errors.New("stream: set " + APIKeyEnv + " and " + APISecretEnv +
	", or " + CustomerEnv + " for a local router")

// Backend is where the acceleration router is, and who is calling it.
type Backend struct {
	// URL is the router's base URL. Empty falls back to Region, then to
	// STREAM_ACCELERATION_URL, then to STREAM_ACCELERATION_REGION, then to the hosted router.
	URL string
	// Region holds this SDK to the hosted router in one region, such as us-east4. It
	// can be set together with URL only when URL names that region.
	Region string
	// CustomerID is the identity every request and every cost row is keyed by. Empty falls
	// back to STREAM_ACCELERATION_CUSTOMER_ID. A router behind the authenticating proxy
	// works this out from the credential instead and ignores what it is told.
	CustomerID string
	// Authenticate says the router is reached through the proxy, so requests carry the app
	// credential below rather than naming a customer. It is on whenever no customer id is
	// set, whenever the router is the hosted one, and whenever a credential is passed in,
	// which is every router but a local one told only who to bill.
	Authenticate bool
	// APIKey and APISecret are the Stream app to authenticate as. Empty falls back to STREAM_API_KEY and STREAM_API_SECRET.
	APIKey    string
	APISecret string
	// UserID is the end user this SDK is acting for, if it is acting for one. Empty speaks
	// for the app itself, which is what a backend does, and is what keeps the per-user
	// daily limits out of it.
	UserID string
	// Token is a credential somebody else minted for UserID to hold, used instead of
	// signing one here.
	//
	// It is what a process without the secret has: a worker handed a user's token, or a test
	// standing in for a device. With it, APISecret is not needed at all, which is the point
	// -- a token is the whole credential and the secret behind it could mint any other.
	Token string
	// ActingFor is the end user a server-side credential speaks for, which is how a backend
	// reaches a conversation that belongs to somebody. Unlike UserID it keeps the app's own
	// credential, so the request is still the server's: what it writes is answered by the
	// model rather than handed to a dispatch worker, and no daily limit is counted.
	ActingFor string
	// Team is the customer's team the work is counted against, such as finance: its spend
	// and its spend limits. 1 to 59 characters of A-Z, a-z, 0-9, _ . and -.
	//
	// It and UsageCategories are signed into every token minted here, and sent in
	// X-Stream-Usage-Categories to a router with no proxy in front of it. A Token handed in
	// carries whatever it was signed with, so neither can be set beside one.
	Team string
	// UsageCategories are the customer's own groupings the work is counted against, such as
	// project:onboarding: at most 10 together with the team, each 1 to 64 characters of
	// A-Z, a-z, 0-9, _ . : and -, and none starting team:. Resolve keeps them sorted, each
	// once.
	UsageCategories []string
	// HTTPClient is used for both the REST calls and the socket handshake. Nil uses the
	// default client.
	HTTPClient *http.Client
}

// Resolve fills in whatever the environment knows and refuses a backend nobody is billed for.
func (b Backend) Resolve() (Backend, error) {
	categories, err := checkedCategories(b.Team, b.UsageCategories)
	if err != nil {
		return b, err
	}
	b.UsageCategories = categories
	if b.Token != "" && (b.Team != "" || len(b.UsageCategories) > 0) {
		return b, errors.New("stream: a token handed in carries the team and usage categories " +
			"it was signed with; sign them into the token instead")
	}
	if b.URL != "" && b.Region != "" {
		// Clients retain the resolved backend and resolve it again for resource calls.
		if url, err := RegionURL(b.Region); err != nil || strings.TrimSuffix(b.URL, "/") != url {
			return b, errors.New("stream: set a URL or a region, not both")
		}
	}
	if b.URL == "" && b.Region == "" {
		b.URL = os.Getenv(URLEnv)
	}
	if b.URL == "" && b.Region == "" {
		b.Region = os.Getenv(RegionEnv)
	}
	if b.URL == "" && b.Region != "" {
		url, err := RegionURL(b.Region)
		if err != nil {
			return b, err
		}
		b.URL = url
	}
	if b.URL == "" {
		b.URL = DefaultURL
	}
	b.URL = strings.TrimSuffix(b.URL, "/")

	if b.CustomerID == "" {
		b.CustomerID = os.Getenv(CustomerEnv)
	}

	// A Stream key and secret are in the environment for video and chat whether or not a
	// local router is being talked to, so a customer id is what says which: sending the
	// credential to a router with nothing in front of it reads as an end user's device.
	// What does not depend on the environment does say: the hosted router, in any region,
	// is always behind the proxy, and a credential passed in was passed in to be used.
	passed := b.APIKey != "" && (b.APISecret != "" || b.Token != "")
	if b.CustomerID == "" || hosted(b.URL) || passed {
		b.Authenticate = true
	}
	if b.Authenticate {
		if b.APIKey == "" {
			b.APIKey = os.Getenv(APIKeyEnv)
		}
		if b.APISecret == "" {
			b.APISecret = os.Getenv(APISecretEnv)
		}
		// A token handed in stands in for the secret, since it is already the thing the
		// secret would have been used to make.
		if b.APIKey == "" || (b.APISecret == "" && b.Token == "") {
			return b, ErrNoCredential
		}
	}
	return b, nil
}

// token mints what the proxy verifies: a token signed with the app's secret, naming a user
// only when this SDK is acting for one.
//
// A token that names nobody speaks for the app itself, which is Stream's `server: true`, and
// is what a backend holds. That is also what leaves the per-user daily limits out of it.
func (b Backend) token() (string, error) {
	if b.Token != "" {
		return b.Token, nil
	}
	now := time.Now()
	claims := jwt.MapClaims{
		"iat": jwt.NewNumericDate(now),
		"exp": jwt.NewNumericDate(now.Add(tokenValidity)),
	}
	if b.UserID == "" {
		claims["server"] = true
	} else {
		claims["user_id"] = b.UserID
	}
	if b.Team != "" {
		claims["team"] = b.Team
	}
	if len(b.UsageCategories) > 0 {
		claims["usage_categories"] = b.UsageCategories
	}
	signed, err := jwt.NewWithClaims(jwt.SigningMethodHS256, claims).SignedString([]byte(b.APISecret))
	if err != nil {
		return "", fmt.Errorf("stream: signing a token for %s: %w", b.APIKey, err)
	}
	return signed, nil
}

// Credentials are what every request to the router carries: the app and a token signed for
// it when going through the proxy, and otherwise who is billed.
func (b Backend) Credentials() (http.Header, error) {
	header := http.Header{}
	header.Set(ClientHeader, ClientSDK)
	if b.ActingFor != "" {
		header.Set(UserHeader, b.ActingFor)
	}
	if !b.Authenticate {
		if b.CustomerID != "" {
			header.Set(CustomerHeader, b.CustomerID)
		}
		if named := b.categoryHeader(); named != "" {
			header.Set(UsageCategoriesHeader, named)
		}
		return header, nil
	}
	token, err := b.token()
	if err != nil {
		return nil, err
	}
	header.Set(APIKeyHeader, b.APIKey)
	header.Set(AuthTypeHeader, AuthTypeJWT)
	header.Set(AuthorizationHeader, token)
	return header, nil
}

// categoryHeader is the team and categories as a router with no proxy in front of it reads
// them, or empty for none.
func (b Backend) categoryHeader() string {
	named := slices.Clone(b.UsageCategories)
	if b.Team != "" {
		named = append(named, teamPrefix+b.Team)
	}
	slices.Sort(named)
	return strings.Join(named, ",")
}

// checkedCategories are the categories each once and sorted, as the proxy keeps them, or an
// error naming the entry the router would refuse.
func checkedCategories(team string, categories []string) ([]string, error) {
	if team != "" && !teamName.MatchString(team) {
		return nil, fmt.Errorf("stream: team %q is not 1 to 59 characters of A-Z, a-z, 0-9, _ . and -", team)
	}
	for _, category := range categories {
		if !categoryName.MatchString(category) {
			return nil, fmt.Errorf("stream: usage category %q is not 1 to 64 characters of A-Z, a-z, 0-9, _ . : and -", category)
		}
		// In any case, as the proxy reads it: Team:finance would be a second team.
		if strings.HasPrefix(strings.ToLower(category), teamPrefix) {
			return nil, fmt.Errorf("stream: usage category %q names a team: use Team for a team", category)
		}
	}
	kept := slices.Compact(slices.Sorted(slices.Values(categories)))
	counted := len(kept)
	if team != "" {
		counted++
	}
	if counted > MaxUsageCategories {
		return nil, fmt.Errorf("stream: at most %d usage categories, the team included; these are %d", MaxUsageCategories, counted)
	}
	if len(kept) == 0 {
		return nil, nil
	}
	return kept, nil
}

// Client is an HTTP client for the generated API, already carrying the credentials.
//
// It reaches the operations this SDK has no method for yet. The generated package is
// regenerated from the router's spec as it changes, so its operation methods carry no
// compatibility promise: a minor release may rename or reshape them. Prefer a method of
// client, agents or stream wherever one exists.
func (b Backend) Client() (*acceleration.ClientWithResponses, error) {
	resolved, err := b.Resolve()
	if err != nil {
		return nil, err
	}

	doer := acceleration.HttpRequestDoer(&http.Client{})
	if resolved.HTTPClient != nil {
		doer = resolved.HTTPClient
	}
	options := []acceleration.ClientOption{
		acceleration.WithHTTPClient(readable{doer: doer}),
		acceleration.WithRequestEditorFn(func(_ context.Context, request *http.Request) error {
			// Minted per request, so a client left idle longer than a token lasts does not
			// wake up holding an expired one.
			credentials, err := resolved.Credentials()
			if err != nil {
				return err
			}
			for name, values := range credentials {
				request.Header[name] = values
			}
			return nil
		}),
	}
	return acceleration.NewClientWithResponses(resolved.URL, options...)
}

// SocketURL is the WebSocket URL for a path on the router.
func (b Backend) SocketURL(path string) string {
	switch {
	case strings.HasPrefix(b.URL, "https://"):
		return "wss://" + strings.TrimPrefix(b.URL, "https://") + path
	case strings.HasPrefix(b.URL, "http://"):
		return "ws://" + strings.TrimPrefix(b.URL, "http://") + path
	default:
		return b.URL + path
	}
}
