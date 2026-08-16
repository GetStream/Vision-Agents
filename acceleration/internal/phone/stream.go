package phone

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"

	getstream "github.com/GetStream/getstream-go/v5"
)

const (
	apiKeyEnvVar    = "STREAM_API_KEY"
	apiSecretEnvVar = "STREAM_API_SECRET"
)

// defaultCallType is the Stream call type a phone call joins.
const defaultCallType = "default"

// callerTemplate names the participant a SIP caller becomes, so per-participant
// transcription has something stable to key on. Stream renders handlebars templates
// against the SIP invite, and the caller's number is the only thing about them that is
// known before they say a word.
const callerTemplate = "sip-{{caller_number}}"

// StreamOptions configures the Stream side of a phone number.
type StreamOptions struct {
	// APIKey defaults to STREAM_API_KEY.
	APIKey string
	// APISecret defaults to STREAM_API_SECRET.
	APISecret string
}

// Stream creates the SIP trunks and routing rules that connect a vendor's numbers to a
// call.
//
// Stream's SIP is inbound only: a trunk is somewhere a vendor sends calls, not somewhere
// calls are placed from. Both directions therefore end at a trunk, with outbound calls
// originated at the vendor and bridged into the same one.
type Stream struct {
	client *getstream.Stream
}

// NewStream validates the credentials and returns a Stream.
func NewStream(options StreamOptions) (*Stream, error) {
	if options.APIKey == "" {
		options.APIKey = os.Getenv(apiKeyEnvVar)
	}
	if options.APISecret == "" {
		options.APISecret = os.Getenv(apiSecretEnvVar)
	}
	if options.APIKey == "" || options.APISecret == "" {
		return nil, errors.New("phone: " + apiKeyEnvVar + " and " + apiSecretEnvVar + " are required")
	}

	client, err := getstream.NewClient(options.APIKey, options.APISecret)
	if err != nil {
		return nil, err
	}
	return &Stream{client: client}, nil
}

// Trunk describes the trunk to create.
type Trunk struct {
	// Name is what the trunk is called in the dashboard.
	Name string
	// Numbers are the numbers the vendor will present on calls to this trunk.
	Numbers []string
	// Password authenticates the vendor. Empty lets Stream generate one, which is then
	// returned in the bridge.
	Password string
	// AllowedIPs are the vendor's signalling addresses, as IPs or CIDR blocks. Leaving it
	// empty accepts calls from anywhere that has the password.
	AllowedIPs []string
}

// CreateTrunk creates an inbound trunk and returns the bridge the vendor should send
// calls to. The password is only readable here, which is why the bridge is returned
// rather than fetched later.
func (s *Stream) CreateTrunk(ctx context.Context, trunk Trunk) (string, Bridge, error) {
	if trunk.Name == "" {
		return "", Bridge{}, errors.New("phone: a trunk needs a name")
	}
	if len(trunk.Numbers) == 0 {
		return "", Bridge{}, errors.New("phone: a trunk needs at least one number")
	}

	request := &getstream.CreateSIPTrunkRequest{
		Name:       trunk.Name,
		Numbers:    trunk.Numbers,
		AllowedIps: trunk.AllowedIPs,
	}
	if trunk.Password != "" {
		request.Password = &trunk.Password
	}

	response, err := s.client.Video().CreateSIPTrunk(ctx, request)
	if err != nil {
		return "", Bridge{}, fmt.Errorf("phone: create sip trunk: %w", err)
	}
	created := response.Data.SipTrunk
	if created == nil {
		return "", Bridge{}, errors.New("phone: stream created no trunk")
	}

	return created.ID, Bridge{
		URI:      sipURI(created.Uri),
		Username: created.Username,
		Password: created.Password,
	}, nil
}

// Route is how calls arriving on a trunk are turned into a call.
type Route struct {
	// Name is what the rule is called in the dashboard.
	Name string
	// TrunkIDs are the trunks the rule applies to.
	TrunkIDs []string
	// CalledNumbers are the numbers this rule answers for.
	CalledNumbers []string
	// CallID is the call every matching caller joins. Leaving it empty gives each caller
	// their own call, named after the number they rang, which is what a support line
	// wants; setting it puts everyone in one call, which is what a conference wants.
	CallID string
	// CallType is the Stream call type. Empty means "default".
	CallType string
}

// CreateRoute points a trunk's calls at a Stream call.
//
// The call id may be a handlebars template, which is how one rule serves every number on
// a trunk without a rule per number.
func (s *Stream) CreateRoute(ctx context.Context, route Route) (string, error) {
	if route.Name == "" {
		return "", errors.New("phone: a routing rule needs a name")
	}
	if len(route.TrunkIDs) == 0 {
		return "", errors.New("phone: a routing rule needs a trunk")
	}
	if len(route.CalledNumbers) == 0 {
		return "", errors.New("phone: a routing rule needs the numbers it answers for")
	}

	callType := route.CallType
	if callType == "" {
		callType = defaultCallType
	}
	callID := route.CallID
	if callID == "" {
		callID = "phone-{{called_number}}"
	}

	response, err := s.client.Video().CreateSIPInboundRoutingRule(ctx,
		&getstream.CreateSIPInboundRoutingRuleRequest{
			Name:          route.Name,
			TrunkIds:      route.TrunkIDs,
			CalledNumbers: route.CalledNumbers,
			// The caller becomes a participant, and their id has to be stable for the
			// length of the call because per-participant transcription is keyed on it.
			CallerConfigs: getstream.SIPCallerConfigsRequest{ID: callerTemplate},
			DirectRoutingConfigs: &getstream.SIPDirectRoutingRuleCallConfigsRequest{
				CallID:   callID,
				CallType: callType,
			},
		})
	if err != nil {
		return "", fmt.Errorf("phone: create sip routing rule: %w", err)
	}
	return response.Data.ID, nil
}

// Client exposes the Stream client, so a caller can reach the parts of the SIP API this
// does not wrap without building a second client.
func (s *Stream) Client() *getstream.Stream { return s.client }

// sipURI makes sure the trunk address is a SIP uri, since Stream reports the host without
// the scheme and a vendor needs the whole thing.
func sipURI(uri string) string {
	if uri == "" {
		return ""
	}
	if strings.HasPrefix(uri, "sip:") || strings.HasPrefix(uri, "sips:") {
		return uri
	}
	return "sip:" + uri
}
