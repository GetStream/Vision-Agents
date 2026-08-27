//go:build integration

package phone

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"testing"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"
	"github.com/golang-jwt/jwt/v5"
	"github.com/stretchr/testify/suite"
)

// Most vendors cannot send a password to a trunk. Twilio, Telnyx and Plivo can name SIP
// digest credentials when they bridge a call; Vonage, Bird, Sinch and Bandwidth have no
// field for one anywhere in their call plans. Whether those four can bridge at all
// therefore comes down to whether a trunk will take an INVITE from an address it knows
// without challenging it, which is what this asks Stream directly.
//
// Stream decides before the call, and the answer is one of three words: "password" means
// challenge it, "accept" means let it in, "no_trunk_found" means there is nothing to let
// it into. Placing a real call to find out would cost money and ring a telephone, so the
// decision is read on its own.
//
// The answer is that an allowlisted address is accepted unchallenged, which is what makes
// the four password-less vendors usable. It also turns out that a trunk with no allowlist
// accepts every address unchallenged, which is why Trunk.AllowedIPs is not optional here.

type SIPAuthIntegrationSuite struct {
	suite.Suite
	ctx    context.Context
	stream *Stream
	// number is a number the trunk answers for. It is never dialled.
	number string
	// allowed is the address the trunk is told to expect calls from.
	allowed string
	// appID names the application, which the pre-auth endpoint wants in the query string
	// because Stream's own SIP layer is what normally calls it.
	appID string
}

func TestSIPAuthIntegrationSuite(t *testing.T) {
	suite.Run(t, new(SIPAuthIntegrationSuite))
}

func (s *SIPAuthIntegrationSuite) SetupSuite() {
	if os.Getenv(apiKeyEnvVar) == "" || os.Getenv(apiSecretEnvVar) == "" {
		s.T().Skip(apiKeyEnvVar + " and " + apiSecretEnvVar + " not set")
	}

	stream, err := NewStream(StreamOptions{})
	s.Require().NoError(err)
	s.stream = stream
	s.number = fmt.Sprintf("+1512555%04d", time.Now().UnixNano()%10_000)
	// Reserved for documentation, so it is nobody's real signalling address.
	s.allowed = "203.0.113.7"

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	app, err := stream.Client().Client.GetApp(ctx, &getstream.GetAppRequest{})
	s.Require().NoError(err)
	s.appID = strconv.Itoa(app.Data.App.ID)
}

func (s *SIPAuthIntegrationSuite) SetupTest() {
	var cancel context.CancelFunc
	s.ctx, cancel = context.WithTimeout(context.Background(), 30*time.Second)
	s.T().Cleanup(cancel)
}

func (s *SIPAuthIntegrationSuite) TestATrunkThatKnowsTheAddressLetsTheCallInUnchallenged() {
	s.trunk("sipauth-allowed", []string{s.allowed + "/32"})

	status, decision := s.resolve(s.allowed)

	s.Equal(http.StatusCreated, status)
	s.Equal("accept", decision.AuthResult,
		"a vendor with no password field can only bridge if its address is enough")
	s.Require().NotNil(decision.TrunkID, "an accepted call still has to name the trunk it joins")
	s.Nil(decision.Password, "an accepted call is not challenged, so there is nothing to answer with")
}

func (s *SIPAuthIntegrationSuite) TestATrunkTurnsAwayAnAddressItDoesNotKnow() {
	// An allowlist replaces the password rather than adding to it: an address that is not
	// on it is refused outright rather than being asked to authenticate. Getting a
	// vendor's signalling ranges wrong therefore fails the call, not just the shortcut.
	s.trunk("sipauth-other", []string{s.allowed + "/32"})

	status, decision := s.resolve("198.51.100.9")

	s.Equal(http.StatusForbidden, status)
	s.Empty(decision.AuthResult, "a refused call never gets as far as a decision")
}

func (s *SIPAuthIntegrationSuite) TestATrunkWithNoAllowlistLetsAnybodyIn() {
	// This is why every trunk made here names the addresses it expects. An empty
	// allowlist does not mean "password only", it means no check at all: anyone who
	// learns the trunk's uri is in the customer's calls.
	s.trunk("sipauth-none", nil)

	status, decision := s.resolve(s.allowed)

	s.Equal(http.StatusCreated, status)
	s.Equal("accept", decision.AuthResult)
}

// trunk makes a trunk that expects calls from allowedIPs, and removes it afterwards.
func (s *SIPAuthIntegrationSuite) trunk(name string, allowedIPs []string) Bridge {
	trunkID, bridge, err := s.stream.CreateTrunk(s.ctx, Trunk{
		Name:       name + "-" + s.number,
		Numbers:    []string{s.number},
		AllowedIPs: allowedIPs,
	})
	s.Require().NoError(err)
	s.T().Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		_, _ = s.stream.Client().Video().DeleteSIPTrunk(ctx, trunkID, &getstream.DeleteSIPTrunkRequest{})
	})
	return bridge
}

// resolve asks Stream what it would do with an INVITE arriving from sourceIP.
//
// The request goes out by hand rather than through the SDK's ResolveSipAuth: the endpoint
// wants app_id, the SDK always sends api_key, and it refuses both at once.
func (s *SIPAuthIntegrationSuite) resolve(sourceIP string) (int, getstream.ResolveSipAuthResponse) {
	client := s.stream.Client().Client

	body, err := json.Marshal(getstream.ResolveSipAuthRequest{
		SipCallerNumber: "+15125550000",
		SipTrunkNumber:  s.number,
		SourceIp:        &sourceIP,
	})
	s.Require().NoError(err)

	address := client.BaseUrl() + "/api/v2/video/sip/auth?app_id=" + s.appID
	request, err := http.NewRequestWithContext(s.ctx, http.MethodPost, address, bytes.NewReader(body))
	s.Require().NoError(err)
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Authorization", s.serverToken())
	request.Header.Set("stream-auth-type", "jwt")

	response, err := http.DefaultClient.Do(request)
	s.Require().NoError(err)
	defer response.Body.Close()

	raw, err := io.ReadAll(response.Body)
	s.Require().NoError(err)
	s.T().Logf("pre-auth for %s answered %d: %s", sourceIP, response.StatusCode, raw)

	var decision getstream.ResolveSipAuthResponse
	s.Require().NoError(json.Unmarshal(raw, &decision))
	return response.StatusCode, decision
}

// serverToken is the server-side JWT Stream authenticates an admin call with.
func (s *SIPAuthIntegrationSuite) serverToken() string {
	token, err := jwt.NewWithClaims(
		jwt.SigningMethodHS256,
		jwt.MapClaims{"server": true},
	).SignedString([]byte(os.Getenv(apiSecretEnvVar)))
	s.Require().NoError(err)
	return token
}
