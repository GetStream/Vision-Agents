package phone

import (
	"fmt"

	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation/chattest"
)

// StreamSuite covers what can be checked without an account: that nothing half-described
// reaches Stream, and that a trunk address a vendor is given is dialable.

// The caller is the participant the routing rule names from callerTemplate. CallerNumber cuts
// exactly what the call hook cut before it moved here (strings.CutPrefix on "sip-"), so
// "sip-" alone is still the caller, with no number.
func (s *PhoneSuite) TestTheCallerIsTheParticipantTheRoutingRuleNamed() {
	number, ok := CallerNumber("sip-+15550001111")
	s.True(ok)
	s.Equal("+15550001111", number)

	number, ok = CallerNumber("sip-")
	s.True(ok)
	s.Empty(number)

	_, ok = CallerNumber("agent-1")
	s.False(ok)
	_, ok = CallerNumber("Sip-+15550001111")
	s.False(ok)
}

func (s *PhoneSuite) TestStreamNeedsItsCredentials() {
	s.T().Setenv("STREAM_API_KEY", "deploy-key")
	s.T().Setenv("STREAM_API_SECRET", "deploy-secret")

	_, err := NewStream(StreamOptions{})

	s.ErrorContains(err, "api key and secret", "the environment is the deployment's app, not this one's")
}

func (s *PhoneSuite) TestATrunkWithoutNumbersIsRefusedBeforeStreamIsAsked() {
	stream := s.stream()

	_, _, err := stream.CreateTrunk(s.ctx, Trunk{Name: "support"})

	s.ErrorContains(err, "at least one number")
}

func (s *PhoneSuite) TestATrunkWithoutANameIsRefused() {
	stream := s.stream()

	_, _, err := stream.CreateTrunk(s.ctx, Trunk{Numbers: []string{"+15125551234"}})

	s.ErrorContains(err, "needs a name")
}

func (s *PhoneSuite) TestARouteNeedsATrunkAndTheNumbersItAnswersFor() {
	stream := s.stream()

	_, err := stream.CreateRoute(s.ctx, Route{Name: "support"})
	s.ErrorContains(err, "needs a trunk")

	_, err = stream.CreateRoute(s.ctx, Route{Name: "support", TrunkIDs: []string{"trunk-1"}})
	s.ErrorContains(err, "numbers it answers for")
}

func (s *PhoneSuite) TestDeletingNoTrunkIsNotAnError() {
	stream := s.stream()

	err := stream.DeleteTrunk(s.ctx, "")

	s.NoError(err)
}

func (s *PhoneSuite) TestDeletingNoRouteIsNotAnError() {
	stream := s.stream()

	err := stream.DeleteRoute(s.ctx, "")

	s.NoError(err)
}

func (s *PhoneSuite) TestTheTrunkAddressGivenToAVendorIsAlwaysDialable() {
	// Stream reports the host without a scheme, and a vendor cannot dial a bare host.
	s.Equal("sip:sip.stream-io-api.com", sipURI("sip.stream-io-api.com"))
	s.Equal("sip:trunk@sip.stream-io-api.com", sipURI("sip:trunk@sip.stream-io-api.com"))
	s.Equal("sips:trunk@sip.stream-io-api.com", sipURI("sips:trunk@sip.stream-io-api.com"))
	s.Empty(sipURI(""))
}

func (s *PhoneSuite) stream() *Stream {
	stream, err := NewStream(StreamOptions{APIKey: "key", APISecret: "secret"})
	s.Require().NoError(err)
	return stream
}

func (s *PhoneSuite) TestATrunkWithNeitherAPasswordNorAddressesGetsARandomPassword() {
	fake := chattest.NewServer(s.T())
	stream := NewStreamFromClient(fake.Client)

	trunkID, bridge, err := stream.CreateTrunk(s.ctx, Trunk{Name: "call", Numbers: []string{"+15125550100"}})
	s.Require().NoError(err)

	password, _ := fake.TrunkAuth(trunkID)
	s.NotEmpty(password)
	s.Equal(password, bridge.Password)
}

func (s *PhoneSuite) TestATrunkWithAddressesGetsNoPassword() {
	fake := chattest.NewServer(s.T())
	stream := NewStreamFromClient(fake.Client)

	trunkID, _, err := stream.CreateTrunk(s.ctx, Trunk{
		Name: "call", Numbers: []string{"+15125550100"}, AllowedIPs: []string{"192.0.2.0/24"},
	})
	s.Require().NoError(err)

	password, allowed := fake.TrunkAuth(trunkID)
	s.Empty(password, "Stream would challenge a vendor that has no way to answer")
	s.Equal([]string{"192.0.2.0/24"}, allowed)
}

func (s *PhoneSuite) TestATrunkGivenAPasswordKeepsIt() {
	fake := chattest.NewServer(s.T())
	stream := NewStreamFromClient(fake.Client)

	trunkID, _, err := stream.CreateTrunk(s.ctx, Trunk{Name: "call", Numbers: []string{"+15125550100"}, Password: "chosen"})
	s.Require().NoError(err)

	password, _ := fake.TrunkAuth(trunkID)
	s.Equal("chosen", password)
}

func (s *PhoneSuite) TestUpdatingATrunkKeepsItsPasswordWhenNoneIsGiven() {
	fake := chattest.NewServer(s.T())
	stream := NewStreamFromClient(fake.Client)
	trunkID, _, err := stream.CreateTrunk(s.ctx, Trunk{Name: "call", Numbers: []string{"+15125550100"}, Password: "chosen"})
	s.Require().NoError(err)

	err = stream.UpdateTrunk(s.ctx, trunkID, Trunk{Name: "call", Numbers: []string{"+15125550100", "+15125550101"}})
	s.Require().NoError(err)

	s.Equal([]string{"+15125550100", "+15125550101"}, fake.TrunkNumbers(trunkID))
	password, _ := fake.TrunkAuth(trunkID)
	s.Equal("chosen", password)
}

func (s *PhoneSuite) TestATrunkTakesNoMoreThanFiftyNumbers() {
	fake := chattest.NewServer(s.T())
	stream := NewStreamFromClient(fake.Client)
	trunkID, _, err := stream.CreateTrunk(s.ctx, Trunk{Name: "call", Numbers: []string{"+15125550100"}})
	s.Require().NoError(err)

	numbers := make([]string, 51)
	for i := range numbers {
		numbers[i] = fmt.Sprintf("+1512555%04d", i)
	}
	s.Error(stream.UpdateTrunk(s.ctx, trunkID, Trunk{Name: "call", Numbers: numbers}))
}

func (s *PhoneSuite) TestUpdatingARouteChangesItsCallID() {
	fake := chattest.NewServer(s.T())
	stream := NewStreamFromClient(fake.Client)
	route := Route{Name: "call", TrunkIDs: []string{"t"}, CalledNumbers: []string{"+15125550100"}}
	ruleID, err := stream.CreateRoute(s.ctx, route)
	s.Require().NoError(err)

	route.CallID = "x"
	s.Require().NoError(stream.UpdateRoute(s.ctx, ruleID, route))

	configs, _ := fake.Rule(ruleID)["direct_routing_configs"].(map[string]any)
	s.Equal("x", configs["call_id"])
}

func (s *PhoneSuite) TestACreatedCallCarriesItsCustom() {
	fake := chattest.NewServer(s.T())
	stream := NewStreamFromClient(fake.Client)

	s.Require().NoError(stream.CreateCall(s.ctx, "agent", "abc", map[string]string{"k": "v"}))

	data, _ := fake.Call("agent", "abc")["data"].(map[string]any)
	custom, _ := data["custom"].(map[string]any)
	s.Equal("v", custom["k"])
}
