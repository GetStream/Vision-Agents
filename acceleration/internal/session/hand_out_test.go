package session

import (
	"testing"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
)

// HandOutSuite is which session answers in a channel a message was handed out from
// (Manager.handedOut), and which app a session a hook started acts in (Manager.hookPin).
type HandOutSuite struct {
	suite.Suite
	manager *Manager
}

func TestHandOutSuite(t *testing.T) {
	suite.Run(t, new(HandOutSuite))
}

func (s *HandOutSuite) SetupTest() {
	s.manager = &Manager{hookPins: map[string]hookPinned{}}
}

// worker is the session a worker opens for a message in channel-1 of customer-1.
func worker() Spec {
	return Spec{CustomerID: "customer-1", AgentID: "channel-1", Text: true, PersistConversation: true, CallerKind: auth.KindServer}
}

// deploymentApp and ownApp are the apps a session acts in: the deployment's, and a customer's
// own registered app 42.
var (
	deploymentApp = streamapp.Bound{Client: &getstream.Stream{}}
	ownApp        = streamapp.Bound{Client: &getstream.Stream{}, Identity: streamapp.Identity{StreamApp: 42, Registered: true}}
)

func (s *HandOutSuite) TestTheWorkersSessionAnswersInTheChannelHandedOutInItsApp() {
	s.manager.HandOut("customer-1", "agent:channel-1", 7, true)
	s.manager.HandOut("customer-2", "agent:channel-2", 42, false)

	cid, handed := s.manager.handedOut(worker(), deploymentApp)
	s.True(handed)
	s.Equal("agent:channel-1", cid)
	other := worker()
	other.CustomerID, other.AgentID = "customer-2", "channel-2"
	cid, handed = s.manager.handedOut(other, ownApp)
	s.True(handed, "a registered app's hand-out, in that app")
	s.Equal("agent:channel-2", cid)
}

func (s *HandOutSuite) TestAnyOtherSessionKeepsAChannelOfItsOwn() {
	s.manager.HandOut("customer-1", "agent:channel-1", 7, true)
	s.manager.HandOut("customer-2", "agent:channel-2", 42, false)
	s.manager.PinHook("customer-3", "agent:channel-3", 42)
	for name, spec := range map[string]func(*Spec){
		"another customer":       func(spec *Spec) { spec.CustomerID = "customer-2" },
		"another channel":        func(spec *Spec) { spec.AgentID = "channel-9" },
		"voice":                  func(spec *Spec) { spec.Text = false },
		"not kept":               func(spec *Spec) { spec.PersistConversation = false },
		"incognito":              func(spec *Spec) { spec.Incognito = true },
		"naming its own channel": func(spec *Spec) { spec.ConversationID = "agent:mine" },
		"a fork":                 func(spec *Spec) { spec.Recall = &Recall{} },
		"an end user":            func(spec *Spec) { spec.CallerKind = auth.KindAuthenticated },
		"acting for a user":      func(spec *Spec) { spec.Caller = routing.Caller{UserID: "sam"} },
		"only an app pin":        func(spec *Spec) { spec.CustomerID, spec.AgentID = "customer-3", "channel-3" },
	} {
		asked := worker()
		spec(&asked)
		_, handed := s.manager.handedOut(asked, deploymentApp)
		s.False(handed, name)
	}
	_, handed := s.manager.handedOut(worker(), ownApp)
	s.False(handed, "a deployment hand-out, to a session in the customer's own app")
	_, handed = s.manager.handedOut(worker(), streamapp.Bound{})
	s.False(handed, "a session with no Stream app")
	registered := worker()
	registered.CustomerID, registered.AgentID = "customer-2", "channel-2"
	_, handed = s.manager.handedOut(registered, deploymentApp)
	s.False(handed, "a registered app's hand-out, to a session in the deployment's app")
	_, handed = s.manager.handedOut(registered, streamapp.Bound{Client: &getstream.Stream{}, Identity: streamapp.Identity{StreamApp: 43, Registered: true}})
	s.False(handed, "a registered app's hand-out, to a session in another app")
}

func (s *HandOutSuite) TestAHandOutIsTheSessionsForAsLongAsAnAppPin() {
	s.manager.HandOut("customer-1", "agent:channel-1", 7, true)
	s.age("customer-1", "agent:channel-1", hookPinFor-time.Minute)
	_, handed := s.manager.handedOut(worker(), deploymentApp)
	s.True(handed)

	s.age("customer-1", "agent:channel-1", hookPinFor+time.Minute)
	_, handed = s.manager.handedOut(worker(), deploymentApp)
	s.False(handed)
}

// A hook the deployment signed pins no app: the session acts in whatever app its customer
// acts in, as on base, where no pin was kept for one.
func (s *HandOutSuite) TestOnlyARegisteredAppsHandOutPinsTheSessionsApp() {
	s.manager.HandOut("customer-1", "agent:channel-1", 7, true)
	s.manager.HandOut("customer-2", "agent:channel-2", 42, false)

	_, pinned := s.manager.hookPin(worker())
	s.False(pinned)
	registered := worker()
	registered.CustomerID, registered.AgentID = "customer-2", "channel-2"
	app, pinned := s.manager.hookPin(registered)
	s.True(pinned)
	s.Equal(int64(42), app)
}

// age dates a hand-out as made ago.
func (s *HandOutSuite) age(customer, cid string, ago time.Duration) {
	pinned := s.manager.hookPins[customer+"\x00"+cid]
	pinned.at = time.Now().Add(-ago)
	s.manager.hookPins[customer+"\x00"+cid] = pinned
}
