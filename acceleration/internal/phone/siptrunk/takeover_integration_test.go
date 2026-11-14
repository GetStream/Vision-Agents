//go:build integration

package siptrunk

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/redis/rueidis"
	"github.com/stretchr/testify/suite"

	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"

	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone/sipbridge"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv/redisready"
)

// redisAddrEnvVar is where the tests look for a Redis to run against.
const redisAddrEnvVar = "ROUTER_REDIS_ADDR"

// What the resuming node's credentials open to, built at runtime so no literal reads as a
// credential.
var (
	resumedTrunkPassword  = strings.Repeat("c", 12)
	resumedBridgePassword = strings.Repeat("d", 12)
)

const (
	testHoldFor    = 600 * time.Millisecond
	testRenewEvery = 100 * time.Millisecond
)

// fakeTakeover stands in for both the phone service, which opens the credentials, and
// sipbridge.Resume, which continues the call.
type fakeTakeover struct {
	mu      sync.Mutex
	configs []sipbridge.ResumeConfig
	states  []sipbridge.CallState
	call    *fakeCall
	err     error
	// lookupErrs are what opening the credentials fails with, the next one each time, until
	// none are left.
	lookupErrs []error
	// beforeLookup, when set, runs before the credentials are opened.
	beforeLookup func()
	// before, when set, runs before the resume answers, and its error is the resume's.
	before func() error
	ended  chan phone.CallStatus
}

func newFakeTakeover() *fakeTakeover {
	return &fakeTakeover{call: newFakeCall(), ended: make(chan phone.CallStatus, 4)}
}

func (t *fakeTakeover) ResumeTrunkCall(_ context.Context, _ phone.TrunkCallRef) (phone.ResumedTrunkCall, error) {
	if t.beforeLookup != nil {
		t.beforeLookup()
	}
	t.mu.Lock()
	var err error
	if len(t.lookupErrs) > 0 {
		err, t.lookupErrs = t.lookupErrs[0], t.lookupErrs[1:]
	}
	t.mu.Unlock()
	if err != nil {
		return phone.ResumedTrunkCall{}, err
	}
	return phone.ResumedTrunkCall{
		Trunk: &phone.SIPTrunk{
			Host: "trunk-b.example.com", Port: 5061, Transport: "tls", Username: "agent-b",
			Password: resumedTrunkPassword, LateOffer: true, Codecs: []string{"PCMA"},
		},
		Bridge: phone.Bridge{URI: "sip:+15550000301@bridge-b.sip.example.com", Username: "stream-b", Password: resumedBridgePassword},
		Ended:  func(status phone.CallStatus) { t.ended <- status },
	}, nil
}

func (t *fakeTakeover) resume(_ context.Context, cfg sipbridge.ResumeConfig, state sipbridge.CallState) (call, error) {
	t.mu.Lock()
	t.configs = append(t.configs, cfg)
	t.states = append(t.states, state)
	t.mu.Unlock()
	if t.before != nil {
		if err := t.before(); err != nil {
			return nil, err
		}
	}
	if t.err != nil {
		return nil, t.err
	}
	return t.call, nil
}

func (t *fakeTakeover) resumes() int {
	t.mu.Lock()
	defer t.mu.Unlock()
	return len(t.configs)
}

// testNode is one router node: its provider, what it dials and how it takes calls over.
type testNode struct {
	provider *Provider
	dialer   *fakeDialer
	takeover *fakeTakeover
	ended    chan phone.CallStatus
	stop     context.CancelFunc
}

type TakeoverSuite struct {
	suite.Suite
	redis  rueidis.Client
	prefix string
	nodes  []*testNode
	// holdFor is how long the claims of the nodes a test starts last, and renewEvery how often
	// they renew and scan.
	holdFor, renewEvery time.Duration
}

func TestTakeoverSuite(t *testing.T) { suite.Run(t, new(TakeoverSuite)) }

func (s *TakeoverSuite) SetupSuite() {
	address := os.Getenv(redisAddrEnvVar)
	if address == "" {
		s.T().Skipf("%s not set", redisAddrEnvVar)
	}
	client, err := redisready.NewClient(rueidis.ClientOption{InitAddress: []string{address}, DisableCache: true})
	s.Require().NoError(err)
	s.redis = client
}

func (s *TakeoverSuite) TearDownSuite() {
	if s.redis != nil {
		s.redis.Close()
	}
}

// SetupTest gives each test keys of its own, so no test cleans up after another.
func (s *TakeoverSuite) SetupTest() {
	s.prefix = "{test:siptrunk:" + uuid.NewString() + "}"
	s.nodes = nil
	s.holdFor = testHoldFor
	s.renewEvery = testRenewEvery
}

func (s *TakeoverSuite) TearDownTest() {
	for _, n := range s.nodes {
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		_ = n.provider.Close(ctx)
		cancel()
		n.stop()
	}
}

// node starts a node named name on the suite's Redis.
func (s *TakeoverSuite) node(name string) *testNode {
	n := &testNode{
		dialer:   &fakeDialer{call: newFakeCall(), dialled: make(chan struct{})},
		takeover: newFakeTakeover(),
		ended:    make(chan phone.CallStatus, 4),
	}
	n.dialer.call.state = callState("a")
	n.provider = newProvider(Options{
		Logger: slog.New(slog.DiscardHandler), Redis: s.redis, Node: name,
		holdFor: s.holdFor, renewEvery: s.renewEvery, prefix: s.prefix,
	}, n.dialer.dial, n.takeover.resume)
	n.provider.SetResumer(n.takeover)
	ctx, stop := context.WithCancel(context.Background())
	n.stop = stop
	n.provider.Start(ctx)
	s.nodes = append(s.nodes, n)
	return n
}

func callState(tag string) sipbridge.CallState {
	return sipbridge.CallState{
		Customer: sipbridge.LegState{
			CallID: "customer-call", LocalTag: "c-" + tag, RemoteTag: "trunk", Transport: "tcp",
			From: "<sip:+15550000301@trunk.example.com>;tag=c-" + tag, To: "<sip:+15550000302@trunk.example.com>;tag=trunk",
			LocalCSeq: 1, RemoteTarget: "sip:+15550000302@198.51.100.7:5060;transport=tcp", LastSDP: []byte("v=0 " + tag),
		},
		Stream: sipbridge.LegState{
			CallID: "stream-call", LocalTag: "s-" + tag, RemoteTag: "stream", Transport: "tcp",
			From: "<sip:+15550000302@bridge.sip.example.com>;tag=s-" + tag, To: "<sip:+15550000301@bridge.sip.example.com>;tag=stream",
			LocalCSeq: 1, RemoteTarget: "sip:+15550000301@10.0.0.7:5060;transport=tcp;alias=10.0.0.7~5060~1",
			RouteSet: []string{"<sip:203.0.113.10;lr>"}, LastSDP: []byte("v=0 stream " + tag),
		},
		Media: sipbridge.MediaAddrs{Customer: "198.51.100.7", Stream: "203.0.113.10"},
	}
}

func ref() *phone.TrunkCallRef {
	return &phone.TrunkCallRef{
		CustomerID: "acme", SIPTrunkID: "trunk-1",
		From: "+15550000301", CallID: "call-1", Token: "outcome-token",
	}
}

// dial places a call on n that another node could continue, and waits until it is recorded.
func (s *TakeoverSuite) dial(n *testNode) string {
	placing := outbound()
	placing.Resume = ref()
	placing.Ended = func(status phone.CallStatus) { n.ended <- status }
	dialed, err := n.provider.Dial(context.Background(), placing)
	s.Require().NoError(err)
	s.Eventually(func() bool { return s.get("call:"+dialed.VendorCallID) != "" }, time.Second, 10*time.Millisecond,
		"the connected call was never recorded")
	return dialed.VendorCallID
}

func (s *TakeoverSuite) get(key string) string {
	value, err := s.redis.Do(context.Background(), s.redis.B().Get().Key(s.prefix+":"+key).Build()).ToString()
	if rueidis.IsRedisNil(err) {
		return ""
	}
	s.Require().NoError(err)
	return value
}

func (s *TakeoverSuite) record(id string) callRecord {
	var rec callRecord
	s.Require().NoError(json.Unmarshal([]byte(s.get("call:"+id)), &rec))
	return rec
}

// lapse is when the call's claim lapses as the set of calls says, and false when it is not in it.
func (s *TakeoverSuite) lapse(id string) (time.Time, bool) {
	score, err := s.redis.Do(context.Background(), s.redis.B().Zscore().Key(s.prefix+":calls").Member(id).Build()).AsFloat64()
	if rueidis.IsRedisNil(err) {
		return time.Time{}, false
	}
	s.Require().NoError(err)
	return time.UnixMilli(int64(score)), true
}

// exists reads a key without failing the test, for checks run off the test's goroutine.
func (s *TakeoverSuite) exists(key string) bool {
	n, err := s.redis.Do(context.Background(), s.redis.B().Exists().Key(s.prefix+":"+key).Build()).AsInt64()
	return err == nil && n == 1
}

func (s *TakeoverSuite) gone(id string) bool {
	_, inSet := s.lapse(id)
	return s.get("call:"+id) == "" && s.get("owner:"+id) == "" && !inSet
}

func (s *TakeoverSuite) noReport(ended <-chan phone.CallStatus) {
	select {
	case status := <-ended:
		s.Failf("a call handed on was reported", "reported %+v", status)
	default:
	}
}

func (s *TakeoverSuite) reported(ended <-chan phone.CallStatus) phone.CallStatus {
	select {
	case status := <-ended:
		return status
	case <-time.After(2 * time.Second):
		s.FailNow("the call never said how it ended")
		return phone.CallStatus{}
	}
}

func (s *TakeoverSuite) TestAConnectedCallIsRecordedAndClaimedByItsNode() {
	a := s.node("a")

	id := s.dial(a)

	rec := s.record(id)
	s.Equal(id, rec.ID)
	s.Equal(*ref(), rec.Ref)
	s.Equal(callState("a"), rec.State)
	s.Equal("a", s.get("owner:"+id))
	lapse, inSet := s.lapse(id)
	s.Require().True(inSet)
	s.True(lapse.After(time.Now()), "a call its node renews does not look lapsed")
}

func (s *TakeoverSuite) TestTheRecordHoldsNoPassword() {
	a := s.node("a")

	id := s.dial(a)

	raw := s.get("call:" + id)
	placed := outbound()
	s.NotContains(raw, placed.Trunk.Password)
	s.NotContains(raw, placed.Bridge.Password)
}

func (s *TakeoverSuite) TestAnEndedCallLeavesNothingInRedis() {
	a := s.node("a")
	id := s.dial(a)

	a.dialer.call.bye()

	s.Equal(phone.CallStatus{Result: phone.CallCompleted, Word: "bye", SIPCode: 200}, s.reported(a.ended))
	s.Eventually(func() bool { return s.gone(id) }, time.Second, 10*time.Millisecond)
}

func (s *TakeoverSuite) TestAChangedSDPRewritesTheRecord() {
	a := s.node("a")
	id := s.dial(a)

	a.dialer.call.setState(callState("changed"))
	a.dialer.mu.Lock()
	onChange := a.dialer.config.OnChange
	a.dialer.mu.Unlock()
	s.Require().NotNil(onChange, "the bridge has somewhere to say the call changed")
	onChange()

	s.Eventually(func() bool { return string(s.record(id).State.Customer.LastSDP) == "v=0 changed" },
		time.Second, 10*time.Millisecond)
	s.Equal(callState("changed"), s.record(id).State)
}

func (s *TakeoverSuite) TestCloseHandsTheCallToAnotherNode() {
	// Claims far longer than Close waits, so the call moves because a gave it up, not
	// because a's claim lapsed.
	s.holdFor = time.Minute
	a, b := s.node("a"), s.node("b")
	id := s.dial(a)

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	s.Require().NoError(a.provider.Close(ctx))

	// a lets go once b holds the claim, which can be just before b resumes.
	s.Eventually(func() bool { return live(b.provider, id) }, time.Second, 10*time.Millisecond)
	s.Require().Equal(1, b.takeover.resumes())
	b.takeover.mu.Lock()
	cfg, state := b.takeover.configs[0], b.takeover.states[0]
	b.takeover.mu.Unlock()
	s.Equal(callState("a"), state)
	s.Equal(sipbridge.CustomerTrunk{
		Host: "trunk-b.example.com", Port: 5061, Transport: "tls", Username: "agent-b",
		Password: resumedTrunkPassword, LateOffer: true, Codecs: []string{"PCMA"},
	}, cfg.CustomerTrunk)
	s.Equal(sipbridge.StreamTrunk{
		URI: "sip:+15550000301@bridge-b.sip.example.com", Username: "stream-b", Password: resumedBridgePassword,
	}, cfg.Stream)
	s.True(a.dialer.call.wasReleased())
	s.False(a.dialer.call.wasHungUp())
	s.noReport(a.ended)
	s.Equal("b", s.get("owner:"+id))
}

func (s *TakeoverSuite) TestClosingTwiceHandsTheCallOverOnce() {
	s.holdFor = time.Minute
	a, b := s.node("a"), s.node("b")
	id := s.dial(a)

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	s.Require().NoError(a.provider.Close(ctx))
	s.Eventually(func() bool { return live(b.provider, id) }, time.Second, 10*time.Millisecond)

	start := time.Now()
	s.Require().NoError(a.provider.Close(ctx))

	s.Less(time.Since(start), 200*time.Millisecond)
	s.Equal(1, b.takeover.resumes())
	s.False(a.dialer.call.wasHungUp())
	s.Equal("b", s.get("owner:"+id))
}

func (s *TakeoverSuite) TestCloseWithNoOtherNodeLeavesTheCallForLater() {
	a := s.node("a")
	id := s.dial(a)

	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Millisecond)
	defer cancel()
	s.ErrorIs(a.provider.Close(ctx), context.DeadlineExceeded)

	s.Eventually(a.dialer.call.wasReleased, time.Second, 10*time.Millisecond)
	s.False(a.dialer.call.wasHungUp())
	s.NotEmpty(s.get("call:" + id))

	b := s.node("b")
	s.Eventually(func() bool { return b.takeover.resumes() == 1 }, 2*time.Second, 10*time.Millisecond)
}

func (s *TakeoverSuite) TestALapsedClaimIsTakenByAScan() {
	a := s.node("a")
	id := s.dial(a)
	b := s.node("b")

	a.stop()

	s.Eventually(func() bool { return b.takeover.resumes() == 1 }, testHoldFor+2*testRenewEvery, 10*time.Millisecond)
	s.Eventually(func() bool { return s.get("owner:"+id) == "b" }, time.Second, 10*time.Millisecond)
}

func (s *TakeoverSuite) TestOnlyOneNodeTakesALapsedCall() {
	a := s.node("a")
	s.dial(a)
	b, c := s.node("b"), s.node("c")

	a.stop()

	s.Eventually(func() bool { return b.takeover.resumes()+c.takeover.resumes() >= 1 }, 2*time.Second, 10*time.Millisecond)
	// Waits on purpose: the check is that no second node takes the call over.
	time.Sleep(testHoldFor + 2*testRenewEvery)
	s.Equal(1, b.takeover.resumes()+c.takeover.resumes())
}

func (s *TakeoverSuite) TestAFailedTakeoverReportsTheCallEndedAndClearsIt() {
	a := s.node("a")
	id := s.dial(a)
	b := s.node("b")
	b.takeover.err = fmt.Errorf("sipbridge: customer leg: %w", &sipbridge.Unanswered{Code: 481, Reason: "Call/Transaction Does Not Exist"})

	a.stop()

	s.Equal(phone.CallStatus{Result: phone.CallCompleted, Word: "bye", SIPCode: 200}, s.reported(b.takeover.ended))
	s.Eventually(func() bool { return s.gone(id) }, time.Second, 10*time.Millisecond)
	s.False(live(b.provider, id))
}

func (s *TakeoverSuite) TestACallWhoseCredentialsCannotBeOpenedIsDropped() {
	a := s.node("a")
	id := s.dial(a)
	b := s.node("b")
	b.takeover.lookupErrs = []error{fmt.Errorf("phone: %w", store.ErrNoSIPTrunk)}

	a.stop()

	s.Eventually(func() bool { return s.gone(id) }, 2*time.Second, 10*time.Millisecond)
	s.Zero(b.takeover.resumes())
}

func (s *TakeoverSuite) TestACallOnAStreamTrunkThatIsGoneIsDropped() {
	a := s.node("a")
	id := s.dial(a)
	b := s.node("b")
	b.takeover.lookupErrs = []error{fmt.Errorf("phone: %w", store.ErrNoStreamTrunk)}

	a.stop()

	s.Eventually(func() bool { return s.gone(id) }, 2*time.Second, 10*time.Millisecond)
	s.Zero(b.takeover.resumes())
}

func (s *TakeoverSuite) TestACallWhoseCredentialsFailToOpenOnceIsTakenLater() {
	a := s.node("a")
	id := s.dial(a)
	b := s.node("b")
	b.takeover.lookupErrs = []error{errors.New("store: sip trunk: connection refused")}

	a.stop()

	s.Eventually(func() bool { return b.takeover.resumes() == 1 }, 3*time.Second, 10*time.Millisecond)
	s.Eventually(func() bool { return s.get("owner:"+id) == "b" }, time.Second, 10*time.Millisecond)
}

func (s *TakeoverSuite) TestANodeThatDiesWhileResumingLeavesTheNextACSeqItDidNotSend() {
	jump := sipbridge.ResumeConfig{}.WithDefaults().CSeqJump
	a := s.node("a")
	id := s.dial(a)
	b := s.node("b")
	release := make(chan struct{})
	defer close(release)
	// b hangs mid-resume, after its re-INVITEs may have gone out, as a node that died would.
	hang := func() error {
		<-release
		return nil
	}
	b.takeover.before = hang

	a.stop()

	s.Eventually(func() bool { return b.takeover.resumes() == 1 }, 2*time.Second, 10*time.Millisecond)
	rec := s.record(id)
	s.Equal(1+jump, rec.State.Customer.LocalCSeq)
	s.Equal(1+jump, rec.State.Stream.LocalCSeq)

	c := s.node("c")
	c.takeover.before = hang
	s.Eventually(func() bool { return c.takeover.resumes() == 1 }, 2*time.Second, 10*time.Millisecond)
	c.takeover.mu.Lock()
	state := c.takeover.states[0]
	c.takeover.mu.Unlock()
	s.Equal(1+jump, state.Customer.LocalCSeq, "c sends past what b may have sent")
	s.Equal(1+jump, state.Stream.LocalCSeq, "c sends past what b may have sent")
	s.Equal(1+2*jump, s.record(id).State.Customer.LocalCSeq)
}

func (s *TakeoverSuite) TestAnUnusableRecordIsDropped() {
	b := s.node("b")
	id := uuid.NewString()
	broken, err := json.Marshal(callRecord{ID: id, Ref: *ref()})
	s.Require().NoError(err)
	s.Require().NoError(s.redis.Do(context.Background(), s.redis.B().Set().Key(s.prefix+":call:"+id).Value(string(broken)).Build()).Error())
	s.Require().NoError(s.redis.Do(context.Background(), s.redis.B().Zadd().Key(s.prefix+":calls").ScoreMember().ScoreMember(0, id).Build()).Error())

	s.Eventually(func() bool { return s.gone(id) }, 2*time.Second, 10*time.Millisecond)
	s.Zero(b.takeover.resumes(), "a record with no dialog to continue is not resumed")
}

func (s *TakeoverSuite) TestACallThatCannotMoveIsNotRecordedAndIsHungUpAtClose() {
	a := s.node("a")
	b := s.node("b")
	dialed, err := a.provider.Dial(context.Background(), outbound())
	s.Require().NoError(err)
	p := a.provider
	s.Eventually(func() bool { return live(p, dialed.VendorCallID) }, time.Second, 10*time.Millisecond)

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	s.Require().NoError(a.provider.Close(ctx))

	s.True(a.dialer.call.wasHungUp())
	s.Empty(s.get("call:" + dialed.VendorCallID))
	s.Zero(b.takeover.resumes())
}

func (s *TakeoverSuite) TestANodeThatLostItsClaimLetsTheCallGo() {
	a := s.node("a")
	id := s.dial(a)

	s.Require().NoError(s.redis.Do(context.Background(), s.redis.B().Set().Key(s.prefix+":owner:"+id).Value("z").Build()).Error())

	s.Eventually(a.dialer.call.wasReleased, time.Second, 10*time.Millisecond)
	s.False(a.dialer.call.wasHungUp())
	s.noReport(a.ended)
	s.Equal("z", s.get("owner:"+id))
}

func (s *TakeoverSuite) TestAByeDuringTheHandoffEndsTheCallOnce() {
	a, b := s.node("a"), s.node("b")
	id := s.dial(a)
	// The BYE reaches a while b is taking the call over.
	b.takeover.beforeLookup = func() {
		a.dialer.call.bye()
		deadline := time.Now().Add(time.Second)
		for s.exists("call:"+id) && time.Now().Before(deadline) {
			time.Sleep(10 * time.Millisecond)
		}
	}
	// As a peer would: a re-INVITE in a dialog still up is answered, one in an ended dialog is
	// refused. A node that left the ended call's record behind is resuming a live call.
	b.takeover.before = func() error {
		if s.exists("call:" + id) {
			return nil
		}
		return &sipbridge.Unanswered{Code: 481, Reason: "Call/Transaction Does Not Exist"}
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	s.Require().NoError(a.provider.Close(ctx))

	s.Equal(phone.CallStatus{Result: phone.CallCompleted, Word: "bye", SIPCode: 200}, s.reported(a.ended))
	s.Eventually(func() bool { return s.gone(id) }, time.Second, 10*time.Millisecond)
	s.Never(func() bool { return len(a.ended) > 0 || live(b.provider, id) }, time.Second, 10*time.Millisecond,
		"the call was reported again, or a node went on holding it")
}

func (s *TakeoverSuite) TestCloseOffersItsCallsWhileATakeoverIsStillResuming() {
	// Claims far longer than the test, so a call moves only because a gave it up.
	s.holdFor = time.Minute
	a := s.node("a")
	held := s.dial(a)
	resuming := make(chan struct{})
	release := make(chan struct{})
	defer close(release)
	a.takeover.before = func() error {
		close(resuming)
		<-release
		return nil
	}
	s.seed(uuid.NewString())
	select {
	case <-resuming:
	case <-time.After(2 * time.Second):
		s.FailNow("a never started taking the seeded call over")
	}
	b := s.node("b")

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	s.Error(a.provider.Close(ctx), "a takeover still running at the deadline")

	s.Eventually(func() bool { return s.get("owner:"+held) == "b" }, time.Second, 10*time.Millisecond)
	s.Eventually(a.dialer.call.wasReleased, time.Second, 10*time.Millisecond)
	s.False(a.dialer.call.wasHungUp())
	s.Equal(1, b.takeover.resumes())
}

func (s *TakeoverSuite) TestACallThatConnectsAsTheNodeStopsIsHandedOver() {
	s.holdFor = time.Minute
	a := s.node("a")
	s.node("b")
	a.dialer.gate = make(chan struct{})
	placing := outbound()
	placing.Resume = ref()
	placing.Ended = func(status phone.CallStatus) { a.ended <- status }
	dialed, err := a.provider.Dial(context.Background(), placing)
	s.Require().NoError(err)
	<-a.dialer.dialled

	closed := make(chan error, 1)
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	go func() { closed <- a.provider.Close(ctx) }()
	p := a.provider
	s.Eventually(func() bool { return p.ctx.Err() != nil }, time.Second, time.Millisecond)
	close(a.dialer.gate)

	select {
	case err := <-closed:
		s.NoError(err)
	case <-time.After(4 * time.Second):
		s.FailNow("Close never returned")
	}
	s.Equal("b", s.get("owner:"+dialed.VendorCallID))
	s.True(a.dialer.call.wasReleased())
	s.noReport(a.ended)
}

func (s *TakeoverSuite) TestAGivenUpCallIsTakenOnTheNoticeBeforeAnyScan() {
	// No claim lapses and no scan runs in the test, so only the notice can move the call.
	s.holdFor, s.renewEvery = 2*time.Minute, time.Minute
	a, b := s.node("a"), s.node("b")
	id := s.dial(a)
	s.Eventually(func() bool {
		subscribers, err := s.redis.Do(context.Background(), s.redis.B().PubsubNumsub().Channel(s.prefix+":free").Build()).AsIntMap()
		return err == nil && subscribers[s.prefix+":free"] > 0
	}, time.Second, 10*time.Millisecond, "b never listened for calls given up")

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	s.Require().NoError(a.provider.Close(ctx))

	s.Equal("b", s.get("owner:"+id))
	s.Equal(1, b.takeover.resumes())
}

func (s *TakeoverSuite) TestADialledCallWhoseRecordCannotBeWrittenLeavesNothingToResume() {
	// A set of calls that is not a set makes the claim fail after it wrote the record and owner,
	// as a reply lost after the script ran would.
	s.Require().NoError(s.redis.Do(context.Background(), s.redis.B().Set().Key(s.prefix+":calls").Value("not a set").Build()).Error())
	a := s.node("a")
	dialed, err := a.provider.Dial(context.Background(), func() phone.Outbound {
		placing := outbound()
		placing.Resume = ref()
		return placing
	}())
	s.Require().NoError(err)
	p := a.provider
	s.Eventually(func() bool { return live(p, dialed.VendorCallID) }, time.Second, 10*time.Millisecond)

	s.Empty(s.get("call:" + dialed.VendorCallID))
	s.Empty(s.get("owner:" + dialed.VendorCallID))
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	s.Require().NoError(a.provider.Close(ctx))
	s.True(a.dialer.call.wasHungUp(), "a call no other node can continue is hung up")
}

func (s *TakeoverSuite) TestATakenOverCallWhoseRecordCannotBeWrittenStaysUnderItsClaim() {
	s.holdFor = time.Minute
	a := s.node("a")
	id := s.dial(a)
	b := s.node("b")
	// Breaks the claim b writes once it resumed, after b won the call.
	b.takeover.before = func() error {
		return s.redis.Do(context.Background(), s.redis.B().Set().Key(s.prefix+":calls").Value("not a set").Build()).Error()
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	s.Require().NoError(a.provider.Close(ctx))
	s.Eventually(func() bool { return live(b.provider, id) }, time.Second, 10*time.Millisecond)

	closeB, cancelB := context.WithTimeout(context.Background(), 300*time.Millisecond)
	defer cancelB()
	_ = b.provider.Close(closeB)

	s.Eventually(b.takeover.call.wasReleased, time.Second, 10*time.Millisecond)
	s.False(b.takeover.call.wasHungUp(), "the node's claim still stands, so the call is offered on, not hung up")
}

// seed records a lapsed call no node holds.
func (s *TakeoverSuite) seed(id string) {
	rec, err := json.Marshal(callRecord{ID: id, Ref: *ref(), State: callState("seeded")})
	s.Require().NoError(err)
	s.Require().NoError(s.redis.Do(context.Background(), s.redis.B().Set().Key(s.prefix+":call:"+id).Value(string(rec)).Build()).Error())
	s.Require().NoError(s.redis.Do(context.Background(), s.redis.B().Zadd().Key(s.prefix+":calls").ScoreMember().ScoreMember(0, id).Build()).Error())
}
