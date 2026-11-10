//go:build integration

package api

import (
	"context"
	"crypto/rand"
	"errors"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/node"
	"github.com/GetStream/Vision-Agents/acceleration/internal/scheduling"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
)

// beforeItsLook is how long a delivery that waits scheduledPoll = an hour is watched for answering
// early: past the 2 s a hardcoded default would wait. It only ever ends a wait that should have
// held, so a loaded machine can make a broken wait pass this test, never a sound one fail.
const beforeItsLook = 3 * time.Second

// looksEvery has the router's deliveries look at a holding session again after d, and puts the
// look back when the test ends. It checks the look's default first.
func (s *ScheduledTasksSuite) looksEvery(d time.Duration) {
	s.Require().Equal(2*time.Second, s.router.scheduledPoll, "the session look's default")
	look := s.router.scheduledPoll
	s.router.scheduledPoll = d
	s.T().Cleanup(func() { s.router.scheduledPoll = look })
}

// refusingPeer is a node the directory says runs the session sessionID, whose hand-off is answered
// 503, as a node that is up but cannot take the delivery does. asked says each hand-off it got;
// move changes the directory's entry for the session, to the node asking (here) or to none.
func (s *ScheduledTasksSuite) refusingPeer(sessionID string) (asked <-chan struct{}, move func(here bool)) {
	got := make(chan struct{}, 16)
	peer := httptest.NewUnstartedServer(nil)
	directory := s.nodeDirectory(peer, s.otherShard, slog.New(slog.DiscardHandler))
	peer.Config.Handler = node.Serve(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		select {
		case got <- struct{}{}:
		default:
		}
		w.WriteHeader(http.StatusServiceUnavailable)
	}), s.router.peerCredentials)
	peer.Start()
	s.T().Cleanup(peer.Close)
	agent := rand.Text()
	directory.Hold(context.Background(), sessionID, agent)
	s.T().Cleanup(func() { directory.Release(context.Background(), sessionID) })
	return got, func(here bool) {
		directory.Release(context.Background(), sessionID)
		if here {
			s.router.directory.Hold(context.Background(), sessionID, agent)
			s.T().Cleanup(func() { s.router.directory.Release(context.Background(), sessionID) })
		}
	}
}

type handOff struct {
	handed bool
	err    error
}

// handingOff starts handDeliveryToPeer for a delivery of sessionID's, and says what it answered.
func (s *ScheduledTasksSuite) handingOff(ctx context.Context, sessionID string) <-chan handOff {
	done := make(chan handOff, 1)
	go func() {
		_, handed, err := s.router.handDeliveryToPeer(ctx, scheduledDelivery{SessionID: sessionID, TaskID: rand.Text()})
		done <- handOff{handed, err}
	}()
	return done
}

// A delivery whose peer did not take it looks at the session's directory entry again, and goes on
// where the entry says: the node running the session lapsed or the session came here, so it is the
// asking node's to deliver and nothing was handed over.
func (s *ScheduledTasksSuite) TestADeliveryAPeerDidNotTakeGoesOnHereOnceTheSessionIsNotThere() {
	for name, here := range map[string]bool{"the entry lapses": false, "the session moves here": true} {
		s.Run(name, func() {
			s.looksEvery(10 * time.Millisecond)
			sessionID := rand.Text()
			asked, move := s.refusingPeer(sessionID)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			done := s.handingOff(ctx, sessionID)
			s.Require().Eventually(func() bool { return len(asked) > 0 }, deliveryWait, 10*time.Millisecond,
				"the peer was never asked")
			select {
			case answered := <-done:
				s.FailNow("the delivery gave up on the peer before the entry moved", "%v", answered)
			case <-time.After(100 * time.Millisecond):
			}

			move(here)

			select {
			case answered := <-done:
				s.NoError(answered.err)
				s.False(answered.handed, "the peer took nothing")
			case <-time.After(deliveryWait):
				s.FailNow("the delivery never saw the session leave the peer")
			}
		})
	}
}

// The wait for a peer's directory entry to move is scheduledPoll: a delivery looks again after it,
// not sooner (a hardcoded or no wait) and not later than it is told. The look is an hour away, so
// an entry that already moved is not seen, and the wait ends only with the delivery's context.
func (s *ScheduledTasksSuite) TestADeliveryWaitsItsLookBeforeLookingForAPeersEntryAgain() {
	s.looksEvery(time.Hour)
	sessionID := rand.Text()
	asked, move := s.refusingPeer(sessionID)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := s.handingOff(ctx, sessionID)
	s.Require().Eventually(func() bool { return len(asked) > 0 }, deliveryWait, 10*time.Millisecond,
		"the peer was never asked")
	move(false)

	select {
	case answered := <-done:
		s.FailNow("the delivery looked again before its look", "%v", answered)
	case <-time.After(beforeItsLook):
	}

	cancel()
	select {
	case answered := <-done:
		s.ErrorIs(answered.err, context.Canceled)
		s.False(answered.handed)
	case <-time.After(deliveryWait):
		s.FailNow("the wait for the next look outlived the delivery's context")
	}
}

// A peer that never takes the delivery and never lets go of the session is waited on for
// scheduledPeerWait, then the delivery fails as the node unreachable. The default is 30 s; the wait
// here is 100 ms, and the answer is waited for well under that default, so a wait that ignores the
// field does not answer in time.
func (s *ScheduledTasksSuite) TestADeliveryGivesUpOnAPeerThatKeepsTheSessionAfterItsWait() {
	s.Require().Equal(30*time.Second, s.router.scheduledPeerWait, "the peer wait's default")
	wait := s.router.scheduledPeerWait
	s.router.scheduledPeerWait = 100 * time.Millisecond
	s.T().Cleanup(func() { s.router.scheduledPeerWait = wait })
	s.looksEvery(10 * time.Millisecond)
	sessionID := rand.Text()
	s.refusingPeer(sessionID)

	select {
	case answered := <-s.handingOff(context.Background(), sessionID):
		s.EqualError(answered.err, "the node running this session could not be reached")
		s.False(answered.handed)
	case <-time.After(10 * time.Second):
		s.FailNow("the delivery never gave up on the peer")
	}
}

// waitingOutACall is a voice session of a user's, and waitOutCall running on it for up to window.
func (s *ScheduledTasksSuite) waitingOutACall(ctx context.Context, window time.Duration) (*session.Session, <-chan error) {
	user := s.data.createUser()
	opened := s.openScheduling(user, s.agentWithScheduling(nil), func(request *CreateSessionRequest) {
		request.StartVoice = pointerTo(true)
	})
	found, live := s.router.sessions.Get(opened.Id, s.ownerOf(user))
	s.Require().True(live)
	done := make(chan error, 1)
	go func() { done <- s.router.waitOutCall(ctx, found, window) }()
	return found, done
}

// A task that comes due mid-call goes on once the call has ended: the delivery is waiting on a
// call when it ends, and sees that at its next look, one scheduledPoll later.
func (s *ScheduledTasksSuite) TestATaskWaitingOutACallGoesOnWhenTheCallEnds() {
	s.looksEvery(10 * time.Millisecond)
	found, done := s.waitingOutACall(context.Background(), time.Hour)
	select {
	case err := <-done:
		s.FailNow("the delivery did not wait for the call", "%v", err)
	case <-time.After(100 * time.Millisecond):
	}

	s.Require().NoError(found.StopVoice(context.Background()))

	select {
	case err := <-done:
		s.NoError(err)
	case <-time.After(deliveryWait):
		s.FailNow("the delivery never saw the call end")
	}
}

// A call that ends the session is not one to wait out: the occurrence goes to reopening.
func (s *ScheduledTasksSuite) TestATaskWaitingOutACallThatEndsTheSessionIsNotHere() {
	s.looksEvery(10 * time.Millisecond)
	found, done := s.waitingOutACall(context.Background(), time.Hour)
	select {
	case err := <-done:
		s.FailNow("the delivery did not wait for the call", "%v", err)
	case <-time.After(100 * time.Millisecond):
	}

	s.Require().NoError(found.Close())

	select {
	case err := <-done:
		s.ErrorIs(err, errScheduledNotHere)
	case <-time.After(deliveryWait):
		s.FailNow("the delivery never saw the session end")
	}
}

// A delivery waiting out a call looks again after scheduledPoll, not sooner (a hardcoded or no
// wait): with the look an hour away and the call's window one second, it holds past the window.
func (s *ScheduledTasksSuite) TestATaskWaitsItsLookBeforeLookingAtACallAgain() {
	s.looksEvery(time.Hour)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	_, done := s.waitingOutACall(ctx, time.Second)

	select {
	case err := <-done:
		s.FailNow("the delivery looked at the call again before its look",
			"%v (on a call: %v)", err, scheduling.ReasonOf(err) == scheduling.ReasonOnACall)
	case <-time.After(beforeItsLook):
	}

	cancel()
	select {
	case err := <-done:
		s.True(errors.Is(err, context.Canceled), "%v", err)
	case <-time.After(deliveryWait):
		s.FailNow("the wait for the next look outlived the delivery's context")
	}
}

// A call that outlasts the wait is looked at every scheduledPoll until the wait is spent, and
// then the user is on a call: the delivery looks again and again, and gives up only at the end.
func (s *ScheduledTasksSuite) TestATaskKeepsLookingAtACallUntilItsWaitIsSpent() {
	s.looksEvery(10 * time.Millisecond)
	_, done := s.waitingOutACall(context.Background(), 200*time.Millisecond)

	select {
	case err := <-done:
		s.Equal(scheduling.ReasonOnACall, scheduling.ReasonOf(err), "%v", err)
	case <-time.After(deliveryWait):
		s.FailNow("the delivery kept waiting past its window")
	}
}
