package streamedge

import (
	"sync"
	"time"

	rtc "github.com/GetStream/getstream-go-webrtc"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
)

func (s *StreamEdgeSuite) TestTheSFUIsTheEdgeNameAndTheHostOfItsURL() {
	state := &rtc.CallState{
		EdgeName: "sfu-a1",
		Url:      "https://sfu-a1.example.com/twirp?token=secret-value",
		Token:    "secret-value",
	}

	s.Equal(agent.SFU{Edge: "sfu-a1", Host: "sfu-a1.example.com"}, sfuOf(state))
}

func (s *StreamEdgeSuite) TestNoStateMeansNoSFU() {
	s.Equal(agent.SFU{}, sfuOf(nil))
}

func (s *StreamEdgeSuite) TestAMoveToAnotherSFUIsReportedOnce() {
	a := agent.SFU{Edge: "sfu-a1", Host: "sfu-a1.example.com"}
	b := agent.SFU{Edge: "sfu-b2", Host: "sfu-b2.example.com"}
	reads := make(chan agent.SFU, 4)
	for _, sfu := range []agent.SFU{a, a, b, b} {
		reads <- sfu
	}
	stop := make(chan struct{})
	type move struct{ was, now agent.SFU }
	var mu sync.Mutex
	var moves []move
	returned := make(chan struct{})

	go func() {
		defer close(returned)
		watchSFU(func() agent.SFU {
			select {
			case sfu := <-reads:
				return sfu
			default:
				return b
			}
		}, time.Millisecond, stop, func(was, now agent.SFU) {
			mu.Lock()
			defer mu.Unlock()
			moves = append(moves, move{was, now})
		})
	}()
	s.Eventually(func() bool { mu.Lock(); defer mu.Unlock(); return len(moves) == 1 },
		time.Second, time.Millisecond, "the move was never reported")
	time.Sleep(20 * time.Millisecond)
	close(stop)

	select {
	case <-returned:
	case <-time.After(time.Second):
		s.Fail("the watch kept going after stop")
	}
	mu.Lock()
	defer mu.Unlock()
	s.Equal([]move{{a, b}}, moves)
}
