package agent

import (
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// holdsHeard has the agent's owner (Options.OnHeard) hold each Heard it is told of until
// release, handing it over on told meanwhile.
func (s *AgentSuite) holdsHeard() (told <-chan Heard, release func()) {
	heard, released := make(chan Heard, 4), make(chan struct{})
	var once sync.Once
	release = func() { once.Do(func() { close(released) }) }
	s.T().Cleanup(release)
	s.onHeard = func(event Heard) {
		heard <- event
		<-released
	}
	return heard, release
}

// toldBeforeReported checks the owner was told of text, and that nothing was reported heard
// while it held the words: the agent tells it before it sends them.
func (s *AgentSuite) toldBeforeReported(told <-chan Heard, release func(), text string) {
	select {
	case heard := <-told:
		s.Equal(text, heard.Text)
	case <-time.After(settleFor):
		s.FailNow("the owner was never told what was heard")
	}
	s.Never(func() bool { return countOf[Heard](s.reported()) > 0 }, 200*time.Millisecond, 5*time.Millisecond,
		"the words were reported heard before the owner was told of them")
	release()
	s.eventually(func() bool { return countOf[Heard](s.reported()) == 1 }, "the words were never reported heard")
}

func (s *AgentSuite) TestTheOwnerIsToldOfACallersWordsBeforeTheyAreReportedHeard() {
	told, release := s.holdsHeard()
	s.join(false)
	alice := stt.Participant{ID: "alice"}
	s.speak(alice)

	s.says(alice, "who is this?")

	s.toldBeforeReported(told, release, "who is this?")
}

func (s *AgentSuite) TestTheOwnerIsToldOfWordsTypedIntoACallBeforeTheyAreReportedHeard() {
	told, release := s.holdsHeard()
	s.join(false)

	go func() { _, _ = s.agent.RespondTo(s.ctx, "my order number is 12", nil, options.LLM{}) }()

	s.toldBeforeReported(told, release, "my order number is 12")
}

func (s *AgentSuite) TestTheOwnerIsToldOfWhatANativeModelHeardBeforeItIsReportedHeard() {
	told, release := s.holdsHeard()
	peer := s.joinNativeWith(func(options *Options) { options.OnHeard = s.onHeard })

	s.nativeSend(peer, `{"type":"conversation.item.input_audio_transcription.completed","transcript":"And for eight people?"}`)

	s.toldBeforeReported(told, release, "And for eight people?")
}
