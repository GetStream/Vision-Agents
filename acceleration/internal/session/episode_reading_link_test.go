//go:build integration

package session

import (
	"encoding/json"
	"time"

	"github.com/google/uuid"

	"github.com/GetStream/Vision-Agents/acceleration/internal/omnichannel"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// A card in progress read for its lines (AI-2716): a thread channel holds every episode of its
// thread, one after another, so a number's SMS thread holds what it texted before a link set
// those episodes aside (store.LinkUserPhone), and an app chat's channel holds each run of the
// person's messages.

// inProgress is the status of a card whose episode is still open.
const inProgress = "in_progress"

// cardsConfig stores the suite's agent config with episode_cards on, as a link sets the
// number's history aside only for such a config (store.LinkUserPhone, retireHistory).
func (s *EpisodeReadingSuite) cardsConfig() {
	config := store.AgentConfig{CustomerID: s.customerID, Name: "cards-" + uuid.NewString(), Mode: store.AgentModeText,
		EpisodeCards: true}
	s.Require().NoError(s.store.CreateAgentConfig(s.ctx, &config))
	s.configID = config.ID
}

// textFrom is an SMS of number in thread at seconds after the test's start, as the channel
// bridge writes it: the episode opened (or found open) at the time the message was queued, a
// moment before Stream Chat made it (channelbridge.writeQueued), then the person's message.
func (s *EpisodeReadingSuite) textFrom(number, thread string, seconds int, text string) omnichannel.Opened {
	at := s.at(seconds)
	opened := s.episode(omnichannel.Episode{Person: s.phone(number), Source: store.EpisodeSMS, ThreadChannel: thread,
		StartedAt: at.Add(-time.Second), StreamAppPK: suiteDeploymentApp})
	s.write(thread, "sms-author", text, nil)
	return opened
}

// endEpisode closes an episode as the idle sweeper does an hour after its last message, before
// its summary.
func (s *EpisodeReadingSuite) endEpisode(opened omnichannel.Opened) {
	_, err := s.store.DB().ExecContext(s.ctx, "UPDATE episodes SET status = ?, ended_at = ? WHERE id = ?",
		store.EpisodeEnded, opened.Episode.StartedAt.Add(time.Hour), opened.Episode.ID)
	s.Require().NoError(err)
}

// cardLines is the text of each line of a card as the model was handed it.
func cardLines(card map[string]any) []string {
	var texts []string
	lines, _ := card["lines"].([]any)
	for _, said := range lines {
		texts = append(texts, said.(map[string]any)["text"].(string))
	}
	return texts
}

// Defect: once anya is linked to a number, her chat reads what the number texted before the
// link through the card of the number's next SMS episode, still in progress, whose lines were
// the last 20 of the thread channel whatever episode they were in. Whether the earlier episode
// had closed at the link or was still open, her chat reads none of it; the text after the link
// is hers once it opened an episode of its own.
func (s *EpisodeReadingSuite) TestALinkedUsersChatReadsNoneOfTheNumbersTextsFromBeforeTheLink() {
	for _, c := range []struct {
		number string
		closed bool
	}{{"+15550100101", true}, {"+15550100102", false}} {
		s.cardsConfig()
		user := "anya-" + uuid.NewString()
		thread := s.thread("sms-author")
		before := s.textFrom(c.number, thread, -7200, "my cat is Saffron")
		if c.closed {
			s.endEpisode(before)
		}
		s.at(-3600)
		chat := s.chatChannel(user)
		s.write(chat, user, "hello", nil)
		s.summarize(s.chatEpisode(user, chat, 0), "Said hello.")
		s.Require().NoError(s.store.LinkUserPhone(s.ctx, &store.UserPhone{CustomerID: s.customerID,
			StreamAppPK: suiteDeploymentApp, UserID: user, Phone: s.phone(c.number).Address}))
		s.textFrom(c.number, thread, -60, "my dog is Pepper")
		s.at(0)
		next := s.chatChannel(user)
		asked := "what are my pets called? " + uuid.NewString()
		s.write(next, user, asked, nil)

		handed, err := json.Marshal(s.handed(s.deploymentChat(next, user, true, asked)))
		s.Require().NoError(err)

		s.NotContains(string(handed), "Saffron", "closed at the link: %v", c.closed)
		s.Contains(string(handed), "Said hello.", "her own chat card is read")
		if c.closed {
			s.Contains(string(handed), "Pepper", "the text after the link is hers")
		}
	}
}

// Defect: the card of an SMS episode in progress hands over the lines of the thread's earlier
// episode too, which has its own card. A call from the number, linked to nobody, reads each
// episode's card with its own lines.
func (s *EpisodeReadingSuite) TestAnSMSCardInProgressHasOnlyItsOwnEpisodesLines() {
	s.cardsConfig()
	thread := s.thread("sms-author")
	s.endEpisode(s.textFrom("+15550100100", thread, -7200, "my cat is Saffron"))
	s.textFrom("+15550100100", thread, -60, "my dog is Pepper")
	s.at(0)

	params, _ := s.voiceSession("sip-+15550100100", true, "hello")

	handed := s.handed(params)
	s.Require().Len(handed, 2)
	s.Equal([]string{"my cat is Saffron"}, cardLines(handed[0]), "the earlier episode, ended")
	s.Equal(inProgress, handed[1]["status"])
	s.Equal([]string{"my dog is Pepper"}, cardLines(handed[1]), "the episode in progress, from its start")
}

// Defect, the sibling in the app chat: an agent channel holds each run of the person's messages
// as an episode of its own (api.chatEpisode), and the card of the run in progress hands over
// the runs before it. Anya's next chat reads each run's card with its own lines.
func (s *EpisodeReadingSuite) TestAChatCardInProgressHasOnlyItsOwnEpisodesLines() {
	s.at(-7200)
	chat := s.chatChannel("anya_123")
	s.write(chat, "anya_123", "my cat is Saffron", nil)
	s.endEpisode(s.chatEpisode("anya_123", chat, 0))
	s.at(-60)
	s.write(chat, "anya_123", "my dog is Pepper", nil)
	s.chatEpisode("anya_123", chat, 0)
	s.at(0)
	next := s.chatChannel("anya_123")
	s.write(next, "anya_123", "what are my pets called?", nil)

	handed := s.handed(s.deploymentChat(next, "anya_123", true, "what are my pets called?"))

	s.Require().Len(handed, 2)
	s.Equal([]string{"my cat is Saffron"}, cardLines(handed[0]), "the earlier run, ended")
	s.Equal(inProgress, handed[1]["status"])
	s.Equal([]string{"my dog is Pepper"}, cardLines(handed[1]), "the run in progress, from its start")
}
