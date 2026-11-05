//go:build integration

package session

import (
	"time"

	"github.com/google/uuid"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// threadStory is the example of the link's history on a new SMS thread of number: the
// allergy on Sunday, the appointment on Monday, a link to a user on Tuesday as history says
// (none when linked is false), and Wednesday's text, which the session answers. Its agent
// config has the cards off, so the session is handed the thread's history and nothing else.
func (s *EpisodeReadingSuite) threadStory(number string, linked bool, history *store.PhoneHistory) string {
	thread := s.thread("sms-author")
	s.textFrom(number, thread, -7200, "allergic to penicillin")
	s.reply(thread, -7190)
	s.textFrom(number, thread, -5000, "appointment at 15:00")
	s.reply(thread, -4990)
	if linked {
		s.linkWith("bob-"+uuid.NewString(), s.phone(number).Address, -3000, history)
	}
	s.textFrom(number, thread, -60, "it is Bob")
	s.reply(thread, -50)
	s.at(0)
	return conversationOf(s.textSession(thread, false, "what do you know about me? "+uuid.NewString()))
}

// reply writes the agent's answer into a thread at seconds after the test's start, as the
// conversation writes it: the people's messages before a reply are history, the ones after the
// last reply are what the session is about to be told.
func (s *EpisodeReadingSuite) reply(thread string, seconds int) {
	at := s.at(seconds).Format(time.RFC3339Nano)
	s.write(thread, "agent-user", "Noted. "+uuid.NewString(), map[string]any{
		"support_runtime": map[string]any{"role": "assistant", "state": "completed", "answer_start": 0,
			"response_started_at": at, "state_started_at": at, "duration_ms": 0},
		"support_message": map[string]any{"schema_version": 1, "sequence": 0, "state": "completed"},
	})
}

// Defect: a session on a linked number's SMS thread is handed everything the thread holds, so
// Bob hears what the number's earlier holder texted. The link's boundary cuts it: by default
// and with false at the link, with phone_history_since from that time, and with true nothing.
// What the number texted after the link is handed in every case.
func (s *EpisodeReadingSuite) TestAThreadSessionOfALinkedNumberIsHandedItsHistoryFromTheLinksBoundary() {
	since := s.now.Add(-6000 * time.Second)
	for _, c := range []struct {
		name    string
		number  string
		history *store.PhoneHistory
		handed  []string
		hidden  []string
	}{
		{"neither field", "+15550100111", nil, nil, []string{"penicillin", "15:00"}},
		{"false", "+15550100112", &store.PhoneHistory{}, nil, []string{"penicillin", "15:00"}},
		{"since", "+15550100113", &store.PhoneHistory{Since: since}, []string{"15:00"}, []string{"penicillin"}},
		{"true", "+15550100114", &store.PhoneHistory{All: true}, []string{"penicillin", "15:00"}, nil},
	} {
		told := s.threadStory(c.number, true, c.history)

		s.Contains(told, "it is Bob", c.name)
		for _, said := range c.handed {
			s.Contains(told, said, c.name)
		}
		for _, said := range c.hidden {
			s.NotContains(told, said, c.name)
		}
	}
}

// The control: a session on the thread of a number linked to nobody is handed the whole
// thread, as the probe of this test at 518ac7e9f (before this change) answered.
func (s *EpisodeReadingSuite) TestAThreadSessionOfAnUnlinkedNumberIsHandedItsWholeHistory() {
	s.linkWith("carl-"+uuid.NewString(), s.phone("+15550100116").Address, -3000, nil)

	told := s.threadStory("+15550100115", false, nil)

	for _, said := range []string{"penicillin", "15:00", "it is Bob"} {
		s.Contains(told, said)
	}
}
