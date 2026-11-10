//go:build integration

package store

import (
	"fmt"
	"sync"
	"time"
)

// answering is a live config of the customer naming the channel types given, made at.
func (s *StoreSuite) answering(customerID string, at time.Time, types ...string) string {
	config := &AgentConfig{CustomerID: customerID, Name: "agent-" + newID()}
	for _, named := range types {
		config.ChatChannels = append(config.ChatChannels, ChatChannel{Type: named})
	}
	s.Require().NoError(s.store.CreateAgentConfig(s.ctx, config))
	_, err := s.store.DB().ExecContext(s.ctx, "UPDATE agent_configs SET created_at = ? WHERE id = ?", at, config.ID)
	s.Require().NoError(err)
	return config.ID
}

// The oldest of the customer's live configs naming the type answers in it: not a newer one,
// another type's, another customer's, a deleted one or a test copy.
func (s *StoreSuite) TestTheOldestConfigNamingAChannelTypeAnswersInIt() {
	customer, other := "acme-"+newID(), "other-"+newID()
	now := time.Now().UTC()
	deleted := s.answering(customer, now.Add(-4*time.Hour), "callouts")
	s.Require().NoError(s.store.DeleteAgentConfig(s.ctx, customer, deleted))
	copied := s.answering(customer, now.Add(-3*time.Hour), "callouts")
	_, err := s.store.DB().ExecContext(s.ctx, "UPDATE agent_configs SET tags = ?::jsonb WHERE id = ?", `{"draft_of": "x"}`, copied)
	s.Require().NoError(err)
	s.answering(other, now.Add(-2*time.Hour), "callouts")
	s.answering(customer, now.Add(-90*time.Minute), "shifts")
	oldest := s.answering(customer, now.Add(-time.Hour), "shifts", "callouts")
	s.answering(customer, now, "callouts")

	config, found, err := s.store.AgentConfigAnsweringChatChannel(s.ctx, customer, "callouts")

	s.Require().NoError(err)
	s.Require().True(found)
	s.Equal(oldest, config.ID)
	s.Equal([]ChatChannel{{Type: "shifts"}, {Type: "callouts"}}, config.ChatChannels)
}

// A type is matched whole: one no config names, or that only begins or ends like one, has none.
func (s *StoreSuite) TestAChannelTypeNoConfigNamesHasNone() {
	customer := "acme-" + newID()
	s.answering(customer, time.Now().UTC(), "callouts")
	s.answering(customer, time.Now().UTC())

	for _, asked := range []string{"call", "callouts2", "Callouts", "agent"} {
		_, found, err := s.store.AgentConfigAnsweringChatChannel(s.ctx, customer, asked)
		s.Require().NoError(err, asked)
		s.False(found, asked)
	}
}

// Two routers are two pools on one database; only one may run a channel's turn, in that app.
func (s *StoreSuite) TestTwoRoutersCannotHoldOneChatChannelsTurnAtOnce() {
	other, err := Open(s.dsn)
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = other.Close() })
	cid, until := "callouts:"+newID(), time.Now().Add(time.Minute)

	first, err := s.store.TakeChatChannelTurn(s.ctx, 7, cid, "router-a", until)
	s.Require().NoError(err)
	second, err := other.TakeChatChannelTurn(s.ctx, 7, cid, "router-b", until)
	s.Require().NoError(err)
	elsewhere, err := other.TakeChatChannelTurn(s.ctx, 8, cid, "router-b", until)
	s.Require().NoError(err)
	s.True(first)
	s.False(second, "router b waits while router a answers")
	s.True(elsewhere, "a channel of the same id in another app is another channel")

	s.Require().NoError(other.ReleaseChatChannelTurn(s.ctx, 7, cid, "router-b"))
	again, err := other.TakeChatChannelTurn(s.ctx, 7, cid, "router-b", until)
	s.Require().NoError(err)
	s.False(again, "a release by a holder that has no lease releases nothing")
	held, err := s.store.TakeChatChannelTurn(s.ctx, 8, cid, "router-c", until)
	s.Require().NoError(err)
	s.False(held, "a release in one app lets go of no lease in another")

	s.Require().NoError(s.store.ReleaseChatChannelTurn(s.ctx, 7, cid, "router-a"))
	after, err := other.TakeChatChannelTurn(s.ctx, 7, cid, "router-b", until)
	s.Require().NoError(err)
	s.True(after)
}

func (s *StoreSuite) TestTwoRoutersAskingAtOnceGetOneChatChannelTurn() {
	cid := "callouts:" + newID()
	pools := []*Store{s.store}
	for range 3 {
		pool, err := Open(s.dsn)
		s.Require().NoError(err)
		s.T().Cleanup(func() { _ = pool.Close() })
		pools = append(pools, pool)
	}
	var wg sync.WaitGroup
	taken := make([]bool, len(pools))
	for i, pool := range pools {
		wg.Add(1)
		go func() {
			defer wg.Done()
			ok, err := pool.TakeChatChannelTurn(s.ctx, 7, cid, fmt.Sprintf("router-%d", i), time.Now().Add(time.Minute))
			s.NoError(err)
			taken[i] = ok
		}()
	}
	wg.Wait()

	count := 0
	for _, ok := range taken {
		if ok {
			count++
		}
	}
	s.Equal(1, count)
}

// A router that stopped mid-turn leaves its lease; it runs out and the next router answers.
func (s *StoreSuite) TestAChatChannelTurnWhoseLeaseRanOutCanBeTaken() {
	cid := "callouts:" + newID()
	_, err := s.store.TakeChatChannelTurn(s.ctx, 7, cid, "router-a", time.Now().Add(time.Minute))
	s.Require().NoError(err)
	_, err = s.store.DB().ExecContext(s.ctx, "UPDATE chat_channel_turns SET turn_until = now() - interval '1 second' WHERE cid = ?", cid)
	s.Require().NoError(err)

	taken, err := s.store.TakeChatChannelTurn(s.ctx, 7, cid, "router-b", time.Now().Add(time.Minute))

	s.Require().NoError(err)
	s.True(taken)
}
