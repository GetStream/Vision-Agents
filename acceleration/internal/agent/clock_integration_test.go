//go:build integration

package agent_test

import (
	"context"
	"log/slog"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// clockAnswerWithin is how long a real model is given to answer one short question.
const clockAnswerWithin = time.Minute

// ClockIntegrationSuite asks a real model about dates it can only know from the clock line,
// with no tool to look the time up: what the clock line says has to be enough.
type ClockIntegrationSuite struct {
	suite.Suite
	ctx context.Context
	llm *llmrouter.Router
}

func TestClockIntegrationSuite(t *testing.T) {
	suite.Run(t, new(ClockIntegrationSuite))
}

func (s *ClockIntegrationSuite) SetupSuite() {
	if os.Getenv("BASETEN_API_KEY") == "" {
		s.T().Skip("BASETEN_API_KEY not set")
	}
	var cancel context.CancelFunc
	s.ctx, cancel = context.WithTimeout(context.Background(), 5*time.Minute)
	s.T().Cleanup(cancel)

	config, err := routing.DefaultConfig()
	s.Require().NoError(err)
	s.llm, err = llmrouter.New(llmrouter.Options{
		Config: config[routing.LLM], Registry: llmrouter.DefaultRegistry(), Logger: slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)
	s.T().Cleanup(s.llm.Close)
}

// ask opens a text agent for a user in zone, empty for one whose zone is not known, and
// returns its answer to question.
func (s *ClockIntegrationSuite) ask(zone, question string) string {
	booking, err := agent.New(agent.Options{
		Text:         true,
		Instructions: "You book appointments for a clinic. Answer in one short sentence.",
		Timezone:     zone,
		CustomerID:   "clock-integration",
		LLM:          s.llm,
		LLMTarget:    "llm-fast",
		Logger:       slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)
	s.Require().NoError(booking.Join(s.ctx))
	s.T().Cleanup(func() { _ = booking.Close() })

	turn, err := booking.RespondTo(s.ctx, question, nil, options.LLM{})
	s.Require().NoError(err)
	deadline := time.After(clockAnswerWithin)
	for {
		select {
		case event := <-booking.Events():
			switch typed := event.(type) {
			case agent.Responded:
				if typed.TurnID == turn && !typed.PendingWork {
					return typed.Text
				}
			case agent.Error:
				s.FailNowf("the agent failed", "%s: %v", typed.Context, typed.Err)
			}
		case <-deadline:
			s.FailNow("the model never answered")
		}
	}
}

func (s *ClockIntegrationSuite) TestTheModelKnowsTomorrowsDateInTheUsersZone() {
	zone, err := time.LoadLocation("Pacific/Auckland")
	s.Require().NoError(err)
	tomorrow := time.Now().In(zone).AddDate(0, 0, 1)

	answer := s.ask("Pacific/Auckland",
		"What is tomorrow's date? Answer with the weekday and the day of the month only.")

	s.Contains(answer, tomorrow.Weekday().String(), answer)
	s.Contains(answer, strconv.Itoa(tomorrow.Day()), answer)
}

func (s *ClockIntegrationSuite) TestTheModelKnowsTodaysDateWithoutTheUsersZone() {
	today := time.Now().UTC()

	answer := s.ask("", "What is today's date? Answer with the weekday, the day and the month only.")

	s.Contains(answer, today.Weekday().String(), answer)
	s.Contains(answer, strconv.Itoa(today.Day()), answer)
	s.Contains(strings.ToLower(answer), strings.ToLower(today.Month().String()), answer)
}

func (s *ClockIntegrationSuite) TestTheModelWorksOutTheDateOfAWeekdayNextWeek() {
	zone, err := time.LoadLocation("Europe/Rome")
	s.Require().NoError(err)
	now := time.Now().In(zone)
	// The Wednesday of next week, Monday to Sunday, which is what "Wednesday next week" means
	// whichever day this runs on.
	monday := now.AddDate(0, 0, 7-(int(now.Weekday())+6)%7)
	wednesday := monday.AddDate(0, 0, 2)

	answer := s.ask("Europe/Rome",
		"I want to come in on Wednesday next week. Which date is that? Answer with the day and the month only.")

	s.Contains(answer, strconv.Itoa(wednesday.Day()), answer)
	s.Contains(strings.ToLower(answer), strings.ToLower(wednesday.Month().String()), answer)
}
