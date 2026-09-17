package typesafe

import (
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/suite"
)

// web stands in for the System One endpoint, so the wire contract can be tested without a key.
type web struct {
	server *httptest.Server

	path   string
	auth   string
	body   map[string]any
	status int
	// calls counts how often the endpoint was reached, which is how a test tells one request
	// carrying several questions from several requests carrying one each.
	calls int
	// respond is the JSON the stub answers with.
	respond string
}

func newWeb() *web {
	stub := &web{
		status:  http.StatusOK,
		respond: `{"model":"jev-1.13.0","answers":{},"usage":{"input_tokens":0,"output_tokens":0}}`,
	}
	stub.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		stub.calls++
		stub.path = r.URL.Path
		stub.auth = r.Header.Get("Authorization")

		raw, _ := io.ReadAll(r.Body)
		stub.body = map[string]any{}
		_ = json.Unmarshal(raw, &stub.body)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(stub.status)
		_, _ = w.Write([]byte(stub.respond))
	}))
	return stub
}

// questions digs the questions back out of what was sent.
func (s *TypeSafeSuite) questions() map[string]any {
	asked, ok := s.web.body["questions"].(map[string]any)
	s.Require().True(ok, "the request carried no questions")
	return asked
}

func (s *TypeSafeSuite) question(id string) map[string]any {
	asked, ok := s.questions()[id].(map[string]any)
	s.Require().True(ok, "the request did not ask %q", id)
	return asked
}

type TypeSafeSuite struct {
	suite.Suite
	ctx    context.Context
	web    *web
	client *Client
}

func TestTypeSafeSuite(t *testing.T) {
	suite.Run(t, new(TypeSafeSuite))
}

func (s *TypeSafeSuite) SetupTest() {
	s.ctx = context.Background()
	s.web = newWeb()
	s.T().Cleanup(s.web.server.Close)

	client, err := New(Options{
		APIKey:  "test-key",
		BaseURL: s.web.server.URL,
		Logger:  slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)
	s.client = client
}

func (s *TypeSafeSuite) TestAKeyIsRequired() {
	s.T().Setenv("TYPESAFE_API_KEY", "")

	_, err := New(Options{})

	s.ErrorContains(err, "TYPESAFE_API_KEY")
}

func (s *TypeSafeSuite) TestTheNewestStableModelIsAskedWhenNoneIsNamed() {
	s.web.respond = `{"model":"jev-1.13.0","answers":{"heard":{"type":"noul","noul":0.8}}}`

	_, err := s.client.Ask(s.ctx, "anything", map[string]Question{
		"heard": Noul("Did anyone speak?", "", ""),
	})
	s.Require().NoError(err)

	s.Equal(DefaultModel, s.client.Model())
	s.Equal(DefaultModel, s.web.body["model"])
	s.Equal("Bearer test-key", s.web.auth)
	s.Equal("/v1/systemone", s.web.path)
}

func (s *TypeSafeSuite) TestEveryQuestionGoesInOneRequest() {
	// Questions in one request share the state's tokens between them rather than paying for
	// it each, which is what makes asking one whose answer may not be needed close to free.
	s.web.respond = `{"model":"jev-1.13.0","answers":{
		"disposition":{"type":"choice","choice":"respond","probabilities":{"respond":0.9,"wait":0.1},"confidence":0.8},
		"floor":{"type":"choice","choice":"continue","probabilities":{"continue":1},"confidence":0.99},
		"addressed":{"type":"noul","noul":0.95}
	}}`

	answers, err := s.client.Ask(s.ctx, map[string]any{"heard": "book a table for four"},
		map[string]Question{
			"disposition": Choice("What should happen to these words?", map[string]string{
				"respond": "A complete thought addressed to the agent.",
				"wait":    "Probably unfinished.",
			}),
			"floor": Choice("Who should hold the floor?", map[string]string{
				"continue": "",
				"stop":     "",
			}),
			"addressed": Noul("Were these words meant for the agent?", "", ""),
		})
	s.Require().NoError(err)

	s.Equal(1, s.web.calls)
	s.Len(s.questions(), 3)
	s.Equal("respond", answers.Answers["disposition"].Chosen)
	s.Equal("continue", answers.Answers["floor"].Chosen)
	s.InDelta(0.95, answers.Answers["addressed"].Yes, 0.001)
	s.InDelta(0.8, answers.Answers["disposition"].Confidence, 0.001)
	s.InDelta(0.9, answers.Answers["disposition"].Probabilities["respond"], 0.001)
}

func (s *TypeSafeSuite) TestAChoiceSendsItsOptionsAndTheirDescriptions() {
	// An option the model was not given cannot be answered with, so what reaches the wire is
	// the whole of what it may say.
	s.web.respond = `{"model":"jev-1.13.0","answers":{"floor":{"type":"choice","choice":"stop"}}}`

	_, err := s.client.Ask(s.ctx, "wait, make it six", map[string]Question{
		"floor": Choice("Who should hold the floor?", map[string]string{
			"stop":     "A correction or a direct interruption.",
			"shorten":  "A related addition.",
			"continue": "",
		}),
	})
	s.Require().NoError(err)

	asked := s.question("floor")
	s.Equal("choice", asked["type"])
	s.Equal("Who should hold the floor?", asked["instructions"])
	criteria, ok := asked["criteria"].(map[string]any)
	s.Require().True(ok)
	s.Equal("A correction or a direct interruption.", criteria["stop"])
	s.Equal("A related addition.", criteria["shorten"])
	s.Contains(criteria, "continue", "an option with no gloss is still an option")
	s.Nil(criteria["continue"])
}

func (s *TypeSafeSuite) TestAScoreSendsItsLevelsInOrder() {
	s.web.respond = `{"model":"jev-1.13.0","answers":{"finished":{"type":"score","score":1.4,
		"legend":{"0":"Mid-word","1":"Mid-sentence","2":"Finished"},
		"probabilities":{"0":0.1,"1":0.4,"2":0.5},"confidence":0.55}}}`

	answers, err := s.client.Ask(s.ctx, "my member id is four four", map[string]Question{
		"finished": Score("How finished is this?", []string{"Mid-word", "Mid-sentence", "Finished"}),
	})
	s.Require().NoError(err)

	levels, ok := s.question("finished")["criteria"].([]any)
	s.Require().True(ok)
	s.Equal([]any{"Mid-word", "Mid-sentence", "Finished"}, levels)
	s.InDelta(1.4, answers.Answers["finished"].Level, 0.001)
	s.Equal("Finished", answers.Answers["finished"].Legend["2"])
}

func (s *TypeSafeSuite) TestANoulSaysWhatYesAndNoMeanOnlyWhenTold() {
	s.web.respond = `{"model":"jev-1.13.0","answers":{
		"menu":{"type":"noul","noul":0.7},"plain":{"type":"noul","noul":0.2}}}`

	_, err := s.client.Ask(s.ctx, "press one for billing", map[string]Question{
		"menu":  Noul("Is this a recorded menu?", "A recording listing options.", "A person talking."),
		"plain": Noul("Is anyone shouting?", "", ""),
	})
	s.Require().NoError(err)

	criteria, ok := s.question("menu")["criteria"].(map[string]any)
	s.Require().True(ok)
	s.Equal("A recording listing options.", criteria["true"])
	s.Equal("A person talking.", criteria["false"])
	s.NotContains(s.question("plain"), "criteria")
}

func (s *TypeSafeSuite) TestTheStateReachesTheWireWithItsPartsNamed() {
	// A question points at a part of the state by path, so the parts have to survive as
	// parts rather than being flattened into one string on the way out.
	s.web.respond = `{"model":"jev-1.13.0","answers":{"heard":{"type":"noul","noul":0.5}}}`

	_, err := s.client.Ask(s.ctx, map[string]any{
		"agent_speaking": true,
		"heard":          "hang on",
	}, map[string]Question{"heard": Noul("Is `heard` an interruption?", "", "")})
	s.Require().NoError(err)

	state, ok := s.web.body["state"].(map[string]any)
	s.Require().True(ok)
	s.Equal(true, state["agent_speaking"])
	s.Equal("hang on", state["heard"])
}

func (s *TypeSafeSuite) TestARequestWithNoQuestionsIsNotSent() {
	_, err := s.client.Ask(s.ctx, "anything", map[string]Question{})

	s.ErrorContains(err, "at least one question")
	s.Zero(s.web.calls)
}

func (s *TypeSafeSuite) TestAQuestionWithNothingAskedIsNotSent() {
	_, err := s.client.Ask(s.ctx, "anything", map[string]Question{
		"floor": {Type: TypeChoice, Instructions: "  "},
	})

	s.ErrorContains(err, "floor")
	s.Zero(s.web.calls)
}

func (s *TypeSafeSuite) TestAnUnansweredQuestionIsAFailureRatherThanAZero() {
	// A missing answer read as a zero value is a floor decision of "" and an agent that does
	// nothing about a caller talking over it, which is worse than the error.
	s.web.respond = `{"model":"jev-1.13.0","answers":{"disposition":{"type":"choice","choice":"respond"}}}`

	_, err := s.client.Ask(s.ctx, "make it six", map[string]Question{
		"disposition": Choice("What now?", map[string]string{"respond": "", "wait": ""}),
		"floor":       Choice("Who has the floor?", map[string]string{"stop": "", "continue": ""}),
	})

	s.ErrorContains(err, "floor")
}

func (s *TypeSafeSuite) TestARateLimitIsWorthAskingAgainAndABadQuestionIsNot() {
	s.web.status = http.StatusTooManyRequests
	s.web.respond = `{"detail":"slow down"}`

	_, err := s.client.Ask(s.ctx, "anything", map[string]Question{
		"heard": Noul("Did anyone speak?", "", ""),
	})

	var refused *StatusError
	s.Require().ErrorAs(err, &refused)
	s.Equal(http.StatusTooManyRequests, refused.StatusCode)
	s.True(refused.Retryable())
	s.ErrorContains(err, "slow down")

	s.web.status = http.StatusUnprocessableEntity
	_, err = s.client.Ask(s.ctx, "anything", map[string]Question{
		"heard": Noul("Did anyone speak?", "", ""),
	})

	s.Require().ErrorAs(err, &refused)
	s.False(refused.Retryable())
}

func (s *TypeSafeSuite) TestTheModelThatAnsweredIsReportedRatherThanTheAliasThatWasAsked() {
	// An alias moves when a release ships, so the answer says which version made it.
	s.web.respond = `{"model":"jev-1.13.0","answers":{"heard":{"type":"noul","noul":0.5}},
		"usage":{"input_tokens":312,"output_tokens":48}}`

	answers, err := s.client.Ask(s.ctx, "anything", map[string]Question{
		"heard": Noul("Did anyone speak?", "", ""),
	})
	s.Require().NoError(err)

	s.Equal("jev-1.13.0", answers.Model)
	s.EqualValues(312, answers.Usage.InputTokens)
	s.EqualValues(48, answers.Usage.OutputTokens)
}
