package systemone

import (
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/decisionmodel"
)

// web stands in for a vendor's endpoint, so the wire contract can be tested without a key.
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
func (s *SystemOneSuite) questions() map[string]any {
	asked, ok := s.web.body["questions"].(map[string]any)
	s.Require().True(ok, "the request carried no questions")
	return asked
}

func (s *SystemOneSuite) question(id string) map[string]any {
	asked, ok := s.questions()[id].(map[string]any)
	s.Require().True(ok, "the request did not ask %q", id)
	return asked
}

// ask puts a state and its questions to the client, which every test here does.
func (s *SystemOneSuite) ask(
	state any, questions map[string]decisionmodel.Question,
) (decisionmodel.Result, error) {
	return s.client.Classify(s.ctx, decisionmodel.Request{State: state, Questions: questions})
}

// vendor is an endpoint no real vendor has, so nothing here passes because it happens to
// match TypeSafe's defaults.
var vendor = Endpoint{
	Provider:     "acme",
	APIKeyEnvVar: "ACME_DECISIONS_API_KEY",
	Path:         "/v2/decide",
	DefaultModel: "judge-latest",
}

type SystemOneSuite struct {
	suite.Suite
	ctx    context.Context
	web    *web
	client *Client
}

func TestSystemOneSuite(t *testing.T) {
	suite.Run(t, new(SystemOneSuite))
}

func (s *SystemOneSuite) SetupTest() {
	s.ctx = context.Background()
	s.web = newWeb()
	s.T().Cleanup(s.web.server.Close)

	client, err := New(vendor, Options{
		APIKey:  "test-key",
		BaseURL: s.web.server.URL,
		Logger:  slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)
	s.client = client
}

func (s *SystemOneSuite) TestAKeyIsRequired() {
	s.T().Setenv(vendor.APIKeyEnvVar, "")

	_, err := New(vendor, Options{})

	s.ErrorContains(err, vendor.APIKeyEnvVar)
}

func (s *SystemOneSuite) TestTheNewestStableModelIsAskedWhenNoneIsNamed() {
	s.web.respond = `{"model":"jev-1.13.0","answers":{"heard":{"type":"noul","noul":0.8}}}`

	_, err := s.ask("anything", map[string]decisionmodel.Question{
		"heard": decisionmodel.Noul("Did anyone speak?", "", ""),
	})
	s.Require().NoError(err)

	s.Equal(vendor.DefaultModel, s.client.Model())
	s.Equal(vendor.DefaultModel, s.web.body["model"])
	s.Equal("Bearer test-key", s.web.auth)
	s.Equal(vendor.Path, s.web.path)
}

func (s *SystemOneSuite) TestTheVendorIsWhatThisIsNamedByRatherThanTheModel() {
	// Stats and health are keyed by the provider name, and "jev" is a model several vendors
	// serve.
	s.Equal("acme", s.client.Provider())
}

func (s *SystemOneSuite) TestAFailureSaysWhichVendorRefused() {
	// Three vendors answer on the same protocol, so an error that did not name one would
	// leave a log reader guessing which key to check.
	s.web.status = http.StatusPaymentRequired
	s.web.respond = `{"error":{"message":"Insufficient credits"}}`

	_, err := s.ask("anything", map[string]decisionmodel.Question{
		"heard": decisionmodel.Noul("Did anyone speak?", "", ""),
	})

	var refused *StatusError
	s.Require().ErrorAs(err, &refused)
	s.Equal("acme", refused.Provider)
	s.ErrorContains(err, "acme: the API returned 402")
	s.False(refused.Retryable(), "credits do not come back by waiting")
}

func (s *SystemOneSuite) TestEveryQuestionGoesInOneRequest() {
	// Questions in one request share the state's tokens between them rather than paying for
	// it each, which is what makes asking one whose answer may not be needed close to free.
	s.web.respond = `{"model":"jev-1.13.0","answers":{
		"disposition":{"type":"choice","choice":"respond","probabilities":{"respond":0.9,"wait":0.1},"confidence":0.8},
		"floor":{"type":"choice","choice":"continue","probabilities":{"continue":1},"confidence":0.99},
		"addressed":{"type":"noul","noul":0.95}
	}}`

	answers, err := s.ask(map[string]any{"heard": "book a table for four"},
		map[string]decisionmodel.Question{
			"disposition": decisionmodel.Choice("What should happen to these words?", map[string]string{
				"respond": "A complete thought addressed to the agent.",
				"wait":    "Probably unfinished.",
			}),
			"floor": decisionmodel.Choice("Who should hold the floor?", map[string]string{
				"continue": "",
				"stop":     "",
			}),
			"addressed": decisionmodel.Noul("Were these words meant for the agent?", "", ""),
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

func (s *SystemOneSuite) TestAChoiceSendsItsOptionsAndTheirDescriptions() {
	// An option the model was not given cannot be answered with, so what reaches the wire is
	// the whole of what it may say.
	s.web.respond = `{"model":"jev-1.13.0","answers":{"floor":{"type":"choice","choice":"stop"}}}`

	_, err := s.ask("wait, make it six", map[string]decisionmodel.Question{
		"floor": decisionmodel.Choice("Who should hold the floor?", map[string]string{
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

func (s *SystemOneSuite) TestAScoreSendsItsLevelsInOrder() {
	s.web.respond = `{"model":"jev-1.13.0","answers":{"finished":{"type":"score","score":1.4,
		"legend":{"0":"Mid-word","1":"Mid-sentence","2":"Finished"},
		"probabilities":{"0":0.1,"1":0.4,"2":0.5},"confidence":0.55}}}`

	answers, err := s.ask("my member id is four four", map[string]decisionmodel.Question{
		"finished": decisionmodel.Score("How finished is this?",
			[]string{"Mid-word", "Mid-sentence", "Finished"}),
	})
	s.Require().NoError(err)

	levels, ok := s.question("finished")["criteria"].([]any)
	s.Require().True(ok)
	s.Equal([]any{"Mid-word", "Mid-sentence", "Finished"}, levels)
	s.InDelta(1.4, answers.Answers["finished"].Level, 0.001)
	s.Equal("Finished", answers.Answers["finished"].Legend["2"])
}

func (s *SystemOneSuite) TestANoulSaysWhatYesAndNoMeanOnlyWhenTold() {
	s.web.respond = `{"model":"jev-1.13.0","answers":{
		"menu":{"type":"noul","noul":0.7},"plain":{"type":"noul","noul":0.2}}}`

	_, err := s.ask("press one for billing", map[string]decisionmodel.Question{
		"menu": decisionmodel.Noul("Is this a recorded menu?",
			"A recording listing options.", "A person talking."),
		"plain": decisionmodel.Noul("Is anyone shouting?", "", ""),
	})
	s.Require().NoError(err)

	criteria, ok := s.question("menu")["criteria"].(map[string]any)
	s.Require().True(ok)
	s.Equal("A recording listing options.", criteria["true"])
	s.Equal("A person talking.", criteria["false"])
	s.NotContains(s.question("plain"), "criteria")
}

func (s *SystemOneSuite) TestTheStateReachesTheWireWithItsPartsNamed() {
	// A question points at a part of the state by path, so the parts have to survive as
	// parts rather than being flattened into one string on the way out.
	s.web.respond = `{"model":"jev-1.13.0","answers":{"heard":{"type":"noul","noul":0.5}}}`

	_, err := s.ask(map[string]any{
		"agent_speaking": true,
		"heard":          "hang on",
	}, map[string]decisionmodel.Question{
		"heard": decisionmodel.Noul("Is `heard` an interruption?", "", ""),
	})
	s.Require().NoError(err)

	state, ok := s.web.body["state"].(map[string]any)
	s.Require().True(ok)
	s.Equal(true, state["agent_speaking"])
	s.Equal("hang on", state["heard"])
}

func (s *SystemOneSuite) TestARequestWithNoQuestionsIsNotSent() {
	_, err := s.ask("anything", map[string]decisionmodel.Question{})

	s.ErrorContains(err, "at least one question")
	s.Zero(s.web.calls)
}

func (s *SystemOneSuite) TestAQuestionWithNothingAskedIsNotSent() {
	_, err := s.ask("anything", map[string]decisionmodel.Question{
		"floor": {Type: decisionmodel.TypeChoice, Instructions: "  "},
	})

	s.ErrorContains(err, "floor")
	s.Zero(s.web.calls)
}

func (s *SystemOneSuite) TestAQuestionOfNoKnownTypeIsNotSent() {
	// A type the API does not know comes back as a 422, which is a round trip spent finding
	// out something the question itself says.
	_, err := s.ask("anything", map[string]decisionmodel.Question{
		"floor": {Type: "vibes", Instructions: "Who has the floor?"},
	})

	s.ErrorContains(err, "vibes")
	s.Zero(s.web.calls)
}

func (s *SystemOneSuite) TestAnUnansweredQuestionIsAFailureRatherThanAZero() {
	// A missing answer read as a zero value is a floor decision of "" and an agent that does
	// nothing about a caller talking over it, which is worse than the error.
	s.web.respond = `{"model":"jev-1.13.0","answers":{"disposition":{"type":"choice","choice":"respond"}}}`

	_, err := s.ask("make it six", map[string]decisionmodel.Question{
		"disposition": decisionmodel.Choice("What now?", map[string]string{"respond": "", "wait": ""}),
		"floor":       decisionmodel.Choice("Who has the floor?", map[string]string{"stop": "", "continue": ""}),
	})

	s.ErrorContains(err, "floor")
}

func (s *SystemOneSuite) TestARateLimitIsWorthAskingAgainAndABadQuestionIsNot() {
	s.web.status = http.StatusTooManyRequests
	s.web.respond = `{"detail":"slow down"}`

	_, err := s.ask("anything", map[string]decisionmodel.Question{
		"heard": decisionmodel.Noul("Did anyone speak?", "", ""),
	})

	var refused *StatusError
	s.Require().ErrorAs(err, &refused)
	s.Equal(http.StatusTooManyRequests, refused.StatusCode)
	s.True(refused.Retryable())
	s.ErrorIs(err, decisionmodel.ErrRateLimited)
	s.ErrorContains(err, "slow down")

	// The outage seen in practice: the model is being moved and comes back on its own.
	s.web.status = http.StatusServiceUnavailable
	s.web.respond = `{"detail":{"error_type":"model_unavailable","message":"The model is unavailable."}}`

	_, err = s.ask("anything", map[string]decisionmodel.Question{
		"heard": decisionmodel.Noul("Did anyone speak?", "", ""),
	})

	s.Require().ErrorAs(err, &refused)
	s.True(refused.Retryable())
	s.ErrorIs(err, decisionmodel.ErrUnavailable)

	s.web.status = http.StatusUnprocessableEntity
	_, err = s.ask("anything", map[string]decisionmodel.Question{
		"heard": decisionmodel.Noul("Did anyone speak?", "", ""),
	})

	s.Require().ErrorAs(err, &refused)
	s.False(refused.Retryable())
	s.NotErrorIs(err, decisionmodel.ErrRateLimited)
	s.NotErrorIs(err, decisionmodel.ErrUnavailable)
}

func (s *SystemOneSuite) TestAnAPINobodyCanReachIsUnavailable() {
	client, err := New(vendor, Options{APIKey: "k", BaseURL: "http://127.0.0.1:1"})
	s.Require().NoError(err)

	_, err = client.Classify(context.Background(), decisionmodel.Request{
		State:     "anything",
		Questions: map[string]decisionmodel.Question{"heard": decisionmodel.Noul("Did anyone speak?", "", "")},
	})

	s.ErrorIs(err, decisionmodel.ErrUnavailable)
}

func (s *SystemOneSuite) TestTheModelThatAnsweredIsReportedRatherThanTheAliasThatWasAsked() {
	// An alias moves when a release ships, so the answer says which version made it.
	s.web.respond = `{"model":"jev-1.13.0","answers":{"heard":{"type":"noul","noul":0.5}},
		"usage":{"input_tokens":312,"output_tokens":48}}`

	answers, err := s.ask("anything", map[string]decisionmodel.Question{
		"heard": decisionmodel.Noul("Did anyone speak?", "", ""),
	})
	s.Require().NoError(err)

	s.Equal("jev-1.13.0", answers.Model)
	s.EqualValues(312, answers.Usage.InputTokens)
	s.EqualValues(48, answers.Usage.OutputTokens)
}
