//go:build integration

package api

import (
	"net/http"
	"testing"

	"github.com/GetStream/Vision-Agents/acceleration/internal/decisionmodel"
)

// ClassifySuite covers asking a classifier several questions at once: how the answers come
// back, which classifier answers, and what a question nothing can answer is refused with.
type ClassifySuite struct {
	RouterSuite
}

func TestClassifySuite(t *testing.T) {
	runSuite(t, new(ClassifySuite))
}

func (s *ClassifySuite) SetupTest() {
	s.useFixture("standard")
}

func (s *ClassifySuite) TestEachQuestionComesBackUnderItsOwnId() {
	result := s.classify(s.allThree(s.complaint()))

	s.Equal("quick", result.Provider)
	s.Equal("judge-2026-09", result.Model, "the version that answered, not the alias")
	s.Require().Len(result.Answers, 3)

	refund := result.Answers["refund"]
	s.Equal(Noul, refund.Type)
	s.InDelta(0.91, *refund.Yes, 1e-9)
	s.Nil(refund.Confidence, "a noul's probability is the answer and has no confidence beside it")

	topic := result.Answers["topic"]
	s.Equal(Choice, topic.Type)
	s.Equal("billing", *topic.Chosen)
	s.InDelta(0.9, (*topic.Probabilities)["billing"], 1e-9)
	s.Nil(topic.Yes)

	urgency := result.Answers["urgency"]
	s.Equal(Score, urgency.Type)
	s.InDelta(1.4, *urgency.Level, 1e-9)
	s.Equal("today", (*urgency.Legend)["2"])

	s.Equal(ClassifyUsage{InputTokens: 406, OutputTokens: 69}, result.Usage)
}

func (s *ClassifySuite) TestEveryQuestionIsAskedInOneRequest() {
	complaint := s.complaint()

	s.classify(s.allThree(complaint))

	asked, ruled := classifiers.quick.ruledOn(complaint)
	s.Require().True(ruled)
	s.Len(asked.Questions, 3)
	s.Equal(decisionmodel.Score("How soon does this need an answer?", []string{"can wait", "this week", "today"}),
		asked.Questions["urgency"])
}

func (s *ClassifySuite) TestAStateWithPartsIsPassedAsItWasSent() {
	s.classify(ClassifyRequest{
		State:     map[string]any{"message": "refund me", "channel": "email"},
		Questions: map[string]ClassifyQuestion{"refund": {Type: Noul, Instructions: "Does `message` ask for money back?"}},
	})

	for _, asked := range classifiers.quick.questions() {
		if parts, ok := asked.State.(map[string]any); ok {
			s.Equal(map[string]any{"message": "refund me", "channel": "email"}, parts)
			return
		}
	}
	s.Fail("the classifier was never asked about a state with parts")
}

func (s *ClassifySuite) TestATargetPicksTheClassifier() {
	complaint := s.complaint()

	result := s.classify(ClassifyRequest{
		Target: pointerTo("careful/judge"), State: complaint,
		Questions: map[string]ClassifyQuestion{"refund": {Type: Noul, Instructions: "Is this a refund request?"}},
	})

	s.Equal("careful", result.Provider)
	_, ruled := classifiers.quick.ruledOn(complaint)
	s.False(ruled, "the default classifier was not asked")
}

func (s *ClassifySuite) TestAFailedJudgementIsAnErrorRatherThanAnEmptyAnswer() {
	status, failure := s.refused(s.allThree("unanswerable"))

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "was not answered")
}

func (s *ClassifySuite) TestAClassifierAskedTooOftenSaysToComeBack() {
	status, failure := s.refused(s.allThree("rate limited"))

	s.Equal(http.StatusTooManyRequests, status)
	s.Contains(failure, "429")
}

func (s *ClassifySuite) TestAClassifierThatIsDownIsAServiceUnavailable() {
	status, _ := s.refused(s.allThree("unavailable"))

	s.Equal(http.StatusServiceUnavailable, status)
}

func (s *ClassifySuite) TestATargetNobodyRoutesIsNotFound() {
	complaint := s.complaint()

	status, failure := s.refused(ClassifyRequest{
		Target: pointerTo("nobody/nothing"), State: complaint,
		Questions: map[string]ClassifyQuestion{"q": {Type: Noul, Instructions: "Is it?"}},
	})

	s.Equal(http.StatusNotFound, status)
	s.Contains(failure, "unknown target")
	_, ruled := classifiers.quick.ruledOn(complaint)
	s.False(ruled)
}

func (s *ClassifySuite) TestAQuestionAboutNothingIsRefused() {
	status, _ := s.refused(ClassifyRequest{
		Questions: map[string]ClassifyQuestion{"q": {Type: Noul, Instructions: "Is it?"}}})

	s.Equal(http.StatusBadRequest, status)
}

func (s *ClassifySuite) TestAStateOfNothingButSpaceIsRefused() {
	status, _ := s.refused(ClassifyRequest{State: "  ",
		Questions: map[string]ClassifyQuestion{"q": {Type: Noul, Instructions: "Is it?"}}})

	s.Equal(http.StatusBadRequest, status)
}

func (s *ClassifySuite) TestAStateWithNoQuestionsAboutItIsRefused() {
	status, _ := s.refused(ClassifyRequest{State: s.complaint(),
		Questions: map[string]ClassifyQuestion{}})

	s.Equal(http.StatusBadRequest, status)
}

func (s *ClassifySuite) TestAQuestionWithNoInstructionsIsRefused() {
	// An id is not part of what is asked, so a question with nothing in its instructions
	// carries no meaning at all.
	status, _ := s.refused(ClassifyRequest{State: s.complaint(),
		Questions: map[string]ClassifyQuestion{"q": {Type: Noul, Instructions: " "}}})

	s.Equal(http.StatusBadRequest, status)
}

func (s *ClassifySuite) TestAQuestionOfAKindNobodyAnswersIsRefused() {
	status, _ := s.refused(ClassifyRequest{State: s.complaint(),
		Questions: map[string]ClassifyQuestion{"q": {Type: "maybe", Instructions: "Is it?"}}})

	s.Equal(http.StatusBadRequest, status)
}

func (s *ClassifySuite) TestAChoiceBetweenOneThingIsRefused() {
	status, _ := s.refused(ClassifyRequest{State: s.complaint(),
		Questions: map[string]ClassifyQuestion{"q": {Type: Choice, Instructions: "Which?",
			Options: &map[string]string{"only": ""}}}})

	s.Equal(http.StatusBadRequest, status)
}

func (s *ClassifySuite) TestAScoreWithOneLevelIsRefused() {
	status, _ := s.refused(ClassifyRequest{State: s.complaint(),
		Questions: map[string]ClassifyQuestion{"q": {Type: Score, Instructions: "How much?",
			Levels: &[]string{"some"}}}})

	s.Equal(http.StatusBadRequest, status)
}

func (s *ClassifySuite) TestOnlyTheCustomersOwnBackendMayClassify() {
	s.assertPosture(serverOnly, func(as *testClient) int {
		status, _ := as.call(http.MethodPost, "/v1/classify", s.allThree(s.complaint()))
		return status
	})
}

// complaint is a support message of this test's own, since the classifiers are shared.
func (s *ClassifySuite) complaint() string {
	return "I was charged twice this month, ticket " + s.utils.uuid()
}

// allThree asks one question of each type about a support message.
func (s *ClassifySuite) allThree(state string) ClassifyRequest {
	return ClassifyRequest{
		State: state,
		Questions: map[string]ClassifyQuestion{
			"refund": {Type: Noul, Instructions: "Is the customer asking for money back?"},
			"topic": {Type: Choice, Instructions: "What is this about?",
				Options: &map[string]string{"billing": "charges and invoices", "other": ""}},
			"urgency": {Type: Score, Instructions: "How soon does this need an answer?",
				Levels: &[]string{"can wait", "this week", "today"}},
		},
	}
}

func (s *ClassifySuite) classify(request ClassifyRequest) ClassifyResult {
	var result ClassifyResult
	s.Require().Equal(http.StatusOK,
		s.serverClient.do(http.MethodPost, "/v1/classify", request, &result))
	return result
}

// refused is the status and the message of a request nothing ruled on.
func (s *ClassifySuite) refused(request ClassifyRequest) (int, string) {
	return s.serverClient.failure(http.MethodPost, "/v1/classify", request)
}
