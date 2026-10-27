//go:build integration

package api

import (
	"net/http"
	"strings"
	"testing"

	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge/ingest"
)

type KnowledgeSuite struct {
	RouterSuite

	// namespace is the knowledge base of the test running, and source is the one file it
	// posts into it. Both are unique, because the base the suite writes to is shared and
	// its passages are keyed by source.
	namespace string
	source    string
}

func TestKnowledgeSuite(t *testing.T) {
	runSuite(t, new(KnowledgeSuite))
}

func (s *KnowledgeSuite) SetupTest() {
	s.useFixture("standard")
	s.namespace = "namespace-" + s.utils.uuid()
	s.source = s.utils.uuid() + ".md"
}

func (s *KnowledgeSuite) TestAPostedDocumentIsListed() {
	s.post("# Pricing\n\nA penny.\n\n# Support\n\nA day.")

	listed := s.documents()
	s.Require().Len(listed, 1)
	s.Equal(s.source, listed[0].Source)
	s.Equal(2, listed[0].Passages)
}

func (s *KnowledgeSuite) TestADocumentReadsBackAsItWasLastPosted() {
	s.post("# Pricing\n\nA penny.")
	s.post("# Pricing\n\nTwo pennies.")

	listed := s.documents()
	s.Require().Len(listed, 1)
	s.Nil(listed[0].Text, "a listing leaves the text out")

	var read IndexedKnowledgeDocument
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet,
		"/v1/agents/knowledge/documents/"+listed[0].Id, nil, &read))
	s.Equal("# Pricing\n\nTwo pennies.", value(read.Text))
}

func (s *KnowledgeSuite) TestADocumentReadsBackAsThePassagesItWasCutInto() {
	s.post("# Pricing\n\nA penny.\n\n# Support\n\nA day.")
	listed := s.documents()
	s.Require().Len(listed, 1)

	var passages []KnowledgePassage
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet,
		"/v1/agents/knowledge/documents/"+listed[0].Id+"/passages", nil, &passages))
	s.Require().Len(passages, 2)
	s.Contains(passages[0].Text, "A penny.")
	s.Contains(passages[1].Text, "A day.")
}

func (s *KnowledgeSuite) TestADocumentPostedShorterLeavesNoOldTailBehind() {
	s.post("# Pricing\n\nA penny.\n\n# Support\n\nA day.")
	s.post("# Pricing\n\nTuppence.")

	listed := s.documents()
	s.Require().Len(listed, 1, "posting the same source again is an edit")
	s.Equal(1, listed[0].Passages)

	_, passages := s.knowledge.stored()
	s.Contains(passages, s.source+"#0")
	s.NotContains(passages, s.source+"#1")
}

func (s *KnowledgeSuite) TestADeletedDocumentIsNoLongerListedOrFound() {
	s.post("# Refunds\n\nThirty days.")
	listed := s.documents()
	s.Require().Len(listed, 1)

	s.Require().Equal(http.StatusNoContent, s.serverClient.do(http.MethodDelete,
		"/v1/agents/knowledge/documents/"+listed[0].Id, nil, nil))

	s.Empty(s.documents())
	_, passages := s.knowledge.stored()
	s.NotContains(passages, s.source+"#0")

	s.Equal(http.StatusNotFound, s.serverClient.do(http.MethodDelete,
		"/v1/agents/knowledge/documents/"+listed[0].Id, nil, nil))
}

func (s *KnowledgeSuite) TestAnotherAppsDocumentIsNeitherReadNorDeleted() {
	s.post("# Refunds\n\nThirty days.")
	listed := s.documents()
	s.Require().Len(listed, 1)

	s.assertHiddenFromOtherApps(func(as *testClient) int {
		return as.do(http.MethodGet, "/v1/agents/knowledge/documents/"+listed[0].Id+"/passages", nil, nil)
	})
	s.assertHiddenFromOtherApps(func(as *testClient) int {
		return as.do(http.MethodDelete, "/v1/agents/knowledge/documents/"+listed[0].Id, nil, nil)
	})
}

func (s *KnowledgeSuite) TestAFiveMegabyteDocumentIsIngested() {
	// Five times what an operation reads by default.
	line := "A penny for every word of the price list.\n\n"
	s.post(strings.Repeat(line, 5<<20/len(line)))

	listed := s.documents()
	s.Require().Len(listed, 1)
	s.Equal(s.source, listed[0].Source)
}

func (s *KnowledgeSuite) TestABodyThatIsNotJSONIsRefusedWithoutRepeatingTheDocument() {
	// Cut short mid-document: refused as one string, which is the customer's text.
	text := strings.Repeat("A penny for every word of the price list. ", 100) + s.utils.uuid()
	status, answer := s.serverClient.call(http.MethodPost, "/v1/agents/knowledge",
		`{"namespace": "`+s.namespace+`", "documents": [{"source": "`+s.source+`", "text": "`+text)

	s.Equal(http.StatusBadRequest, status)
	s.NotContains(string(answer), text[len(text)-36:])
}

func (s *KnowledgeSuite) TestASourceLongerThan256CharactersIsRefused() {
	s.source = strings.Repeat("a", 257)
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/knowledge", map[string]any{
		"namespace": s.namespace,
		"documents": []map[string]string{{"source": s.source, "text": "# Pricing\n\nA penny."}},
	})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "expected length <= 256")
	s.Empty(s.documents())
}

func (s *KnowledgeSuite) TestOnlyTheAppsOwnBackendMayPostADocument() {
	s.assertPosture(serverOnly, func(as *testClient) int {
		return as.do(http.MethodPost, "/v1/agents/knowledge", map[string]any{
			"namespace": s.namespace,
			"documents": []map[string]string{{"source": s.utils.uuid() + ".md", "text": "# Pricing\n\nA penny."}},
		}, nil)
	})
}

func (s *KnowledgeSuite) TestADocumentWithALongSourceIsWrittenReadTrimmedAndDeleted() {
	// Over 61 bytes, so its ids were 64 bytes and up, which turbopuffer refuses.
	s.source = "https://getstream.io/chat/docs/sdk/android/" + s.utils.uuid() + "/push-notifications/"
	s.post("# Pricing\n\nA penny.\n\n# Support\n\nA day.")
	listed := s.documents()
	s.Require().Len(listed, 1)
	s.Equal(s.source, listed[0].Source)

	var passages []KnowledgePassage
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet,
		"/v1/agents/knowledge/documents/"+listed[0].Id+"/passages", nil, &passages))
	s.Require().Len(passages, 2)
	s.Contains(passages[0].Text, "A penny.")
	s.Contains(passages[1].Text, "A day.")
	s.Equal(s.source+" > Pricing", passages[0].Source, "the passage still says where it came from")
	s.ElementsMatch([]string{passages[0].Id, passages[1].Id}, s.storedOfSource())

	s.post("# Pricing\n\nTuppence.")
	s.Equal([]string{passages[0].Id}, s.storedOfSource(), "the old tail is no longer findable")

	s.Require().Equal(http.StatusNoContent, s.serverClient.do(http.MethodDelete,
		"/v1/agents/knowledge/documents/"+listed[0].Id, nil, nil))
	s.Empty(s.storedOfSource())
}

// storedOfSource is the id of every passage stored from the test's file, each checked to
// be short enough for turbopuffer to take.
func (s *KnowledgeSuite) storedOfSource() []string {
	_, passages := s.knowledge.stored()
	var stored []string
	for id, passage := range passages {
		if strings.HasPrefix(passage.Source, s.source) {
			s.Less(len(id), 64, id)
			stored = append(stored, id)
		}
	}
	return stored
}

// post writes the test's one file into its knowledge base.
func (s *KnowledgeSuite) post(text string) {
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost, "/v1/agents/knowledge",
		map[string]any{
			"namespace": s.namespace,
			"documents": []map[string]string{{"source": s.source, "text": text}},
		}, nil))
}

// documents lists what the test's knowledge base holds.
func (s *KnowledgeSuite) documents() []IndexedKnowledgeDocument {
	var listed []IndexedKnowledgeDocument
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet,
		"/v1/agents/knowledge/documents?namespace="+s.namespace, nil, &listed))
	return listed
}

// KnowledgeLimitsSuite runs a router whose knowledge limits are not the defaults.
type KnowledgeLimitsSuite struct {
	RouterSuite
}

func TestKnowledgeLimitsSuite(t *testing.T) {
	runSuite(t, new(KnowledgeLimitsSuite))
}

func (s *KnowledgeLimitsSuite) SetupSuite() {
	s.knowledgeLimits = ingest.Limits{MinChunk: 800, MaxHeading: 6}
	s.RouterSuite.SetupSuite()
}

func (s *KnowledgeLimitsSuite) SetupTest() {
	s.useFixture("standard")
}

func (s *KnowledgeLimitsSuite) TestADocumentIsCutByTheRoutersLimits() {
	namespace, source := "namespace-"+s.utils.uuid(), s.utils.uuid()+".md"
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost, "/v1/agents/knowledge",
		map[string]any{
			"namespace":  namespace,
			"chunk_size": 1,
			"documents": []map[string]string{{
				"source": source,
				"text":   "# Refund policy\n\n" + strings.Repeat("Refunds take thirty days.\n\n", 200),
			}},
		}, nil))
	var listed []IndexedKnowledgeDocument
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet,
		"/v1/agents/knowledge/documents?namespace="+namespace, nil, &listed))
	s.Require().Len(listed, 1)

	var passages []KnowledgePassage
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet,
		"/v1/agents/knowledge/documents/"+listed[0].Id+"/passages", nil, &passages))

	s.Require().Greater(len(passages), 1)
	for _, passage := range passages[:len(passages)-1] {
		s.Greater(len(passage.Text), 700, "a chunk_size of 1 is raised to the router's 800")
	}
	for _, passage := range passages[1:] {
		s.True(strings.HasPrefix(passage.Text, "## Refund\n\n"), "the heading is cut to the router's 6 bytes")
	}
}
