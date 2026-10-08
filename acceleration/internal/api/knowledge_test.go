//go:build integration

package api

import (
	"net/http"
	"testing"
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

func (s *KnowledgeSuite) TestOnlyTheAppsOwnBackendMayPostADocument() {
	s.assertPosture(serverOnly, func(as *testClient) int {
		return as.do(http.MethodPost, "/v1/agents/knowledge", map[string]any{
			"namespace": s.namespace,
			"documents": []map[string]string{{"source": s.utils.uuid() + ".md", "text": "# Pricing\n\nA penny."}},
		}, nil)
	})
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
