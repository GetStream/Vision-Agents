package api

import (
	"context"
	"net/http"
	"sync"

	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sttrouter"
)

// base is a knowledge base kept in memory, keyed the way a real one is. Passages replace
// whatever is already stored under their id, which is the property the endpoint depends on
// for posting a document twice to be an edit rather than a second copy.
type base struct {
	mu        sync.Mutex
	namespace string
	passages  map[string]knowledge.Document
}

func newBase() *base {
	return &base{passages: map[string]knowledge.Document{}}
}

func (b *base) Upsert(_ context.Context, namespace string, documents []knowledge.Document) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	b.namespace = namespace
	for _, document := range documents {
		b.passages[document.ID] = document
	}
	return nil
}

func (b *base) Delete(_ context.Context, _ string, ids []string) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	for _, id := range ids {
		delete(b.passages, id)
	}
	return nil
}

func (b *base) stored() (string, map[string]knowledge.Document) {
	b.mu.Lock()
	defer b.mu.Unlock()

	copied := make(map[string]knowledge.Document, len(b.passages))
	for id, document := range b.passages {
		copied[id] = document
	}
	return b.namespace, copied
}

// withKnowledge rebuilds the suite's server with somewhere to write.
func (s *ServerSuite) withKnowledge() *base {
	config, err := routing.DefaultConfig()
	s.Require().NoError(err)

	speech, err := sttrouter.New(sttrouter.Options{
		Config:   config[routing.STT],
		Registry: sttrouter.DefaultRegistry(),
	})
	s.Require().NoError(err)
	s.T().Cleanup(speech.Close)

	written := newBase()
	server, err := NewServer(Options{
		Routers:   map[routing.Modality]routing.Inspector{routing.STT: speech},
		Knowledge: written,
	})
	s.Require().NoError(err)
	s.handler = server.Handler()
	return written
}

func (s *ServerSuite) TestIngestingKnowledgeRequiresTheCustomerHeader() {
	recorder := s.post("/v1/agents/knowledge", "",
		`{"namespace":"docs","documents":[{"source":"a.md","text":"# A\n\nsomething"}]}`)

	s.Equal(http.StatusUnauthorized, recorder.Code)
}

func (s *ServerSuite) TestWithoutAProviderThereIsNowhereToWrite() {
	recorder := s.post("/v1/agents/knowledge", "acme",
		`{"namespace":"docs","documents":[{"source":"a.md","text":"# A\n\nsomething"}]}`)

	s.Equal(http.StatusBadRequest, recorder.Code)

	var failure Error
	s.decode(recorder, &failure)
	s.Contains(failure.Error, "no provider configured")
}

func (s *ServerSuite) TestADocumentIsCutIntoPassagesAndWritten() {
	written := s.withKnowledge()

	recorder := s.post("/v1/agents/knowledge", "acme", `{
		"namespace": "docs",
		"documents": [{"source": "pricing.md", "text": "# Pricing\n\nA call costs a penny.\n\n# Support\n\nWe answer within a day.\n"}]
	}`)

	s.Equal(http.StatusOK, recorder.Code)

	var result IngestedKnowledge
	s.decode(recorder, &result)
	s.Equal("docs", result.Namespace)
	s.Equal(1, result.Documents)
	s.Equal(2, result.Passages, "the document is cut at its headings")

	namespace, passages := written.stored()
	s.Equal("docs", namespace)
	s.Len(passages, 2)
	s.Contains(passages, "pricing.md#0", "a passage is keyed by where it came from")
	s.Equal("pricing.md > Pricing", passages["pricing.md#0"].Source)
}

func (s *ServerSuite) TestPostingADocumentAgainReplacesWhatItWroteBefore() {
	written := s.withKnowledge()

	first := s.post("/v1/agents/knowledge", "acme",
		`{"namespace":"docs","documents":[{"source":"pricing.md","text":"# Pricing\n\nA call costs a penny."}]}`)
	s.Equal(http.StatusOK, first.Code)

	second := s.post("/v1/agents/knowledge", "acme",
		`{"namespace":"docs","documents":[{"source":"pricing.md","text":"# Pricing\n\nA call costs tuppence."}]}`)
	s.Equal(http.StatusOK, second.Code)

	_, passages := written.stored()
	s.Len(passages, 1, "editing a document does not leave two versions of it to be found")
	s.Contains(passages["pricing.md#0"].Text, "tuppence")
}

func (s *ServerSuite) TestKnowledgeIsNeverSharedSoANamespaceIsRequired() {
	s.withKnowledge()

	recorder := s.post("/v1/agents/knowledge", "acme",
		`{"namespace":"  ","documents":[{"source":"a.md","text":"# A\n\nsomething"}]}`)

	s.Equal(http.StatusBadRequest, recorder.Code)

	var failure Error
	s.decode(recorder, &failure)
	s.Contains(failure.Error, "namespace")
}

func (s *ServerSuite) TestADocumentWithNoSourceCannotBeKeyed() {
	s.withKnowledge()

	recorder := s.post("/v1/agents/knowledge", "acme",
		`{"namespace":"docs","documents":[{"source":"","text":"# A\n\nsomething"}]}`)

	s.Equal(http.StatusBadRequest, recorder.Code)

	var failure Error
	s.decode(recorder, &failure)
	s.Contains(failure.Error, "source")
}

func (s *ServerSuite) TestDocumentsWithNothingInThemAreRefusedRatherThanWrittenEmpty() {
	written := s.withKnowledge()

	recorder := s.post("/v1/agents/knowledge", "acme",
		`{"namespace":"docs","documents":[{"source":"blank.md","text":"   \n"}]}`)

	s.Equal(http.StatusBadRequest, recorder.Code)

	_, passages := written.stored()
	s.Empty(passages)
}

func (s *ServerSuite) TestTheChunkSizeCanBeMadeSmallerThanTheDefault() {
	written := s.withKnowledge()

	long := "# Routing\n"
	for range 20 {
		long += "\nThe router fails over between providers.\n"
	}

	recorder := s.post("/v1/agents/knowledge", "acme",
		`{"namespace":"docs","chunk_size":200,"documents":[{"source":"routing.md","text":`+
			quote(long)+`}]}`)

	s.Equal(http.StatusOK, recorder.Code)

	var result IngestedKnowledge
	s.decode(recorder, &result)
	s.Greater(result.Passages, 1, "a smaller chunk cuts the same document into more of them")

	_, passages := written.stored()
	s.Len(passages, result.Passages)
}

func (s *ServerSuite) TestAddingAKnowledgeUrlRequiresTheCustomerHeader() {
	recorder := s.post("/v1/agents/knowledge/urls", "",
		`{"namespace":"docs","url":"https://example.com/pricing"}`)

	s.Equal(http.StatusUnauthorized, recorder.Code)
}

func (s *ServerSuite) TestWithoutACrawlerAUrlWouldBeASubscriptionNothingHonours() {
	recorder := s.post("/v1/agents/knowledge/urls", "acme",
		`{"namespace":"docs","url":"https://example.com/pricing"}`)

	s.Equal(http.StatusBadRequest, recorder.Code)

	var failure Error
	s.decode(recorder, &failure)
	s.Contains(failure.Error, "knowledge urls are not available")
}

func (s *ServerSuite) TestListingKnowledgeUrlsSaysWhatTheDeploymentIsMissing() {
	recorder := s.get("/v1/agents/knowledge/urls", "acme")

	s.Equal(http.StatusBadRequest, recorder.Code)

	var failure Error
	s.decode(recorder, &failure)
	s.Contains(failure.Error, "no database or no way to read a page")
}

// quote renders a string as a JSON one, for building a body by hand.
func quote(text string) string {
	quoted := `"`
	for _, r := range text {
		switch r {
		case '"':
			quoted += `\"`
		case '\n':
			quoted += `\n`
		default:
			quoted += string(r)
		}
	}
	return quoted + `"`
}
