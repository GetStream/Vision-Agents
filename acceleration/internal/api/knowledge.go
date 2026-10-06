package api

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge"
	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge/ingest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/danielgtaylor/huma/v2"
)

// noKnowledge is what the paths say when the deployment has no knowledge provider. Filling
// a base that nothing can read is not worth pretending to do.
var noKnowledge = coded{codeNotConfigured, "knowledge is not available: no provider configured"}

// noKnowledgeDocuments is what the document paths say on a deployment that cannot list
// them: it takes a database to remember one and a knowledge base to remove it from.
var noKnowledgeDocuments = coded{codeNotConfigured, "knowledge documents are not available: no database or no knowledge provider configured"}

// unknownKnowledgeDocument is what a caller is told about a document that is not theirs,
// which is the same thing they are told about one that never existed.
var unknownKnowledgeDocument = coded{codeKnowledgeDocNotFound, "no such knowledge document"}

// ingestKnowledge fills a knowledge base with what the business wrote down.
//
// The documents are cut into passages here rather than by the caller, so a file read off
// disk by cmd/knowledge and one posted by an SDK are cut the same way and can replace each
// other.
func (s *Server) ingestKnowledge(ctx context.Context, request *ingestKnowledgeRequest) (*ingestKnowledgeResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, missingCustomer()
	}
	if s.knowledge == nil {
		return nil, invalidRequest(noKnowledge)
	}
	if request.Body == nil {
		return nil, invalidRequest("a request body is required")
	}

	namespace := strings.TrimSpace(request.Body.Namespace)
	if namespace == "" {
		return nil, invalidRequest("a namespace is required, knowledge is never shared")
	}

	read, passages, err := s.fillKnowledge(ctx, customerID, namespace, request.Body.Documents, request.Body.ChunkSize)
	if err != nil {
		return nil, invalidRequest(err.Error())
	}

	s.logger.Info("filled a knowledge base",
		"namespace", namespace, "documents", read, "passages", passages)
	return &ingestKnowledgeResponse{Body: IngestedKnowledge{Namespace: namespace,
		Documents: read,
		Passages:  passages}}, nil
}

// listKnowledgeDocuments returns the documents the calling customer's knowledge bases were
// filled with, most recently written first.
func (s *Server) listKnowledgeDocuments(ctx context.Context, request *listKnowledgeDocumentsRequest) (*listKnowledgeDocumentsResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, missingCustomer()
	}
	if s.store == nil || s.knowledge == nil {
		return nil, invalidRequest(noKnowledgeDocuments)
	}

	namespace := ""
	if request.Namespace.ptr() != nil {
		namespace = strings.TrimSpace(*request.Namespace.ptr())
	}

	stored, err := s.store.CustomerKnowledgeDocuments(ctx, customerID, namespace)
	if err != nil {
		return nil, err
	}

	listed := make([]IndexedKnowledgeDocument, 0, len(stored))
	for _, document := range stored {
		listed = append(listed, indexedKnowledgeDocumentOf(document))
	}
	return &listKnowledgeDocumentsResponse{Body: listed}, nil
}

// getKnowledgeDocument returns one document with the text it was posted as.
func (s *Server) getKnowledgeDocument(ctx context.Context, request *getKnowledgeDocumentRequest) (*getKnowledgeDocumentResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, missingCustomer()
	}
	if s.store == nil || s.knowledge == nil {
		return nil, invalidRequest(noKnowledgeDocuments)
	}

	document, err := s.store.KnowledgeDocument(ctx, customerID, request.Id)
	if err != nil {
		return nil, notFound(unknownKnowledgeDocument)
	}
	read := indexedKnowledgeDocumentOf(document)
	read.Text = &document.Text
	return &getKnowledgeDocumentResponse{Body: read}, nil
}

// deleteKnowledgeDocument takes a document out of its knowledge base, passages and all.
func (s *Server) deleteKnowledgeDocument(ctx context.Context, request *deleteKnowledgeDocumentRequest) (*struct{}, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, missingCustomer()
	}
	if s.store == nil || s.knowledge == nil {
		return nil, invalidRequest(noKnowledgeDocuments)
	}

	document, err := s.store.KnowledgeDocument(ctx, customerID, request.Id)
	if err != nil {
		return nil, notFound(unknownKnowledgeDocument)
	}
	if err := s.removeKnowledgeDocument(ctx, document); err != nil {
		return nil, err
	}
	return nil, nil
}

// listKnowledgeDocumentPassages reads back what a document was cut into.
func (s *Server) listKnowledgeDocumentPassages(ctx context.Context, request *listKnowledgeDocumentPassagesRequest) (*listKnowledgeDocumentPassagesResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, missingCustomer()
	}
	if s.store == nil || s.knowledge == nil {
		return nil, invalidRequest(noKnowledgeDocuments)
	}

	document, err := s.store.KnowledgeDocument(ctx, customerID, request.Id)
	if err != nil {
		return nil, notFound(unknownKnowledgeDocument)
	}
	passages, err := s.knowledgePassages(ctx, customerID, document.Namespace, document.Source, document.Passages)
	if err != nil {
		return nil, err
	}
	return &listKnowledgeDocumentPassagesResponse{Body: passages}, nil
}

// knowledgePassages reads a source's passages back in the order it was cut into them.
func (s *Server) knowledgePassages(
	ctx context.Context, customerID, namespace, source string, count int,
) ([]KnowledgePassage, error) {
	found, err := s.knowledge.Fetch(ctx, knowledge.Scoped(customerID, namespace), ingest.IDs(source, 0, count))
	if err != nil {
		return nil, err
	}
	passages := make([]KnowledgePassage, 0, len(found))
	for _, document := range found {
		passages = append(passages, KnowledgePassage{Id: document.ID, Source: document.Source, Text: document.Text})
	}
	return passages, nil
}

// fillKnowledge cuts documents into passages and writes them. The count of documents
// actually read can be less than what was sent: a file of only whitespace is skipped.
//
// With a database, each document is recorded with how many passages it became, and what
// a shorter version no longer covers is removed, so an edit does not leave its old tail
// behind to be found.
func (s *Server) fillKnowledge(
	ctx context.Context, customerID, namespace string, documents []KnowledgeDocument, chunkSize *int,
) (int, int, error) {
	size := ingest.DefaultChunk
	if chunkSize != nil && *chunkSize > 0 {
		size = *chunkSize
	}

	var passages []knowledge.Document
	var written []store.KnowledgeDocument
	for _, document := range documents {
		source := strings.TrimSpace(document.Source)
		if source == "" {
			return 0, 0, stack.Wrap(errors.New("every document needs a source, which is what its passages are keyed by"))
		}
		// A document that is only whitespace is skipped rather than refused: a directory
		// posted whole often has one in it, and failing the lot over it helps nobody.
		if strings.TrimSpace(document.Text) == "" {
			continue
		}
		cut := ingest.Split(source, document.Text, size)
		passages = append(passages, cut...)
		written = append(written, store.KnowledgeDocument{
			CustomerID: customerID,
			Namespace:  namespace,
			Source:     source,
			Passages:   len(cut),
			Text:       document.Text,
		})
	}

	if len(passages) == 0 {
		return 0, 0, stack.Wrap(errors.New("there is nothing to read in these documents"))
	}
	base := knowledge.Scoped(customerID, namespace)
	if err := s.knowledge.Upsert(ctx, base, passages); err != nil {
		return 0, 0, err
	}
	if s.store == nil {
		return len(written), len(passages), nil
	}

	stored, err := s.store.CustomerKnowledgeDocuments(ctx, customerID, namespace)
	if err != nil {
		return 0, 0, err
	}
	before := make(map[string]int, len(stored))
	for _, document := range stored {
		before[document.Source] = document.Passages
	}
	for i := range written {
		document := &written[i]
		stale := ingest.IDs(document.Source, document.Passages, before[document.Source])
		if err := s.knowledge.Delete(ctx, base, stale); err != nil {
			return 0, 0, err
		}
		if err := s.store.SaveKnowledgeDocument(ctx, document); err != nil {
			return 0, 0, err
		}
	}
	return len(written), len(passages), nil
}

// forgetKnowledge removes every document in a knowledge base that is not among these,
// which is what a synced directory losing a file means.
func (s *Server) forgetKnowledge(
	ctx context.Context, customerID, namespace string, documents []KnowledgeDocument,
) error {
	kept := make(map[string]struct{}, len(documents))
	for _, document := range documents {
		if strings.TrimSpace(document.Text) != "" {
			kept[strings.TrimSpace(document.Source)] = struct{}{}
		}
	}

	stored, err := s.store.CustomerKnowledgeDocuments(ctx, customerID, namespace)
	if err != nil {
		return err
	}
	for _, document := range stored {
		if _, ok := kept[document.Source]; ok {
			continue
		}
		if err := s.removeKnowledgeDocument(ctx, document); err != nil {
			return err
		}
	}
	return nil
}

// removeKnowledgeDocument deletes a document's passages and then its row. The passages go
// first: a document we have forgotten but are still answering out of is the failure worth
// avoiding.
func (s *Server) removeKnowledgeDocument(ctx context.Context, document store.KnowledgeDocument) error {
	ids := ingest.IDs(document.Source, 0, document.Passages)
	base := knowledge.Scoped(document.CustomerID, document.Namespace)
	if err := s.knowledge.Delete(ctx, base, ids); err != nil {
		return err
	}
	return s.store.DeleteKnowledgeDocument(ctx, document.CustomerID, document.ID)
}

// indexedKnowledgeDocumentOf is the stored row as the API describes it.
func indexedKnowledgeDocumentOf(document store.KnowledgeDocument) IndexedKnowledgeDocument {
	return IndexedKnowledgeDocument{
		Id:        document.ID,
		Namespace: document.Namespace,
		Source:    document.Source,
		Passages:  document.Passages,
		CreatedAt: document.CreatedAt,
		UpdatedAt: document.UpdatedAt,
	}
}

// registerKnowledge declares the operations served in knowledge.go.
func (s *Server) registerKnowledge(api huma.API) {
	huma.Register(api, huma.Operation{
		OperationID: "ingestKnowledge",
		Method:      http.MethodPost,
		Path:        "/v1/agents/knowledge",
		Summary:     "Fill a knowledge base with what the business wrote down",
		Description: "The writing half of the lookup a config's knowledge_namespace gives an agent. Each " +
			"document is cut into passages here rather than by the caller, so a file read off disk " +
			"by the command and one posted by an SDK are cut the same way and can replace each " +
			"other.\n" +
			"Passages are keyed by the source and the position they came from, so posting a document " +
			"again after editing it replaces that document's passages rather than leaving two " +
			"versions of them to be found. A namespace is never shared between agents, and the " +
			"caller names it.\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an end " +
			"user's device.",
		Responses: map[string]*huma.Response{
			"200": {Description: "The passages were written"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden},
	}, s.ingestKnowledge)
	huma.Register(api, huma.Operation{
		OperationID: "listKnowledgeDocuments",
		Method:      http.MethodGet,
		Path:        "/v1/agents/knowledge/documents",
		Summary:     "The documents a knowledge base was filled with",
		Description: "What was posted to /v1/agents/knowledge or synced from an agent directory's knowledge " +
			"folder, one entry per source. Pages are listed at /v1/agents/knowledge/urls.",
		Responses: map[string]*huma.Response{
			"200": {Description: "The customer's documents, most recently written first"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden},
	}, s.listKnowledgeDocuments)
	huma.Register(api, huma.Operation{
		OperationID: "getKnowledgeDocument",
		Method:      http.MethodGet,
		Path:        "/v1/agents/knowledge/documents/{id}",
		Summary:     "One document, with the text it was posted as",
		Description: "What to start from when editing it: post it again under the same source to replace it. " +
			"A document written before its text was kept comes back without one.",
		Responses: map[string]*huma.Response{
			"200": {Description: "The document"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.getKnowledgeDocument)
	huma.Register(api, huma.Operation{
		OperationID: "deleteKnowledgeDocument",
		Method:      http.MethodDelete,
		Path:        "/v1/agents/knowledge/documents/{id}",
		Summary:     "Take a document out of a knowledge base",
		Description: "The passages it was cut into are removed too, so the agent stops answering out of it.\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an end " +
			"user's device.",
		DefaultStatus: http.StatusNoContent,
		Responses: map[string]*huma.Response{
			"204": {Description: "The document and its passages are gone"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.deleteKnowledgeDocument)
	huma.Register(api, huma.Operation{
		OperationID: "listKnowledgeDocumentPassages",
		Method:      http.MethodGet,
		Path:        "/v1/agents/knowledge/documents/{id}/passages",
		Summary:     "What a document was cut into, in order",
		Description: "The passages as the agent finds them, which is what shows what a lookup can answer out " +
			"of this document.\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an end " +
			"user's device.",
		Responses: map[string]*huma.Response{
			"200": {Description: "The passages"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.listKnowledgeDocumentPassages)
}

type ingestKnowledgeRequest struct {
	Body *IngestKnowledgeRequest `required:"true"`
}

type ingestKnowledgeResponse struct {
	Body IngestedKnowledge
}

type listKnowledgeDocumentsRequest struct {
	Namespace optionalParam[string] `query:"namespace" doc:"One knowledge base. Omit to list every document the customer has."`
}

type listKnowledgeDocumentsResponse struct {
	Body []IndexedKnowledgeDocument `nullable:"false"`
}

type getKnowledgeDocumentRequest struct {
	Id string `path:"id" doc:"The resource, as returned when it was created."`
}

type getKnowledgeDocumentResponse struct {
	Body IndexedKnowledgeDocument
}

type deleteKnowledgeDocumentRequest struct {
	Id string `path:"id" doc:"The resource, as returned when it was created."`
}

type listKnowledgeDocumentPassagesRequest struct {
	Id string `path:"id" doc:"The resource, as returned when it was created."`
}

type listKnowledgeDocumentPassagesResponse struct {
	Body []KnowledgePassage `nullable:"false"`
}

// IndexedKnowledgeDocument is the IndexedKnowledgeDocument schema.
type IndexedKnowledgeDocument struct {
	CreatedAt time.Time `json:"created_at"`
	Id        string    `json:"id"`
	Namespace string    `json:"namespace"`
	Passages  int       `json:"passages" doc:"How many passages it was last cut into."`
	Source    string    `json:"source" doc:"What the document was posted as, and what its passages are keyed by." example:"pricing.md"`
	Text      *string   `json:"text,omitempty" doc:"The document as it was last posted. Only reading one document fills it in, and one written before its text was kept has none."`
	UpdatedAt time.Time `json:"updated_at" doc:"When it was last written."`
}

func (*IndexedKnowledgeDocument) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Properties["passages"].Format = ""
	return schema
}

// IngestKnowledgeRequest is the IngestKnowledgeRequest schema.
type IngestKnowledgeRequest struct {
	ChunkSize *int                `json:"chunk_size,omitempty" doc:"Characters per passage. Zero is the default, which is small enough that several passages fit in front of a model and large enough that one still answers the question on its own."`
	Documents []KnowledgeDocument `json:"documents" minItems:"1" nullable:"false"`
	Namespace string              `json:"namespace" doc:"The knowledge base to write into, which is what a config's knowledge_namespace names. Knowledge is never shared, so there is no default." example:"docs"`
}

func (*IngestKnowledgeRequest) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Properties["chunk_size"].Format = ""
	return schema
}

// IngestedKnowledge is the IngestedKnowledge schema.
type IngestedKnowledge struct {
	Documents int    `json:"documents" doc:"How many documents were read."`
	Namespace string `json:"namespace"`
	Passages  int    `json:"passages" doc:"How many passages they were cut into and written as."`
}

func (*IngestedKnowledge) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Properties["documents"].Format = ""
	schema.Properties["passages"].Format = ""
	return schema
}
