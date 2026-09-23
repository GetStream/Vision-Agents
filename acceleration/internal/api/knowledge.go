package api

import (
	"context"
	"errors"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge"
	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge/ingest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// noKnowledge is what the paths say when the deployment has no knowledge provider. Filling
// a base that nothing can read is not worth pretending to do.
const noKnowledge = "knowledge is not available: no provider configured"

// noKnowledgeDocuments is what the document paths say on a deployment that cannot list
// them: it takes a database to remember one and a knowledge base to remove it from.
const noKnowledgeDocuments = "knowledge documents are not available: no database or no knowledge provider configured"

// unknownKnowledgeDocument is what a caller is told about a document that is not theirs,
// which is the same thing they are told about one that never existed.
const unknownKnowledgeDocument = "no such knowledge document"

// IngestKnowledge fills a knowledge base with what the business wrote down.
//
// The documents are cut into passages here rather than by the caller, so a file read off
// disk by cmd/knowledge and one posted by an SDK are cut the same way and can replace each
// other.
func (s *Server) IngestKnowledge(ctx context.Context, request IngestKnowledgeRequestObject) (IngestKnowledgeResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return IngestKnowledge401JSONResponse{missingCustomer()}, nil
	}
	if s.knowledge == nil {
		return IngestKnowledge400JSONResponse{badRequest(noKnowledge)}, nil
	}
	if request.Body == nil {
		return IngestKnowledge400JSONResponse{badRequest("a request body is required")}, nil
	}

	namespace := strings.TrimSpace(request.Body.Namespace)
	if namespace == "" {
		return IngestKnowledge400JSONResponse{
			badRequest("a namespace is required, knowledge is never shared"),
		}, nil
	}

	read, passages, err := s.fillKnowledge(ctx, customerID, namespace, request.Body.Documents, request.Body.ChunkSize)
	if err != nil {
		return IngestKnowledge400JSONResponse{badRequest(err.Error())}, nil
	}

	s.logger.Info("filled a knowledge base",
		"namespace", namespace, "documents", read, "passages", passages)
	return IngestKnowledge200JSONResponse{
		Namespace: namespace,
		Documents: read,
		Passages:  passages,
	}, nil
}

// ListKnowledgeDocuments returns the documents the calling customer's knowledge bases were
// filled with, most recently written first.
func (s *Server) ListKnowledgeDocuments(
	ctx context.Context, request ListKnowledgeDocumentsRequestObject,
) (ListKnowledgeDocumentsResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return ListKnowledgeDocuments401JSONResponse{missingCustomer()}, nil
	}
	if s.store == nil || s.knowledge == nil {
		return ListKnowledgeDocuments400JSONResponse{badRequest(noKnowledgeDocuments)}, nil
	}

	namespace := ""
	if request.Params.Namespace != nil {
		namespace = strings.TrimSpace(*request.Params.Namespace)
	}

	stored, err := s.store.CustomerKnowledgeDocuments(ctx, customerID, namespace)
	if err != nil {
		return nil, err
	}

	listed := make([]IndexedKnowledgeDocument, 0, len(stored))
	for _, document := range stored {
		listed = append(listed, indexedKnowledgeDocumentOf(document))
	}
	return ListKnowledgeDocuments200JSONResponse(listed), nil
}

// GetKnowledgeDocument returns one document with the text it was posted as.
func (s *Server) GetKnowledgeDocument(
	ctx context.Context, request GetKnowledgeDocumentRequestObject,
) (GetKnowledgeDocumentResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return GetKnowledgeDocument401JSONResponse{missingCustomer()}, nil
	}
	if s.store == nil || s.knowledge == nil {
		return GetKnowledgeDocument400JSONResponse{badRequest(noKnowledgeDocuments)}, nil
	}

	document, err := s.store.KnowledgeDocument(ctx, customerID, request.Id)
	if err != nil {
		return GetKnowledgeDocument404JSONResponse{NotFoundJSONResponse{Error: unknownKnowledgeDocument}}, nil
	}
	read := indexedKnowledgeDocumentOf(document)
	read.Text = &document.Text
	return GetKnowledgeDocument200JSONResponse(read), nil
}

// DeleteKnowledgeDocument takes a document out of its knowledge base, passages and all.
func (s *Server) DeleteKnowledgeDocument(
	ctx context.Context, request DeleteKnowledgeDocumentRequestObject,
) (DeleteKnowledgeDocumentResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return DeleteKnowledgeDocument401JSONResponse{missingCustomer()}, nil
	}
	if s.store == nil || s.knowledge == nil {
		return DeleteKnowledgeDocument400JSONResponse{badRequest(noKnowledgeDocuments)}, nil
	}

	document, err := s.store.KnowledgeDocument(ctx, customerID, request.Id)
	if err != nil {
		return DeleteKnowledgeDocument404JSONResponse{NotFoundJSONResponse{Error: unknownKnowledgeDocument}}, nil
	}
	if err := s.removeKnowledgeDocument(ctx, document); err != nil {
		return nil, err
	}
	return DeleteKnowledgeDocument204Response{}, nil
}

// ListKnowledgeDocumentPassages reads back what a document was cut into.
func (s *Server) ListKnowledgeDocumentPassages(
	ctx context.Context, request ListKnowledgeDocumentPassagesRequestObject,
) (ListKnowledgeDocumentPassagesResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return ListKnowledgeDocumentPassages401JSONResponse{missingCustomer()}, nil
	}
	if s.store == nil || s.knowledge == nil {
		return ListKnowledgeDocumentPassages400JSONResponse{badRequest(noKnowledgeDocuments)}, nil
	}

	document, err := s.store.KnowledgeDocument(ctx, customerID, request.Id)
	if err != nil {
		return ListKnowledgeDocumentPassages404JSONResponse{NotFoundJSONResponse{Error: unknownKnowledgeDocument}}, nil
	}
	passages, err := s.knowledgePassages(ctx, customerID, document.Namespace, document.Source, document.Passages)
	if err != nil {
		return nil, err
	}
	return ListKnowledgeDocumentPassages200JSONResponse(passages), nil
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
			return 0, 0, errors.New("every document needs a source, which is what its passages are keyed by")
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
		return 0, 0, errors.New("there is nothing to read in these documents")
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
