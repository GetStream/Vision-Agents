package api

import (
	"context"
	"errors"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge"
	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge/ingest"
)

// noKnowledge is what the paths say when the deployment has no knowledge provider. Filling
// a base that nothing can read is not worth pretending to do.
const noKnowledge = "knowledge is not available: no provider configured"

// IngestKnowledge fills a knowledge base with what the business wrote down.
//
// The documents are cut into passages here rather than by the caller, so a file read off
// disk by cmd/knowledge and one posted by an SDK are cut the same way and can replace each
// other.
func (s *Server) IngestKnowledge(ctx context.Context, request IngestKnowledgeRequestObject) (IngestKnowledgeResponseObject, error) {
	if _, ok := CustomerFrom(ctx); !ok {
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

	read, passages, err := s.fillKnowledge(ctx, namespace, request.Body.Documents, request.Body.ChunkSize)
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

// fillKnowledge cuts documents into passages and writes them. The count of documents
// actually read can be less than what was sent: a file of only whitespace is skipped.
func (s *Server) fillKnowledge(ctx context.Context, namespace string, documents []KnowledgeDocument, chunkSize *int) (int, int, error) {
	size := ingest.DefaultChunk
	if chunkSize != nil && *chunkSize > 0 {
		size = *chunkSize
	}

	var passages []knowledge.Document
	read := 0
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
		read++
		passages = append(passages, ingest.Split(source, document.Text, size)...)
	}

	if len(passages) == 0 {
		return 0, 0, errors.New("there is nothing to read in these documents")
	}
	if err := s.knowledge.Upsert(ctx, namespace, passages); err != nil {
		return 0, 0, err
	}
	return read, len(passages), nil
}
