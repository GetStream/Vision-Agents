package api

import (
	"context"
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

	size := ingest.DefaultChunk
	if request.Body.ChunkSize != nil && *request.Body.ChunkSize > 0 {
		size = *request.Body.ChunkSize
	}

	var passages []knowledge.Document
	read := 0
	for _, document := range request.Body.Documents {
		source := strings.TrimSpace(document.Source)
		if source == "" {
			return IngestKnowledge400JSONResponse{
				badRequest("every document needs a source, which is what its passages are keyed by"),
			}, nil
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
		return IngestKnowledge400JSONResponse{
			badRequest("there is nothing to read in these documents"),
		}, nil
	}
	if err := s.knowledge.Upsert(ctx, namespace, passages); err != nil {
		return IngestKnowledge400JSONResponse{badRequest(err.Error())}, nil
	}

	s.logger.Info("filled a knowledge base",
		"namespace", namespace, "documents", read, "passages", len(passages))
	return IngestKnowledge200JSONResponse{
		Namespace: namespace,
		Documents: read,
		Passages:  len(passages),
	}, nil
}
