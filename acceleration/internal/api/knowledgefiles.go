package api

import (
	"encoding/json"
	"errors"
	"io"
	"mime"
	"net/http"
	"path"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge/files"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// knowledgeFilesPath is where a file is uploaded to a knowledge base.
const knowledgeFilesPath = "/v1/agents/knowledge/files"

// maxFormOverhead is the room a multipart body takes around its file: the boundaries, a
// part's headers and any small field sent beside it.
const maxFormOverhead = 1 << 20

// maxKnowledgeSource is the longest source, as KnowledgeDocument.Source allows a posted one.
const maxKnowledgeSource = 256

// fileReadTimeout is how long the file may take to arrive: 50 MB at under a megabit a
// second, as an upload from a shop's connection can be.
const fileReadTimeout = 10 * time.Minute

var errNoKnowledgeFiles = notConfigured("this deployment does not take files for knowledge: it needs a database, a knowledge " +
	"base, a queue and a bucket to keep the files in (ROUTER_KNOWLEDGE_BUCKET_URL)")

// errFileTooLarge is a file past files.MaxBytes.
var errFileTooLarge = errors.New("api: the file is larger than a knowledge upload takes")

// uploadKnowledgeFile takes a file to read into a knowledge base. It answers once the file is
// kept and queued, with the document pending; the worker reads it and the document says
// how that went.
func (s *Server) uploadKnowledgeFile(w http.ResponseWriter, r *http.Request) {
	customerID, known := CustomerFrom(r.Context())
	if !known {
		writeError(w, errMissingCustomer)
		return
	}
	if s.refuseClientSide(w, r) {
		return
	}
	if s.knowledgeFiles == nil || s.store == nil || s.knowledge == nil {
		writeError(w, errNoKnowledgeFiles)
		return
	}
	namespace := strings.TrimSpace(r.URL.Query().Get("namespace"))
	if namespace == "" {
		writeError(w, invalidRequest("a namespace is required, knowledge is never shared"))
		return
	}

	_ = http.NewResponseController(w).SetReadDeadline(time.Now().Add(fileReadTimeout))
	r.Body = http.MaxBytesReader(w, r.Body, files.MaxBytes+maxFormOverhead)
	form, err := r.MultipartReader()
	if err != nil {
		writeError(w, invalidRequest("send the file as multipart/form-data, in a part named file"))
		return
	}
	var part io.ReadCloser
	var name, sent string
	for {
		next, err := form.NextPart()
		if errors.Is(err, io.EOF) {
			writeError(w, invalidRequest("the form has no part named file"))
			return
		}
		if err != nil {
			writeError(w, uploadFailure(err))
			return
		}
		if next.FormName() != "file" {
			_ = next.Close()
			continue
		}
		part, name, sent = next, next.FileName(), next.Header.Get("Content-Type")
		break
	}
	defer part.Close()

	source := strings.TrimSpace(r.URL.Query().Get("source"))
	if source == "" {
		source = strings.TrimSpace(name)
	}
	if source == "" {
		writeError(w, invalidRequest("the file part has no filename, so pass source to say what its passages are keyed by"))
		return
	}
	if len(source) > maxKnowledgeSource {
		writeError(w, invalidRequest("a source is at most 256 bytes"))
		return
	}

	document, err := s.knowledgeFiles.Upload(r.Context(), files.Upload{
		CustomerID:  customerID,
		Namespace:   namespace,
		Source:      source,
		ContentType: contentTypeOf(sent, source),
		Body:        &capped{reader: part, left: files.MaxBytes},
	})
	if refusal, ok := files.Refusal(err); ok {
		writeError(w, invalidRequest(refusal))
		return
	}
	if err != nil {
		writeError(w, uploadFailure(err))
		return
	}

	action := store.AuditUpdated
	if document.CreatedAt.Equal(document.UpdatedAt) {
		action = store.AuditCreated
	}
	s.audit(r.Context(), auditRecord{
		ResourceType: store.AuditKnowledge, ResourceID: document.ID, ResourceName: document.Source,
		AgentID: s.agentOfKnowledge(r.Context(), customerID, namespace), Action: action,
		Changes: auditDiff(nil, map[string]any{
			"namespace": document.Namespace, "source": document.Source,
			"content_type": document.ContentType, "bytes": document.Bytes,
		}),
	})
	s.logger.Info("took a file for a knowledge base",
		"namespace", namespace, "source", source, "bytes", document.Bytes)
	w.Header().Set("Location", "/v1/agents/knowledge/documents/"+document.ID)
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(http.StatusAccepted)
	_ = json.NewEncoder(w).Encode(indexedKnowledgeDocumentOf(document))
}

// uploadFailure is the answer to an upload that failed while its body was read or kept.
func uploadFailure(err error) APIError {
	var tooLarge *http.MaxBytesError
	if errors.Is(err, errFileTooLarge) || errors.As(err, &tooLarge) {
		return payloadTooLarge("a knowledge file is at most 50 MB: split a larger one, or compress its pictures")
	}
	return unavailable("the file could not be kept: " + err.Error())
}

// contentTypeOf is the type the file was sent as, or the one its name says when the sender
// named none. It is recorded rather than trusted: the file is read by its bytes.
func contentTypeOf(sent, source string) string {
	if media, _, err := mime.ParseMediaType(sent); err == nil && media != "application/octet-stream" {
		return media
	}
	extension := strings.ToLower(path.Ext(source))
	if known, ok := knowledgeFileTypes[extension]; ok {
		return known
	}
	if byName := mime.TypeByExtension(extension); byName != "" {
		media, _, _ := mime.ParseMediaType(byName)
		return media
	}
	return "application/octet-stream"
}

// knowledgeFileTypes are the types of the files knowledge reads, which the system's own
// table of types may not know.
var knowledgeFileTypes = map[string]string{
	".pdf":  "application/pdf",
	".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
	".doc":  "application/msword",
	".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
	".odt":  "application/vnd.oasis.opendocument.text",
	".rtf":  "application/rtf",
	".md":   "text/markdown",
	".txt":  "text/plain",
}

// capped reads at most left bytes, and fails past them rather than stopping short, so a
// file over the limit is refused instead of kept cut off.
type capped struct {
	reader io.Reader
	left   int64
}

func (c *capped) Read(p []byte) (int, error) {
	if c.left < 0 {
		return 0, errFileTooLarge
	}
	if int64(len(p)) > c.left+1 {
		p = p[:c.left+1]
	}
	n, err := c.reader.Read(p)
	c.left -= int64(n)
	if c.left < 0 {
		return n, errFileTooLarge
	}
	return n, err
}
