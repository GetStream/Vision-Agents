package api

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"path"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/danielgtaylor/huma/v2"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/decisionrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge/extract"
	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge/files"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/ocrrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/policy"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// FileSource is a file to answer questions about.
type FileSource struct {
	Url  string  `json:"url" doc:"Public HTTP(S) URL or base64 data URI, such as data:application/pdf;base64,.... A PDF or a Word document (DOCX or DOC), told apart by its bytes. At most 20 MiB fetched, or 8 MiB as a data URI. The router fetches a URL itself, and refuses one that resolves to a private or loopback address."`
	Name *string `json:"name,omitempty" doc:"What the file is called, shown on the person's message and to the model. Left out, the last part of the URL's path, or document for a data URI." maxLength:"255"`
}

func (*FileSource) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "A file sent with a question. The router reads it as text once, and the agent answers from it on this turn and every turn of the conversation after it."
	return schema
}

const (
	// maxFetchedDocument is the most a file fetched by URL may be.
	maxFetchedDocument = 20 << 20
	// maxInlineDocument is the most a file sent as a data URI may be: base64 makes it a third
	// larger, so it stays inside maxInlineAudioBody with room for the rest of the request.
	maxInlineDocument = 8 << 20
	// maxDocumentPages is the most pages a file sent with a question may have.
	maxDocumentPages = 100
	// documentReadTimeout bounds reading one file while the person waits.
	documentReadTimeout = 20 * time.Second
)

// NewDocumentReader is the reader of the files sent with a question: one of its own, so a
// question never waits behind a manual being read into a knowledge base.
func NewDocumentReader(options extract.Options) *extract.Extractor {
	options.MaxPages, options.Timeout = maxDocumentPages, documentReadTimeout
	return extract.New(options)
}

var (
	errNoDocuments = notConfigured("this deployment does not read files sent with a question")
	// errDocumentWithVoiceNote keeps a request within its body limit: a file and a voice note
	// as data URIs would not fit together.
	errDocumentWithVoiceNote = invalidRequest("a file and a voice note go in separate responses")
	errDocumentNative        = invalidRequest("a speech-to-speech session takes no files: the model hears and speaks for itself and reads nothing")
	errDocumentShared        = invalidRequest("a file can only be sent in a person's own conversation, not in a thread or a channel others write in")
	errDocumentRefused       = APIError{Type: ErrorTypeInvalidRequest, Code: codePromptInjection,
		Message: "the file reads as a prompt injection, so the agent did not read it"}
	errDocumentUnscreened = unavailable("the file could not be screened for prompt injection, so the agent did not read it; try again")
)

func documentTooLarge(message string) APIError {
	return APIError{Type: ErrorTypePayloadTooLarge, Code: codeDocumentTooLarge, Message: message}
}

func documentUnreadable(message string) APIError {
	return APIError{Type: ErrorTypeInvalidRequest, Code: codeDocumentUnreadable, Message: message}
}

// documentFormats are the formats a file sent with a question may be.
var documentFormats = map[extract.Format]string{
	extract.PDF:  "application/pdf",
	extract.DOCX: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
	extract.DOC:  "application/msword",
}

// sentDocument is a file sent with a question: as it was sent, for the person's message to
// show it, and as the agent reads it.
type sentDocument struct {
	file sandbox.File
	read agent.Document
	sha  string
}

// readDocument fetches the file sent to found, reads it as text and screens it. Everything that
// can refuse it does so here, before the turn starts and before anything is billed. The retry of
// a question already taken is only fetched, for its digest to be matched: it never reaches the
// model, and the file its first sending pinned leaves it no room to be weighed again.
func (s *Server) readDocument(ctx context.Context, found *session.Session, source FileSource, retry bool) (sentDocument, error) {
	if s.documents == nil {
		return sentDocument{}, errNoDocuments
	}
	spec := found.Spec()
	switch {
	case spec.Native():
		return sentDocument{}, errDocumentNative
	case spec.Dispatched || spec.Shared():
		return sentDocument{}, errDocumentShared
	}

	// Fetched, read and screened while the person waits, so all of it within one deadline.
	ctx, cancel := context.WithTimeout(ctx, inlineDeadline)
	defer cancel()
	data, name, failure := s.fetchDocument(ctx, source)
	if failure != nil {
		return sentDocument{}, failure
	}
	digest := sha256.Sum256(data)
	if retry {
		return sentDocument{file: sandbox.File{Name: name, MIME: http.DetectContentType(data), Data: data}, sha: hex.EncodeToString(digest[:])}, nil
	}
	read, err := s.documents.Extract(ctx, name, data)
	if err != nil {
		return sentDocument{}, documentFailure(err)
	}
	mime, accepted := documentFormats[read.Format]
	if !accepted {
		return sentDocument{}, newAPIError(ErrorTypeUnsupportedMediaType,
			fmt.Sprintf("the file is %s: a PDF or a Word document (DOCX or DOC) can be sent", read.Format))
	}
	if failure := s.readDocumentPictures(ctx, spec, &read); failure != nil {
		return sentDocument{}, failure
	}
	if !hasText(read) {
		return sentDocument{}, documentUnreadable("the file has no text in it")
	}
	text := withUnreadPages(read)
	if room, runes := found.DocumentRoom(), utf8.RuneCountInString(text); runes > room {
		return sentDocument{}, documentTooLarge(fmt.Sprintf(
			"the file holds %d characters of text, and the conversation has room for %d more", runes, room))
	}

	owner := routing.Owner{CustomerID: spec.CustomerID, AgentID: spec.AgentID, CallID: spec.CallID, Region: spec.Region,
		Tags: spec.Tags, Attribution: spec.Attribution}
	// The name goes ahead of every reply with the text, so it is screened with it.
	err = s.policies.ScreenDocument(ctx, s.decisionModel(), owner, name+"\n\n"+text)
	if answer, money := moneyError(err); money {
		return sentDocument{}, answer
	}
	switch {
	case errors.Is(err, policy.ErrPromptInjection):
		return sentDocument{}, errDocumentRefused
	case err != nil:
		return sentDocument{}, errDocumentUnscreened
	}

	return sentDocument{
		file: sandbox.File{Name: name, MIME: mime, Data: data},
		read: agent.Document{ID: store.NewID(), Name: name, Pages: len(read.Pages), Text: text},
		sha:  hex.EncodeToString(digest[:]),
	}, nil
}

// decisionModel is the classifier the prompt injection screen asks, nil when there is none.
func (s *Server) decisionModel() *decisionrouter.Router {
	if s.streams == nil {
		return nil
	}
	return s.streams.DecisionModel
}

// keepDocument records the file sent with the turn responseID, so a conversation carried on
// or forked reads it again. An incognito session, or one recording nothing, keeps it only
// while it runs. A file that cannot be kept leaves the turn answered and says so in the log,
// as a session's other records do.
func (s *Server) keepDocument(ctx context.Context, found *session.Session, responseID string, sent sentDocument) {
	spec := found.Spec()
	if s.store == nil || spec.Incognito || responseID == "" {
		return
	}
	// The turn is running whether or not the caller is still there to hear it started.
	err := s.store.AddConversationDocument(context.WithoutCancel(ctx), &store.ConversationDocument{
		ID: sent.read.ID, CustomerID: spec.CustomerID, OwnerUserID: spec.Caller.UserID, SessionID: found.ID(),
		ConversationID: spec.ConversationID, ResponseID: responseID, Name: sent.read.Name, Pages: sent.read.Pages,
		SHA256: sent.sha, Markdown: sent.read.Text,
	})
	if err != nil {
		s.logger.Error("could not keep a file sent with a question", "session", found.ID(), "response", responseID, "error", err)
	}
}

// Pictures of a file sent with a question are read with the person waiting, so far fewer of
// them than of a knowledge file.
const (
	// maxDocumentScans is the most scanned pages read from their pictures.
	maxDocumentScans = 10
	// maxDocumentPictures is the most pictures read in all: scans, then pages with charts on
	// them, then the pictures a Word file holds.
	maxDocumentPictures = 10
)

// readDocumentPictures reads a file's scanned pages, and its charts and pictures, as text,
// through the OCR models knowledge reads them with, billed to the session's customer. A
// deployment with no OCR leaves them unread, and the model is told which (withUnreadPages).
func (s *Server) readDocumentPictures(ctx context.Context, spec session.Spec, read *extract.Document) error {
	if s.streams == nil || s.streams.OCR == nil {
		return nil
	}
	ctx = routing.WithAttribution(ctx, spec.Attribution)
	request := ocrrouter.Request{CustomerID: spec.CustomerID, Tags: spec.Tags, Target: files.FiguresTarget}
	refused := func(err error) bool { _, money := moneyError(err); return money }
	err := files.ReadPictures(ctx, s.streams.OCR, request, read,
		files.PictureLimits{Scanned: maxDocumentScans, Reads: maxDocumentPictures}, refused,
		s.logger.With("customer_id", spec.CustomerID))
	var tooMany *files.TooManyScansError
	switch {
	case err == nil:
		return nil
	case errors.As(err, &tooMany):
		return documentTooLarge(fmt.Sprintf("the file has %d pages that are only a picture, and at most %d are read: send a copy with a text layer",
			tooMany.Scanned, tooMany.Limit))
	}
	if answer, money := moneyError(err); money {
		return answer
	}
	return modelFailed(err)
}

// fetchDocument is the bytes of the file at source, a data URI or a URL, and its name.
func (s *Server) fetchDocument(ctx context.Context, source FileSource) ([]byte, string, error) {
	name := strings.TrimSpace(value(source.Name))
	if strings.HasPrefix(source.Url, "data:") {
		parsed, err := llm.ParseDataURI(source.Url)
		if err != nil {
			return nil, "", invalidRequest("the file is not a base64 data URI with a type")
		}
		if len(parsed.Data) > maxInlineDocument {
			return nil, "", documentTooLarge(fmt.Sprintf("a file sent as a data URI is at most %d MiB: send a URL for one up to %d MiB",
				maxInlineDocument>>20, maxFetchedDocument>>20))
		}
		return parsed.Data, documentName(name, "document"), nil
	}
	data, _, err := fetchAtMost(ctx, s.imageHTTP, source.Url, maxFetchedDocument)
	if errors.As(err, new(fetchTooLargeError)) {
		return nil, "", documentTooLarge(fmt.Sprintf("a file fetched by URL is at most %d MiB", maxFetchedDocument>>20))
	}
	if err != nil {
		return nil, "", invalidRequest(fmt.Sprintf("the file could not be fetched: %v", err))
	}
	fallback := "document"
	if parsed, err := url.Parse(source.Url); err == nil && path.Base(parsed.Path) != "/" && path.Base(parsed.Path) != "." {
		fallback = path.Base(parsed.Path)
	}
	return data, documentName(name, fallback), nil
}

// maxDocumentName is the most runes of a file's name kept, as FileSource.name allows.
const maxDocumentName = 255

// documentName is a file's name as it is uploaded: never a path, never empty, and no longer
// than a name sent in the request may be, since one taken from a URL is the URL's.
func documentName(name, fallback string) string {
	name = path.Base(strings.ReplaceAll(strings.TrimSpace(name), "\\", "/"))
	if name == "" || name == "." || name == "/" {
		name = fallback
	}
	if runes := []rune(name); len(runes) > maxDocumentName {
		name = string(runes[:maxDocumentName])
	}
	return name
}

// documentFailure is the answer to a file the reader refused.
func documentFailure(err error) error {
	if errors.Is(err, context.DeadlineExceeded) {
		return documentTooLarge(fmt.Sprintf("the file took longer than the %s a file sent with a question may take to read", documentReadTimeout))
	}
	reason, ok := extract.ReasonOf(err)
	if !ok {
		return err
	}
	switch reason {
	case extract.TooLarge:
		return documentTooLarge(fmt.Sprintf("the file is larger than a file sent with a question may be (at most %d pages): %v", maxDocumentPages, err))
	case extract.Unsupported:
		return newAPIError(ErrorTypeUnsupportedMediaType, "the file is no format that can be read: a PDF or a Word document (DOCX or DOC) can be sent")
	case extract.Encrypted:
		return documentUnreadable("the file needs a password to open")
	default:
		return documentUnreadable(fmt.Sprintf("the file could not be read: %v", err))
	}
}

// hasText reports whether any page of a file has words on it.
func hasText(read extract.Document) bool {
	for _, page := range read.Pages {
		if strings.TrimSpace(page.Text) != "" {
			return true
		}
	}
	return false
}

// withUnreadPages is a file's text, followed by which of its pages are pictures nobody read,
// so the model says it cannot see them rather than guessing at them.
func withUnreadPages(read extract.Document) string {
	text := read.Markdown()
	if scanned := read.Scanned(); len(scanned) > 0 {
		text += "\n\n[" + pageList(scanned, "is", "are") + " only a picture, which was not read]"
	}
	if figured := unreadFigures(read); len(figured) > 0 {
		text += "\n\n[" + pageList(figured, "has", "have") + " charts, diagrams or pictures, which were not read]"
	}
	return text
}

// unreadFigures are the pages with charts on them that nothing read.
func unreadFigures(read extract.Document) []int {
	var numbers []int
	for _, page := range read.Pages {
		if page.Figured && strings.TrimSpace(page.Figures) == "" {
			numbers = append(numbers, page.Number)
		}
	}
	return numbers
}

// pageList names pages as a sentence opens, with its verb: "Page 3 is", "Pages 3, 7 are".
func pageList(numbers []int, one, several string) string {
	words := make([]string, 0, len(numbers))
	for _, number := range numbers {
		words = append(words, strconv.Itoa(number))
	}
	if len(words) == 1 {
		return "Page " + words[0] + " " + one
	}
	return "Pages " + strings.Join(words, ", ") + " " + several
}
