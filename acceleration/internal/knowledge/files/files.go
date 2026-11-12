// Package files reads the files a business uploads to a knowledge base, a PDF manual or a
// Word handbook, into the passages a lookup finds.
//
// A file is read in the background rather than in the request that sent it. A 50 MB manual
// takes seconds to read, and minutes when its pages are scans that are read from their
// pictures, so the upload is kept in the knowledge bucket, its document recorded as pending,
// and a task queued on asynq. The worker reads the file into Markdown with extract, has each
// scanned page, each page with charts or diagrams on it and each picture in it read by the
// OCR router, so what a figure shows is in the passages too, writes the passages as a posted
// document's are written, and deletes the file: the bucket only ever holds what is still
// to be read.
package files

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"path"
	"slices"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/hibiken/asynq"

	"github.com/GetStream/Vision-Agents/acceleration/internal/blob"
	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge/extract"
	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge/ingest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/metered"
	"github.com/GetStream/Vision-Agents/acceleration/internal/ocrrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/policy"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// TaskRead is the task that reads one uploaded file into its knowledge base.
const TaskRead = "knowledge:file:read"

// MaxBytes is the largest file one upload takes: a manual of a few hundred pages with its
// pictures in it.
const MaxBytes = 50 << 20

// MaxScannedPages is the most pages of one file read from their pictures. Each costs a
// fraction of a cent and a few seconds, and a file of more scans than this is better split
// than read for most of an hour.
const MaxScannedPages = 500

// MaxPictureReads is the most pictures of one file a model reads: its scanned pages first,
// then its pages with figures on them, then the pictures it holds. A file with more is
// read with the rest of its figures left out, rather than refused.
const MaxPictureReads = 500

// FiguresTarget is the shortcut a file's pictures are read through: models that can say
// what a chart or a diagram shows, and Mistral OCR for a scan's text when none answers.
const FiguresTarget = "ocr-figures"

const (
	// defaultQueue is a queue of its own: an asynq worker given a task it has no handler for
	// fails it, so the files and the urls are not read from one queue.
	defaultQueue = "knowledge-files"
	// readTimeout is reading the file, then every picture in it four at a time.
	readTimeout = 30 * time.Minute
	// maxRetry is how many more times a read that failed for a reason that may pass is tried
	// before the document is marked failed.
	maxRetry = 3
	// concurrency is how many files are read at once. A PDF is read in an engine of its own
	// with up to 512 MiB, so few.
	concurrency = 2
)

// Reader reads a page from its picture. *ocrrouter.Router is one.
type Reader interface {
	Read(ctx context.Context, request ocrrouter.Request) (ocrrouter.Reading, error)
}

// Options configures a Service. Everything but OCR is required.
type Options struct {
	Store *store.Store
	// Redis is the broker the reads are queued on.
	Redis asynq.RedisConnOpt
	// Bucket keeps a file until it is read.
	Bucket *blob.Bucket
	// Extractor reads the files. The service owns it, and Close releases it.
	Extractor *extract.Extractor
	// Knowledge is where the passages are written, each call recorded under the key that
	// uploaded the file.
	Knowledge *metered.Knowledge
	// OCR reads scanned pages and figures. Without it, a file with no text but its scans
	// is refused, and figures are left out.
	OCR Reader
	// ChunkSize is how much of a file goes in one passage. Zero is the ingest default.
	ChunkSize int
	// Limits bound how much larger a file's passages can be than the file.
	Limits ingest.Limits
	// CheckInterval is how often the worker looks for reads that are due, and how long a
	// failed one waits before it is tried again. Zero is asynq's defaults.
	CheckInterval time.Duration
	// Queue is the asynq queue the reads go through, "knowledge-files" when empty.
	Queue  string
	Logger *slog.Logger
}

// Service takes uploads and is the worker that reads them.
type Service struct {
	store     *store.Store
	queue     *asynq.Client
	queueName string
	worker    *asynq.Server
	bucket    *blob.Bucket
	extractor *extract.Extractor
	knowledge *metered.Knowledge
	ocr       Reader
	chunkSize int
	limits    ingest.Limits
	logger    *slog.Logger
}

// New validates the options and returns a Service. Nothing is read until Start.
func New(options Options) (*Service, error) {
	switch {
	case options.Store == nil:
		return nil, errors.New("files: a store is required")
	case options.Redis == nil:
		return nil, errors.New("files: a redis to queue the reads on is required")
	case options.Bucket == nil:
		return nil, errors.New("files: a bucket to keep the files in is required")
	case options.Extractor == nil:
		return nil, errors.New("files: something has to read the files")
	case options.Knowledge == nil:
		return nil, errors.New("files: a knowledge base is required")
	}
	if options.ChunkSize <= 0 {
		options.ChunkSize = ingest.DefaultChunk
	}
	if options.Logger == nil {
		options.Logger = slog.Default()
	}
	if options.Queue == "" {
		options.Queue = defaultQueue
	}

	config := asynq.Config{
		Concurrency: concurrency,
		Queues:      map[string]int{options.Queue: 1},
		LogLevel:    asynq.WarnLevel,
	}
	if interval := options.CheckInterval; interval > 0 {
		config.TaskCheckInterval = interval
		config.DelayedTaskCheckInterval = interval
		config.RetryDelayFunc = func(int, error, *asynq.Task) time.Duration { return interval }
	}

	return &Service{
		store:     options.Store,
		queue:     asynq.NewClient(options.Redis),
		queueName: options.Queue,
		worker:    asynq.NewServer(options.Redis, config),
		bucket:    options.Bucket,
		extractor: options.Extractor,
		knowledge: options.Knowledge,
		ocr:       options.OCR,
		chunkSize: options.ChunkSize,
		limits:    options.Limits,
		logger:    options.Logger,
	}, nil
}

// Start begins reading the files queued, including any left from before a restart.
func (s *Service) Start() error {
	mux := asynq.NewServeMux()
	mux.HandleFunc(TaskRead, s.process)
	return s.worker.Start(mux)
}

// Close lets the reads in flight finish, then stops the worker, the queue and the extractor.
func (s *Service) Close() error {
	s.worker.Shutdown()
	return errors.Join(s.queue.Close(), s.extractor.Close())
}

// Upload is a file sent to a knowledge base.
type Upload struct {
	CustomerID string
	Namespace  string
	// Source is what the passages are keyed by, the file's name unless the sender named it.
	Source string
	// ContentType is the file's type as the sender named it. The file is read by its bytes.
	ContentType string
	// Body is the file, which the caller caps at MaxBytes.
	Body io.Reader
}

// ErrEmpty is a file with nothing in it.
var ErrEmpty = errors.New("files: the file is empty")

// Upload keeps the file, records its document as pending and queues it to be read, for
// whoever ctx says sent it. A file whose first bytes are no format that can be read is
// refused with an *extract.Error before anything is kept. The document comes back with the
// passages it still has from before, which a lookup finds until the file's replace them.
func (s *Service) Upload(ctx context.Context, upload Upload) (store.KnowledgeDocument, error) {
	body := bufio.NewReaderSize(upload.Body, extract.SniffBytes)
	head, err := body.Peek(extract.SniffBytes)
	if err != nil && !errors.Is(err, io.EOF) {
		return store.KnowledgeDocument{}, err
	}
	if len(head) == 0 {
		return store.KnowledgeDocument{}, stack.Wrap(ErrEmpty)
	}
	if err := extract.Sniff(head); err != nil {
		return store.KnowledgeDocument{}, err
	}

	key := objectKey(upload.CustomerID, upload.Source)
	written, err := s.bucket.Write(ctx, key, upload.ContentType, body)
	if err != nil {
		return store.KnowledgeDocument{}, err
	}
	document := store.KnowledgeDocument{
		CustomerID:  upload.CustomerID,
		Namespace:   upload.Namespace,
		Source:      upload.Source,
		ContentType: upload.ContentType,
		Bytes:       written,
		ObjectKey:   key,
	}
	if err := s.store.PendKnowledgeFile(ctx, &document); err != nil {
		return store.KnowledgeDocument{}, errors.Join(err, s.bucket.Delete(context.WithoutCancel(ctx), key))
	}
	if err := s.enqueue(ctx, document); err != nil {
		_, failErr := s.store.FailKnowledgeFile(context.WithoutCancel(ctx), document.CustomerID, document.ID, key,
			"The file could not be queued to be read. Upload it again.", true)
		return store.KnowledgeDocument{}, errors.Join(err, failErr, s.bucket.Delete(context.WithoutCancel(ctx), key))
	}
	return document, nil
}

// Forget deletes the file of a document removed before it was read. The read finds the
// document gone and stops; this frees the bucket sooner.
func (s *Service) Forget(ctx context.Context, document store.KnowledgeDocument) error {
	if document.ObjectKey == "" {
		return nil
	}
	return s.bucket.Delete(ctx, document.ObjectKey)
}

// objectKey is where an upload is kept: under its customer, so a bucket listing reads as
// whose files are waiting, and under an id of its own, so a second upload of one source
// is a second file and the read of the first can tell it was overtaken.
func objectKey(customerID, source string) string {
	name := path.Base(strings.ReplaceAll(source, "\\", "/"))
	if name == "." || name == "/" || name == "" {
		name = "file"
	}
	return customerID + "/" + uuid.NewString() + "/" + name
}

// uploadOrigin is whose work a read is when no key was carried in the task.
var uploadOrigin = routing.JobOrigin("upload")

// readPayload names the file a task reads: the document, and the object it was uploaded
// as, which is how the read knows the document is still waiting for this file.
type readPayload struct {
	CustomerID  string               `json:"customer_id"`
	ID          string               `json:"id"`
	ObjectKey   string               `json:"object_key"`
	Attribution *routing.Attribution `json:"attribution,omitempty"`
}

func (p readPayload) owner() routing.Owner {
	owner := routing.Owner{CustomerID: p.CustomerID}
	if p.Attribution != nil {
		owner.Attribution = *p.Attribution
	}
	if owner.APIKey == "" {
		owner.APIKey = uploadOrigin
	}
	return owner
}

func (s *Service) enqueue(ctx context.Context, document store.KnowledgeDocument) error {
	queued := readPayload{CustomerID: document.CustomerID, ID: document.ID, ObjectKey: document.ObjectKey}
	if asked := routing.AttributionFrom(ctx); asked.APIKey != "" {
		queued.Attribution = &asked
	}
	payload, err := json.Marshal(queued)
	if err != nil {
		return stack.Wrap(fmt.Errorf("files: queue a read of %s: %w", document.Source, err))
	}
	_, err = s.queue.EnqueueContext(ctx, asynq.NewTask(TaskRead, payload),
		asynq.Queue(s.queueName),
		asynq.TaskID(document.ObjectKey),
		asynq.MaxRetry(maxRetry),
		asynq.Timeout(readTimeout),
	)
	if err != nil {
		return stack.Wrap(fmt.Errorf("files: queue a read of %s: %w", document.Source, err))
	}
	return nil
}

// errOvertaken ends a read of a file whose document was removed, or uploaded or posted
// again, while it waited or was read.
var errOvertaken = errors.New("files: the document no longer waits for this file")

// unreadable is a file no attempt will read, with what its sender is told.
type unreadable struct{ reason string }

func (u *unreadable) Error() string { return u.reason }

// unpaid is a read refused because the customer may not spend money now: no credit, a
// spend limit reached, a suspended organization. Unlike other errors its reason is the
// customer's to read, and it may pass, so it is tried again.
type unpaid struct{ err error }

func (u *unpaid) Error() string { return u.err.Error() }

func (u *unpaid) Unwrap() error { return u.err }

// why is the refusal as the customer reads it, without the package that made it, as the API
// answers it.
func (u *unpaid) why() string { return strings.TrimPrefix(u.err.Error(), "policy: ") }

// spendRefusals are the refusals of an admission that are the customer's own to read and to
// fix. A spend check that could not be shared is the router's, and reads as any other error.
var spendRefusals = []error{
	policy.ErrSuspended, policy.ErrNoCredits, policy.ErrSystemLimit, policy.ErrBudgetSpent, policy.ErrUserLimit,
}

// process is the worker's half of an upload: it reads the file, writes its passages,
// records what happened and deletes the file.
//
// A read that failed for a reason that may pass, a vendor down or a write refused, is
// returned to asynq to be tried again, with the reason on the row meanwhile. A file that
// cannot be read, encrypted or damaged or without text, fails at once. Only a last failure
// marks the document failed, and either way the file is deleted once nothing will read it.
func (s *Service) process(ctx context.Context, task *asynq.Task) error {
	var payload readPayload
	if err := json.Unmarshal(task.Payload(), &payload); err != nil {
		return fmt.Errorf("files: read the task: %v: %w", err, asynq.SkipRetry)
	}
	forget := func() {
		if err := s.bucket.Delete(context.WithoutCancel(ctx), payload.ObjectKey); err != nil {
			s.logger.Warn("could not delete an uploaded file", "key", payload.ObjectKey, "error", err)
		}
	}

	document, err := s.waiting(ctx, payload)
	if errors.Is(err, errOvertaken) {
		forget()
		return nil
	}
	if err != nil {
		return err
	}

	owned := s.knowledge.For(payload.owner())
	ctx = routing.WithAttribution(ctx, payload.owner().Attribution)
	err = s.read(ctx, owned, payload, &document)
	if errors.Is(err, errOvertaken) {
		forget()
		return nil
	}
	if err == nil {
		forget()
		s.logger.Info("read a file into a knowledge base",
			"source", document.Source, "namespace", document.Namespace,
			"pages", document.Pages, "passages", document.Passages)
		return nil
	}

	retried, _ := asynq.GetRetryCount(ctx)
	limit, _ := asynq.GetMaxRetry(ctx)
	var cannot *unreadable
	var refused *unpaid
	last := errors.As(err, &cannot) || retried >= limit
	// Only the reasons that are the customer's own are written for them: the rest, a
	// bucket path or a vendor's answer, stay in the log.
	reason := "The file could not be read yet, and is tried again."
	switch {
	case cannot != nil:
		reason = cannot.reason
	case errors.As(err, &refused) && last:
		reason = "The file was not read, as nothing can be spent now: " + refused.why() + ". Upload it again once it can."
	case refused != nil:
		reason = "The file is not read yet, as nothing can be spent now: " + refused.why() + ". It is tried again."
	case last:
		reason = "The file could not be read. Upload it again."
	}
	s.logger.Warn("could not read a file into a knowledge base",
		"source", document.Source, "namespace", document.Namespace, "attempt", retried+1, "last", last, "error", err)
	if _, failErr := s.store.FailKnowledgeFile(ctx, payload.CustomerID, payload.ID, payload.ObjectKey, reason, last); failErr != nil {
		return errors.Join(err, failErr)
	}
	if !last {
		return err
	}
	forget()
	return nil
}

// waiting is the document, while it still waits for the payload's file.
func (s *Service) waiting(ctx context.Context, payload readPayload) (store.KnowledgeDocument, error) {
	document, err := s.store.KnowledgeDocument(ctx, payload.CustomerID, payload.ID)
	if errors.Is(err, store.ErrNoKnowledgeDocument) {
		return store.KnowledgeDocument{}, errOvertaken
	}
	if err != nil {
		return store.KnowledgeDocument{}, err
	}
	if document.ObjectKey != payload.ObjectKey {
		return store.KnowledgeDocument{}, errOvertaken
	}
	return document, nil
}

// read reads the file, writes its passages, and removes what the document's last version
// left past the end of this one. Writing first means a lookup landing mid-update reads
// the new passages or the old ones rather than a gap.
func (s *Service) read(ctx context.Context, owned metered.OwnedKnowledge, payload readPayload, document *store.KnowledgeDocument) error {
	// Reading scans and writing passages are paid for, so a customer whose write would be
	// refused is not read for.
	if err := owned.Admit(ctx); err != nil {
		if slices.ContainsFunc(spendRefusals, func(refusal error) bool { return errors.Is(err, refusal) }) {
			return &unpaid{err: err}
		}
		return err
	}
	data, err := s.bucket.Read(ctx, payload.ObjectKey)
	if err != nil {
		return err
	}
	read, err := s.extractor.Extract(ctx, document.Source, data)
	if err != nil {
		if reason, ok := extract.ReasonOf(err); ok {
			return &unreadable{reason: refusal(reason, err)}
		}
		return err
	}
	if err := s.readPictures(ctx, payload, &read); err != nil {
		return err
	}

	markdown := read.Markdown()
	passages := ingest.Split(document.Source, markdown, s.chunkSize, s.limits)
	if len(passages) == 0 {
		return &unreadable{reason: "The file has no text in it."}
	}

	// A read takes long enough for the document to be removed or replaced meanwhile, and
	// passages written for it then would bring it back. It is looked for again once
	// written, since it can go between the look and the write.
	if _, err := s.waiting(ctx, payload); err != nil {
		return err
	}
	if err := owned.Upsert(ctx, document.Namespace, passages); err != nil {
		return err
	}
	current, err := s.waiting(ctx, payload)
	if errors.Is(err, errOvertaken) {
		if _, gone := s.store.KnowledgeDocument(ctx, payload.CustomerID, payload.ID); errors.Is(gone, store.ErrNoKnowledgeDocument) {
			if taken := owned.Delete(ctx, document.Namespace, ingest.IDs(document.Source, 0, len(passages))); taken != nil {
				return fmt.Errorf("files: take back what was read for a document removed meanwhile: %w", taken)
			}
		}
		return err
	}
	if err != nil {
		return err
	}
	// A file that got shorter would otherwise leave its old tail behind, still findable
	// and no longer in the file it claims to come from.
	if err := owned.Delete(ctx, document.Namespace, ingest.IDs(document.Source, len(passages), current.Passages)); err != nil {
		return err
	}

	document.Text = markdown
	document.Passages = len(passages)
	document.Pages = pagesOf(read)
	finished, err := s.store.FinishKnowledgeFile(ctx, document, payload.ObjectKey)
	if err != nil {
		return err
	}
	if !finished {
		return errOvertaken
	}
	return nil
}

// readPictures has every scanned page read for its text and figures, every page with
// figures on it read for them, and every picture in the file read for what it shows, a
// few at a time, through FiguresTarget (ReadPictures). A file with scans and nothing to
// read them is refused only when the scans are all it has: the pages with text are still
// worth looking up.
//
// A scan that cannot be read fails the read, to be tried again, since the page's text is
// missing without it. A figure that cannot be read is left out: every model in
// FiguresTarget failed it, which is more likely the picture than an outage, and reading
// the file again would only pay for its scans twice.
func (s *Service) readPictures(ctx context.Context, payload readPayload, read *extract.Document) error {
	scanned := read.Scanned()
	if s.ocr == nil {
		if len(scanned) > 0 && len(scanned) == len(read.Pages) {
			return &unreadable{reason: "The file has no text in it: its pages are scanned pictures, " +
				"and reading text from pictures is not available here. Upload a copy with a text layer."}
		}
		return nil
	}
	refusedSpend := func(err error) bool {
		return slices.ContainsFunc(spendRefusals, func(refusal error) bool { return errors.Is(err, refusal) })
	}
	err := ReadPictures(ctx, s.ocr, ocrrouter.Request{CustomerID: payload.CustomerID, Target: FiguresTarget}, read,
		PictureLimits{Scanned: MaxScannedPages, Reads: MaxPictureReads}, refusedSpend,
		s.logger.With("customer_id", payload.CustomerID))
	var tooMany *TooManyScansError
	switch {
	case errors.As(err, &tooMany):
		return &unreadable{reason: fmt.Sprintf("The file has %d scanned pages, more than the %d read from their "+
			"pictures. Split it, or upload a copy with a text layer.", tooMany.Scanned, tooMany.Limit)}
	case refusedSpend(err):
		return &unpaid{err: spentOf(err)}
	}
	return err
}

// spentOf is the spend refusal err holds, which is what its customer is told.
func spentOf(err error) error {
	for _, refusal := range spendRefusals {
		if errors.Is(err, refusal) {
			return refusal
		}
	}
	return err
}

// pagesOf is how many pages the file had, 0 for one without pages.
func pagesOf(read extract.Document) int {
	pages := 0
	for _, page := range read.Pages {
		pages = max(pages, page.Number)
	}
	return pages
}

// refusal is what the sender of a file extract could not read is told.
func refusal(reason extract.Reason, err error) string {
	var failed *extract.Error
	detail := ""
	if errors.As(err, &failed) && failed.Err != nil {
		detail = failed.Err.Error()
	}
	switch reason {
	case extract.Encrypted:
		return "The file is protected with a password. Save a copy without the password and upload that."
	case extract.Corrupt:
		return "The file is damaged and cannot be read (" + detail + "). Export it again and upload the new copy."
	case extract.TooLarge:
		return "The file is too large to read: " + detail + "."
	default:
		return "This kind of file cannot be read (" + detail + "). Upload a PDF, a Word document (DOCX or DOC), " +
			"an Excel workbook (XLSX), an RTF or OpenDocument text file, or Markdown or plain text."
	}
}

// Refusal is what the sender of a file refused at upload is told, for an error Upload
// returned, and false for an error that is not about the file.
func Refusal(err error) (string, bool) {
	if errors.Is(err, ErrEmpty) {
		return "The file is empty.", true
	}
	if reason, ok := extract.ReasonOf(err); ok {
		return refusal(reason, err), true
	}
	return "", false
}
