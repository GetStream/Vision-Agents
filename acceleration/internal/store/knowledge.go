package store

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"time"

	"github.com/uptrace/bun"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// CreateKnowledgeURL stores a new page to keep a knowledge base filled from, and fills in
// its id and timestamps. It starts pending, because nothing has been read yet.
func (s *Store) CreateKnowledgeURL(ctx context.Context, page *KnowledgeURL) error {
	if page.CustomerID == "" {
		return stack.Wrap(errors.New("store: customer id is required"))
	}
	if page.Namespace == "" {
		return stack.Wrap(errors.New("store: a namespace is required, knowledge is never shared"))
	}
	if page.URL == "" {
		return stack.Wrap(errors.New("store: a knowledge url needs a url"))
	}

	page.ID = newID()
	if page.State == "" {
		page.State = KnowledgeURLPending
	}
	now := time.Now().UTC()
	page.CreatedAt = now
	page.UpdatedAt = now
	page.DeletedAt = nil

	if _, err := s.db.NewInsert().Model(page).Exec(ctx); err != nil {
		return stack.Wrap(fmt.Errorf("store: create knowledge url: %w", err))
	}
	return nil
}

// SubscribedKnowledgeURL returns the page a knowledge base is already filled from at this
// url, if it is.
func (s *Store) SubscribedKnowledgeURL(
	ctx context.Context, customerID, namespace, address string,
) (KnowledgeURL, bool, error) {
	if customerID == "" || namespace == "" || address == "" {
		return KnowledgeURL{}, false, stack.Wrap(errors.New("store: a customer, a namespace and a url are required"))
	}

	var page KnowledgeURL
	err := s.db.NewSelect().Model(&page).
		Where("customer_id = ?", customerID).
		Where("namespace = ?", namespace).
		Where("url = ?", address).
		Where("deleted_at IS NULL").
		Limit(1).
		Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return KnowledgeURL{}, false, nil
	}
	if err != nil {
		return KnowledgeURL{}, false, stack.Wrap(fmt.Errorf("store: subscribed knowledge url: %w", err))
	}
	return page, true, nil
}

// SaveKnowledgeURL records what reading a page made of it: where it got to, what it was
// called, how many passages it became and when. The declaration is saved with it, so
// subscribing to a page again can say something different about it.
func (s *Store) SaveKnowledgeURL(ctx context.Context, page *KnowledgeURL) error {
	if page.CustomerID == "" || page.ID == "" {
		return stack.Wrap(errors.New("store: a customer and a knowledge url id are required"))
	}

	page.UpdatedAt = time.Now().UTC()

	result, err := s.db.NewUpdate().Model(page).
		Column("title", "declared_title", "description", "refresh_hours",
			"state", "error", "passages", "last_indexed_at", "updated_at").
		Where("id = ?", page.ID).
		Where("customer_id = ?", page.CustomerID).
		Where("deleted_at IS NULL").
		Exec(ctx)
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: save knowledge url: %w", err))
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: save knowledge url: %w", err))
	}
	if affected == 0 {
		return unknownKnowledgeURL(page.ID)
	}
	return nil
}

// DeleteKnowledgeURL marks a page as gone. Its passages are the caller's to remove, since
// they are not in this database.
func (s *Store) DeleteKnowledgeURL(ctx context.Context, customerID, id string) error {
	if customerID == "" || id == "" {
		return stack.Wrap(errors.New("store: a customer and a knowledge url id are required"))
	}

	result, err := s.db.NewUpdate().Model((*KnowledgeURL)(nil)).
		Set("deleted_at = ?", time.Now().UTC()).
		Where("id = ?", id).
		Where("customer_id = ?", customerID).
		Where("deleted_at IS NULL").
		Exec(ctx)
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: delete knowledge url: %w", err))
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: delete knowledge url: %w", err))
	}
	if affected == 0 {
		return unknownKnowledgeURL(id)
	}
	return nil
}

// KnowledgeURL returns one page a customer subscribes to.
func (s *Store) KnowledgeURL(ctx context.Context, customerID, id string) (KnowledgeURL, error) {
	if customerID == "" || id == "" {
		return KnowledgeURL{}, stack.Wrap(errors.New("store: a customer and a knowledge url id are required"))
	}

	var page KnowledgeURL
	err := s.db.NewSelect().Model(&page).
		Where("id = ?", id).
		Where("customer_id = ?", customerID).
		Where("deleted_at IS NULL").
		Limit(1).
		Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return KnowledgeURL{}, unknownKnowledgeURL(id)
	}
	if err != nil {
		return KnowledgeURL{}, stack.Wrap(fmt.Errorf("store: knowledge url: %w", err))
	}
	return page, nil
}

// DueKnowledgeURLs returns pages, of every customer, whose refresh interval has passed since
// they were last read. A read that failed counts, since every read is saved, so a broken
// page is tried once an interval rather than on every sweep. A page still waiting on its
// first read is not due: that read is already queued.
func (s *Store) DueKnowledgeURLs(ctx context.Context, now time.Time, limit int) ([]KnowledgeURL, error) {
	var pages []KnowledgeURL
	err := s.db.NewSelect().Model(&pages).
		Where("deleted_at IS NULL").
		Where("refresh_hours > 0").
		Where("state <> ?", KnowledgeURLPending).
		Where("updated_at + refresh_hours * interval '1 hour' <= ?", now).
		Order("updated_at ASC").
		Limit(limit).
		Scan(ctx)
	if err != nil {
		return nil, fmt.Errorf("store: due knowledge urls: %w", err)
	}
	return pages, nil
}

// CustomerKnowledgeURLs returns the pages a customer subscribes to, newest first. An empty
// namespace returns every one of them, which is what a caller listing them all wants.
func (s *Store) CustomerKnowledgeURLs(ctx context.Context, customerID, namespace string) ([]KnowledgeURL, error) {
	if customerID == "" {
		return nil, stack.Wrap(errors.New("store: customer id is required"))
	}

	var pages []KnowledgeURL
	query := s.db.NewSelect().Model(&pages).
		Where("customer_id = ?", customerID).
		Where("deleted_at IS NULL")
	if namespace != "" {
		query = query.Where("namespace = ?", namespace)
	}
	if err := query.Order("created_at DESC").Scan(ctx); err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: customer knowledge urls: %w", err))
	}
	return pages, nil
}

// ErrNoKnowledgeURL is what asking for a page that is not there returns, which is what
// tells a subscription removed since from a database that did not answer.
var ErrNoKnowledgeURL = errors.New("store: no such knowledge url")

func unknownKnowledgeURL(id string) error {
	return stack.Wrap(fmt.Errorf("%w: %s", ErrNoKnowledgeURL, id))
}

// SaveKnowledgeDocument records how many passages a document was cut into. A source the
// knowledge base already has is updated in place and keeps its id, since posting it again
// replaced the same passages rather than adding a second copy. The row takes the state and
// error the caller sets.
func (s *Store) SaveKnowledgeDocument(ctx context.Context, document *KnowledgeDocument) error {
	if document.CustomerID == "" {
		return stack.Wrap(errors.New("store: customer id is required"))
	}
	if document.Namespace == "" {
		return stack.Wrap(errors.New("store: a namespace is required, knowledge is never shared"))
	}
	if document.Source == "" {
		return stack.Wrap(errors.New("store: a knowledge document needs a source"))
	}

	document.ID = newID()
	now := time.Now().UTC()
	document.CreatedAt = now
	document.UpdatedAt = now
	document.ObjectKey = ""

	// Text posted over a file still waiting to be read is the newer of the two, so the row
	// lets go of the file and the read that finds it gone leaves the row alone.
	_, err := s.db.NewInsert().Model(document).
		On("CONFLICT (customer_id, namespace, source) DO UPDATE").
		Set("passages = EXCLUDED.passages").
		Set("text = EXCLUDED.text").
		Set("state = EXCLUDED.state").
		Set("error = EXCLUDED.error").
		Set("pages = EXCLUDED.pages").
		Set("content_type = EXCLUDED.content_type").
		Set("bytes = EXCLUDED.bytes").
		Set("object_key = EXCLUDED.object_key").
		Set("updated_at = EXCLUDED.updated_at").
		Returning("id, created_at").
		Exec(ctx)
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: save knowledge document: %w", err))
	}
	return nil
}

// PendKnowledgeFile records a file uploaded to a knowledge base, waiting in the bucket under
// document.ObjectKey to be read. A source the base already has keeps its id and its passages,
// which are still looked up until the file's replace them. The row comes back with the
// passages and pages it still has.
func (s *Store) PendKnowledgeFile(ctx context.Context, document *KnowledgeDocument) error {
	if document.CustomerID == "" || document.Namespace == "" || document.Source == "" {
		return stack.Wrap(errors.New("store: a customer, a namespace and a source are required"))
	}
	if document.ObjectKey == "" {
		return stack.Wrap(errors.New("store: a file waiting to be read needs its object key"))
	}

	document.ID = newID()
	now := time.Now().UTC()
	document.CreatedAt = now
	document.UpdatedAt = now
	document.State = KnowledgeDocumentPending
	document.Error = ""

	_, err := s.db.NewInsert().Model(document).
		On("CONFLICT (customer_id, namespace, source) DO UPDATE").
		Set("state = EXCLUDED.state").
		Set("error = EXCLUDED.error").
		Set("content_type = EXCLUDED.content_type").
		Set("bytes = EXCLUDED.bytes").
		Set("object_key = EXCLUDED.object_key").
		Set("updated_at = EXCLUDED.updated_at").
		Returning("id, created_at, passages, pages").
		Exec(ctx)
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: pend knowledge file: %w", err))
	}
	return nil
}

// CoverKnowledgeFile raises the document's passage count to cover passages about to be
// written for its file, under the same condition as FinishKnowledgeFile, so a write that
// stops part way leaves a row whose delete removes every passage either version wrote.
func (s *Store) CoverKnowledgeFile(ctx context.Context, customerID, id, objectKey string, passages int) (bool, error) {
	result, err := s.db.NewUpdate().Model((*KnowledgeDocument)(nil)).
		Set("passages = GREATEST(passages, ?)", passages).
		Where("id = ?", id).
		Where("customer_id = ?", customerID).
		Where("object_key = ?", objectKey).
		Exec(ctx)
	if err != nil {
		return false, stack.Wrap(fmt.Errorf("store: cover knowledge file: %w", err))
	}
	return updatedOne(result, "cover knowledge file")
}

// FinishKnowledgeFile records a file read into its passages, as long as the row still waits
// for that file. It is false when the document was removed, or uploaded or posted again,
// while the file was read: the row is no longer this read's to write.
func (s *Store) FinishKnowledgeFile(ctx context.Context, document *KnowledgeDocument, objectKey string) (bool, error) {
	document.UpdatedAt = time.Now().UTC()
	document.State = KnowledgeDocumentIndexed
	document.Error = ""
	document.ObjectKey = ""
	result, err := s.db.NewUpdate().Model(document).
		Column("text", "passages", "pages", "state", "error", "object_key", "updated_at").
		Where("id = ?", document.ID).
		Where("customer_id = ?", document.CustomerID).
		Where("object_key = ?", objectKey).
		Exec(ctx)
	if err != nil {
		return false, stack.Wrap(fmt.Errorf("store: finish knowledge file: %w", err))
	}
	return updatedOne(result, "finish knowledge file")
}

// FailKnowledgeFile records why a file could not be read, under the same condition as
// FinishKnowledgeFile. A last failure marks the document failed and lets go of the file;
// an earlier one leaves it pending, to be tried again.
func (s *Store) FailKnowledgeFile(ctx context.Context, customerID, id, objectKey, reason string, last bool) (bool, error) {
	query := s.db.NewUpdate().Model((*KnowledgeDocument)(nil)).
		Set("error = ?", reason).
		Set("updated_at = ?", time.Now().UTC()).
		Where("id = ?", id).
		Where("customer_id = ?", customerID).
		Where("object_key = ?", objectKey)
	if last {
		query = query.Set("state = ?", KnowledgeDocumentFailed).Set("object_key = ''")
	}
	result, err := query.Exec(ctx)
	if err != nil {
		return false, stack.Wrap(fmt.Errorf("store: fail knowledge file: %w", err))
	}
	return updatedOne(result, "fail knowledge file")
}

func updatedOne(result sql.Result, doing string) (bool, error) {
	affected, err := result.RowsAffected()
	if err != nil {
		return false, stack.Wrap(fmt.Errorf("store: %s: %w", doing, err))
	}
	return affected == 1, nil
}

// The accounts a knowledge base is kept in (KnowledgePlace.Account).
const (
	// KnowledgeAccountStream is the deployment's own account at the knowledge vendor.
	KnowledgeAccountStream = "stream"
	// KnowledgeAccountCustomer is the customer's own, on the key they keep for the base's region.
	KnowledgeAccountCustomer = "customer"
)

// KnowledgePlace is where one of a customer's knowledge bases is kept: a region, and an account
// in it.
type KnowledgePlace struct {
	Region  string
	Account string
}

// KnowledgeNamespace is where one of a customer's knowledge bases is kept.
type KnowledgeNamespace struct {
	bun.BaseModel `bun:"table:knowledge_namespaces"`

	CustomerID string    `bun:"customer_id,pk"`
	Namespace  string    `bun:"namespace,pk"`
	Region     string    `bun:"region,notnull"`
	Account    string    `bun:"account,notnull"`
	CreatedAt  time.Time `bun:"created_at,notnull"`
}

// Place is where the base is kept.
func (n KnowledgeNamespace) Place() KnowledgePlace {
	return KnowledgePlace{Region: n.Region, Account: n.Account}
}

// KnowledgePlace is where a customer's knowledge base is kept, and false for one nothing has
// been written to through the router.
func (s *Store) KnowledgePlace(ctx context.Context, customerID, namespace string) (KnowledgePlace, bool, error) {
	var kept KnowledgeNamespace
	err := s.db.NewSelect().Model(&kept).
		Where("customer_id = ?", customerID).
		Where("namespace = ?", namespace).
		Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return KnowledgePlace{}, false, nil
	}
	if err != nil {
		return KnowledgePlace{}, false, stack.Wrap(fmt.Errorf("store: knowledge place: %w", err))
	}
	return kept.Place(), true, nil
}

// ClaimKnowledgePlace records that a customer's knowledge base is kept at place, unless it is
// kept somewhere already, and returns where it is kept: of two first writes at once, in two
// regions or accounts, the one that lands first is where the base stays.
func (s *Store) ClaimKnowledgePlace(ctx context.Context, customerID, namespace string, place KnowledgePlace) (KnowledgePlace, error) {
	if customerID == "" || namespace == "" {
		return KnowledgePlace{}, stack.Wrap(errors.New("store: a customer and a namespace are required"))
	}
	if place.Account == "" {
		place.Account = KnowledgeAccountStream
	}
	kept := KnowledgeNamespace{CustomerID: customerID, Namespace: namespace, Region: place.Region, Account: place.Account,
		CreatedAt: time.Now().UTC()}
	// A no-op update, so RETURNING answers the row that was already there.
	_, err := s.db.NewInsert().Model(&kept).
		On("CONFLICT (customer_id, namespace) DO UPDATE").
		Set("customer_id = EXCLUDED.customer_id").
		Returning("region, account").
		Exec(ctx)
	if err != nil {
		return KnowledgePlace{}, stack.Wrap(fmt.Errorf("store: claim knowledge place: %w", err))
	}
	return kept.Place(), nil
}

// ForgetKnowledgePlace forgets where a customer's knowledge base is kept, once it is dropped.
func (s *Store) ForgetKnowledgePlace(ctx context.Context, customerID, namespace string) error {
	_, err := s.db.NewDelete().Model((*KnowledgeNamespace)(nil)).
		Where("customer_id = ?", customerID).
		Where("namespace = ?", namespace).
		Exec(ctx)
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: forget knowledge place: %w", err))
	}
	return nil
}

// KnowledgeInCustomerAccounts is every knowledge base kept in a customer's own account, by
// customer and then namespace: what the hourly storage job reads on each one's key, since the
// deployment's account does not list them.
func (s *Store) KnowledgeInCustomerAccounts(ctx context.Context) ([]KnowledgeNamespace, error) {
	kept := []KnowledgeNamespace{}
	err := s.db.NewSelect().Model(&kept).
		Where("account = ?", KnowledgeAccountCustomer).
		Order("customer_id", "namespace").
		Scan(ctx)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: knowledge in customer accounts: %w", err))
	}
	return kept, nil
}

// KnowledgeDocument returns one document a customer filled a knowledge base with.
func (s *Store) KnowledgeDocument(ctx context.Context, customerID, id string) (KnowledgeDocument, error) {
	if customerID == "" || id == "" {
		return KnowledgeDocument{}, stack.Wrap(errors.New("store: a customer and a knowledge document id are required"))
	}

	var document KnowledgeDocument
	err := s.db.NewSelect().Model(&document).
		Where("id = ?", id).
		Where("customer_id = ?", customerID).
		Limit(1).
		Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return KnowledgeDocument{}, unknownKnowledgeDocument(id)
	}
	if err != nil {
		return KnowledgeDocument{}, stack.Wrap(fmt.Errorf("store: knowledge document: %w", err))
	}
	return document, nil
}

// CustomerKnowledgeDocuments returns the documents a customer filled its knowledge bases
// with, most recently written first. An empty namespace returns every one of them. Their
// text is left out: KnowledgeDocument reads it.
func (s *Store) CustomerKnowledgeDocuments(
	ctx context.Context, customerID, namespace string,
) ([]KnowledgeDocument, error) {
	return s.customerKnowledgeDocuments(ctx, customerID, namespace, false)
}

// CustomerKnowledgeDocumentsWithText returns the same documents with the text they were
// posted as. It is for the callers that compare the text rather than list it: what a sync
// would write over, and what a change records. A listing reads the lighter one.
func (s *Store) CustomerKnowledgeDocumentsWithText(
	ctx context.Context, customerID, namespace string,
) ([]KnowledgeDocument, error) {
	return s.customerKnowledgeDocuments(ctx, customerID, namespace, true)
}

func (s *Store) customerKnowledgeDocuments(
	ctx context.Context, customerID, namespace string, withText bool,
) ([]KnowledgeDocument, error) {
	if customerID == "" {
		return nil, stack.Wrap(errors.New("store: customer id is required"))
	}

	var documents []KnowledgeDocument
	query := s.db.NewSelect().Model(&documents).Where("customer_id = ?", customerID)
	if !withText {
		query = query.ExcludeColumn("text")
	}
	if namespace != "" {
		query = query.Where("namespace = ?", namespace)
	}
	if err := query.Order("updated_at DESC").Scan(ctx); err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: customer knowledge documents: %w", err))
	}
	return documents, nil
}

// DeleteKnowledgeDocument forgets a document. Its passages are the caller's to remove,
// since they are not in this database.
func (s *Store) DeleteKnowledgeDocument(ctx context.Context, customerID, id string) error {
	if customerID == "" || id == "" {
		return stack.Wrap(errors.New("store: a customer and a knowledge document id are required"))
	}

	result, err := s.db.NewDelete().Model((*KnowledgeDocument)(nil)).
		Where("id = ?", id).
		Where("customer_id = ?", customerID).
		Exec(ctx)
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: delete knowledge document: %w", err))
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: delete knowledge document: %w", err))
	}
	if affected == 0 {
		return unknownKnowledgeDocument(id)
	}
	return nil
}

// DeleteKnowledgeBase forgets every document and marks every page of one knowledge base as
// gone, and returns how many of each there were. Their passages are the caller's to remove,
// since they are not in this database.
func (s *Store) DeleteKnowledgeBase(ctx context.Context, customerID, namespace string) (documents, pages int64, err error) {
	if customerID == "" || namespace == "" {
		return 0, 0, stack.Wrap(errors.New("store: a customer and a namespace are required"))
	}

	err = s.db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		deleted, err := tx.NewDelete().Model((*KnowledgeDocument)(nil)).
			Where("customer_id = ?", customerID).
			Where("namespace = ?", namespace).
			Exec(ctx)
		if err != nil {
			return err
		}
		if documents, err = deleted.RowsAffected(); err != nil {
			return err
		}
		marked, err := tx.NewUpdate().Model((*KnowledgeURL)(nil)).
			Set("deleted_at = ?", time.Now().UTC()).
			Where("customer_id = ?", customerID).
			Where("namespace = ?", namespace).
			Where("deleted_at IS NULL").
			Exec(ctx)
		if err != nil {
			return err
		}
		pages, err = marked.RowsAffected()
		return err
	})
	if err != nil {
		return 0, 0, stack.Wrap(fmt.Errorf("store: delete knowledge base: %w", err))
	}
	return documents, pages, nil
}

// ErrNoKnowledgeDocument is what asking for a document that is not there returns, which is
// what tells a document removed while its file was read from a database that did not answer.
var ErrNoKnowledgeDocument = errors.New("store: no such knowledge document")

func unknownKnowledgeDocument(id string) error {
	return stack.Wrap(fmt.Errorf("%w: %s", ErrNoKnowledgeDocument, id))
}
