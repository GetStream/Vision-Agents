package store

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"time"
)

// CreateKnowledgeURL stores a new page to keep a knowledge base filled from, and fills in
// its id and timestamps. It starts pending, because nothing has been read yet.
func (s *Store) CreateKnowledgeURL(ctx context.Context, page *KnowledgeURL) error {
	if page.CustomerID == "" {
		return errors.New("store: customer id is required")
	}
	if page.Namespace == "" {
		return errors.New("store: a namespace is required, knowledge is never shared")
	}
	if page.URL == "" {
		return errors.New("store: a knowledge url needs a url")
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
		return fmt.Errorf("store: create knowledge url: %w", err)
	}
	return nil
}

// SaveKnowledgeURL records what reading a page made of it: where it got to, what it was
// called, how many passages it became and when.
func (s *Store) SaveKnowledgeURL(ctx context.Context, page *KnowledgeURL) error {
	if page.CustomerID == "" || page.ID == "" {
		return errors.New("store: a customer and a knowledge url id are required")
	}

	page.UpdatedAt = time.Now().UTC()

	result, err := s.db.NewUpdate().Model(page).
		Column("title", "state", "error", "passages", "last_indexed_at", "updated_at").
		Where("id = ?", page.ID).
		Where("customer_id = ?", page.CustomerID).
		Where("deleted_at IS NULL").
		Exec(ctx)
	if err != nil {
		return fmt.Errorf("store: save knowledge url: %w", err)
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("store: save knowledge url: %w", err)
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
		return errors.New("store: a customer and a knowledge url id are required")
	}

	result, err := s.db.NewUpdate().Model((*KnowledgeURL)(nil)).
		Set("deleted_at = ?", time.Now().UTC()).
		Where("id = ?", id).
		Where("customer_id = ?", customerID).
		Where("deleted_at IS NULL").
		Exec(ctx)
	if err != nil {
		return fmt.Errorf("store: delete knowledge url: %w", err)
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("store: delete knowledge url: %w", err)
	}
	if affected == 0 {
		return unknownKnowledgeURL(id)
	}
	return nil
}

// KnowledgeURL returns one page a customer subscribes to.
func (s *Store) KnowledgeURL(ctx context.Context, customerID, id string) (KnowledgeURL, error) {
	if customerID == "" || id == "" {
		return KnowledgeURL{}, errors.New("store: a customer and a knowledge url id are required")
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
		return KnowledgeURL{}, fmt.Errorf("store: knowledge url: %w", err)
	}
	return page, nil
}

// CustomerKnowledgeURLs returns the pages a customer subscribes to, newest first. An empty
// namespace returns every one of them, which is what a caller listing them all wants.
func (s *Store) CustomerKnowledgeURLs(ctx context.Context, customerID, namespace string) ([]KnowledgeURL, error) {
	if customerID == "" {
		return nil, errors.New("store: customer id is required")
	}

	var pages []KnowledgeURL
	query := s.db.NewSelect().Model(&pages).
		Where("customer_id = ?", customerID).
		Where("deleted_at IS NULL")
	if namespace != "" {
		query = query.Where("namespace = ?", namespace)
	}
	if err := query.Order("created_at DESC").Scan(ctx); err != nil {
		return nil, fmt.Errorf("store: customer knowledge urls: %w", err)
	}
	return pages, nil
}

func unknownKnowledgeURL(id string) error {
	return fmt.Errorf("store: there is no knowledge url %s", id)
}
