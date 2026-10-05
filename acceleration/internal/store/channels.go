package store

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"time"
)

// ErrUnknownChannelAccount is a webhook token nobody connected, or one whose account has
// since been disconnected.
var ErrUnknownChannelAccount = errors.New("store: there is no such channel account")

// ChannelAccounts returns the lines one app has connected, oldest first.
func (s *Store) ChannelAccounts(ctx context.Context, customerID string) ([]ChannelAccount, error) {
	if customerID == "" {
		return nil, errors.New("store: a customer is required")
	}

	var accounts []ChannelAccount
	err := s.db.NewSelect().Model(&accounts).
		Where("customer_id = ?", customerID).
		Where("deleted_at IS NULL").
		Order("created_at").
		Scan(ctx)
	if err != nil {
		return nil, fmt.Errorf("store: channel accounts: %w", err)
	}
	return accounts, nil
}

// ChannelAccount is the line one app connected on a channel for a number.
func (s *Store) ChannelAccount(ctx context.Context, customerID, kind, e164 string) (ChannelAccount, error) {
	if customerID == "" || kind == "" || e164 == "" {
		return ChannelAccount{}, ErrUnknownChannelAccount
	}

	var account ChannelAccount
	err := s.db.NewSelect().Model(&account).
		Where("customer_id = ?", customerID).
		Where("kind = ?", kind).
		Where("e164 = ?", e164).
		Where("deleted_at IS NULL").
		Limit(1).
		Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return ChannelAccount{}, ErrUnknownChannelAccount
	}
	if err != nil {
		return ChannelAccount{}, fmt.Errorf("store: channel account: %w", err)
	}
	return account, nil
}

// ChannelAccountByToken finds the line a delivery is addressed to.
func (s *Store) ChannelAccountByToken(ctx context.Context, token string) (ChannelAccount, error) {
	if token == "" {
		return ChannelAccount{}, ErrUnknownChannelAccount
	}

	var account ChannelAccount
	err := s.db.NewSelect().Model(&account).
		Where("token = ?", token).
		Where("deleted_at IS NULL").
		Limit(1).
		Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return ChannelAccount{}, ErrUnknownChannelAccount
	}
	if err != nil {
		return ChannelAccount{}, fmt.Errorf("store: channel account by token: %w", err)
	}
	return account, nil
}

// SaveChannelAccount connects a line, or replaces the credentials of one already connected.
// The token survives a reconnection, so the URL the provider was given goes on working.
func (s *Store) SaveChannelAccount(ctx context.Context, account *ChannelAccount) error {
	if account.CustomerID == "" || account.Kind == "" || account.E164 == "" || account.Token == "" {
		return errors.New("store: a customer, a kind, a number and a token are required")
	}
	now := time.Now().UTC()
	account.UpdatedAt = now

	existing, err := s.ChannelAccount(ctx, account.CustomerID, account.Kind, account.E164)
	switch {
	case errors.Is(err, ErrUnknownChannelAccount):
		account.ID = newID()
		account.CreatedAt = now
		if _, err := s.db.NewInsert().Model(account).Exec(ctx); err != nil {
			return fmt.Errorf("store: connect channel account: %w", err)
		}
		return nil
	case err != nil:
		return err
	}

	account.ID = existing.ID
	account.Token = existing.Token
	account.CreatedAt = existing.CreatedAt
	_, err = s.db.NewUpdate().Model(account).
		Column("account_id", "secrets_sealed", "kek_version", "updated_at").
		Where("id = ?", account.ID).
		Where("deleted_at IS NULL").
		Exec(ctx)
	if err != nil {
		return fmt.Errorf("store: save channel account: %w", err)
	}
	return nil
}

// DeleteChannelAccount disconnects a line, so deliveries to it are refused. The credentials
// go with it: a line nobody answers on has no use for them.
func (s *Store) DeleteChannelAccount(ctx context.Context, customerID, kind, e164 string) error {
	result, err := s.db.NewUpdate().Model((*ChannelAccount)(nil)).
		Set("deleted_at = ?", time.Now().UTC()).
		Set("secrets_sealed = NULL").
		Where("customer_id = ?", customerID).
		Where("kind = ?", kind).
		Where("e164 = ?", e164).
		Where("deleted_at IS NULL").
		Exec(ctx)
	if err != nil {
		return fmt.Errorf("store: disconnect channel account: %w", err)
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("store: disconnect channel account: %w", err)
	}
	if affected == 0 {
		return ErrUnknownChannelAccount
	}
	return nil
}

// ConfigOnChannel is the agent reachable on a number, and false when none is.
//
// It answers both questions there are about a line: which agent a message to it is for, and
// whether another agent has already claimed it. Two agents on one number would both answer
// every message, so the second one to ask for it is refused.
func (s *Store) ConfigOnChannel(ctx context.Context, customerID, kind, number string) (AgentConfig, bool, error) {
	if customerID == "" || kind == "" || number == "" {
		return AgentConfig{}, false, nil
	}

	var config AgentConfig
	err := s.db.NewSelect().Model(&config).
		Where("customer_id = ?", customerID).
		Where("channels -> ? ->> 'number' = ?", kind, number).
		Where("deleted_at IS NULL").
		Limit(1).
		Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return AgentConfig{}, false, nil
	}
	if err != nil {
		return AgentConfig{}, false, fmt.Errorf("store: config on channel: %w", err)
	}
	return config, true, nil
}

// ErrUnknownChannelIdentity is a number nobody has tied to an end user yet.
var ErrUnknownChannelIdentity = errors.New("store: this number belongs to nobody yet")

// ChannelIdentity is who a number is to one agent.
func (s *Store) ChannelIdentity(ctx context.Context, customerID, configID, kind, address string) (ChannelIdentity, error) {
	if customerID == "" || configID == "" || kind == "" || address == "" {
		return ChannelIdentity{}, ErrUnknownChannelIdentity
	}

	var identity ChannelIdentity
	err := s.db.NewSelect().Model(&identity).
		Where("customer_id = ?", customerID).
		Where("config_id = ?", configID).
		Where("kind = ?", kind).
		Where("address = ?", address).
		Limit(1).
		Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return ChannelIdentity{}, ErrUnknownChannelIdentity
	}
	if err != nil {
		return ChannelIdentity{}, fmt.Errorf("store: channel identity: %w", err)
	}
	return identity, nil
}

// SaveChannelIdentity records who a number is, or moves its conversation on.
func (s *Store) SaveChannelIdentity(ctx context.Context, identity *ChannelIdentity) error {
	if identity.CustomerID == "" || identity.ConfigID == "" || identity.Kind == "" ||
		identity.Address == "" || identity.UserID == "" {
		return errors.New("store: a customer, an agent, a kind, an address and a user are required")
	}
	now := time.Now().UTC()
	identity.UpdatedAt = now
	if identity.ID == "" {
		identity.ID = newID()
		identity.CreatedAt = now
	}

	_, err := s.db.NewInsert().Model(identity).
		On("CONFLICT (customer_id, config_id, kind, address) DO UPDATE").
		Set("user_id = EXCLUDED.user_id").
		Set("conversation_id = EXCLUDED.conversation_id").
		Set("updated_at = EXCLUDED.updated_at").
		Exec(ctx)
	if err != nil {
		return fmt.Errorf("store: save channel identity: %w", err)
	}
	return nil
}

// SaveChannelLink stores a code somebody may text to claim their number.
func (s *Store) SaveChannelLink(ctx context.Context, link *ChannelLink) error {
	if link.Code == "" || link.CustomerID == "" || link.ConfigID == "" || link.UserID == "" {
		return errors.New("store: a code, a customer, an agent and a user are required")
	}
	link.CreatedAt = time.Now().UTC()
	if _, err := s.db.NewInsert().Model(link).Exec(ctx); err != nil {
		return fmt.Errorf("store: save channel link: %w", err)
	}
	return nil
}

// ClaimChannelLink spends a code. A code already spent or past its time is no code at all,
// so a number cannot be claimed twice over by somebody who saw the message go by.
func (s *Store) ClaimChannelLink(ctx context.Context, customerID, code string) (ChannelLink, error) {
	if customerID == "" || code == "" {
		return ChannelLink{}, ErrUnknownChannelLink
	}

	now := time.Now().UTC()
	var link ChannelLink
	result, err := s.db.NewUpdate().Model(&link).
		Set("used_at = ?", now).
		Where("code = ?", code).
		Where("customer_id = ?", customerID).
		Where("used_at IS NULL").
		Where("expires_at > ?", now).
		Returning("*").
		Exec(ctx)
	if err != nil {
		return ChannelLink{}, fmt.Errorf("store: claim channel link: %w", err)
	}
	// An update that matched nothing is not an error to the driver, so the count is what
	// says the code was already spent, has run out, or was never anybody's.
	affected, err := result.RowsAffected()
	if err != nil {
		return ChannelLink{}, fmt.Errorf("store: claim channel link: %w", err)
	}
	if affected == 0 {
		return ChannelLink{}, ErrUnknownChannelLink
	}
	return link, nil
}

// ErrUnknownChannelLink is a code nobody was given, one already spent, or one past its time.
var ErrUnknownChannelLink = errors.New("store: there is no such code")

// ClaimChannelMessage records a message as delivered. False is one already taken, which a
// provider retrying a slow delivery sends.
func (s *Store) ClaimChannelMessage(ctx context.Context, accountID, messageID string) (bool, error) {
	result, err := s.db.NewRaw(
		"INSERT INTO channel_deliveries (account_id, message_id, received_at) "+
			"VALUES (?, ?, ?) ON CONFLICT DO NOTHING",
		accountID, messageID, time.Now().UTC()).Exec(ctx)
	if err != nil {
		return false, fmt.Errorf("store: claim channel message: %w", err)
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return false, fmt.Errorf("store: claim channel message: %w", err)
	}
	return affected == 1, nil
}
