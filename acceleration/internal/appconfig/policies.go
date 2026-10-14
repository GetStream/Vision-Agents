package appconfig

import (
	"context"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// Policy returns a scope's document.
func (s *Store) Policy(ctx context.Context, scope store.PolicyScope, id string) (store.PolicyDocument, error) {
	return read(ctx, s, key("policy", string(scope), id), func(ctx context.Context) (store.PolicyDocument, error) {
		return s.db.Policy(ctx, scope, id)
	})
}

// SavePolicy replaces a scope's document and drops what every replica had cached of it.
func (s *Store) SavePolicy(ctx context.Context, scope store.PolicyScope, id string, document store.PolicyDocument) error {
	if err := s.db.SavePolicy(ctx, scope, id, document); err != nil {
		return err
	}
	s.forget(ctx, key("policy", string(scope), id))
	return nil
}

// ChangeAppQuota replaces an app's daily limit per end user, records the change, and drops
// what every replica had cached of the app's policy.
func (s *Store) ChangeAppQuota(ctx context.Context, customerID, changedBy, reason string, quota *store.Quota) (store.AppQuotaChange, error) {
	change, err := s.db.ChangeAppQuota(ctx, customerID, changedBy, reason, quota)
	if err != nil {
		return store.AppQuotaChange{}, err
	}
	s.forget(ctx, key("policy", string(store.ScopeApp), customerID))
	return change, nil
}

// OrganizationOf returns the organization an app was last seen under.
func (s *Store) OrganizationOf(ctx context.Context, appID string) (string, error) {
	return read(ctx, s, key("organization-of", appID), func(ctx context.Context) (string, error) {
		return s.db.OrganizationOf(ctx, appID)
	})
}

// JoinOrganization records that an app was seen under an organization.
func (s *Store) JoinOrganization(ctx context.Context, appID, organizationID string) error {
	if err := s.db.JoinOrganization(ctx, appID, organizationID); err != nil {
		return err
	}
	s.forget(ctx, key("organization-of", appID))
	return nil
}
