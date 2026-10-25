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
	s.forget(ctx, policyKeys(scope, id)...)
	return nil
}

// DeletePolicy removes a scope's document, drops what every replica had cached of it, and
// reports whether there was one.
func (s *Store) DeletePolicy(ctx context.Context, scope store.PolicyScope, id string) (bool, error) {
	deleted, err := s.db.DeletePolicy(ctx, scope, id)
	if err != nil {
		return false, err
	}
	s.forget(ctx, policyKeys(scope, id)...)
	return deleted, nil
}

// HasUserPolicies reports whether any user of an app has a policy of their own, so a user
// of an app with none is answered without a lookup of their own.
func (s *Store) HasUserPolicies(ctx context.Context, appID string) (bool, error) {
	return read(ctx, s, key("user-policies", appID), func(ctx context.Context) (bool, error) {
		return s.db.HasUserPolicies(ctx, appID)
	})
}

// policyKeys are the cached values a write of one scope's document changes.
func policyKeys(scope store.PolicyScope, id string) []string {
	keys := []string{key("policy", string(scope), id)}
	if scope == store.ScopeUser {
		if appID, _, ok := store.SplitUserScopeID(id); ok {
			keys = append(keys, key("user-policies", appID))
		}
	}
	return keys
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
