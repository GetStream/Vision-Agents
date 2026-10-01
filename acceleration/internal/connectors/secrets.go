package connectors

import (
	"encoding/json"
	"errors"
	"fmt"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/mcp"
)

// Credentials are encrypted as one versioned bundle so refresh-token rotation cannot leave
// an access token paired with an older refresh token.
type Credentials struct {
	AuthType          string `json:"auth_type"`
	AuthHeader        string `json:"-"`
	AccessToken       string `json:"access_token"`
	RefreshToken      string `json:"refresh_token,omitempty"`
	APIKey            string `json:"api_key,omitempty"`
	OAuthClientID     string `json:"oauth_client_id"`
	OAuthClientSecret string `json:"oauth_client_secret,omitempty"`
	ClientAuthMethod  string `json:"client_auth_method"`
	OAuthIssuer       string `json:"oauth_issuer,omitempty"`
	TokenEndpoint     string `json:"token_endpoint"`
	RefreshEndpoint   string `json:"refresh_endpoint,omitempty"`
	Resource          string `json:"resource,omitempty"`
}

// AuthorizationAttempt is the sensitive state needed to finish one OAuth redirect.
type AuthorizationAttempt struct {
	ConnectionID   string                   `json:"connection_id"`
	Revision       int                      `json:"revision"`
	ConnectorID    string                   `json:"connector_id"`
	BrowserBinding string                   `json:"browser_binding"`
	Pending        mcp.PendingAuthorization `json:"pending"`
}

// SealCredentials encrypts a connection's complete OAuth grant, bound to its tenant, row,
// and credential revision.
func SealCredentials(sealer *auth.Sealer, customerID, connectionID string, revision int, credentials Credentials) ([]byte, error) {
	if credentials.AuthType == "" {
		credentials.AuthType = AuthOAuth2
	}
	if sealer == nil || customerID == "" || connectionID == "" || revision < 1 {
		return nil, errors.New("connectors: sealer, tenant, connection id and revision are required")
	}
	switch credentials.AuthType {
	case AuthOAuth2, AuthBearer:
		if credentials.AccessToken == "" {
			return nil, errors.New("connectors: an access token is required")
		}
	case AuthAPIKey:
		if credentials.APIKey == "" {
			return nil, errors.New("connectors: an API key is required")
		}
	default:
		return nil, fmt.Errorf("connectors: unsupported credential type %q", credentials.AuthType)
	}
	raw, err := json.Marshal(credentials)
	if err != nil {
		return nil, fmt.Errorf("connectors: encode credentials: %w", err)
	}
	return sealer.SealWithAAD(string(raw), credentialAAD(customerID, connectionID, revision))
}

// OpenCredentials decrypts a connection's grant only for its owning row.
func OpenCredentials(sealer *auth.Sealer, customerID, connectionID string, revision, kekVersion int, sealed []byte) (Credentials, error) {
	if sealer == nil || customerID == "" || connectionID == "" || revision < 1 || kekVersion < 1 || len(sealed) == 0 {
		return Credentials{}, errors.New("connectors: encrypted credentials are unavailable")
	}
	plain, err := sealer.OpenWithAADVersion(sealed, credentialAAD(customerID, connectionID, revision), kekVersion)
	if err != nil {
		return Credentials{}, fmt.Errorf("connectors: open credentials: %w", err)
	}
	var credentials Credentials
	if err := json.Unmarshal([]byte(plain), &credentials); err != nil {
		return Credentials{}, fmt.Errorf("connectors: decode credentials: %w", err)
	}
	if credentials.AuthType == "" {
		credentials.AuthType = AuthOAuth2
	}
	return credentials, nil
}

// SealAuthorizationAttempt protects state, PKCE verifier and customer OAuth client secret.
func SealAuthorizationAttempt(sealer *auth.Sealer, attemptID string, attempt AuthorizationAttempt) ([]byte, error) {
	if sealer == nil || attemptID == "" || attempt.ConnectionID == "" || attempt.Pending.State == "" {
		return nil, errors.New("connectors: sealer and authorization attempt are required")
	}
	raw, err := json.Marshal(attempt)
	if err != nil {
		return nil, fmt.Errorf("connectors: encode authorization attempt: %w", err)
	}
	return sealer.SealWithAAD(string(raw), authorizationAAD(attemptID))
}

// OpenAuthorizationAttempt decrypts an attempt only when its row identity matches.
func OpenAuthorizationAttempt(sealer *auth.Sealer, attemptID string, kekVersion int, sealed []byte) (AuthorizationAttempt, error) {
	if sealer == nil || attemptID == "" || kekVersion < 1 || len(sealed) == 0 {
		return AuthorizationAttempt{}, errors.New("connectors: encrypted authorization attempt is unavailable")
	}
	plain, err := sealer.OpenWithAADVersion(sealed, authorizationAAD(attemptID), kekVersion)
	if err != nil {
		return AuthorizationAttempt{}, fmt.Errorf("connectors: open authorization attempt: %w", err)
	}
	var attempt AuthorizationAttempt
	if err := json.Unmarshal([]byte(plain), &attempt); err != nil {
		return AuthorizationAttempt{}, fmt.Errorf("connectors: decode authorization attempt: %w", err)
	}
	return attempt, nil
}

func credentialAAD(customerID, id string, revision int) []byte {
	return []byte(fmt.Sprintf("accelerate:connector-credentials:v1:%s:%s:%d", customerID, id, revision))
}

func authorizationAAD(id string) []byte {
	return []byte("accelerate:connector-authorization:v1:" + id)
}
