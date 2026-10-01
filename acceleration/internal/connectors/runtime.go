package connectors

import (
	"context"
	"errors"
	"net/http"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/mcp"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

const (
	AuthNone   = "none"
	AuthBearer = "bearer"
	AuthAPIKey = "api_key"
	AuthOAuth2 = "oauth2"
)

var ErrReauthorizationRequired = errors.New("connectors: this account needs to be reauthorized")
var ErrCredentialTemporarilyUnavailable = errors.New("connectors: credential could not be refreshed before expiry")

// ResolveCredentials checks connection state on every outbound MCP request and refreshes
// expiring OAuth grants under a database advisory lock safe across router replicas.
func ResolveCredentials(
	ctx context.Context,
	db *store.Store,
	sealer *auth.Sealer,
	customerID, connectionID string,
	authenticator *mcp.OAuthClient,
) (Credentials, error) {
	if db == nil {
		return Credentials{}, errors.New("connectors: connector storage is unavailable")
	}
	var resolved Credentials
	var grantErr error
	err := db.WithLockedConnectorConnection(ctx, customerID, connectionID, func(connection *store.ConnectorConnection, checkpoint func() error) (bool, error) {
		if connection.Status != store.ConnectorConnected {
			grantErr = ErrReauthorizationRequired
			return false, nil
		}
		authType := connection.AuthType
		if authType == "" {
			authType = AuthOAuth2
		}
		if authType == AuthNone {
			resolved = Credentials{AuthType: AuthNone}
			return false, nil
		}
		if sealer == nil {
			grantErr = errors.New("connectors: encrypted credential storage is unavailable")
			return false, nil
		}
		credentials, err := OpenCredentials(sealer, customerID, connection.ID, connection.Revision,
			connection.CredentialKEKVersion, connection.CredentialSealed)
		if err != nil || credentials.AuthType != authType {
			connection.Status = store.ConnectorNeedsReauth
			connection.LastError = "Encrypted credentials could not be opened; reconnect required"
			grantErr = ErrReauthorizationRequired
			return true, nil
		}
		keyRotated := false
		if connection.CredentialKEKVersion != sealer.CurrentVersion() {
			sealed, err := SealCredentials(sealer, customerID, connection.ID, connection.Revision, credentials)
			if err != nil {
				return false, err
			}
			connection.CredentialSealed = sealed
			connection.CredentialKEKVersion = sealer.CurrentVersion()
			keyRotated = true
		}
		if authType != AuthOAuth2 {
			credentials.AuthHeader = connection.AuthHeader
			resolved = credentials
			return keyRotated, nil
		}
		if connection.ExpiresAt != nil && time.Until(*connection.ExpiresAt) < time.Minute {
			if credentials.RefreshToken == "" {
				connection.Status = store.ConnectorNeedsReauth
				connection.LastError = "OAuth access expired without a refresh token; reconnect required"
				grantErr = ErrReauthorizationRequired
				return true, nil
			}
			if authenticator == nil {
				authenticator = &mcp.OAuthClient{}
			}
			pending := mcp.PendingAuthorization{
				ClientID:         credentials.OAuthClientID,
				ClientSecret:     credentials.OAuthClientSecret,
				ClientAuthMethod: credentials.ClientAuthMethod,
				Issuer:           credentials.OAuthIssuer,
				TokenEndpoint:    credentials.TokenEndpoint,
				RefreshEndpoint:  credentials.RefreshEndpoint,
				Resource:         credentials.Resource,
			}
			connection.Status = store.ConnectorNeedsReauth
			connection.LastError = "OAuth refresh did not finish durably; reconnect required"
			if err := checkpoint(); err != nil {
				return false, err
			}
			refreshed, err := authenticator.RefreshWithClient(ctx, pending, credentials.RefreshToken)
			if err != nil {
				if errors.Is(err, mcp.ErrOAuthInvalidGrant) {
					connection.Status = store.ConnectorNeedsReauth
					connection.LastError = "OAuth grant was rejected; reconnect required"
					grantErr = ErrReauthorizationRequired
					return true, nil
				}
				if errors.Is(err, mcp.ErrOAuthRefreshUncertain) {
					grantErr = ErrReauthorizationRequired
					return false, nil
				}
				connection.Status = store.ConnectorConnected
				connection.LastError = "OAuth refresh is temporarily unavailable"
				if connection.ExpiresAt == nil || time.Now().Before(*connection.ExpiresAt) {
					resolved = credentials
					return true, nil
				}
				grantErr = ErrCredentialTemporarilyUnavailable
				return true, nil
			}
			credentials.AccessToken = refreshed.AccessToken
			credentials.RefreshToken = refreshed.RefreshToken
			nextRevision := connection.Revision + 1
			sealed, err := SealCredentials(sealer, customerID, connection.ID, nextRevision, credentials)
			if err != nil {
				return false, err
			}
			connection.CredentialSealed = sealed
			connection.CredentialKEKVersion = sealer.CurrentVersion()
			connection.ExpiresAt = refreshed.ExpiresAt
			connection.Status = store.ConnectorConnected
			connection.LastError = ""
			connection.Revision = nextRevision
			returnValue := credentials
			resolved = returnValue
			return true, nil
		}
		resolved = credentials
		return keyRotated, nil
	})
	if err != nil {
		return Credentials{}, err
	}
	if grantErr != nil {
		return Credentials{}, grantErr
	}
	if resolved.AuthType == "" {
		return Credentials{}, ErrReauthorizationRequired
	}
	return resolved, nil
}

// ResolveAccessToken is the OAuth/bearer convenience wrapper for callers that only need
// an Authorization bearer value.
func ResolveAccessToken(
	ctx context.Context,
	db *store.Store,
	sealer *auth.Sealer,
	customerID, connectionID string,
	authenticator *mcp.OAuthClient,
) (string, error) {
	credentials, err := ResolveCredentials(ctx, db, sealer, customerID, connectionID, authenticator)
	if err != nil {
		return "", err
	}
	if credentials.AuthType != AuthOAuth2 && credentials.AuthType != AuthBearer {
		return "", errors.New("connectors: connection does not use bearer authorization")
	}
	return credentials.AccessToken, nil
}

// AuthorizeRequest injects a connection credential only at the bound MCP endpoint.
func AuthorizeRequest(
	ctx context.Context,
	db *store.Store,
	sealer *auth.Sealer,
	customerID, connectionID string,
	authenticator *mcp.OAuthClient,
	request *http.Request,
) error {
	credentials, err := ResolveCredentials(ctx, db, sealer, customerID, connectionID, authenticator)
	if err != nil {
		return err
	}
	switch credentials.AuthType {
	case AuthNone:
		return nil
	case AuthOAuth2, AuthBearer:
		if credentials.AccessToken == "" {
			return ErrReauthorizationRequired
		}
		request.Header.Set("Authorization", "Bearer "+credentials.AccessToken)
		return nil
	case AuthAPIKey:
		if credentials.AuthHeader == "" || credentials.APIKey == "" {
			return ErrReauthorizationRequired
		}
		request.Header.Set(credentials.AuthHeader, credentials.APIKey)
		return nil
	default:
		return errors.New("connectors: unsupported authorization mode")
	}
}
