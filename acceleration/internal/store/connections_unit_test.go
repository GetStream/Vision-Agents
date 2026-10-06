package store

import (
	"context"
	"net/http"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
)

// testScheme is a scheme that is registered under a name and does nothing else: the store
// asks only whether a name is registered. No real scheme is registered on accelerate yet.
type testScheme string

func (n testScheme) Name() string { return string(n) }

func (testScheme) Begin(context.Context, core.BeginInput) (core.BeginOutput, error) {
	return core.BeginOutput{Done: true}, nil
}

func (testScheme) Complete(context.Context, core.CompleteInput) (core.StoredCredentials, core.AccountInfo, error) {
	return core.StoredCredentials{}, core.AccountInfo{}, nil
}

func (testScheme) Retrieve(_ context.Context, stored core.StoredCredentials, _ core.ResolvedManifest, _ core.RetrieveOptions) (core.AccessCredential, core.StoredCredentials, error) {
	return core.AccessCredential{}, stored, nil
}

func (testScheme) Wrap(base http.RoundTripper, _ core.AccessCredential) http.RoundTripper {
	return base
}

func (testScheme) Classify(*http.Response, []byte, error) core.Outcome { return core.Outcome{} }

func (testScheme) Revoke(context.Context, core.StoredCredentials, core.ResolvedManifest) error {
	return nil
}

// testSchemes is the registry the tests create connections against. acme lists test_key
// and test_mtls; test_unlisted is registered but listed by no test manifest.
var testSchemes = core.Registry{Schemes: map[string]core.Scheme{
	"test_key":      testScheme("test_key"),
	"test_mtls":     testScheme("test_mtls"),
	"test_unlisted": testScheme("test_unlisted"),
}}

// appConnection is an app-owned connection to the acme test built-in at revision 1.
func appConnection() *ConnectorConnection {
	return &ConnectorConnection{
		CustomerID:         "acme-app",
		ConnectorID:        "acme",
		DefinitionRevision: 1,
		OwnerType:          OwnerApp,
		AuthScheme:         "test_key",
	}
}

func TestAConnectionWithAnUnregisteredAuthSchemeIsRefused(t *testing.T) {
	connection := appConnection()
	connection.AuthScheme = "oauth2_code"

	err := (&Store{}).CreateConnectorConnection(context.Background(), testSchemes, connection)

	require.ErrorIs(t, err, ErrUnregisteredScheme)
	require.ErrorContains(t, err, `"oauth2_code"`)
}

func TestAConnectionWithAnUnregisteredTLSSchemeIsRefused(t *testing.T) {
	connection := appConnection()
	connection.TLSScheme = "mtls"

	err := (&Store{}).CreateConnectorConnection(context.Background(), testSchemes, connection)

	require.ErrorIs(t, err, ErrUnregisteredScheme)
	require.ErrorContains(t, err, `tls scheme "mtls"`)
}

func TestAnEmptyRegistryRegistersNoScheme(t *testing.T) {
	err := (&Store{}).CreateConnectorConnection(context.Background(), core.Registry{}, appConnection())

	require.ErrorIs(t, err, ErrUnregisteredScheme)
}

func TestAnAppOwnedConnectionNamingAUserIsRefused(t *testing.T) {
	connection := appConnection()
	connection.OwnerID = "alice"

	err := (&Store{}).CreateConnectorConnection(context.Background(), testSchemes, connection)

	require.ErrorContains(t, err, "an app owner has no id")
}

func TestAUserOwnedConnectionNamingNoUserIsRefused(t *testing.T) {
	connection := appConnection()
	connection.OwnerType = OwnerUser

	err := (&Store{}).CreateConnectorConnection(context.Background(), testSchemes, connection)

	require.ErrorContains(t, err, "a user owner has one")
}

func TestAConnectionIsNotCreatedWithCredentials(t *testing.T) {
	connection := appConnection()
	connection.CredentialsSealed = []byte("sealed elsewhere")

	err := (&Store{}).CreateConnectorConnection(context.Background(), testSchemes, connection)

	require.ErrorContains(t, err, "after it exists")
}

func TestListingConnectionsNeedsAValidOwner(t *testing.T) {
	_, err := (&Store{}).ConnectorConnectionsByOwner(context.Background(), "acme-app", ConnectionFilter{OwnerType: OwnerUser})

	require.ErrorContains(t, err, "a user owner has one")
}

func TestAnAttemptOfAnUnknownKindIsRefused(t *testing.T) {
	attempt := &ConnectorAuthorizationAttempt{
		ID: "attempt", CustomerID: "acme-app", ConnectionID: "connection", StateHash: AuthorizationStateHash("state"),
		Kind: "install", AttemptSealed: []byte("sealed"), KEKVersion: 1, ExpiresAt: time.Now().Add(time.Minute),
	}

	err := (&Store{}).CreateConnectorAuthorizationAttempt(context.Background(), attempt)

	require.ErrorContains(t, err, `kind "install"`)
}

func TestAnAttemptThatHasAlreadyExpiredIsRefused(t *testing.T) {
	attempt := &ConnectorAuthorizationAttempt{
		ID: "attempt", CustomerID: "acme-app", ConnectionID: "connection", StateHash: AuthorizationStateHash("state"),
		Kind: AttemptConsent, AttemptSealed: []byte("sealed"), KEKVersion: 1, ExpiresAt: time.Now().Add(-time.Second),
	}

	err := (&Store{}).CreateConnectorAuthorizationAttempt(context.Background(), attempt)

	require.ErrorContains(t, err, "expires in the future")
}

func TestAnUnsealedAttemptIsRefused(t *testing.T) {
	attempt := &ConnectorAuthorizationAttempt{
		ID: "attempt", CustomerID: "acme-app", ConnectionID: "connection", StateHash: AuthorizationStateHash("state"),
		Kind: AttemptConsent, KEKVersion: 1, ExpiresAt: time.Now().Add(time.Minute),
	}

	err := (&Store{}).CreateConnectorAuthorizationAttempt(context.Background(), attempt)

	require.ErrorContains(t, err, "an attempt is sealed")
}

func TestTheStateHashIsNotTheState(t *testing.T) {
	hash := AuthorizationStateHash("bearer-state")

	require.Len(t, hash, 64, "hex of a SHA-256")
	require.NotContains(t, hash, "bearer-state")
	require.Equal(t, hash, AuthorizationStateHash("bearer-state"), "the same state finds the same row")
}

func TestAHandoffWithoutTheBlobItReadIsRefused(t *testing.T) {
	err := (&Store{}).HandOffConnectorAuthorizationAttempt(context.Background(), "attempt", nil, []byte("sealed"), 1)

	require.ErrorContains(t, err, "the blob it read")
}
