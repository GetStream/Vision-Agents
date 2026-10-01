package connectors

import (
	"testing"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/stretchr/testify/require"
)

func TestCredentialsAreBoundToTenantConnectionAndRevision(t *testing.T) {
	sealer, err := auth.NewSealer("test-key-encryption-key")
	require.NoError(t, err)

	want := Credentials{
		AuthType:    AuthOAuth2,
		AccessToken: "access", RefreshToken: "refresh", OAuthClientID: "client",
		RefreshEndpoint: "https://provider.example/token", Resource: "https://provider.example/mcp",
	}
	sealed, err := SealCredentials(sealer, "tenant-a", "connection-1", 4, want)
	require.NoError(t, err)

	got, err := OpenCredentials(sealer, "tenant-a", "connection-1", 4, sealer.CurrentVersion(), sealed)
	require.NoError(t, err)
	require.Equal(t, want, got)

	for _, identity := range []struct {
		customerID   string
		connectionID string
		revision     int
	}{
		{customerID: "tenant-b", connectionID: "connection-1", revision: 4},
		{customerID: "tenant-a", connectionID: "connection-2", revision: 4},
		{customerID: "tenant-a", connectionID: "connection-1", revision: 5},
	} {
		_, err := OpenCredentials(sealer, identity.customerID, identity.connectionID, identity.revision, sealer.CurrentVersion(), sealed)
		require.Error(t, err)
	}
}

func TestStaticBearerAndAPIKeyCredentialsAreEncrypted(t *testing.T) {
	sealer, err := auth.NewSealer("test-key-encryption-key")
	require.NoError(t, err)

	for _, want := range []Credentials{
		{AuthType: AuthBearer, AccessToken: "bearer-secret"},
		{AuthType: AuthAPIKey, APIKey: "api-key-secret"},
	} {
		sealed, err := SealCredentials(sealer, "tenant-a", "connection-1", 2, want)
		require.NoError(t, err)
		got, err := OpenCredentials(sealer, "tenant-a", "connection-1", 2, sealer.CurrentVersion(), sealed)
		require.NoError(t, err)
		require.Equal(t, want, got)
	}
}

func TestCredentialsCanBeRewrappedAcrossKEKVersions(t *testing.T) {
	old, err := auth.NewSealer("old-connector-key")
	require.NoError(t, err)
	want := Credentials{AuthType: AuthOAuth2, AccessToken: "old-access-token", RefreshToken: "old-refresh-token"}
	sealed, err := SealCredentials(old, "tenant-a", "connection-1", 7, want)
	require.NoError(t, err)

	rotated, err := auth.NewSealerWithKeyring(2, map[int]string{1: "old-connector-key", 2: "new-connector-key"})
	require.NoError(t, err)
	opened, err := OpenCredentials(rotated, "tenant-a", "connection-1", 7, 1, sealed)
	require.NoError(t, err)
	require.Equal(t, want, opened)

	rewrapped, err := SealCredentials(rotated, "tenant-a", "connection-1", 7, opened)
	require.NoError(t, err)
	opened, err = OpenCredentials(rotated, "tenant-a", "connection-1", 7, 2, rewrapped)
	require.NoError(t, err)
	require.Equal(t, want, opened)
}
