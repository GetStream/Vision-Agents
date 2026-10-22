package main

import (
	"context"
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"log/slog"
	"net"
	"os"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"cloud.google.com/go/kms/apiv1/kmspb"
	"github.com/stretchr/testify/suite"
	"google.golang.org/api/option"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/stats"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/test/bufconn"

	"github.com/GetStream/Vision-Agents/acceleration/internal/appconfig"
	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// ConnectorSealerSuite covers how the router builds the keyring the secrets it holds are
// sealed under, connector credentials and Stream app keys, starting from the hosted
// deployment's proxy mode with no key at all.
type ConnectorSealerSuite struct {
	suite.Suite
	settings config.Config
}

func TestConnectorSealerSuite(t *testing.T) {
	suite.Run(t, new(ConnectorSealerSuite))
}

func (s *ConnectorSealerSuite) SetupTest() {
	// The keyring is read from every ROUTER_AUTH_KEK_V<n> in the environment, so any the
	// shell running the tests happens to hold are cleared too.
	variables := []string{authKEKEnvVar, authKEKVersionEnvVar}
	for _, variable := range os.Environ() {
		if name, _, _ := strings.Cut(variable, "="); strings.HasPrefix(name, authKEKEnvVar+"_V") {
			variables = append(variables, name)
		}
	}
	for _, variable := range variables {
		s.T().Setenv(variable, "")
		s.Require().NoError(os.Unsetenv(variable))
	}
	s.settings = config.Defaults()
	s.settings.Auth.Mode = string(auth.Proxy)
}

func (s *ConnectorSealerSuite) TestProxyStartsWithoutAKEKWhenConnectorsAreOff() {
	sealer, err := newSecretSealer(s.T().Context(), s.settings)
	s.Require().NoError(err)
	s.Nil(sealer)

	_, err = newAuthenticator(s.T().Context(), s.settings, nil, sealer, slog.New(slog.DiscardHandler))
	s.NoError(err)
}

func (s *ConnectorSealerSuite) TestAKeyringIsBuiltWhenSetEvenIfNothingRequiresIt() {
	// Channels seal a provider's credentials under the keyring when there is one, though
	// they do not refuse to start without it, so a key set only as ROUTER_AUTH_KEK_V1 has
	// to reach them as much as connectors and app mode.
	s.T().Setenv("ROUTER_AUTH_KEK_V1", "first-key")

	sealer, err := newSecretSealer(s.T().Context(), s.settings)
	s.Require().NoError(err)
	s.Require().NotNil(sealer)
	s.Equal(1, sealer.CurrentVersion())
}

func (s *ConnectorSealerSuite) TestAKeyringNothingRequiresIsStillChecked() {
	s.T().Setenv("ROUTER_AUTH_KEK_V1", "first-key")
	s.T().Setenv("ROUTER_AUTH_KEK_VERSION", "two")

	_, err := newSecretSealer(s.T().Context(), s.settings)
	s.ErrorContains(err, "channels need ROUTER_AUTH_KEK_VERSION to be a positive integer")
}

func (s *ConnectorSealerSuite) TestProxyWithConnectorsOnRefusesToStartWithoutAKeyring() {
	s.settings.Connectors.Enabled = true

	_, err := newSecretSealer(s.T().Context(), s.settings)
	s.ErrorContains(err, "connectors.enabled needs a key encryption keyring")
	s.NotContains(err.Error(), "stream.tenancy")
	s.ErrorContains(err, "ROUTER_AUTH_KEK_V1")
}

func (s *ConnectorSealerSuite) TestProxyWithConnectorsOnBuildsTheSealerFromTheKeyring() {
	s.settings.Connectors.Enabled = true
	s.T().Setenv("ROUTER_AUTH_KEK_V1", "first-key")

	sealer, err := newSecretSealer(s.T().Context(), s.settings)
	s.Require().NoError(err)
	s.Equal(1, sealer.CurrentVersion())
}

func (s *ConnectorSealerSuite) TestAuthKEKIsVersionOne() {
	s.settings.Connectors.Enabled = true
	s.settings.Auth.KEK = "first-key"
	apiSecrets, err := auth.NewSealer("first-key")
	s.Require().NoError(err)
	sealed, err := apiSecrets.Seal("vas_live_s3cret")
	s.Require().NoError(err)

	sealer, err := newSecretSealer(s.T().Context(), s.settings)
	s.Require().NoError(err)
	opened, err := sealer.OpenWithAADVersion(sealed, nil, 1)
	s.Require().NoError(err)
	s.Equal("vas_live_s3cret", opened)
}

func (s *ConnectorSealerSuite) TestARowSealedUnderVersionOneOpensAfterVersionTwoIsAdded() {
	s.settings.Connectors.Enabled = true
	s.T().Setenv("ROUTER_AUTH_KEK_V1", "first-key")
	before, err := newSecretSealer(s.T().Context(), s.settings)
	s.Require().NoError(err)
	sealed, err := before.SealWithAAD("connector secret", []byte("connection"))
	s.Require().NoError(err)

	s.T().Setenv("ROUTER_AUTH_KEK_V2", "second-key")
	s.T().Setenv("ROUTER_AUTH_KEK_VERSION", "2")
	after, err := newSecretSealer(s.T().Context(), s.settings)
	s.Require().NoError(err)
	s.Equal(2, after.CurrentVersion())
	opened, err := after.OpenWithAADVersion(sealed, []byte("connection"), 1)
	s.Require().NoError(err)
	s.Equal("connector secret", opened)
}

func (s *ConnectorSealerSuite) TestAVersionWithoutItsKeyIsRefused() {
	s.settings.Connectors.Enabled = true
	s.T().Setenv("ROUTER_AUTH_KEK_V1", "first-key")
	s.T().Setenv("ROUTER_AUTH_KEK_VERSION", "2")

	_, err := newSecretSealer(s.T().Context(), s.settings)
	s.ErrorContains(err, "needs a key for the version ROUTER_AUTH_KEK_VERSION names")
	s.ErrorContains(err, "versions set: [1]")
}

func (s *ConnectorSealerSuite) TestAVersionThatIsNotANumberIsRefused() {
	s.settings.Connectors.Enabled = true
	s.T().Setenv("ROUTER_AUTH_KEK_V1", "first-key")
	s.T().Setenv("ROUTER_AUTH_KEK_VERSION", "two")

	_, err := newSecretSealer(s.T().Context(), s.settings)
	s.ErrorContains(err, "ROUTER_AUTH_KEK_VERSION to be a positive integer")
}

func (s *ConnectorSealerSuite) TestAKeyPastedIntoTheVersionIsNotRepeatedInTheError() {
	s.settings.Connectors.Enabled = true
	s.T().Setenv("ROUTER_AUTH_KEK_V1", "first-key")
	s.T().Setenv("ROUTER_AUTH_KEK_VERSION", "pasted-key-encryption-key")

	_, err := newSecretSealer(s.T().Context(), s.settings)
	s.Require().Error(err)
	s.NotContains(err.Error(), "pasted-key-encryption-key")
}

func (s *ConnectorSealerSuite) TestAnAllDigitKeyPastedIntoTheVersionIsNotRepeatedInTheError() {
	s.settings.Connectors.Enabled = true
	s.T().Setenv("ROUTER_AUTH_KEK_V1", "first-key")
	s.T().Setenv("ROUTER_AUTH_KEK_VERSION", "8473019385746201")

	_, err := newSecretSealer(s.T().Context(), s.settings)
	s.Require().Error(err)
	s.NotContains(err.Error(), "8473019385746201")
}

func (s *ConnectorSealerSuite) TestMovingTheVersionBackKeepsNewerKeysLoaded() {
	s.settings.Connectors.Enabled = true
	s.T().Setenv("ROUTER_AUTH_KEK_V1", "first-key")
	s.T().Setenv("ROUTER_AUTH_KEK_V2", "second-key")
	s.T().Setenv("ROUTER_AUTH_KEK_VERSION", "2")
	forward, err := newSecretSealer(s.T().Context(), s.settings)
	s.Require().NoError(err)
	sealed, err := forward.SealWithAAD("connector secret", []byte("connection"))
	s.Require().NoError(err)

	s.T().Setenv("ROUTER_AUTH_KEK_VERSION", "1")
	back, err := newSecretSealer(s.T().Context(), s.settings)
	s.Require().NoError(err)
	s.Equal(1, back.CurrentVersion())
	opened, err := back.OpenWithAADVersion(sealed, []byte("connection"), 2)
	s.Require().NoError(err)
	s.Equal("connector secret", opened)
}

func (s *ConnectorSealerSuite) TestOnlyThePlainSpellingOfAVersionIsAKey() {
	s.settings.Connectors.Enabled = true
	s.T().Setenv("ROUTER_AUTH_KEK_V1", "first-key")
	s.T().Setenv("ROUTER_AUTH_KEK_V01", "another-key")

	sealer, err := newSecretSealer(s.T().Context(), s.settings)
	s.Require().NoError(err)
	sealed, err := sealer.SealWithAAD("connector secret", nil)
	s.Require().NoError(err)
	withFirst, err := auth.NewSealerWithKeyring(1, map[int]string{1: "first-key"})
	s.Require().NoError(err)
	opened, err := withFirst.OpenWithAADVersion(sealed, nil, 1)
	s.Require().NoError(err)
	s.Equal("connector secret", opened)
}

func (s *ConnectorSealerSuite) TestTwoDifferentVersionOneKeysAreRefused() {
	s.settings.Connectors.Enabled = true
	s.settings.Auth.KEK = "first-key"
	s.T().Setenv("ROUTER_AUTH_KEK_V1", "another-key")

	_, err := newSecretSealer(s.T().Context(), s.settings)
	s.ErrorContains(err, "both version 1 and differ")
}

func (s *ConnectorSealerSuite) TestAppTenancyBuildsTheSealerInProxyMode() {
	// Every Stream app's keys are sealed under the keyring, whatever decides who calls.
	s.settings.Stream.Tenancy = config.TenancyApp
	s.T().Setenv("ROUTER_AUTH_KEK_V1", "first-key")

	sealer, err := newSecretSealer(s.T().Context(), s.settings)

	s.Require().NoError(err)
	s.Require().NotNil(sealer)
	s.Equal(1, sealer.CurrentVersion())
}

func (s *ConnectorSealerSuite) TestAppTenancyWithoutAKeyringRefusesToStart() {
	s.settings.Stream.Tenancy = config.TenancyApp

	_, err := newSecretSealer(s.T().Context(), s.settings)

	s.ErrorContains(err, "stream.tenancy=app needs a key encryption keyring")
	s.ErrorContains(err, "ROUTER_AUTH_KEK_V1")
}

func (s *ConnectorSealerSuite) TestARefusalNamesEverySettingThatNeedsTheKeyring() {
	s.settings.Connectors.Enabled = true
	s.settings.Stream.Tenancy = config.TenancyApp

	_, err := newSecretSealer(s.T().Context(), s.settings)

	s.ErrorContains(err, "connectors.enabled and stream.tenancy=app need a key encryption keyring")
}

func (s *ConnectorSealerSuite) TestProxyStartsWithoutAKeyWhenNothingNeedsOne() {
	s.settings.Stream.Tenancy = config.TenancyDeployment

	sealer, err := newSecretSealer(s.T().Context(), s.settings)

	s.Require().NoError(err)
	s.Nil(sealer)
}

// sealedOnBase is "connector secret" sealed with AAD "connection" under a keyring of
// ROUTER_AUTH_KEK_V1=first-key by the router before a key could be wrapped, and
// derivedOnBase is that keyring's Derive("peer"): rows and peers a deployment already has.
const (
	sealedOnBase  = "5NUVi8EqT1yQd+3xwCz8LZEphIS6BhczMLcM4IRuXg+j8tbqVShrG0lYfvI="
	derivedOnBase = "b6b80eceb6a97ac5f0dc6f9a7fb263e6a1e1f4bf4343aec0ee1778f62784b90b"
)

// localKMS stands in for a cloud KMS: it wraps under a key of its own with AES-GCM and, like
// Cloud KMS, unwraps only with the AAD a key was wrapped with.
type localKMS struct {
	aead cipher.AEAD
}

func newLocalKMS(s *ConnectorSealerSuite) *localKMS {
	key := sha256.Sum256([]byte("local kms"))
	block, err := aes.NewCipher(key[:])
	s.Require().NoError(err)
	aead, err := cipher.NewGCM(block)
	s.Require().NoError(err)
	return &localKMS{aead: aead}
}

func (k *localKMS) wrap(plaintext, aad string) string {
	nonce := make([]byte, k.aead.NonceSize())
	_, _ = rand.Read(nonce)
	return base64.StdEncoding.EncodeToString(k.aead.Seal(nonce, nonce, []byte(plaintext), []byte(aad)))
}

func (k *localKMS) Unwrap(_ context.Context, ciphertext, aad []byte) ([]byte, error) {
	nonce, sealed := ciphertext[:k.aead.NonceSize()], ciphertext[k.aead.NonceSize():]
	return k.aead.Open(nil, nonce, sealed, aad)
}

func (s *ConnectorSealerSuite) requireOpensWhatTheBaseSealed(sealer *auth.Sealer) {
	sealed, err := base64.StdEncoding.DecodeString(sealedOnBase)
	s.Require().NoError(err)
	opened, err := sealer.OpenWithAADVersion(sealed, []byte("connection"), 1)
	s.Require().NoError(err)
	s.Equal("connector secret", opened)
	current, _ := sealer.Derive("peer")
	s.Equal(derivedOnBase, hex.EncodeToString(current))
}

func (s *ConnectorSealerSuite) TestAPlainKeyringOpensWhatItSealedBefore() {
	s.settings.Connectors.Enabled = true
	s.T().Setenv("ROUTER_AUTH_KEK_V1", "first-key")

	sealer, err := newSecretSealer(s.T().Context(), s.settings)
	s.Require().NoError(err)
	s.requireOpensWhatTheBaseSealed(sealer)
}

func (s *ConnectorSealerSuite) TestAWrappedKeyOpensWhatThePlainOneSealed() {
	kms := newLocalKMS(s)
	s.T().Setenv("ROUTER_AUTH_KEK_V1_WRAPPED", kms.wrap("first-key", "router-kek-v1"))

	sealer, err := readKeyring(s.T().Context(), s.settings, kms, "the test needs")
	s.Require().NoError(err)
	s.Equal(1, sealer.CurrentVersion())
	s.requireOpensWhatTheBaseSealed(sealer)
}

func (s *ConnectorSealerSuite) TestAWrappedKeyCanSealNewRows() {
	kms := newLocalKMS(s)
	s.T().Setenv("ROUTER_AUTH_KEK_V1", "first-key")
	s.T().Setenv("ROUTER_AUTH_KEK_V2_WRAPPED", kms.wrap("second-key", "router-kek-v2"))
	s.T().Setenv("ROUTER_AUTH_KEK_VERSION", "2")

	sealer, err := readKeyring(s.T().Context(), s.settings, kms, "the test needs")
	s.Require().NoError(err)
	s.Equal(2, sealer.CurrentVersion())
	sealed, err := sealer.SealWithAAD("connector secret", []byte("connection"))
	s.Require().NoError(err)
	plain, err := auth.NewSealerWithKeyring(2, map[int]string{2: "second-key"})
	s.Require().NoError(err)
	opened, err := plain.OpenWithAADVersion(sealed, []byte("connection"), 2)
	s.Require().NoError(err)
	s.Equal("connector secret", opened)
}

func (s *ConnectorSealerSuite) TestAWrappedKeyUnwrapsOnlyUnderItsOwnVersion() {
	kms := newLocalKMS(s)
	s.T().Setenv("ROUTER_AUTH_KEK_V1", "first-key")
	wrapped := kms.wrap("first-key", "router-kek-v1")
	s.T().Setenv("ROUTER_AUTH_KEK_V2_WRAPPED", wrapped)

	_, err := readKeyring(s.T().Context(), s.settings, kms, "the test needs")
	s.ErrorContains(err, "ROUTER_AUTH_KEK_V2_WRAPPED: unwrap")
	s.NotContains(err.Error(), wrapped)
	s.NotContains(err.Error(), "first-key")
}

func (s *ConnectorSealerSuite) TestAWrappedKeyThatUnwrapsToNothingIsRefused() {
	kms := newLocalKMS(s)
	s.T().Setenv("ROUTER_AUTH_KEK_V1_WRAPPED", kms.wrap("", "router-kek-v1"))

	_, err := readKeyring(s.T().Context(), s.settings, kms, "the test needs")
	s.ErrorContains(err, "ROUTER_AUTH_KEK_V1_WRAPPED: unwrapped to an empty key")
}

func (s *ConnectorSealerSuite) TestAVersionSetBothPlainAndWrappedIsRefused() {
	kms := newLocalKMS(s)
	s.T().Setenv("ROUTER_AUTH_KEK_V1", "first-key")
	s.T().Setenv("ROUTER_AUTH_KEK_V1_WRAPPED", kms.wrap("first-key", "router-kek-v1"))

	_, err := readKeyring(s.T().Context(), s.settings, kms, "the test needs")
	s.ErrorContains(err, "ROUTER_AUTH_KEK_V1_WRAPPED and a plain version 1 key")
	s.ErrorContains(err, "are both set")
	s.NotContains(err.Error(), "first-key")
}

func (s *ConnectorSealerSuite) TestAWrappedKeyWithoutAKMSKeyIsRefused() {
	s.settings.Connectors.Enabled = true
	s.T().Setenv("ROUTER_AUTH_KEK_V1_WRAPPED", newLocalKMS(s).wrap("first-key", "router-kek-v1"))

	_, err := newSecretSealer(s.T().Context(), s.settings)
	s.ErrorContains(err, "ROUTER_AUTH_KEK_V1_WRAPPED is set, but auth.kms_key names no KMS key")
}

func (s *ConnectorSealerSuite) TestAWrappedKeyThatIsNotBase64IsNotRepeatedInTheError() {
	s.T().Setenv("ROUTER_AUTH_KEK_V1_WRAPPED", "first-key pasted here")

	_, err := readKeyring(s.T().Context(), s.settings, newLocalKMS(s), "the test needs")
	s.ErrorContains(err, "ROUTER_AUTH_KEK_V1_WRAPPED: not standard base64")
	s.NotContains(err.Error(), "first-key")
}

// testKMSKey is the CryptoKey the fake Cloud KMS holds, in the resource name form
// auth.kms_key takes.
const testKMSKey = "projects/p/locations/global/keyRings/router/cryptoKeys/kek"

// fakeCloudKMS answers Decrypt for testKMSKey as Cloud KMS does, unwrapping with localKMS
// only under the AAD a key was wrapped with. It counts the Decrypts a start makes and the
// connections still open to it.
type fakeCloudKMS struct {
	kmspb.UnimplementedKeyManagementServiceServer
	kms      *localKMS
	refusal  error
	decrypts atomic.Int64
	open     atomic.Int64
}

func (f *fakeCloudKMS) Decrypt(ctx context.Context, request *kmspb.DecryptRequest) (*kmspb.DecryptResponse, error) {
	f.decrypts.Add(1)
	if f.refusal != nil {
		return nil, f.refusal
	}
	if request.GetName() != testKMSKey {
		return nil, status.Error(codes.NotFound, "no such CryptoKey")
	}
	plaintext, err := f.kms.Unwrap(ctx, request.GetCiphertext(), request.GetAdditionalAuthenticatedData())
	if err != nil {
		return nil, status.Error(codes.InvalidArgument, "decryption failed")
	}
	return &kmspb.DecryptResponse{Plaintext: plaintext}, nil
}

func (f *fakeCloudKMS) TagConn(ctx context.Context, _ *stats.ConnTagInfo) context.Context { return ctx }

func (f *fakeCloudKMS) HandleConn(_ context.Context, event stats.ConnStats) {
	switch event.(type) {
	case *stats.ConnBegin:
		f.open.Add(1)
	case *stats.ConnEnd:
		f.open.Add(-1)
	}
}

func (f *fakeCloudKMS) TagRPC(ctx context.Context, _ *stats.RPCTagInfo) context.Context { return ctx }

func (f *fakeCloudKMS) HandleRPC(context.Context, stats.RPCStats) {}

// startFakeCloudKMS serves a fakeCloudKMS in process, over an in-memory listener, and points
// every KMS client the router opens at it, with no credentials and no TLS.
func (s *ConnectorSealerSuite) startFakeCloudKMS() *fakeCloudKMS {
	fake := &fakeCloudKMS{kms: newLocalKMS(s)}
	listener := bufconn.Listen(1 << 20)
	server := grpc.NewServer(grpc.StatsHandler(fake))
	kmspb.RegisterKeyManagementServiceServer(server, fake)
	go func() { _ = server.Serve(listener) }()
	s.T().Cleanup(server.Stop)
	kmsClientOptions = []option.ClientOption{
		option.WithEndpoint("passthrough:///cloudkms"),
		option.WithoutAuthentication(),
		option.WithGRPCDialOption(grpc.WithTransportCredentials(insecure.NewCredentials())),
		option.WithGRPCDialOption(grpc.WithContextDialer(func(ctx context.Context, _ string) (net.Conn, error) {
			return listener.DialContext(ctx)
		})),
	}
	s.T().Cleanup(func() { kmsClientOptions = nil })
	s.settings.Auth.KMSKey = testKMSKey
	return fake
}

func (s *ConnectorSealerSuite) TestAKeyCloudKMSUnwrapsOpensWhatThePlainOneSealed() {
	fake := s.startFakeCloudKMS()
	s.settings.Connectors.Enabled = true
	s.T().Setenv("ROUTER_AUTH_KEK_V1_WRAPPED", fake.kms.wrap("first-key", "router-kek-v1"))

	sealer, err := newSecretSealer(s.T().Context(), s.settings)
	s.Require().NoError(err)
	s.Equal(1, sealer.CurrentVersion())
	s.requireOpensWhatTheBaseSealed(sealer)
	s.Equal(int64(1), fake.decrypts.Load())
	s.Eventually(func() bool { return fake.open.Load() == 0 }, 5*time.Second, 10*time.Millisecond,
		"the KMS client is closed once the keyring is read")
}

func (s *ConnectorSealerSuite) TestACloudKMSRefusalFailsStartupWithoutTheKey() {
	fake := s.startFakeCloudKMS()
	fake.refusal = status.Error(codes.PermissionDenied, "permission cloudkms.cryptoKeyVersions.useToDecrypt denied")
	s.settings.Connectors.Enabled = true
	wrapped := fake.kms.wrap("first-key", "router-kek-v1")
	s.T().Setenv("ROUTER_AUTH_KEK_V1_WRAPPED", wrapped)

	_, err := newSecretSealer(s.T().Context(), s.settings)
	s.ErrorContains(err, "ROUTER_AUTH_KEK_V1_WRAPPED: unwrap: cloud kms: decrypt with "+testKMSKey)
	s.ErrorContains(err, "useToDecrypt denied")
	s.NotContains(err.Error(), wrapped)
	s.NotContains(err.Error(), "first-key")
}

func (s *ConnectorSealerSuite) TestAKeyCloudKMSWrappedForAnotherVersionFailsStartup() {
	fake := s.startFakeCloudKMS()
	s.settings.Connectors.Enabled = true
	wrapped := fake.kms.wrap("first-key", "router-kek-v2")
	s.T().Setenv("ROUTER_AUTH_KEK_V1_WRAPPED", wrapped)

	_, err := newSecretSealer(s.T().Context(), s.settings)
	s.ErrorContains(err, "ROUTER_AUTH_KEK_V1_WRAPPED: unwrap: cloud kms: decrypt with "+testKMSKey)
	s.ErrorContains(err, "decryption failed")
	s.NotContains(err.Error(), wrapped)
	s.NotContains(err.Error(), "first-key")
}

// apiKeyConfigs is the store api_key mode looks keys up in. Nothing here looks one up, and
// opening a store connects to nothing until a query runs, so no database is needed.
func (s *ConnectorSealerSuite) apiKeyConfigs() *appconfig.Store {
	s.settings.Auth.Mode = string(auth.APIKey)
	pg, err := store.Open("postgres://nobody@127.0.0.1:1/unused?sslmode=disable")
	s.Require().NoError(err)
	s.T().Cleanup(func() { pg.Close() })
	configs, err := appconfig.New(appconfig.Options{Store: pg})
	s.Require().NoError(err)
	return configs
}

func (s *ConnectorSealerSuite) TestAnAPIKeyStartUnwrapsTheKeyOnce() {
	fake := s.startFakeCloudKMS()
	configs := s.apiKeyConfigs()
	s.T().Setenv("ROUTER_AUTH_KEK_V1_WRAPPED", fake.kms.wrap("first-key", "router-kek-v1"))

	secrets, err := newSecretSealer(s.T().Context(), s.settings)
	s.Require().NoError(err)
	authenticator, err := newAuthenticator(s.T().Context(), s.settings, configs, secrets, slog.New(slog.DiscardHandler))
	s.Require().NoError(err)
	s.NotNil(authenticator)
	s.Equal(int64(1), fake.decrypts.Load())
}

func (s *ConnectorSealerSuite) TestAnAPIKeyStartWithoutAKeyIsRefusedAsBefore() {
	configs := s.apiKeyConfigs()

	secrets, err := newSecretSealer(s.T().Context(), s.settings)
	s.Require().NoError(err)
	_, err = newAuthenticator(s.T().Context(), s.settings, configs, secrets, slog.New(slog.DiscardHandler))
	s.EqualError(err, "auth.mode=api_key needs a key encryption keyring to seal the secrets it holds: "+
		"set ROUTER_AUTH_KEK_V1 (ROUTER_AUTH_KEK is version 1)")
}

func (s *ConnectorSealerSuite) TestAnAPIKeyStartWithAPlainKeyAuthenticates() {
	configs := s.apiKeyConfigs()
	s.T().Setenv("ROUTER_AUTH_KEK_V1", "first-key")

	secrets, err := newSecretSealer(s.T().Context(), s.settings)
	s.Require().NoError(err)
	authenticator, err := newAuthenticator(s.T().Context(), s.settings, configs, secrets, slog.New(slog.DiscardHandler))
	s.Require().NoError(err)
	s.NotNil(authenticator)
}
