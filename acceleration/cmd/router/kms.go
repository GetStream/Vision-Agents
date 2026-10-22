package main

import (
	"context"
	"encoding/base64"
	"errors"
	"fmt"

	kms "cloud.google.com/go/kms/apiv1"
	"cloud.google.com/go/kms/apiv1/kmspb"
	"google.golang.org/api/option"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// kekAADFormat is the additional authenticated data a key of version n is wrapped with,
// router-kek-v<n>. It binds the ciphertext to its version, so one put under another
// version's variable does not unwrap. The string is this router's choice: Cloud KMS takes
// any AAD up to 64 KiB and decrypts only with the one it encrypted with
// (https://docs.cloud.google.com/kms/docs/additional-authenticated-data). The operator
// wrapping a key passes the same string.
const kekAADFormat = "router-kek-v%d"

// kekUnwrapper decrypts a key encryption key a KMS wrapped, so the key itself is in no
// variable or secret the deployment holds.
type kekUnwrapper interface {
	// Unwrap returns the plaintext of ciphertext, wrapped with aad as its additional
	// authenticated data.
	Unwrap(ctx context.Context, ciphertext, aad []byte) ([]byte, error)
}

// kmsClientOptions are added to the options of the KMS client the router opens: none in the
// binary, which reaches Cloud KMS with Application Default Credentials. A test points them at
// a fake server.
var kmsClientOptions []option.ClientOption

// cloudKMS unwraps with Decrypt under one Cloud KMS key. The client's default Decrypt call
// gives up after 60 s and retries only Unavailable and DeadlineExceeded
// (defaultKeyManagementCallOptions in cloud.google.com/go/kms/apiv1), so a KMS that does not
// answer fails startup rather than holding it.
type cloudKMS struct {
	client *kms.KeyManagementClient
	// key is the CryptoKey's resource name, projects/<p>/locations/<l>/keyRings/<r>/cryptoKeys/<k>:
	// Decrypt takes the key, never a version, and the server finds the version
	// (kmspb.DecryptRequest.Name, cloud.google.com/go/kms v1.26.0).
	key string
}

// newKEKUnwrapper opens a Cloud KMS client for key, the CryptoKey auth.kms_key names. The
// caller closes it.
func newKEKUnwrapper(ctx context.Context, key string) (*cloudKMS, error) {
	client, err := kms.NewKeyManagementClient(ctx, kmsClientOptions...)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("auth.kms_key: open a Cloud KMS client: %w", err))
	}
	return &cloudKMS{client: client, key: key}, nil
}

func (c *cloudKMS) Unwrap(ctx context.Context, ciphertext, aad []byte) ([]byte, error) {
	response, err := c.client.Decrypt(ctx, &kmspb.DecryptRequest{
		Name:                        c.key,
		Ciphertext:                  ciphertext,
		AdditionalAuthenticatedData: aad,
	})
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("cloud kms: decrypt with %s: %w", c.key, err))
	}
	return response.GetPlaintext(), nil
}

// Close closes the client's connection.
func (c *cloudKMS) Close() error {
	return c.client.Close()
}

// unwrapKEK unwraps the base64 ciphertext of the key of version n into the key itself, the
// same string a plain ROUTER_AUTH_KEK_V<n> holds, so the rows it sealed still open.
func unwrapKEK(ctx context.Context, unwrapper kekUnwrapper, version int, encoded string) (string, error) {
	ciphertext, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		// The value is left out: a plain key pasted into the wrong variable would otherwise
		// reach the logs and Sentry with this error.
		return "", errors.New("not standard base64")
	}
	plaintext, err := unwrapper.Unwrap(ctx, ciphertext, fmt.Appendf(nil, kekAADFormat, version))
	if err != nil {
		return "", fmt.Errorf("unwrap: %w", err)
	}
	if len(plaintext) == 0 {
		return "", errors.New("unwrapped to an empty key")
	}
	return string(plaintext), nil
}
