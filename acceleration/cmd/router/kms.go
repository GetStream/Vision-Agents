package main

import (
	"context"
	"encoding/base64"
	"errors"
	"fmt"

	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
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

// newKEKUnwrapper is the unwrapper for the KMS key auth.kms_key names, or nil when it is
// unset.
func newKEKUnwrapper(_ context.Context, settings config.Config) (kekUnwrapper, error) {
	if settings.Auth.KMSKey == "" {
		return nil, nil
	}
	// Refused rather than ignored: a deployment that set it has taken its plain key away,
	// and starting without the wrapped one leaves every sealed row unreadable.
	return nil, errors.New("auth.kms_key is set, but this build has no KMS client to unwrap a key with: " +
		"unset it and set " + authKEKEnvVar + "_V<n>")
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
	return string(plaintext), nil
}
