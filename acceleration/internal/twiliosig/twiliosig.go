// Package twiliosig checks that a request came from Twilio. Twilio signs every webhook with
// HMAC-SHA1, keyed by the account's Auth Token, over the URL it called followed by the POST
// parameters (https://www.twilio.com/docs/usage/webhooks/webhooks-security).
package twiliosig

import (
	"crypto/hmac"
	"crypto/sha1"
	"encoding/base64"
	"errors"
	"maps"
	"net/url"
	"slices"
	"strings"
	"unicode"
)

// AuthTokenEnvVar holds the Auth Token of the Twilio account whose webhooks the router checks.
const AuthTokenEnvVar = "TWILIO_WEBHOOK_AUTH_TOKEN"

// SignatureHeader carries Twilio's base64 signature of a webhook.
const SignatureHeader = "X-Twilio-Signature"

// CheckAuthToken refuses a token that cannot be the one Twilio signs with. A token carrying
// whitespace, such as the trailing newline of a secret mounted from a file, would fail every
// delivery with nothing in the log but rejections.
func CheckAuthToken(token string) error {
	if strings.ContainsFunc(token, unicode.IsSpace) {
		return errors.New("twilio: " + AuthTokenEnvVar + " has whitespace in it")
	}
	return nil
}

// Verify checks the signature of a request to fullURL, the URL exactly as Twilio called it with
// its query string, carrying the POST parameters params.
func Verify(token, fullURL string, params url.Values, signature string) error {
	// HMAC takes an empty key, and anyone can sign with that.
	if token == "" {
		return errors.New("no Twilio auth token to check the request against")
	}
	if !hmac.Equal([]byte(Sign(token, fullURL, params)), []byte(signature)) {
		return errors.New("the request is not signed by Twilio")
	}
	return nil
}

// Sign is the signature Twilio sends: the URL, then each parameter's name and value with the
// names sorted. A name sent more than once adds each of its distinct values, sorted, as
// Twilio's Python and Node SDKs sign it (twilio-python twilio/request_validator.py
// compute_signature, twilio-node src/webhooks/webhooks.ts toFormUrlEncodedParam).
func Sign(token, fullURL string, params url.Values) string {
	signed := []byte(fullURL)
	for _, name := range slices.Sorted(maps.Keys(params)) {
		values := slices.Clone(params[name])
		slices.Sort(values)
		for _, value := range slices.Compact(values) {
			signed = append(append(signed, name...), value...)
		}
	}
	mac := hmac.New(sha1.New, []byte(token))
	mac.Write(signed)
	return base64.StdEncoding.EncodeToString(mac.Sum(nil))
}
