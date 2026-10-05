package channels

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// sendTimeout bounds one request to a provider. Sending is a few hundred milliseconds of
// work, and a provider that has stopped answering should not hold the turn open.
const sendTimeout = 30 * time.Second

// maxRefusalBytes is how much of a provider's complaint is reported. Enough to say which
// field it did not like, short enough not to put a page of HTML in a log line.
const maxRefusalBytes = 1 << 10

// post sends a JSON body with a bearer token. All three providers authenticate this way.
func post(ctx context.Context, client *http.Client, url, token string, body any) error {
	return send(ctx, client, http.MethodPost, url, token, body)
}

// patch is how a provider's own record of a number is changed.
func patch(ctx context.Context, client *http.Client, url, token string, body any) error {
	return send(ctx, client, http.MethodPatch, url, token, body)
}

func send(ctx context.Context, client *http.Client, method, url, token string, body any) error {
	raw, err := json.Marshal(body)
	if err != nil {
		return fmt.Errorf("channels: %s %s: %w", method, url, err)
	}
	ctx, cancel := context.WithTimeout(ctx, sendTimeout)
	defer cancel()
	request, err := http.NewRequestWithContext(ctx, method, url, bytes.NewReader(raw))
	if err != nil {
		return fmt.Errorf("channels: %s %s: %w", method, url, err)
	}
	request.Header.Set("Content-Type", "application/json")
	if token != "" {
		request.Header.Set("Authorization", "Bearer "+token)
	}
	if client == nil {
		client = http.DefaultClient
	}

	response, err := client.Do(request)
	if err != nil {
		return fmt.Errorf("channels: %s %s: %w", method, url, err)
	}
	defer response.Body.Close()
	if response.StatusCode >= http.StatusBadRequest {
		// The provider's own words are carried through: what is wrong with a send is
		// something only it knows, and it is usually a number that cannot be written to.
		said, _ := io.ReadAll(io.LimitReader(response.Body, maxRefusalBytes))
		return fmt.Errorf("channels: %s refused the request: %s: %s",
			url, response.Status, bytes.TrimSpace(said))
	}
	_, _ = io.Copy(io.Discard, io.LimitReader(response.Body, maxRefusalBytes))
	return nil
}
