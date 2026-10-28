package voices

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/keypool"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// defaultTimeout bounds a clone. It is generous because the providers read the whole
// recording before answering, and short because nobody is listening to a call while this
// happens.
const defaultTimeout = 2 * time.Minute

// form builds a multipart body. Every provider takes the recordings this way, and they
// differ only in what the file field is called and what goes beside it.
type form struct {
	body   bytes.Buffer
	writer *multipart.Writer
}

func newForm() *form {
	built := &form{}
	built.writer = multipart.NewWriter(&built.body)
	return built
}

// field adds a text value, skipping the ones that are empty: a provider that treats a
// blank description as a description is one we would have to work around.
func (f *form) field(name, value string) error {
	if value == "" {
		return nil
	}
	return stack.Wrap(f.writer.WriteField(name, value))
}

// file adds a recording under a field name.
func (f *form) file(field string, sample Sample) error {
	part, err := f.writer.CreateFormFile(field, sample.Name)
	if err != nil {
		return stack.Wrap(err)
	}
	_, err = part.Write(sample.Content)
	return stack.Wrap(err)
}

// done closes the body and reports the content type that describes it.
func (f *form) done() (io.Reader, string, error) {
	if err := f.writer.Close(); err != nil {
		return nil, "", stack.Wrap(err)
	}
	return &f.body, f.writer.FormDataContentType(), nil
}

// client is the HTTP client a provider uses. Every request that carries the key goes out
// through keys, put where auth says, so a 401 or 402 is counted as it is on a call, and a
// key the pool rests (a 429, a spent quota) rests for the calls too. It always sends on the
// first key and never moves to another: a cloned voice belongs to the account of the key
// that made it, and is looked up and deleted there.
func client(keys *keypool.Pool, auth keypool.Auth, timeout time.Duration) *http.Client {
	if timeout == 0 {
		timeout = defaultTimeout
	}
	return keys.First().WrapClient(&http.Client{Timeout: timeout}, auth)
}

// speak posts a JSON synthesis request and reads the audio it answers with. The content
// type is the one that was asked for, since not every provider labels what it sends back.
func speak(
	ctx context.Context,
	httpClient *http.Client,
	provider, url, contentType string,
	header http.Header,
	payload any,
) (Speech, error) {
	body, err := json.Marshal(payload)
	if err != nil {
		return Speech{}, stack.Wrap(err)
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return Speech{}, stack.Wrap(err)
	}
	request.Header = header
	request.Header.Set("Content-Type", "application/json")

	response, err := httpClient.Do(request)
	if err != nil {
		return Speech{}, stack.Wrap(fmt.Errorf("voices: %s speak: %w", provider, err))
	}
	defer response.Body.Close()

	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return Speech{}, refused(provider, response)
	}
	audio, err := io.ReadAll(response.Body)
	if err != nil {
		return Speech{}, stack.Wrap(fmt.Errorf("voices: %s speak: %w", provider, err))
	}
	if len(audio) == 0 {
		return Speech{}, stack.Wrap(fmt.Errorf("voices: %s answered with no audio", provider))
	}
	return Speech{Audio: audio, ContentType: contentType}, nil
}

// fetchPreview downloads a sample a vendor has already made. The content type is taken
// from the response when it labels one, since a preview is a file rather than the raw PCM
// a session streams and the browser has to be told which.
func fetchPreview(
	ctx context.Context,
	httpClient *http.Client,
	provider, url string,
	header http.Header,
	contentType string,
) (Speech, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return Speech{}, stack.Wrap(err)
	}
	if header != nil {
		request.Header = header
	}

	response, err := httpClient.Do(request)
	if err != nil {
		return Speech{}, stack.Wrap(fmt.Errorf("voices: %s preview: %w", provider, err))
	}
	defer response.Body.Close()

	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return Speech{}, refused(provider, response)
	}
	audio, err := io.ReadAll(response.Body)
	if err != nil {
		return Speech{}, stack.Wrap(fmt.Errorf("voices: %s preview: %w", provider, err))
	}
	if len(audio) == 0 {
		return Speech{}, stack.Wrap(fmt.Errorf("voices: %s answered with no audio", provider))
	}
	// Only a label that claims to be audio is believed: ElevenLabs serves its samples from
	// a bucket that calls every one of them text/plain, and a browser handed that plays
	// nothing at all.
	if labelled := response.Header.Get("Content-Type"); strings.HasPrefix(labelled, "audio/") {
		contentType = labelled
	}
	return Speech{Audio: audio, ContentType: contentType}, nil
}

// labelTags collects the labels worth showing beside a voice, skipping the empty ones.
func labelTags(labels map[string]string, keys ...string) []string {
	var tags []string
	for _, key := range keys {
		if value := labels[key]; value != "" {
			tags = append(tags, value)
		}
	}
	return tags
}

// refused turns a non-2xx response into an error that says what the provider said, which
// is the only thing that makes a rejected clone actionable.
func refused(provider string, response *http.Response) error {
	body, _ := io.ReadAll(io.LimitReader(response.Body, 2048))
	return stack.Wrap(fmt.Errorf("voices: %s refused the voice (http %d): %s",
		provider, response.StatusCode, bytes.TrimSpace(body)))
}

// listRefused is refused for a listing. A list names no voice, so a 401 or 403 on one is
// about the key: staging's Inworld key answered the list with a 403 until its scopes were
// widened on 2026-10-19, and "refused the voice" sent people looking for a voice nobody
// had asked for. Any other status is worded as refused words it.
func listRefused(provider string, response *http.Response) error {
	if response.StatusCode != http.StatusUnauthorized && response.StatusCode != http.StatusForbidden {
		return refused(provider, response)
	}
	body, _ := io.ReadAll(io.LimitReader(response.Body, 2048))
	return stack.Wrap(fmt.Errorf("voices: the %s key cannot list voices (http %d): %s",
		provider, response.StatusCode, bytes.TrimSpace(body)))
}
