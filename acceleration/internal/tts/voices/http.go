package voices

import (
	"bytes"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"time"
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
	return f.writer.WriteField(name, value)
}

// file adds a recording under a field name.
func (f *form) file(field string, sample Sample) error {
	part, err := f.writer.CreateFormFile(field, sample.Name)
	if err != nil {
		return err
	}
	_, err = part.Write(sample.Content)
	return err
}

// done closes the body and reports the content type that describes it.
func (f *form) done() (io.Reader, string, error) {
	if err := f.writer.Close(); err != nil {
		return nil, "", err
	}
	return &f.body, f.writer.FormDataContentType(), nil
}

// client is the HTTP client a cloner uses. It is a field rather than the default so a test
// can point one at a server of its own.
func client(timeout time.Duration) *http.Client {
	if timeout == 0 {
		timeout = defaultTimeout
	}
	return &http.Client{Timeout: timeout}
}

// refused turns a non-2xx response into an error that says what the provider said, which
// is the only thing that makes a rejected clone actionable.
func refused(provider string, response *http.Response) error {
	body, _ := io.ReadAll(io.LimitReader(response.Body, 2048))
	return fmt.Errorf("voices: %s refused the voice (http %d): %s",
		provider, response.StatusCode, bytes.TrimSpace(body))
}
