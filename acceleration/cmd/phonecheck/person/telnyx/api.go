package telnyx

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"
)

// apiBase is a variable so tests can point it at a fake.
var apiBase = "https://api.telnyx.com"

type api struct {
	base, key, account, appID string
	http                      *http.Client
}

// newAPI reads the credentials. The person app is only needed to place a call.
func newAPI() (api, error) {
	key, account := os.Getenv("TELNYX_API_KEY"), os.Getenv("TELNYX_ACCOUNT_SID")
	switch {
	case key == "":
		return api{}, errors.New("TELNYX_API_KEY is not set")
	case account == "":
		return api{}, errors.New("TELNYX_ACCOUNT_SID is not set")
	}
	return api{
		base: apiBase, key: key, account: account, appID: os.Getenv("TELNYX_PERSON_CONNECTION_ID"),
		http: &http.Client{Timeout: 30 * time.Second},
	}, nil
}

// callRecord keeps the few fields of Telnyx's call record the adapter uses.
type callRecord struct {
	Status    string     `json:"status"`
	StartTime telnyxTime `json:"start_time"`
	EndTime   telnyxTime `json:"end_time"`
}

// telnyxTime reads Telnyx's RFC 2822 timestamps; an absent or null one is the zero time.
type telnyxTime struct{ time.Time }

func (t *telnyxTime) UnmarshalJSON(b []byte) error {
	var s *string
	if err := json.Unmarshal(b, &s); err != nil || s == nil || *s == "" {
		return err
	}
	parsed, err := time.Parse(time.RFC1123Z, *s)
	t.Time = parsed
	return err
}

// placeCall starts a call through the person's TeXML app and returns its call sid.
func (a api) placeCall(ctx context.Context, from, to, texml string) (string, error) {
	if a.appID == "" {
		return "", errors.New("TELNYX_PERSON_CONNECTION_ID is not set")
	}
	body, err := json.Marshal(map[string]string{"ApplicationSid": a.appID, "From": from, "To": to, "Texml": texml})
	if err != nil {
		return "", err
	}
	var placed struct {
		CallSID string `json:"call_sid"`
		SID     string `json:"sid"`
	}
	if err := a.do(ctx, http.MethodPost, "", "application/json", bytes.NewReader(body), &placed); err != nil {
		return "", err
	}
	if placed.CallSID != "" {
		return placed.CallSID, nil
	}
	return placed.SID, nil
}

func (a api) call(ctx context.Context, sid string) (callRecord, error) {
	var rec callRecord
	err := a.do(ctx, http.MethodGet, "/"+url.PathEscape(sid), "", nil, &rec)
	return rec, err
}

func (a api) hangup(ctx context.Context, sid string) error {
	form := strings.NewReader(url.Values{"Status": {"completed"}}.Encode())
	return a.do(ctx, http.MethodPost, "/"+url.PathEscape(sid), "application/x-www-form-urlencoded", form, nil)
}

// errNotFound marks a 404, which on a call just placed only means it is not readable yet.
var errNotFound = errors.New("not found")

// do sends one request under the account's TeXML Calls. Errors carry the status and
// Telnyx's message, never the request or the key.
func (a api) do(ctx context.Context, method, path, contentType string, body io.Reader, out any) error {
	target := a.base + "/v2/texml/Accounts/" + a.account + "/Calls" + path
	req, err := http.NewRequestWithContext(ctx, method, target, body)
	if err != nil {
		return fmt.Errorf("telnyx %s calls%s: %w", method, path, err)
	}
	req.Header.Set("Authorization", "Bearer "+a.key)
	req.Header.Set("Accept", "application/json")
	if contentType != "" {
		req.Header.Set("Content-Type", contentType)
	}
	resp, err := a.http.Do(req)
	if err != nil {
		return fmt.Errorf("telnyx %s calls%s: %w", method, path, errors.Unwrap(err))
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode >= 300 {
		if resp.StatusCode == http.StatusNotFound {
			return fmt.Errorf("telnyx %s calls%s: %s: %s: %w", method, path, resp.Status, apiMessage(resp.Body), errNotFound)
		}
		return fmt.Errorf("telnyx %s calls%s: %s: %s", method, path, resp.Status, apiMessage(resp.Body))
	}
	if out == nil {
		return nil
	}
	return json.NewDecoder(resp.Body).Decode(out)
}

func apiMessage(body io.Reader) string {
	var failure struct {
		Errors []struct {
			Detail string `json:"detail"`
		} `json:"errors"`
	}
	_ = json.NewDecoder(io.LimitReader(body, 4096)).Decode(&failure)
	if len(failure.Errors) == 0 {
		return ""
	}
	return failure.Errors[0].Detail
}
