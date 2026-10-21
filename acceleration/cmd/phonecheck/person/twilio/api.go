package twilio

import (
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
var apiBase = "https://api.twilio.com"

type api struct {
	base, sid, token string
	http             *http.Client
}

func newAPI() (api, error) {
	sid, token := os.Getenv("TWILIO_ACCOUNT_SID"), os.Getenv("TWILIO_AUTH_TOKEN")
	switch {
	case sid == "":
		return api{}, errors.New("TWILIO_ACCOUNT_SID is not set")
	case token == "":
		return api{}, errors.New("TWILIO_AUTH_TOKEN is not set")
	}
	return api{base: apiBase, sid: sid, token: token, http: &http.Client{Timeout: 30 * time.Second}}, nil
}

// callRecord keeps the few fields of Twilio's call record the adapter uses.
type callRecord struct {
	SID       string     `json:"sid"`
	Status    string     `json:"status"`
	StartTime twilioTime `json:"start_time"`
	EndTime   twilioTime `json:"end_time"`
}

// twilioTime reads Twilio's RFC 2822 timestamps; an absent or null one is the zero time.
type twilioTime struct{ time.Time }

func (t *twilioTime) UnmarshalJSON(b []byte) error {
	var s *string
	if err := json.Unmarshal(b, &s); err != nil || s == nil || *s == "" {
		return err
	}
	parsed, err := time.Parse(time.RFC1123Z, *s)
	t.Time = parsed
	return err
}

func (a api) placeCall(ctx context.Context, from, to, twiml string) (string, error) {
	var created callRecord
	form := url.Values{"From": {from}, "To": {to}, "Twiml": {twiml}}
	if err := a.do(ctx, http.MethodPost, "/Calls.json", nil, form, &created); err != nil {
		return "", err
	}
	return created.SID, nil
}

func (a api) call(ctx context.Context, sid string) (callRecord, error) {
	var rec callRecord
	err := a.do(ctx, http.MethodGet, "/Calls/"+sid+".json", nil, nil, &rec)
	return rec, err
}

// callsTo lists the calls to a number that started at or after since. Twilio filters by
// date only, so the time is filtered here.
func (a api) callsTo(ctx context.Context, e164 string, since time.Time) ([]callRecord, error) {
	query := url.Values{"To": {e164}, "StartTime>=": {since.UTC().Format(time.DateOnly)}}
	var list struct {
		Calls []callRecord `json:"calls"`
	}
	if err := a.do(ctx, http.MethodGet, "/Calls.json", query, nil, &list); err != nil {
		return nil, err
	}
	// Twilio stamps whole seconds, so a call started just after since can read as just before.
	from := since.Truncate(time.Second)
	var recent []callRecord
	for _, rec := range list.Calls {
		if !rec.StartTime.Before(from) {
			recent = append(recent, rec)
		}
	}
	return recent, nil
}

func (a api) complete(ctx context.Context, sid string) error {
	return a.do(ctx, http.MethodPost, "/Calls/"+sid+".json", nil, url.Values{"Status": {"completed"}}, nil)
}

// do sends one request under the account's Calls. Errors carry the status and Twilio's
// message, never the request or its credentials.
func (a api) do(ctx context.Context, method, path string, query, form url.Values, out any) error {
	target := a.base + "/2010-04-01/Accounts/" + a.sid + path
	if len(query) > 0 {
		target += "?" + query.Encode()
	}
	var body io.Reader
	if form != nil {
		body = strings.NewReader(form.Encode())
	}
	req, err := http.NewRequestWithContext(ctx, method, target, body)
	if err != nil {
		return fmt.Errorf("twilio %s %s: %w", method, path, err)
	}
	req.SetBasicAuth(a.sid, a.token)
	if form != nil {
		req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	}
	resp, err := a.http.Do(req)
	if err != nil {
		return fmt.Errorf("twilio %s %s: %w", method, path, errors.Unwrap(err))
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode >= 300 {
		return fmt.Errorf("twilio %s %s: %s: %s", method, path, resp.Status, apiMessage(resp.Body))
	}
	if out == nil {
		return nil
	}
	return json.NewDecoder(resp.Body).Decode(out)
}

func apiMessage(body io.Reader) string {
	var failure struct {
		Message string `json:"message"`
	}
	_ = json.NewDecoder(io.LimitReader(body, 4096)).Decode(&failure)
	return failure.Message
}
