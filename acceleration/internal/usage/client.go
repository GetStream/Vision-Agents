package usage

import (
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"slices"
	"strings"
)

// The service's paths.
const (
	PathEvents    = "/v1/events"
	PathWindows   = "/v1/windows"
	PathStandings = "/v1/standings"
	// PathManage prefixes the management API, which takes the manage key and never the
	// ingest token.
	PathManage = "/v1/manage"
)

const (
	// maxErrorBody is how much of a failed answer an error repeats.
	maxErrorBody = 1 << 10
	// shipBytes is the most event JSON one batch carries, a quarter under MaxBodyBytes.
	shipBytes = MaxBodyBytes * 3 / 4
	// batchOpen and batchClose wrap a batch's events in its body.
	batchOpen  = `{"events":[`
	batchClose = "]}"
)

// Client calls the usage service. It is safe for concurrent use.
type Client struct {
	url   string
	token string
	http  *http.Client
}

// ClientOptions configure a Client.
type ClientOptions struct {
	// URL is the service's base URL, such as http://usage:8070.
	URL string
	// Token is the service's ingest token.
	Token string
	// HTTP makes the calls; nil is http.DefaultClient. Bound each call with its context.
	HTTP *http.Client
}

// NewClient returns a client of the service at options.URL.
func NewClient(options ClientOptions) (*Client, error) {
	if options.URL == "" || options.Token == "" {
		return nil, errors.New("usage: a client needs the service's URL and its ingest token")
	}
	if options.HTTP == nil {
		options.HTTP = http.DefaultClient
	}
	return &Client{url: strings.TrimRight(options.URL, "/"), token: options.Token, http: options.HTTP}, nil
}

// StatusError is an answer of the service other than 200. A 5xx is the service failing, and
// the same call is safe to make again; a 4xx is the caller's mistake.
type StatusError struct {
	Status int
	Body   string
}

func (e *StatusError) Error() string {
	return fmt.Sprintf("usage: the service answered %d: %s", e.Status, e.Body)
}

// Ship sends events, in batches of at most MaxEvents and shipBytes, and sums their acks: a
// rejection's index is into events, and the ack is mirrored when every batch was. An event
// bigger than shipBytes goes alone, and one no body can hold is not sent but rejected here
// as ReasonTooLarge, since the service would refuse it, and every retry, with a 413. On an
// error some batches may have landed; shipping all of them again is safe, since the service
// records a usage id once.
func (c *Client) Ship(ctx context.Context, events []Event) (Ack, error) {
	ack := Ack{Rejected: []Rejection{}, Mirrored: true}
	var batch [][]byte
	// positions is each batched event's index in events.
	var positions []int
	size := 0
	send := func() error {
		if len(batch) == 0 {
			return nil
		}
		payload := append([]byte(batchOpen), bytes.Join(batch, []byte(","))...)
		var part Ack
		if err := c.post(ctx, PathEvents, append(payload, batchClose...), &part); err != nil {
			return err
		}
		ack.Accepted += part.Accepted
		ack.Duplicates += part.Duplicates
		ack.Mirrored = ack.Mirrored && part.Mirrored
		for _, rejection := range part.Rejected {
			rejection.Index = positions[rejection.Index]
			ack.Rejected = append(ack.Rejected, rejection)
		}
		batch, positions, size = batch[:0], positions[:0], 0
		return nil
	}
	for i, event := range events {
		encoded, err := json.Marshal(event)
		if err != nil {
			return Ack{}, fmt.Errorf("usage: encode %s: %w", PathEvents, err)
		}
		if len(batchOpen)+len(encoded)+len(batchClose) > MaxBodyBytes {
			ack.Rejected = append(ack.Rejected, Rejection{
				Index: i, UsageID: event.UsageID, Reason: ReasonTooLarge,
				Message: fmt.Sprintf("the event is %d bytes of JSON, and a batch at most %d: not sent", len(encoded), MaxBodyBytes),
			})
			continue
		}
		if len(batch) == MaxEvents || (len(batch) > 0 && size+len(encoded)+1 > shipBytes) {
			if err := send(); err != nil {
				return Ack{}, err
			}
		}
		batch = append(batch, encoded)
		positions = append(positions, i)
		size += len(encoded) + 1
	}
	if err := send(); err != nil {
		return Ack{}, err
	}
	slices.SortFunc(ack.Rejected, func(a, b Rejection) int { return cmp.Compare(a.Index, b.Index) })
	return ack, nil
}

// Windows reads what the subjects spent in their current windows, in calls of at most
// MaxSubjects subjects.
func (c *Client) Windows(ctx context.Context, subjects []SubjectRef) ([]WindowSpend, error) {
	var spends []WindowSpend
	for start := 0; start < len(subjects); start += MaxSubjects {
		var answer Windows
		if err := c.call(ctx, PathWindows, WindowsQuery{Subjects: subjects[start:min(start+MaxSubjects, len(subjects))]}, &answer); err != nil {
			return nil, err
		}
		spends = append(spends, answer.Windows...)
	}
	return spends, nil
}

// Standings reads the organizations' standings, in calls of at most MaxOrganizations.
func (c *Client) Standings(ctx context.Context, query StandingQuery) ([]Standing, error) {
	var standings []Standing
	organizations := query.Organizations
	for start := 0; start < len(organizations); start += MaxOrganizations {
		var answer Standings
		part := StandingQuery{Organizations: organizations[start:min(start+MaxOrganizations, len(organizations))]}
		if err := c.call(ctx, PathStandings, part, &answer); err != nil {
			return nil, err
		}
		standings = append(standings, answer.Standings...)
	}
	return standings, nil
}

// call posts body to path and decodes a 200 into answer.
func (c *Client) call(ctx context.Context, path string, body, answer any) error {
	payload, err := json.Marshal(body)
	if err != nil {
		return fmt.Errorf("usage: encode %s: %w", path, err)
	}
	return c.post(ctx, path, payload, answer)
}

// post posts payload to path and decodes a 200 into answer.
func (c *Client) post(ctx context.Context, path string, payload []byte, answer any) error {
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, c.url+path, bytes.NewReader(payload))
	if err != nil {
		return fmt.Errorf("usage: %s: %w", path, err)
	}
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Authorization", "Bearer "+c.token)
	response, err := c.http.Do(request)
	if err != nil {
		return fmt.Errorf("usage: %s: %w", path, err)
	}
	defer response.Body.Close() //nolint:errcheck // a read-only body
	if response.StatusCode != http.StatusOK {
		text, _ := io.ReadAll(io.LimitReader(response.Body, maxErrorBody))
		return &StatusError{Status: response.StatusCode, Body: strings.TrimSpace(string(text))}
	}
	if err := json.NewDecoder(response.Body).Decode(answer); err != nil {
		return fmt.Errorf("usage: decode %s: %w", path, err)
	}
	return nil
}
