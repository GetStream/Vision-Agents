package main

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
	"time"
)

const telnyxAPI = "https://api.telnyx.com"

// TelnyxCaller places a Telnyx call the way the bench needs it: one leg to the bench, one
// to the number called, bridged together at Telnyx.
//
// The phone package's Telnyx Dial sends the bridge address as link_to, which Telnyx reads
// as another call's control id, so the called number rings and nothing is ever bridged.
// Call Control does have what is needed, in three requests rather than one: dial the bench,
// dial the number linked to that call, and bridge them once both have answered.
type TelnyxCaller struct {
	apiKey string
	// app is the Call Control Application calls are placed on.
	app    string
	client *http.Client
}

// NewTelnyxCaller reads TELNYX_API_KEY and TELNYX_CALL_CONTROL_ID, falling back to
// TELNYX_CONNECTION_ID for an account whose one connection does both.
func NewTelnyxCaller(client *http.Client) (*TelnyxCaller, error) {
	caller := &TelnyxCaller{
		apiKey: os.Getenv("TELNYX_API_KEY"),
		app:    os.Getenv("TELNYX_CALL_CONTROL_ID"),
		client: client,
	}
	if caller.app == "" {
		caller.app = os.Getenv("TELNYX_CONNECTION_ID")
	}
	if caller.apiKey == "" || caller.app == "" {
		return nil, errors.New("telnyx: TELNYX_API_KEY and TELNYX_CALL_CONTROL_ID are required to place calls")
	}
	return caller, nil
}

// Dial calls to from from and returns the call's control id. linkTo, when set, puts the
// call in the same session as that one, which is what lets the two be bridged.
func (c *TelnyxCaller) Dial(ctx context.Context, from, to, linkTo string, timeout time.Duration) (string, error) {
	request := map[string]any{
		"connection_id": c.app,
		"from":          from,
		"to":            to,
		"timeout_secs":  int(timeout.Seconds()),
	}
	if linkTo != "" {
		request["link_to"] = linkTo
	}
	var response struct {
		Data struct {
			CallControlID string `json:"call_control_id"`
		} `json:"data"`
	}
	if err := c.post(ctx, "/v2/calls", request, &response); err != nil {
		return "", err
	}
	return response.Data.CallControlID, nil
}

// Bridge joins two answered calls' audio.
func (c *TelnyxCaller) Bridge(ctx context.Context, callControlID, other string) error {
	path := "/v2/calls/" + url.PathEscape(callControlID) + "/actions/bridge"
	return c.post(ctx, path, map[string]any{"call_control_id": other}, nil)
}

// Hangup ends a call, for one the bench gave up on before it was bridged.
func (c *TelnyxCaller) Hangup(ctx context.Context, callControlID string) error {
	return c.post(ctx, "/v2/calls/"+url.PathEscape(callControlID)+"/actions/hangup", map[string]any{}, nil)
}

func (c *TelnyxCaller) post(ctx context.Context, path string, body, into any) error {
	encoded, err := json.Marshal(body)
	if err != nil {
		return fmt.Errorf("telnyx %s: %w", path, err)
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, telnyxAPI+path, bytes.NewReader(encoded))
	if err != nil {
		return fmt.Errorf("telnyx %s: %w", path, err)
	}
	request.Header.Set("Authorization", "Bearer "+c.apiKey)
	request.Header.Set("Content-Type", "application/json")

	response, err := c.client.Do(request)
	if err != nil {
		return fmt.Errorf("telnyx %s: %w", path, err)
	}
	defer response.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(response.Body, 1<<20))
	if err != nil {
		return fmt.Errorf("telnyx %s: %w", path, err)
	}
	if response.StatusCode >= 300 {
		return fmt.Errorf("telnyx %s: %s: %s", path, response.Status, raw)
	}
	if into == nil {
		return nil
	}
	if err := json.Unmarshal(raw, into); err != nil {
		return fmt.Errorf("telnyx %s: %w", path, err)
	}
	return nil
}
