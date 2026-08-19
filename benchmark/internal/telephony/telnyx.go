package telephony

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/benchmark/internal/audio"
)

const defaultBaseURL = "https://api.telnyx.com"

// Client talks to Telnyx Call Control.
type Client struct {
	apiKey       string
	connectionID string
	baseURL      string
	http         *http.Client
}

// NewClient reads TELNYX_API_KEY and TELNYX_CONNECTION_ID when unset.
func NewClient() (*Client, error) {
	key := os.Getenv("TELNYX_API_KEY")
	if key == "" {
		return nil, fmt.Errorf("telephony: TELNYX_API_KEY is required")
	}
	conn := os.Getenv("TELNYX_CONNECTION_ID")
	if conn == "" {
		conn = os.Getenv("TELNYX_CALL_CONTROL_APP_ID")
	}
	if conn == "" {
		return nil, fmt.Errorf("telephony: TELNYX_CONNECTION_ID or TELNYX_CALL_CONTROL_APP_ID is required")
	}
	return &Client{
		apiKey:       key,
		connectionID: conn,
		baseURL:      defaultBaseURL,
		http:         &http.Client{Timeout: 30 * time.Second},
	}, nil
}

// DialRequest is an outbound call with a media stream attached.
type DialRequest struct {
	From      string
	To        string
	StreamURL string
}

type envelope[T any] struct {
	Data T `json:"data"`
}

type dialedCall struct {
	CallControlID string `json:"call_control_id"`
	IsAlive       bool   `json:"is_alive"`
}

// Dial places an outbound call and attaches a bidirectional PCMU stream.
func (c *Client) Dial(ctx context.Context, req DialRequest) (string, error) {
	body := map[string]any{
		"connection_id":              c.connectionID,
		"from":                       req.From,
		"to":                         req.To,
		"stream_url":                 req.StreamURL,
		"stream_track":               "inbound_track",
		"stream_bidirectional_mode":  "rtp",
		"stream_bidirectional_codec": "PCMU",
	}
	var response envelope[dialedCall]
	if err := c.do(ctx, http.MethodPost, "/v2/calls", body, &response); err != nil {
		return "", err
	}
	return response.Data.CallControlID, nil
}

// Hangup ends a live call.
func (c *Client) Hangup(ctx context.Context, callControlID string) error {
	if callControlID == "" {
		return nil
	}
	path := "/v2/calls/" + callControlID + "/actions/hangup"
	return c.do(ctx, http.MethodPost, path, map[string]any{}, nil)
}

func (c *Client) do(ctx context.Context, method, path string, body, into any) error {
	var payload io.Reader
	if body != nil {
		encoded, err := json.Marshal(body)
		if err != nil {
			return err
		}
		payload = bytes.NewReader(encoded)
	}
	req, err := http.NewRequestWithContext(ctx, method, c.baseURL+path, payload)
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	resp, err := c.http.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
	if resp.StatusCode >= 300 {
		return fmt.Errorf("telnyx %s %s: HTTP %d: %s", method, path, resp.StatusCode, strings.TrimSpace(string(raw)))
	}
	if into == nil || len(raw) == 0 {
		return nil
	}
	return json.Unmarshal(raw, into)
}

func decodePayload(b64 string) ([]int16, error) {
	raw, err := base64.StdEncoding.DecodeString(b64)
	if err != nil {
		return nil, err
	}
	return audio.DecodeUlaw(raw), nil
}

func encodePayload(ulaw []byte) string {
	return base64.StdEncoding.EncodeToString(ulaw)
}
