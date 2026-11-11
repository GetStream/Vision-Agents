package limitnotice

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"time"
)

// slackAPI is Slack's Web API.
const slackAPI = "https://slack.com/api"

// Slack posts to one channel with a bot token holding chat:write, as a member of the channel.
type Slack struct {
	token   string
	channel string
	baseURL string
	client  *http.Client
}

// NewSlack posts to channel, an id such as C0123456789, with token.
func NewSlack(token, channel string) *Slack {
	return NewSlackAt(token, channel, slackAPI)
}

// NewSlackAt is NewSlack against baseURL, for a Slack stand-in in tests.
func NewSlackAt(token, channel, baseURL string) *Slack {
	return &Slack{token: token, channel: channel, baseURL: baseURL, client: &http.Client{Timeout: 10 * time.Second}}
}

// Post sends text with chat.postMessage.
func (s *Slack) Post(ctx context.Context, text string) error {
	body, err := json.Marshal(map[string]any{"channel": s.channel, "text": text, "unfurl_links": false})
	if err != nil {
		return err
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, s.baseURL+"/chat.postMessage", bytes.NewReader(body))
	if err != nil {
		return err
	}
	request.Header.Set("Content-Type", "application/json; charset=utf-8")
	request.Header.Set("Authorization", "Bearer "+s.token)
	response, err := s.client.Do(request)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	var answer struct {
		OK    bool   `json:"ok"`
		Error string `json:"error"`
	}
	if err := json.NewDecoder(io.LimitReader(response.Body, 1<<16)).Decode(&answer); err != nil {
		return fmt.Errorf("slack: chat.postMessage answered %s", response.Status)
	}
	if !answer.OK {
		return errors.New("slack: chat.postMessage: " + answer.Error)
	}
	return nil
}
