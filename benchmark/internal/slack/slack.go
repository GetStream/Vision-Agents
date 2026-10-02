// Package slack posts a Voicebench digest to a channel with a bot token.
package slack

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
)

const defaultBaseURL = "https://slack.com/api"

// Client posts as a Slack app's bot. The app needs chat:write and files:write, and has to
// be in the channel.
type Client struct {
	Token string
	// BaseURL is Slack's Web API, and is only set by tests.
	BaseURL string
	HTTP    *http.Client
}

// File is something to attach to the message.
type File struct {
	Name  string
	Title string
	Data  []byte
}

// Post sends text to the channel with the files attached to the same message.
//
// Slack uploads in three steps: ask for an upload URL per file, send the bytes there,
// then complete the upload, which is what shares the files and posts the text with them.
func (c Client) Post(ctx context.Context, channel, text string, files []File) error {
	if c.Token == "" {
		return errors.New("slack: a bot token is required")
	}
	if channel == "" {
		return errors.New("slack: a channel id is required")
	}
	if len(files) == 0 {
		var posted struct{}
		return c.call(ctx, "chat.postMessage", url.Values{"channel": {channel}, "text": {text}}, &posted)
	}

	type uploaded struct {
		ID    string `json:"id"`
		Title string `json:"title"`
	}
	var done []uploaded
	for _, file := range files {
		var ticket struct {
			UploadURL string `json:"upload_url"`
			FileID    string `json:"file_id"`
		}
		form := url.Values{"filename": {file.Name}, "length": {strconv.Itoa(len(file.Data))}}
		if err := c.call(ctx, "files.getUploadURLExternal", form, &ticket); err != nil {
			return err
		}
		if err := c.upload(ctx, ticket.UploadURL, file.Data); err != nil {
			return fmt.Errorf("slack: upload %s: %w", file.Name, err)
		}
		done = append(done, uploaded{ID: ticket.FileID, Title: file.Title})
	}
	listed, err := json.Marshal(done)
	if err != nil {
		return err
	}
	var completed struct{}
	return c.call(ctx, "files.completeUploadExternal", url.Values{
		"files":           {string(listed)},
		"channel_id":      {channel},
		"initial_comment": {text},
	}, &completed)
}

// call posts a form to a Web API method and decodes the answer into out. Slack answers
// 200 for its own errors, so ok is what says whether the call worked.
func (c Client) call(ctx context.Context, method string, form url.Values, out any) error {
	base := c.BaseURL
	if base == "" {
		base = defaultBaseURL
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, base+"/"+method, strings.NewReader(form.Encode()))
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+c.Token)
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	resp, err := c.client().Do(req)
	if err != nil {
		return fmt.Errorf("slack: %s: %w", method, err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("slack: %s: %w", method, err)
	}
	var status struct {
		OK     bool   `json:"ok"`
		Error  string `json:"error"`
		Needed string `json:"needed"`
	}
	if err := json.Unmarshal(raw, &status); err != nil {
		return fmt.Errorf("slack: %s: HTTP %d: %s", method, resp.StatusCode, raw)
	}
	if !status.OK {
		if status.Needed != "" {
			return fmt.Errorf("slack: %s: %s (the app needs the %s scope)", method, status.Error, status.Needed)
		}
		return fmt.Errorf("slack: %s: %s", method, status.Error)
	}
	return json.Unmarshal(raw, out)
}

func (c Client) upload(ctx context.Context, target string, data []byte) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, target, bytes.NewReader(data))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/octet-stream")
	resp, err := c.client().Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("HTTP %d: %s", resp.StatusCode, body)
	}
	return nil
}

func (c Client) client() *http.Client {
	if c.HTTP != nil {
		return c.HTTP
	}
	return http.DefaultClient
}
