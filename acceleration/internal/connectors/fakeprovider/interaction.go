package fakeprovider

import (
	"bytes"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"io"
	"net/http"
	"strconv"
	"time"
)

// DeliverInteraction posts body to target as Slack posts an interaction, such as a
// block_actions button click or a view_submission, to an app's interactivity Request URL: «in
// the form application/x-www-form-urlencoded», the JSON in a payload parameter
// (https://docs.slack.dev/interactivity/handling-user-interaction, opened October 6, 2026).
// body is the form, already encoded. It is signed as Deliver signs an event; that Slack signs
// an interaction the same way is not on that page or on the request verification page, which
// names the Events API, shortcuts and slash commands. # unverified
//
// It returns the status and body the endpoint answered, and the headers it sent, so a test
// can compare what reached a receiver further on.
func (s *Server) DeliverInteraction(target, signingSecret string, body []byte) (int, string, http.Header) {
	s.t.Helper()
	timestamp := strconv.FormatInt(time.Now().Unix(), 10)
	mac := hmac.New(sha256.New, []byte(signingSecret))
	mac.Write([]byte("v0:" + timestamp + ":"))
	mac.Write(body)
	request, err := http.NewRequest(http.MethodPost, target, bytes.NewReader(body))
	if err != nil {
		s.t.Fatalf("fakeprovider: %v", err)
	}
	request.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	request.Header.Set("X-Slack-Request-Timestamp", timestamp)
	request.Header.Set("X-Slack-Signature", "v0="+hex.EncodeToString(mac.Sum(nil)))
	sent := request.Header.Clone()
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		s.t.Fatalf("fakeprovider: deliver an interaction: %v", err)
	}
	defer response.Body.Close()
	answer, _ := io.ReadAll(response.Body)
	return response.StatusCode, string(answer), sent
}
