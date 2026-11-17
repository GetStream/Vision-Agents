package fakeprovider

import (
	"bytes"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// Slack's pages the channel personality follows, opened October 6, 2026:
//
//	[events]  https://docs.slack.dev/apis/events-api/
//	[verify]  https://docs.slack.dev/authentication/verifying-requests-from-slack
//	[post]    https://docs.slack.dev/reference/methods/chat.postMessage
const (
	// PathChatPostMessage is where SlackChannel takes a reply: Web API methods are POSTed to
	// https://slack.com/api/<method> ([post]).
	PathChatPostMessage = "/api/chat.postMessage"
)

// Post is one message chat.postMessage took.
type Post struct {
	Channel  string
	ThreadTS string
	Text     string
	// Token is the bearer token it was posted with. Compare it with ==, never print it.
	Token string
}

// InstallBot is the bot token a finished install of the app into the workspace TeamID hands
// over, as oauth.v2.access's access_token: a grant of its own with no expiry, since a bot token
// without token rotation does not expire (https://docs.slack.dev/authentication/using-token-rotation).
// It is what a connection's stored credentials hold, without a consent run for it.
func (s *Server) InstallBot() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	token := synthetic("xoxb")
	// A year stands for no expiry: the server's clock never moves that far in a test.
	s.access[token] = &accessToken{grant: &grant{clientID: s.ClientID, account: s.UserID}, expires: s.now().Add(365 * 24 * time.Hour)}
	return token
}

// RevokeBot ends the grant of a token InstallBot issued, as an admin uninstalling the app or
// revoking its token does.
func (s *Server) RevokeBot(token string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if issued := s.access[token]; issued != nil {
		issued.grant.revoked = true
	}
}

// FailPosts has the next n calls to chat.postMessage answer HTTP 503 and post nothing, as a
// provider that is down does (RFC 9110 section 15.6.4).
func (s *Server) FailPosts(n int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.failPosts = n
}

// RefusePosts has the next n calls to chat.postMessage answer HTTP 200 with «"ok": false» and
// the error name code, and post nothing, as Slack refuses a post for a reason other than the
// token, such as not_in_channel, «Cannot post user messages to a channel they are not in»
// ([post], opened October 9, 2026).
func (s *Server) RefusePosts(n int, code string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.refusedPosts, s.refusal = n, code
}

// GarblePosts has the next n calls to chat.postMessage post their message and answer HTTP 200
// with a body that is not JSON, as an answer cut off or rewritten on its way back would be.
func (s *Server) GarblePosts(n int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.garbledPosts = n
}

// Posts are the messages chat.postMessage took, oldest first.
func (s *Server) Posts() []Post {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]Post(nil), s.posts...)
}

// Deliver posts body to target as Slack delivers an Events API event to an app's Request
// URL, signed with signingSecret ([verify]: v0= and the hex HMAC-SHA256 of
// v0:{timestamp}:{body}, the timestamp in X-Slack-Request-Timestamp). retry is the attempt
// number Slack sends as X-Slack-Retry-Num on a retry, 1 to 3, with X-Slack-Retry-Reason;
// 0 is the first delivery, which carries neither ([events], «Retries»). It returns the
// status and body the endpoint answered.
func (s *Server) Deliver(target, signingSecret string, body []byte, retry int) (int, string) {
	s.t.Helper()
	timestamp := strconv.FormatInt(time.Now().Unix(), 10)
	mac := hmac.New(sha256.New, []byte(signingSecret))
	mac.Write([]byte("v0:" + timestamp + ":"))
	mac.Write(body)
	request, err := http.NewRequest(http.MethodPost, target, bytes.NewReader(body))
	if err != nil {
		s.t.Fatalf("fakeprovider: %v", err)
	}
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("X-Slack-Request-Timestamp", timestamp)
	request.Header.Set("X-Slack-Signature", "v0="+hex.EncodeToString(mac.Sum(nil)))
	if retry > 0 {
		request.Header.Set("X-Slack-Retry-Num", strconv.Itoa(retry))
		// [events]: one of the reasons Slack names.
		request.Header.Set("X-Slack-Retry-Reason", "http_timeout")
	}
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		s.t.Fatalf("fakeprovider: deliver: %v", err)
	}
	defer response.Body.Close()
	answer, _ := io.ReadAll(response.Body)
	return response.StatusCode, string(answer)
}

// chatPostMessage answers as Slack's chat.postMessage does ([post]): JSON in, the token as a
// bearer header, and every answer HTTP 200, with ok false and an error name for a refusal.
func (s *Server) chatPostMessage(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.is(SlackChannel) {
		http.NotFound(w, r)
		return
	}
	if s.failPosts > 0 {
		s.failPosts--
		w.WriteHeader(http.StatusServiceUnavailable)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	if s.refusedPosts > 0 {
		s.refusedPosts--
		_ = json.NewEncoder(w).Encode(map[string]any{"ok": false, "error": s.refusal})
		return
	}
	presented, found := strings.CutPrefix(r.Header.Get("Authorization"), "Bearer ")
	if !found {
		// [post] error names: not_authed is «No authentication token provided».
		_ = json.NewEncoder(w).Encode(map[string]any{"ok": false, "error": "not_authed"})
		return
	}
	token := s.access[presented]
	if token == nil || token.grant.revoked || s.now().After(token.expires) {
		// [post]: invalid_auth, «Some aspect of authentication cannot be validated».
		_ = json.NewEncoder(w).Encode(map[string]any{"ok": false, "error": "invalid_auth"})
		return
	}
	var sent struct {
		Channel  string `json:"channel"`
		ThreadTS string `json:"thread_ts"`
		Text     string `json:"text"`
	}
	if err := json.NewDecoder(r.Body).Decode(&sent); err != nil || sent.Channel == "" {
		// [post]: channel_not_found, «Value passed for channel was invalid».
		_ = json.NewEncoder(w).Encode(map[string]any{"ok": false, "error": "channel_not_found"})
		return
	}
	s.posts = append(s.posts, Post{Channel: sent.Channel, ThreadTS: sent.ThreadTS, Text: sent.Text, Token: presented})
	if s.garbledPosts > 0 {
		s.garbledPosts--
		_, _ = w.Write([]byte("<html>posted</html>"))
		return
	}
	ts := strconv.FormatInt(s.now().Unix(), 10) + "." + syntheticDigits(6)
	// [post]'s example answer: the message as posted, with the bot's id, which is how its
	// own message comes back as an event the bridge skips (bot_id).
	_ = json.NewEncoder(w).Encode(map[string]any{
		"ok": true, "channel": sent.Channel, "ts": ts,
		"message": map[string]any{"type": "message", "text": sent.Text, "bot_id": "B" + strings.ToUpper(synthetic("bot")[4:14]), "ts": ts},
	})
}
