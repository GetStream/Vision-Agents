package egress

import (
	"bytes"
	"encoding/base64"
	"io"
	"strings"
)

// authorization is an Authorization header value whose credential is a placeholder, in one
// of the forms code sends a token in: Bearer <placeholder>, token <placeholder> (GitHub's,
// https://docs.github.com/en/rest/authentication/authenticating-to-the-rest-api), or Basic
// whose password is the placeholder, as git sends it.
type authorization struct {
	scheme string
	// user is Basic's user name; basic is false for the other two.
	user  string
	basic bool
}

// placed reads value as an authorization carrying placeholder. Anything else, the
// placeholder elsewhere in it included, is not one.
func placed(value, placeholder string) (authorization, bool) {
	scheme, credential, _ := strings.Cut(value, " ")
	switch {
	case strings.EqualFold(scheme, "Bearer"), strings.EqualFold(scheme, "token"):
		return authorization{scheme: scheme}, credential == placeholder
	case strings.EqualFold(scheme, "Basic"):
		raw, err := base64.StdEncoding.DecodeString(credential)
		user, password, _ := strings.Cut(string(raw), ":")
		return authorization{scheme: scheme, user: user, basic: true}, err == nil && password == placeholder
	}
	return authorization{}, false
}

// credential is what follows the scheme with secret in the placeholder's place.
func (a authorization) credential(secret string) string {
	if a.basic {
		return base64.StdEncoding.EncodeToString([]byte(a.user + ":" + secret))
	}
	return secret
}

// rewriting is a response body with every olds[i] replaced by news[i] as it streams, so a
// host echoing the request gives the sandbox back its placeholder, not the token.
type rewriting struct {
	body       io.ReadCloser
	olds, news [][]byte
	// longest is the longest of olds: a shorter tail of what was read may be the start of
	// one, so it waits for the next read.
	longest int
	buf     []byte
	in, out []byte
	ended   bool
}

func newRewriting(body io.ReadCloser, pairs ...string) *rewriting {
	r := &rewriting{body: body, buf: make([]byte, 32<<10)}
	for i := 0; i < len(pairs); i += 2 {
		r.olds, r.news = append(r.olds, []byte(pairs[i])), append(r.news, []byte(pairs[i+1]))
		r.longest = max(r.longest, len(pairs[i]))
	}
	return r
}

func (r *rewriting) Read(p []byte) (int, error) {
	for len(r.out) == 0 {
		if r.ended {
			return 0, io.EOF
		}
		n, err := r.body.Read(r.buf)
		r.in = append(r.in, r.buf[:n]...)
		if err == io.EOF {
			r.ended = true
		} else if err != nil {
			return 0, err
		}
		r.rewrite()
	}
	n := copy(p, r.out)
	r.out = r.out[n:]
	return n, nil
}

// rewrite moves what was read to out, replaced, and keeps back a tail that may be the start
// of an old until the body ends.
func (r *rewriting) rewrite() {
	for {
		at, which := -1, 0
		for i, old := range r.olds {
			if found := bytes.Index(r.in, old); found >= 0 && (at < 0 || found < at) {
				at, which = found, i
			}
		}
		if at < 0 {
			break
		}
		r.out = append(append(r.out, r.in[:at]...), r.news[which]...)
		r.in = r.in[at+len(r.olds[which]):]
	}
	keep := 0
	if !r.ended {
		keep = min(len(r.in), r.longest-1)
	}
	r.out = append(r.out, r.in[:len(r.in)-keep]...)
	r.in = append([]byte(nil), r.in[len(r.in)-keep:]...)
}

func (r *rewriting) Close() error {
	return r.body.Close()
}
