// Package keypool spreads a provider's requests across several API keys, so one key that
// is rate limited does not take the provider down with it.
//
// Keys come from the variable a provider already reads, comma separated, and from the same
// name with a number after it: OPENAI_API_KEY=k1,k2 and OPENAI_API_KEY_1=k3 make a pool of
// three. Requests take the keys in turn. A key answered with 429 is put on cooldown for as
// long as Retry-After asks, and one whose quota is spent for MaxCooldown, and the request is
// sent again at once with the next key; only when every key has been refused does the
// refusal reach the router, which then fails over to another provider as it always did.
//
// A pool is shared by every provider built from the same keys, so a key one session wore
// out is not handed to the next.
package keypool

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"os"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"
)

// DefaultCooldown is how long a key rests after a 429 that does not say for how long.
const DefaultCooldown = 30 * time.Second

// MaxCooldown caps what a Retry-After can ask for, so one bad header cannot retire a key
// for the day.
const MaxCooldown = 15 * time.Minute

// drainLimit is how much of a refused response is read before it is closed, so its
// connection can be reused.
const drainLimit = 4 << 10

// Parse splits each value on commas and returns the keys in order, trimmed, without blanks
// or repeats.
func Parse(raw ...string) []string {
	var keys []string
	for _, value := range raw {
		for key := range strings.SplitSeq(value, ",") {
			key = strings.TrimSpace(key)
			if key != "" && !slices.Contains(keys, key) {
				keys = append(keys, key)
			}
		}
	}
	return keys
}

// Env reads the keys of the first of names that holds any: NAME itself, then NAME_1,
// NAME_2 and on in numeric order.
func Env(names ...string) []string {
	for _, name := range names {
		if name == "" {
			continue
		}
		if keys := Parse(numbered(name)...); len(keys) > 0 {
			return keys
		}
	}
	return nil
}

// Getenv is Env joined with commas, for a provider that keeps its key in a string.
func Getenv(names ...string) string {
	return strings.Join(Env(names...), ",")
}

// numbered is NAME followed by the values of NAME_<n>, in the order of n.
func numbered(name string) []string {
	type entry struct {
		n     int
		value string
	}
	var entries []entry
	prefix := name + "_"
	for _, pair := range os.Environ() {
		key, value, _ := strings.Cut(pair, "=")
		suffix, ok := strings.CutPrefix(key, prefix)
		if !ok {
			continue
		}
		n, err := strconv.Atoi(suffix)
		if err != nil || n < 0 || strconv.Itoa(n) != suffix {
			continue
		}
		entries = append(entries, entry{n: n, value: value})
	}
	slices.SortFunc(entries, func(a, b entry) int { return a.n - b.n })

	values := []string{os.Getenv(name)}
	for _, e := range entries {
		values = append(values, e.value)
	}
	return values
}

var (
	sharedMu sync.Mutex
	shared   = map[string]*Pool{}
)

// quotaExceededStatus is the status label of a refusal that came as a websocket frame.
const quotaExceededStatus = "quota_exceeded"

// FromEnv is the shared pool for explicit when it names any key, or else for the keys Env
// finds under envVars. The first of envVars names the pool's key variable in its metrics.
func FromEnv(explicit string, envVars ...string) *Pool {
	keys := Parse(explicit)
	if len(keys) == 0 {
		keys = Env(envVars...)
	}
	id := strings.Join(keys, ",")

	sharedMu.Lock()
	defer sharedMu.Unlock()
	pool, ok := shared[id]
	if !ok {
		pool = &Pool{ring: newRing(keys)}
		shared[id] = pool
	}
	if len(envVars) > 0 {
		pool.name(envVars[0])
	}
	return pool
}

// New returns a pool of its own over the keys in raw, shared with nobody.
func New(raw ...string) *Pool {
	return &Pool{ring: newRing(Parse(raw...))}
}

// Pool is a set of keys for one provider, taken in turn and rested after a 429.
type Pool struct {
	*ring
	// pin is set on a pool made by Pin.
	pin *pin
}

// ring is the state every pool over the same keys shares: whose turn it is and which keys
// are resting.
type ring struct {
	keys []string
	now  func() time.Time

	mu   sync.Mutex
	next int
	// until is when each key's cooldown ends, zero for a key that is not resting.
	until []time.Time
	// keyVar is the variable the keys are read from, "" until a FromEnv names one.
	keyVar string
}

type pin struct {
	mu  sync.Mutex
	key string
}

func newRing(keys []string) *ring {
	return &ring{keys: keys, now: time.Now, until: make([]time.Time, len(keys))}
}

// Pin is a view of the pool that keeps to one key, taken in turn the first time it is
// asked for, and moves to the next only once that key is resting. It shares the pool's
// turn and cooldowns.
//
// It is for a session whose requests build on state the provider keeps under the key that
// made it: a stored response continued on another project's key is not found, and a
// resumed socket is refused.
func (p *Pool) Pin() *Pool {
	return &Pool{ring: p.ring, pin: &pin{}}
}

// Len is how many keys the pool holds.
func (p *Pool) Len() int { return len(p.keys) }

// Peek is the first key, without taking a turn, or "" for an empty pool.
func (p *Pool) Peek() string {
	if len(p.keys) == 0 {
		return ""
	}
	return p.keys[0]
}

// Next takes the next key that is not resting, or for a pin the key it keeps to while
// that key is not resting. When every key is, it is the one whose cooldown ends first: a
// request sent on it may still go through, and failing it here would only fail it sooner.
// It is "" for an empty pool.
func (p *Pool) Next() string {
	if p.pin == nil {
		return p.ring.take()
	}
	p.pin.mu.Lock()
	defer p.pin.mu.Unlock()
	if p.pin.key == "" || p.ring.resting(p.pin.key) {
		p.pin.key = p.ring.take()
	}
	return p.pin.key
}

func (r *ring) take() string {
	if len(r.keys) == 0 {
		return ""
	}
	r.mu.Lock()
	defer r.mu.Unlock()

	now := r.now()
	soonest := -1
	for range r.keys {
		index := r.next
		r.next = (r.next + 1) % len(r.keys)
		if !r.until[index].After(now) {
			return r.keys[index]
		}
		if soonest < 0 || r.until[index].Before(r.until[soonest]) {
			soonest = index
		}
	}
	return r.keys[soonest]
}

// name gives the ring its key variable unless it has one. A pool is shared by key set, so
// one built from keys passed in (the Gemini LLM is given GOOGLE_API_KEY's keys) takes its
// name from the provider that reads the same keys from their variable, whichever is built
// first.
func (r *ring) name(keyVar string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.keyVar == "" {
		r.keyVar = keyVar
	}
}

// countRefused counts a refused attempt under the pool's key variable, never its key.
func (r *ring) countRefused(status string) {
	r.mu.Lock()
	keyVar := r.keyVar
	r.mu.Unlock()
	if keyVar == "" {
		keyVar = unnamedKeyVar
	}
	refusedAttempts.WithLabelValues(keyVar, status).Inc()
}

func (r *ring) resting(key string) bool {
	index := slices.Index(r.keys, key)
	if index < 0 {
		return false
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.until[index].After(r.now())
}

// Cooldown rests key for d, at most MaxCooldown. A key resting longer already keeps its
// longer rest.
func (p *Pool) Cooldown(key string, d time.Duration) {
	index := slices.Index(p.keys, key)
	if index < 0 || d <= 0 {
		return
	}
	until := p.now().Add(min(d, MaxCooldown))

	p.mu.Lock()
	defer p.mu.Unlock()
	if until.After(p.until[index]) {
		p.until[index] = until
	}
}

// QuotaExceeded rests key for as long as a pool can, after a websocket frame said its quota
// is spent, and counts it as refused under the status "quota_exceeded": the frame has no
// HTTP status.
func (p *Pool) QuotaExceeded(key string) {
	p.countRefused(quotaExceededStatus)
	p.Cooldown(key, MaxCooldown)
}

// CooldownFromResponse rests key for as long as the response's Retry-After asks, or for
// DefaultCooldown when it does not say.
func (p *Pool) CooldownFromResponse(key string, response *http.Response) {
	p.Cooldown(key, p.retryAfter(response))
}

func (p *Pool) retryAfter(response *http.Response) time.Duration {
	if response == nil {
		return DefaultCooldown
	}
	if ms, err := strconv.ParseFloat(response.Header.Get("Retry-After-Ms"), 64); err == nil && ms > 0 {
		return time.Duration(ms * float64(time.Millisecond))
	}
	value := response.Header.Get("Retry-After")
	if seconds, err := strconv.ParseFloat(value, 64); err == nil && seconds > 0 {
		return time.Duration(seconds * float64(time.Second))
	}
	if at, err := http.ParseTime(value); err == nil {
		if d := at.Sub(p.now()); d > 0 {
			return d
		}
	}
	return DefaultCooldown
}

// refused reports whether a response says its key cannot be used for now, and for how long
// the key should rest. That is a 429, for as long as Retry-After asks, or a 401 or 402
// whose body says the key's quota is spent, which no retry soon will change. The body is
// read only far enough to tell, and is left for the caller to read whole.
func (p *Pool) refused(response *http.Response) (time.Duration, bool) {
	switch response.StatusCode {
	case http.StatusTooManyRequests:
		return p.retryAfter(response), true
	case http.StatusUnauthorized, http.StatusPaymentRequired:
		if response.Body == nil {
			return 0, false
		}
		peeked, _ := io.ReadAll(io.LimitReader(response.Body, drainLimit))
		response.Body = peekedBody{Reader: io.MultiReader(bytes.NewReader(peeked), response.Body), Closer: response.Body}
		if bytes.Contains(peeked, []byte("quota_exceeded")) {
			return MaxCooldown, true
		}
	}
	return 0, false
}

// rejectsKey reports whether a status says the provider would not take the key at all: a
// 401 for a key it does not know, a 402 for an account out of credit, such as Gemini's
// "prepayment credits are depleted". Each is counted as refused whatever its body says, so a
// key to replace or top up shows up, even when refused leaves it in turn.
func rejectsKey(status int) bool {
	return status == http.StatusUnauthorized || status == http.StatusPaymentRequired
}

// peekedBody is a body with what was read of it put back in front.
type peekedBody struct {
	io.Reader
	io.Closer
}

// Send calls send with a key from the pool, and again with the next key for as long as
// the key it was given is refused and there are keys it has not been given. A refused
// response it moves past is drained and closed; the last one is returned as it came, so
// the caller fails over as it would have on any refusal.
func (p *Pool) Send(ctx context.Context, send func(key string) (*http.Response, error)) (*http.Response, error) {
	return p.sendTimes(ctx, max(1, p.Len()), send)
}

func (p *Pool) sendTimes(ctx context.Context, attempts int, send func(key string) (*http.Response, error)) (*http.Response, error) {
	for attempt := 0; ; attempt++ {
		key := p.Next()
		response, err := send(key)
		if err != nil || key == "" {
			return response, err
		}
		rest, refused := p.refused(response)
		if refused || rejectsKey(response.StatusCode) {
			p.countRefused(strconv.Itoa(response.StatusCode))
		}
		if !refused {
			return response, nil
		}
		p.Cooldown(key, rest)

		if attempt >= attempts-1 || ctx.Err() != nil {
			// The cooldown already holds the wait; a client that read it again would sleep
			// on a key the pool has rested instead of failing over to another provider.
			if p.Len() > 1 {
				response.Header.Del("Retry-After")
				response.Header.Del("Retry-After-Ms")
			}
			return response, nil
		}
		_, _ = io.Copy(io.Discard, io.LimitReader(response.Body, drainLimit))
		_ = response.Body.Close()
	}
}

// Auth is where a provider puts its key on a request: in a header after a prefix, or in a
// query parameter.
type Auth struct {
	header string
	prefix string
	query  string
}

// Bearer is a key sent as a bearer token.
var Bearer = Header("Authorization", "Bearer ")

// Header is a key sent in the header name, after prefix.
func Header(name, prefix string) Auth { return Auth{header: name, prefix: prefix} }

// Query is a key sent as the query parameter param.
func Query(param string) Auth { return Auth{query: param} }

// carried reports whether the request already holds a key where this puts one. Only such
// a request has its key swapped: one without was not meant to carry it, and might be going
// somewhere the key must not.
func (a Auth) carried(request *http.Request) bool {
	if a.query != "" {
		return request.URL.Query().Has(a.query)
	}
	return request.Header.Get(a.header) != ""
}

func (a Auth) set(request *http.Request, key string) {
	if a.query == "" {
		request.Header.Set(a.header, a.prefix+key)
		return
	}
	u := *request.URL
	query := u.Query()
	query.Set(a.query, key)
	u.RawQuery = query.Encode()
	request.URL = &u
}

// Doer is what sends a request: an *http.Client, or the SDK clients that take one.
type Doer interface {
	Do(*http.Request) (*http.Response, error)
}

// WrapDoer sends every request that carries a key through base on a key from the pool,
// moving to the next key on a 429. A nil base is http.DefaultClient.
func (p *Pool) WrapDoer(base Doer, auth Auth) Doer {
	if base == nil {
		base = http.DefaultClient
	}
	return wrapped{send: base.Do, pool: p, auth: auth}
}

// WrapClient is a copy of client whose transport sends every request that carries a key on
// a key from the pool, moving to the next key on a 429. A nil client is a new default one.
func (p *Pool) WrapClient(client *http.Client, auth Auth) *http.Client {
	copied := &http.Client{}
	if client != nil {
		*copied = *client
	}
	base := copied.Transport
	if base == nil {
		base = http.DefaultTransport
	}
	copied.Transport = wrapped{send: base.RoundTrip, pool: p, auth: auth}
	return copied
}

// wrapped is both the Doer and the RoundTripper the pool puts in front of a client.
type wrapped struct {
	send func(*http.Request) (*http.Response, error)
	pool *Pool
	auth Auth
}

func (w wrapped) Do(request *http.Request) (*http.Response, error) { return w.do(request) }

func (w wrapped) RoundTrip(request *http.Request) (*http.Response, error) { return w.do(request) }

// do sends the request on one key after another. A body that cannot be read again is
// sent once, on whichever key's turn it is.
func (w wrapped) do(request *http.Request) (*http.Response, error) {
	if !w.auth.carried(request) {
		return w.send(request)
	}
	attempts := max(1, w.pool.Len())
	if request.Body != nil && request.Body != http.NoBody && request.GetBody == nil {
		attempts = 1
	}
	first := true
	return w.pool.sendTimes(request.Context(), attempts, func(key string) (*http.Response, error) {
		attempt := request.Clone(request.Context())
		if !first && request.GetBody != nil {
			body, err := request.GetBody()
			if err != nil {
				return nil, err
			}
			attempt.Body = body
		}
		first = false
		if key != "" {
			w.auth.set(attempt, key)
		}
		return w.send(attempt)
	})
}

// Dial opens a websocket on a key from the pool, moving to the next key when the handshake
// is refused the way Send moves on. build gives the URL and headers for a key. It returns
// the key the connection was opened with, for a protocol that also sends it in a frame.
func (p *Pool) Dial(
	ctx context.Context,
	dialer *websocket.Dialer,
	build func(key string) (string, http.Header),
) (*websocket.Conn, *http.Response, string, error) {
	attempts := max(1, p.Len())
	for attempt := 0; ; attempt++ {
		key := p.Next()
		endpoint, header := build(key)
		conn, response, err := dialer.DialContext(ctx, endpoint, header)
		if err == nil || response == nil || key == "" {
			return conn, response, key, err
		}
		rest, refused := p.refused(response)
		if refused || rejectsKey(response.StatusCode) {
			p.countRefused(strconv.Itoa(response.StatusCode))
		}
		if !refused {
			return conn, response, key, err
		}
		p.Cooldown(key, rest)
		if attempt >= attempts-1 || ctx.Err() != nil {
			return conn, response, key, err
		}
	}
}
