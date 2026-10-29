// Package live keeps the fast-moving routing state in Redis: how healthy each provider
// looks right now, and how much traffic each customer is currently pushing.
//
// A routing decision must never wait on Postgres, so health is written here on every
// request and read back as a small ranked view.
//
// Every key is scoped by modality, so the same provider serving both speech-to-text and
// text-to-speech is ranked separately for each.
package live

import (
	"context"
	"errors"
	"fmt"
	"math"
	"strconv"
	"time"

	"github.com/redis/rueidis"
	"github.com/redis/rueidis/rueidisotel"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// Health is the recent behaviour of one provider and model.
type Health struct {
	Provider string
	Model    string
	// Requests and Errors count only the current window.
	Requests int64
	Errors   int64
	// LatencyMsAvg is the mean latency over the current window.
	LatencyMsAvg float64
	// Available is false once the window holds enough requests to judge (Options.MinRequests)
	// and their error rate crosses the configured threshold, and for one window after a fatal
	// failure however few requests it holds.
	Available bool
}

// SuccessRate returns successes over total, or 1 when nothing has been seen yet, so an
// unused provider is not punished for having no history.
func (h Health) SuccessRate() float64 {
	if h.Requests == 0 {
		return 1
	}
	return float64(h.Requests-h.Errors) / float64(h.Requests)
}

// ErrorRate returns errors over total, or 0 when nothing has been seen yet. It is
// computed directly rather than as 1-SuccessRate so exact ratios like 1-in-10 compare
// cleanly against a threshold.
func (h Health) ErrorRate() float64 {
	if h.Requests == 0 {
		return 0
	}
	return float64(h.Errors) / float64(h.Requests)
}

// Options configures the client.
type Options struct {
	// Address is a host:port, for example localhost:6379.
	Address  string
	Username string
	Password string
	// Shard is this shard's own Redis. Empty is the one above.
	Shard Shard
	// Window is how long health observations stay relevant.
	Window time.Duration
	// MaxErrorRate is the error rate at or above which a provider is treated as
	// unavailable, once the window holds MinRequests requests.
	MaxErrorRate float64
	// MinRequests is how many requests the window must hold before its error rate can make a
	// provider unavailable. Zero is DefaultMinRequests.
	MinRequests int64
}

// DefaultMinRequests is the fewest requests in a window whose error rate is trusted to take a
// provider out.
//
// Health is kept per shard and the outcome lasts a whole window, so a rate read off one or two
// requests would let a single failure take a provider out for every caller on the shard. Three
// is the fewest at which one failure alone stays under the default 50% threshold while two
// still cross it. A fatal failure is not a rate and is not held to this.
const DefaultMinRequests = 3

// Shard is where a shard's own Redis is.
type Shard struct {
	Address  string
	Username string
	Password string
}

// Client stores live counters and provider health.
type Client struct {
	// redis is the Redis every shard shares, and shard this shard's: the same client when
	// the deployment has one. Provider health and a customer's live counters are the
	// shard's, since latency is regional.
	redis   rueidis.Client
	shard   rueidis.Client
	options Options
}

// New connects to Redis and applies defaults for any unset option.
func New(options Options) (*Client, error) {
	if options.Address == "" {
		return nil, errors.New("live: redis address is required")
	}
	if options.Window == 0 {
		options.Window = 5 * time.Minute
	}
	if options.MaxErrorRate == 0 {
		options.MaxErrorRate = 0.5
	}
	if options.MinRequests == 0 {
		options.MinRequests = DefaultMinRequests
	}

	client, err := connect(options.Address, options.Username, options.Password)
	if err != nil {
		return nil, err
	}
	shard := client
	if options.Shard.Address != "" && options.Shard.Address != options.Address {
		if shard, err = connect(options.Shard.Address, options.Shard.Username, options.Shard.Password); err != nil {
			client.Close()
			return nil, fmt.Errorf("live: shard: %w", err)
		}
	}
	return &Client{redis: client, shard: shard, options: options}, nil
}

func connect(address, username, password string) (rueidis.Client, error) {
	client, err := rueidisotel.NewClient(rueidis.ClientOption{
		InitAddress:  []string{address},
		Username:     username,
		Password:     password,
		DisableCache: true,
	})
	if err != nil {
		return nil, fmt.Errorf("live: connect to redis: %w", err)
	}
	return client, nil
}

// Redis exposes the shared client for commands this package does not wrap.
func (c *Client) Redis() rueidis.Client { return c.redis }

// ShardRedis exposes this shard's client, for what only this shard's nodes need to hear.
func (c *Client) ShardRedis() rueidis.Client { return c.shard }

// Close releases the connections.
func (c *Client) Close() {
	if c.shard != c.redis {
		c.shard.Close()
	}
	c.redis.Close()
}

// Ping verifies both connections are usable.
func (c *Client) Ping(ctx context.Context) error {
	if err := c.redis.Do(ctx, c.redis.B().Ping().Build()).Error(); err != nil {
		return stack.Wrap(err)
	}
	return stack.Wrap(c.shard.Do(ctx, c.shard.B().Ping().Build()).Error())
}

// Usage is one request's contribution to the live counters.
type Usage struct {
	Modality          string
	CustomerID        string
	Provider          string
	Model             string
	LatencyMs         float64
	AudioMs           int64
	Characters        int64
	InputTokens       int64
	CachedInputTokens int64
	OutputTokens      int64
	CostMicros        int64
	Success           bool
	// Cancelled marks a request its caller gave up on. It says nothing about the provider, so
	// it is no request or error in the provider's health or the customer's counts, and the
	// time it ran is not a latency. Whatever it generated before it was cut off was spent all
	// the same, and is added to what the customer has used.
	Cancelled bool
	// Fatal marks a failure the provider gives every caller, such as an account it refuses.
	Fatal bool
	// Private marks work done on the customer's own key at the provider. What it says is about
	// their account, not the provider, so it is counted for the customer and left out of the
	// provider's health, which every customer is routed by.
	Private bool
}

// spent reports whether the request used anything at all.
func (u Usage) spent() bool {
	return u.AudioMs != 0 || u.Characters != 0 || u.InputTokens != 0 ||
		u.CachedInputTokens != 0 || u.OutputTokens != 0 || u.CostMicros != 0
}

// CustomerUsage is what one customer has spent in the current window.
type CustomerUsage struct {
	Requests          int64
	Errors            int64
	AudioMs           int64
	Characters        int64
	InputTokens       int64
	CachedInputTokens int64
	OutputTokens      int64
	CostMicros        int64
}

// RecordRequest updates provider health and the customer's live counters in one round
// trip. Every counter is given the health window as its TTL, so health reflects recent
// behaviour instead of all history and nothing has to be trimmed.
func (c *Client) RecordRequest(ctx context.Context, usage Usage) error {
	return c.RecordRequests(ctx, usage)
}

// RecordRequests is RecordRequest for many requests, in one round trip.
func (c *Client) RecordRequests(ctx context.Context, usages ...Usage) error {
	var commands []rueidis.Completed
	for _, usage := range usages {
		commands = append(commands, c.requestCommands(usage)...)
	}
	if len(commands) == 0 {
		return nil
	}

	for _, response := range c.shard.DoMulti(ctx, commands...) {
		if err := response.Error(); err != nil {
			return fmt.Errorf("live: record request: %w", err)
		}
	}
	return nil
}

// requestCommands are what one request adds to its provider's health and its customer's counters.
func (c *Client) requestCommands(usage Usage) []rueidis.Completed {
	healthKey := healthKey(usage.Modality, usage.Provider, usage.Model)
	customerKey := customerKey(usage.Modality, usage.CustomerID)
	ttl := int64(c.options.Window.Seconds())

	var commands []rueidis.Completed
	measured := !usage.Cancelled && !usage.Private
	if measured {
		commands = append(commands,
			c.shard.B().Hincrby().Key(healthKey).Field("requests").Increment(1).Build(),
			c.shard.B().Hincrbyfloat().Key(healthKey).Field("latency_ms_total").Increment(usage.LatencyMs).Build(),
			c.shard.B().Expire().Key(healthKey).Seconds(ttl).Build(),
		)
	}
	if !usage.Cancelled {
		commands = append(commands, c.shard.B().Hincrby().Key(customerKey).Field("requests").Increment(1).Build())
	}
	// A cancelled request that generated nothing adds nothing, so it writes nothing.
	if !usage.Cancelled || usage.spent() {
		commands = append(commands,
			c.shard.B().Hincrby().Key(customerKey).Field("audio_ms").Increment(usage.AudioMs).Build(),
			c.shard.B().Hincrby().Key(customerKey).Field("characters").Increment(usage.Characters).Build(),
			c.shard.B().Hincrby().Key(customerKey).Field("input_tokens").Increment(usage.InputTokens).Build(),
			c.shard.B().Hincrby().Key(customerKey).Field("cached_input_tokens").Increment(usage.CachedInputTokens).Build(),
			c.shard.B().Hincrby().Key(customerKey).Field("output_tokens").Increment(usage.OutputTokens).Build(),
			c.shard.B().Hincrby().Key(customerKey).Field("cost_micros").Increment(usage.CostMicros).Build(),
			c.shard.B().Expire().Key(customerKey).Seconds(ttl).Build(),
		)
	}
	if !usage.Success && measured {
		commands = append(commands, c.shard.B().Hincrby().Key(healthKey).Field("errors").Increment(1).Build())
	}
	if !usage.Success && !usage.Cancelled {
		commands = append(commands, c.shard.B().Hincrby().Key(customerKey).Field("errors").Increment(1).Build())
	}
	if usage.Fatal && !usage.Private {
		// A time rather than a count: the key's expiry moves with every request, so a count
		// would hold a provider that keeps being asked unavailable for good.
		commands = append(commands, c.shard.B().Hset().Key(healthKey).FieldValue().
			FieldValue("fatal_at_ms", strconv.FormatInt(time.Now().UnixMilli(), 10)).Build())
	}
	return commands
}

// Health returns the recent behaviour of one provider and model within a modality. An
// unseen provider comes back available with no history rather than as an error.
func (c *Client) Health(ctx context.Context, modality, provider, model string) (Health, error) {
	entries, err := c.shard.Do(ctx, c.shard.B().Hgetall().Key(healthKey(modality, provider, model)).Build()).AsStrMap()
	if err != nil {
		return Health{}, stack.Wrap(fmt.Errorf("live: read health: %w", err))
	}

	health := Health{Provider: provider, Model: model, Available: true}
	health.Requests = parseInt(entries["requests"])
	health.Errors = parseInt(entries["errors"])
	if health.Requests > 0 {
		health.LatencyMsAvg = parseFloat(entries["latency_ms_total"]) / float64(health.Requests)
	}
	// Too few requests are not a rate: until there are enough, an unseen provider and one that
	// failed once look alike.
	health.Available = health.Requests < c.options.MinRequests || health.ErrorRate() < c.options.MaxErrorRate
	// A fatal failure is one every caller would get, so no number of successes outweighs it.
	if fatalAt := parseInt(entries["fatal_at_ms"]); fatalAt > 0 &&
		time.Since(time.UnixMilli(fatalAt)) < c.options.Window {
		health.Available = false
	}

	return health, nil
}

// Usage returns what one customer has spent on a modality in the current window.
func (c *Client) Usage(ctx context.Context, modality, customerID string) (CustomerUsage, error) {
	entries, err := c.shard.Do(ctx, c.shard.B().Hgetall().Key(customerKey(modality, customerID)).Build()).AsStrMap()
	if err != nil {
		return CustomerUsage{}, fmt.Errorf("live: read customer usage: %w", err)
	}

	return CustomerUsage{
		Requests:          parseInt(entries["requests"]),
		Errors:            parseInt(entries["errors"]),
		AudioMs:           parseInt(entries["audio_ms"]),
		Characters:        parseInt(entries["characters"]),
		InputTokens:       parseInt(entries["input_tokens"]),
		CachedInputTokens: parseInt(entries["cached_input_tokens"]),
		OutputTokens:      parseInt(entries["output_tokens"]),
		CostMicros:        parseInt(entries["cost_micros"]),
	}, nil
}

func healthKey(modality, provider, model string) string {
	return "health:" + modality + ":" + provider + ":" + model
}

func customerKey(modality, customerID string) string {
	return "customer:" + modality + ":" + customerID
}

func parseInt(value string) int64 {
	parsed, err := strconv.ParseInt(value, 10, 64)
	if err != nil {
		return 0
	}
	return parsed
}

func parseFloat(value string) float64 {
	parsed, err := strconv.ParseFloat(value, 64)
	if err != nil || math.IsNaN(parsed) {
		return 0
	}
	return parsed
}
