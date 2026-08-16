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
	// Available is false once the error rate crosses the configured threshold.
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
	Address string
	// Window is how long health observations stay relevant.
	Window time.Duration
	// MaxErrorRate is the error rate at or above which a provider is treated as
	// unavailable.
	MaxErrorRate float64
}

// Client stores live counters and provider health.
type Client struct {
	redis   rueidis.Client
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

	client, err := rueidis.NewClient(rueidis.ClientOption{
		InitAddress:  []string{options.Address},
		DisableCache: true,
	})
	if err != nil {
		return nil, fmt.Errorf("live: connect to redis: %w", err)
	}

	return &Client{redis: client, options: options}, nil
}

// Redis exposes the underlying client for commands this package does not wrap.
func (c *Client) Redis() rueidis.Client { return c.redis }

// Close releases the connection.
func (c *Client) Close() { c.redis.Close() }

// Ping verifies the connection is usable.
func (c *Client) Ping(ctx context.Context) error {
	return c.redis.Do(ctx, c.redis.B().Ping().Build()).Error()
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
	healthKey := healthKey(usage.Modality, usage.Provider, usage.Model)
	customerKey := customerKey(usage.Modality, usage.CustomerID)
	ttl := int64(c.options.Window.Seconds())

	commands := []rueidis.Completed{
		c.redis.B().Hincrby().Key(healthKey).Field("requests").Increment(1).Build(),
		c.redis.B().Hincrbyfloat().Key(healthKey).Field("latency_ms_total").Increment(usage.LatencyMs).Build(),
		c.redis.B().Expire().Key(healthKey).Seconds(ttl).Build(),
		c.redis.B().Hincrby().Key(customerKey).Field("requests").Increment(1).Build(),
		c.redis.B().Hincrby().Key(customerKey).Field("audio_ms").Increment(usage.AudioMs).Build(),
		c.redis.B().Hincrby().Key(customerKey).Field("characters").Increment(usage.Characters).Build(),
		c.redis.B().Hincrby().Key(customerKey).Field("input_tokens").Increment(usage.InputTokens).Build(),
		c.redis.B().Hincrby().Key(customerKey).Field("cached_input_tokens").Increment(usage.CachedInputTokens).Build(),
		c.redis.B().Hincrby().Key(customerKey).Field("output_tokens").Increment(usage.OutputTokens).Build(),
		c.redis.B().Hincrby().Key(customerKey).Field("cost_micros").Increment(usage.CostMicros).Build(),
		c.redis.B().Expire().Key(customerKey).Seconds(ttl).Build(),
	}
	if !usage.Success {
		commands = append(commands,
			c.redis.B().Hincrby().Key(healthKey).Field("errors").Increment(1).Build(),
			c.redis.B().Hincrby().Key(customerKey).Field("errors").Increment(1).Build(),
		)
	}

	for _, response := range c.redis.DoMulti(ctx, commands...) {
		if err := response.Error(); err != nil {
			return fmt.Errorf("live: record request: %w", err)
		}
	}
	return nil
}

// Health returns the recent behaviour of one provider and model within a modality. An
// unseen provider comes back available with no history rather than as an error.
func (c *Client) Health(ctx context.Context, modality, provider, model string) (Health, error) {
	entries, err := c.redis.Do(ctx, c.redis.B().Hgetall().Key(healthKey(modality, provider, model)).Build()).AsStrMap()
	if err != nil {
		return Health{}, fmt.Errorf("live: read health: %w", err)
	}

	health := Health{Provider: provider, Model: model, Available: true}
	health.Requests = parseInt(entries["requests"])
	health.Errors = parseInt(entries["errors"])
	if health.Requests > 0 {
		health.LatencyMsAvg = parseFloat(entries["latency_ms_total"]) / float64(health.Requests)
	}
	health.Available = health.ErrorRate() < c.options.MaxErrorRate

	return health, nil
}

// Usage returns what one customer has spent on a modality in the current window.
func (c *Client) Usage(ctx context.Context, modality, customerID string) (CustomerUsage, error) {
	entries, err := c.redis.Do(ctx, c.redis.B().Hgetall().Key(customerKey(modality, customerID)).Build()).AsStrMap()
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
