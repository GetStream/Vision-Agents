package keypool

import (
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"

	"github.com/GetStream/Vision-Agents/acceleration/internal/alerts"
)

// unnamedKeyVar is the key_var of a pool no FromEnv has named: one from New, or from keys
// passed in that no provider reads from a variable.
const unnamedKeyVar = "unknown"

// key_var is the variable a pool's keys are read from (DEEPGRAM_API_KEY), a constant in the
// code, and status the refused response's HTTP status (refused, rejectsKey) or a socket
// refusal's name (QuotaExceeded, InvalidKey, OutOfFunds): both fixed sets. A key is never a label.
var refusedAttempts = promauto.NewCounterVec(prometheus.CounterOpts{
	Namespace: "router",
	Name:      "keypool_refused_attempts_total",
	Help:      "Requests and handshakes a provider refused on a pooled key, by the variable the keys come from and the refusal's HTTP status (every 401 and 402, a 400 or 403 whose body says the key is not valid, and 429), or, for a socket frame or close after the handshake, quota_exceeded, invalid_key (a key the provider does not take) or 402 (a frame for want of funds); a 429 or a spent quota moves to the next key or, on the last, is handed back to fail over, any other 400, 401, 402 or 403 counted is handed back at once",
}, []string{"key_var", "status"})

// Alerts are the conditions of a provider refusing one of the router's keys, one alert per key
// variable and status at most every limit of the loop that runs them, such as
// ELEVENLABS_API_KEY/quota_exceeded: a key is never named. It does not page: the request moves to
// the next key or fails over, and the key is to top up or replace.
func Alerts(limit time.Duration) []alerts.Condition {
	return []alerts.Condition{{Name: "key_refused", Level: alerts.Warn,
		Items: alerts.IncreasedPer(refusedAttempts, limit, []string{"key_var", "status"}, nil)}}
}
