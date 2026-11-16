// Package keypooltest reads the key pool's metrics for the tests of the providers that count
// into them.
package keypooltest

import (
	"strings"
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"
)

// Refused is how many refusals the counter holds under keyVar and status, 0 for a series not
// yet counted. The counter is the process's, so a test compares it before and after.
func Refused(t testing.TB, keyVar, status string) float64 {
	t.Helper()
	value, _ := series(t, "router_keypool_refused_attempts_total", map[string]string{"key_var": keyVar, "status": status})
	return value
}

// RateLimited is how many rate-limit refusals the vendor counter holds under keyVar and key,
// 0 for a series not yet counted.
func RateLimited(t testing.TB, keyVar, key string) float64 {
	t.Helper()
	value, _ := series(t, "router_vendor_rate_limited_total", map[string]string{"key_var": keyVar, "key": key})
	return value
}

// Remaining is the headroom gauge under keyVar, key and limit, and whether it has a series.
func Remaining(t testing.TB, keyVar, key, limit string) (float64, bool) {
	t.Helper()
	return series(t, "router_vendor_ratelimit_remaining", map[string]string{"key_var": keyVar, "key": key, "limit": limit})
}

// Series is every series of the vendor metric name under keyVar, by its other labels joined
// with "/" in their name order, so a test can tell no series was made.
func Series(t testing.TB, name, keyVar string) map[string]float64 {
	t.Helper()
	found := map[string]float64{}
	for _, metric := range metrics(t, name) {
		labels, rest := map[string]string{}, []string{}
		for _, pair := range metric.GetLabel() {
			labels[pair.GetName()] = pair.GetValue()
			if pair.GetName() != "key_var" {
				rest = append(rest, pair.GetValue())
			}
		}
		if labels["key_var"] == keyVar {
			found[strings.Join(rest, "/")] = value(metric)
		}
	}
	return found
}

// series is the value of the series of name with exactly labels, and whether it exists.
func series(t testing.TB, name string, labels map[string]string) (float64, bool) {
	for _, metric := range metrics(t, name) {
		if len(metric.GetLabel()) != len(labels) {
			continue
		}
		matches := true
		for _, pair := range metric.GetLabel() {
			if labels[pair.GetName()] != pair.GetValue() {
				matches = false
			}
		}
		if matches {
			return value(metric), true
		}
	}
	return 0, false
}

func metrics(t testing.TB, name string) []*dto.Metric {
	families, err := prometheus.DefaultGatherer.Gather()
	if err != nil {
		t.Fatal(err)
	}
	for _, family := range families {
		if family.GetName() == name {
			return family.GetMetric()
		}
	}
	return nil
}

func value(metric *dto.Metric) float64 {
	if metric.GetGauge() != nil {
		return metric.GetGauge().GetValue()
	}
	return metric.GetCounter().GetValue()
}
