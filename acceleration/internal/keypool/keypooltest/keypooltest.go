// Package keypooltest reads the key pool's refusal counter for the tests of the providers that
// count into it.
package keypooltest

import (
	"testing"

	"github.com/prometheus/client_golang/prometheus"
)

// Refused is how many refusals the counter holds under keyVar and status, 0 for a series not
// yet counted. The counter is the process's, so a test compares it before and after.
func Refused(t testing.TB, keyVar, status string) float64 {
	t.Helper()
	families, err := prometheus.DefaultGatherer.Gather()
	if err != nil {
		t.Fatal(err)
	}
	for _, family := range families {
		if family.GetName() != "router_keypool_refused_attempts_total" {
			continue
		}
		for _, metric := range family.GetMetric() {
			labels := map[string]string{}
			for _, pair := range metric.GetLabel() {
				labels[pair.GetName()] = pair.GetValue()
			}
			if labels["key_var"] == keyVar && labels["status"] == status {
				return metric.GetCounter().GetValue()
			}
		}
	}
	return 0
}
