// Package failovertest reads the router's failover counter for the tests of the paths that
// count into it.
package failovertest

import (
	"testing"

	"github.com/prometheus/client_golang/prometheus"
)

// Of is every failover counted against provider, by "modality stage reason". The
// counter is the process's, so a test names a provider no other test does.
func Of(t testing.TB, provider string) map[string]float64 {
	t.Helper()
	families, err := prometheus.DefaultGatherer.Gather()
	if err != nil {
		t.Fatal(err)
	}
	counted := map[string]float64{}
	for _, family := range families {
		if family.GetName() != "router_failovers_total" {
			continue
		}
		for _, metric := range family.GetMetric() {
			labels := map[string]string{}
			for _, label := range metric.GetLabel() {
				labels[label.GetName()] = label.GetValue()
			}
			if labels["provider"] == provider {
				counted[labels["modality"]+" "+labels["stage"]+" "+labels["reason"]] += metric.GetCounter().GetValue()
			}
		}
	}
	return counted
}
