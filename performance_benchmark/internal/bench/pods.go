package bench

import (
	"context"
	"encoding/json"
	"fmt"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

// kubectlTimeout bounds one reading. A cluster that is not answering must not hold up a
// step: the numbers are context, not the measurement.
const kubectlTimeout = 15 * time.Second

// Pods is what the router's pods were doing while a step ran, read from the cluster. It
// is all zero for a run that was given no namespace, such as one against a local router.
type Pods struct {
	// MilliCPU and MemoryMiB are the busiest reading taken during the step, summed over
	// the pods: the peak is what decides whether the next step fits.
	MilliCPU  int `json:"milli_cpu"`
	MemoryMiB int `json:"memory_mib"`
	// Restarts is how many times a pod restarted during the step, which is the clearest
	// way a router says it has had enough.
	Restarts int `json:"restarts"`
	// Readings is how many samples went into this, so an empty one is told from a quiet
	// one.
	Readings int `json:"readings"`
}

// CPU is the peak reading as cores, for the report.
func (p Pods) CPU() string {
	if p.Readings == 0 {
		return "-"
	}
	return fmt.Sprintf("%.2f", float64(p.MilliCPU)/1000)
}

// Memory is the peak reading in mebibytes, for the report.
func (p Pods) Memory() string {
	if p.Readings == 0 {
		return "-"
	}
	return fmt.Sprintf("%dMi", p.MemoryMiB)
}

// Cluster reads what the router's pods are doing, through kubectl.
//
// It shells out rather than holding a Kubernetes client because a benchmark run already
// needs a working kubectl context to be looking at the right cluster at all, and one of
// those is a dependency and a credential chain this does not otherwise need.
type Cluster struct {
	// Namespace holds the router. Empty turns the readings off.
	Namespace string
	// Selector narrows the pods to the router's own.
	Selector string
	// Context is the kubectl context, or empty for the current one.
	Context string
}

// Watch samples the pods until the context ends, and returns the peak.
func (c Cluster) Watch(ctx context.Context, every time.Duration) Pods {
	if c.Namespace == "" {
		return Pods{}
	}

	peak := Pods{}
	started, _ := c.restarts(ctx)
	ticker := time.NewTicker(every)
	defer ticker.Stop()
	for {
		cpu, memory, err := c.top(ctx)
		if err == nil {
			peak.Readings++
			peak.MilliCPU = max(peak.MilliCPU, cpu)
			peak.MemoryMiB = max(peak.MemoryMiB, memory)
		}
		select {
		case <-ticker.C:
		case <-ctx.Done():
			if now, err := c.restarts(context.WithoutCancel(ctx)); err == nil {
				peak.Restarts = now - started
			}
			return peak
		}
	}
}

// top is the pods' current CPU and memory, summed.
func (c Cluster) top(ctx context.Context) (int, int, error) {
	out, err := c.run(ctx, "top", "pod", "--no-headers")
	if err != nil {
		return 0, 0, err
	}

	var milliCPU, memoryMiB int
	for _, line := range strings.Split(strings.TrimSpace(out), "\n") {
		fields := strings.Fields(line)
		if len(fields) < 3 {
			continue
		}
		milliCPU += quantity(fields[1], "m")
		memoryMiB += quantity(fields[2], "Mi")
	}
	return milliCPU, memoryMiB, nil
}

// restarts is how many times the pods have restarted since they were created.
func (c Cluster) restarts(ctx context.Context) (int, error) {
	out, err := c.run(ctx, "get", "pods", "-o", "json")
	if err != nil {
		return 0, err
	}

	var listed struct {
		Items []struct {
			Status struct {
				ContainerStatuses []struct {
					RestartCount int `json:"restartCount"`
				} `json:"containerStatuses"`
			} `json:"status"`
		} `json:"items"`
	}
	if err := json.Unmarshal([]byte(out), &listed); err != nil {
		return 0, err
	}
	total := 0
	for _, pod := range listed.Items {
		for _, container := range pod.Status.ContainerStatuses {
			total += container.RestartCount
		}
	}
	return total, nil
}

func (c Cluster) run(ctx context.Context, args ...string) (string, error) {
	ctx, cancel := context.WithTimeout(ctx, kubectlTimeout)
	defer cancel()

	full := []string{"-n", c.Namespace}
	if c.Context != "" {
		full = append(full, "--context", c.Context)
	}
	full = append(full, args...)
	if c.Selector != "" {
		full = append(full, "-l", c.Selector)
	}
	out, err := exec.CommandContext(ctx, "kubectl", full...).Output()
	if err != nil {
		return "", fmt.Errorf("bench: kubectl %s: %w", strings.Join(full, " "), err)
	}
	return string(out), nil
}

// quantity reads a kubectl quantity such as 250m or 384Mi, in the unit asked for.
func quantity(raw, unit string) int {
	value, err := strconv.Atoi(strings.TrimSuffix(raw, unit))
	if err != nil {
		return 0
	}
	return value
}
