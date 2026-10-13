package bench

import (
	"runtime"
	"syscall"
	"time"
)

// hostCPU measures how much of this machine the benchmark itself used over a stretch.
//
// It matters because every caller runs here: a hundred WebRTC legs encoding Opus will
// saturate a laptop long before they trouble a router, and a reply time measured on a
// machine with nothing left is the caller's own wait rather than the agent's.
type hostCPU struct {
	at    time.Time
	spent time.Duration
}

func watchHost() hostCPU {
	return hostCPU{at: time.Now(), spent: processCPU()}
}

// since is what this process used as a share of the machine's cores.
func (h hostCPU) since() HostCPU {
	elapsed := time.Since(h.at)
	cores := runtime.NumCPU()
	if elapsed <= 0 || cores == 0 {
		return HostCPU{Cores: cores}
	}
	used := processCPU() - h.spent
	return HostCPU{Busy: float64(used) / float64(elapsed) / float64(cores), Cores: cores}
}

// processCPU is the user and system time this process has spent.
func processCPU() time.Duration {
	var usage syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &usage); err != nil {
		return 0
	}
	return time.Duration(usage.Utime.Nano()) + time.Duration(usage.Stime.Nano())
}
