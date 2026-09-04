package target

import (
	"context"
	"os"
	"strings"
)

const defaultAccelURL = "http://127.0.0.1:8080"

// Accelerated is the Python SDK plus stream.Accelerated. Function calling stays
// in Python. An optional --bin spawns the router the SDK talks to.
type Accelerated struct {
	Python
	Bin string
}

func (a *Accelerated) Prepare(ctx context.Context) (func(), error) {
	a.Pipeline = "accelerated"
	var stops []func()
	combine := func() {
		for i := len(stops) - 1; i >= 0; i-- {
			stops[i]()
		}
	}
	if a.Spawn && (a.Bin != "" || os.Getenv("ACCEL_ROUTER") != "") {
		routerURL := a.routerURL()
		stopRouter, err := StartRouter(ctx, a.Bin, routerURL)
		if err != nil {
			return nil, err
		}
		stops = append(stops, stopRouter)
		a.Env = append(a.Env, "STREAM_ACCELERATION_URL="+routerURL)
		a.logger().Info("spawned accel router for accelerated target", "url", routerURL)
	}
	stopPython, err := a.Python.Prepare(ctx)
	if err != nil {
		combine()
		return nil, err
	}
	stops = append(stops, stopPython)
	return combine, nil
}

func (a *Accelerated) routerURL() string {
	if u := os.Getenv("STREAM_ACCELERATION_URL"); u != "" {
		return strings.TrimRight(u, "/")
	}
	return defaultAccelURL
}
