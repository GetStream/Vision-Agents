package main

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/research"
)

func main() {
	if e := run(); e != nil {
		log.Fatal(e)
	}
}
func run() error {
	data, e := os.ReadFile("/opt/support/profile.json")
	if e != nil {
		return e
	}
	var p research.Profile
	if e = json.Unmarshal(data, &p); e != nil {
		return e
	}
	token, e := os.ReadFile("/opt/support/worker-token")
	if e != nil {
		return e
	}
	key, e := os.ReadFile("/opt/support/cursor-key")
	if e != nil {
		return e
	}
	_ = os.Setenv("CURSOR_API_KEY", string(key))
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	startup, cancel := context.WithTimeout(ctx, 60*time.Second)
	defer cancel()
	worker, e := research.NewWorker(startup, p, research.Root, string(token))
	if e != nil {
		return e
	}
	defer worker.Close()
	server := &http.Server{Addr: ":8081", Handler: worker, ReadHeaderTimeout: 10 * time.Second, IdleTimeout: 60 * time.Second}
	go func() {
		<-ctx.Done()
		c, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = server.Shutdown(c)
	}()
	e = server.ListenAndServe()
	if e == http.ErrServerClosed {
		return nil
	}
	return e
}
