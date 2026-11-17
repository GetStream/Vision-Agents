package main

import (
	"context"
	"sync"
)

// Scope says which numbers call which.
type Scope string

const (
	// ScopeCity has every number call every other number in its own city.
	ScopeCity Scope = "city"
	// ScopeAll has every number call every other number anywhere.
	ScopeAll Scope = "all"
)

// Task is one call to place.
type Task struct {
	Run    int
	Caller Line
	Callee Line
}

// Plan is every call of a run: each ordered pair of numbers in scope, runs times over.
//
// The repeats are the outer loop, so each cell's calls are spread across the whole run
// rather than placed back to back, and a bad few minutes at one carrier lands on every
// cell a little instead of on one cell entirely.
func Plan(lines Lines, scope Scope, runs int) []Task {
	var tasks []Task
	for run := 1; run <= runs; run++ {
		for _, caller := range lines {
			for _, callee := range lines {
				if caller.E164 == callee.E164 {
					continue
				}
				if scope == ScopeCity && caller.City != callee.City {
					continue
				}
				tasks = append(tasks, Task{Run: run, Caller: caller, Callee: callee})
			}
		}
	}
	return tasks
}

// Execute places the calls, at most parallel at once and never two on the same number: a
// number already on a call would answer the second one busy, and the bench could not tell
// which leg belonged to which call. Each result is handed to record as it finishes.
func Execute(ctx context.Context, tasks []Task, parallel int, probe func(context.Context, Task) Result, record func(Result)) {
	var mu sync.Mutex
	busy := map[string]bool{}
	pending := append([]Task(nil), tasks...)
	freed := sync.NewCond(&mu)

	// take blocks until some pending call's numbers are both free, and claims them.
	take := func() (Task, bool) {
		mu.Lock()
		defer mu.Unlock()
		for {
			if ctx.Err() != nil || len(pending) == 0 {
				return Task{}, false
			}
			for i, task := range pending {
				if !busy[task.Caller.E164] && !busy[task.Callee.E164] {
					busy[task.Caller.E164] = true
					busy[task.Callee.E164] = true
					pending = append(pending[:i], pending[i+1:]...)
					return task, true
				}
			}
			freed.Wait()
		}
	}
	release := func(task Task) {
		mu.Lock()
		defer mu.Unlock()
		delete(busy, task.Caller.E164)
		delete(busy, task.Callee.E164)
		freed.Broadcast()
	}

	stop := context.AfterFunc(ctx, func() {
		mu.Lock()
		defer mu.Unlock()
		freed.Broadcast()
	})
	defer stop()

	var workers sync.WaitGroup
	var recording sync.Mutex
	for range max(parallel, 1) {
		workers.Go(func() {
			for {
				task, ok := take()
				if !ok {
					return
				}
				result := probe(ctx, task)
				release(task)
				recording.Lock()
				record(result)
				recording.Unlock()
			}
		})
	}
	workers.Wait()
}
