package main

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"
)

type MatrixSuite struct {
	suite.Suite
}

func TestMatrixSuite(t *testing.T) {
	suite.Run(t, new(MatrixSuite))
}

func (s *MatrixSuite) TestThePlanCallsEveryOtherNumberInTheCityRunsTimes() {
	lines := Lines{
		{Vendor: "twilio", City: "Chicago", E164: "+13125550001"},
		{Vendor: "telnyx", City: "Chicago", E164: "+13125550002"},
		{Vendor: "twilio", City: "Houston", E164: "+17135550001"},
		{Vendor: "telnyx", City: "Houston", E164: "+17135550002"},
	}

	tasks := Plan(lines, ScopeCity, 10)

	s.Len(tasks, 2*2*10)
	for _, task := range tasks {
		s.NotEqual(task.Caller.E164, task.Callee.E164)
		s.Equal(task.Caller.City, task.Callee.City)
	}
	s.Len(Plan(lines, ScopeAll, 1), 4*3)
}

func (s *MatrixSuite) TestANumberIsNeverOnTwoCallsAtOnce() {
	lines := Lines{
		{Vendor: "twilio", City: "Chicago", E164: "+1"},
		{Vendor: "twilio", City: "Chicago", E164: "+2"},
		{Vendor: "telnyx", City: "Chicago", E164: "+3"},
		{Vendor: "telnyx", City: "Chicago", E164: "+4"},
	}
	tasks := Plan(lines, ScopeCity, 3)

	var mu sync.Mutex
	onCall := map[string]int{}
	overlapped := false
	var results []Result
	Execute(context.Background(), tasks, 8, func(_ context.Context, task Task) Result {
		mu.Lock()
		onCall[task.Caller.E164]++
		onCall[task.Callee.E164]++
		if onCall[task.Caller.E164] > 1 || onCall[task.Callee.E164] > 1 {
			overlapped = true
		}
		mu.Unlock()
		time.Sleep(5 * time.Millisecond)
		mu.Lock()
		onCall[task.Caller.E164]--
		onCall[task.Callee.E164]--
		mu.Unlock()
		return Result{Caller: task.Caller, Callee: task.Callee}
	}, func(result Result) { results = append(results, result) })

	s.False(overlapped)
	s.Len(results, len(tasks))
}
