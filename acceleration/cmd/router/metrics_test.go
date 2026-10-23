package main

import (
	"bytes"
	"io"
	"log/slog"
	"net"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"
)

// logged is what a logger wrote, safe to read while the logger is still in use.
type logged struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (l *logged) Write(p []byte) (int, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.buf.Write(p)
}

func (l *logged) String() string {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.buf.String()
}

type MetricsSuite struct {
	suite.Suite
	addr string
}

func TestMetricsSuite(t *testing.T) {
	suite.Run(t, new(MetricsSuite))
}

// SetupTest finds a free port: the listener is opened by address, so it cannot be told to
// pick one.
func (s *MetricsSuite) SetupTest() {
	free, err := net.Listen("tcp", "127.0.0.1:0")
	s.Require().NoError(err)
	s.addr = free.Addr().String()
	s.Require().NoError(free.Close())
}

// scrape asks for the metrics the way Prometheus does.
func (s *MetricsSuite) scrape() (*http.Response, error) {
	request, err := http.NewRequest(http.MethodGet, "http://"+s.addr+"/metrics", nil)
	s.Require().NoError(err)
	request.Header.Set("Accept", "application/openmetrics-text;version=1.0.0")
	return http.DefaultClient.Do(request)
}

func (s *MetricsSuite) TestMetricsAreServedAsOpenMetrics() {
	stop, _ := serveMetrics(s.addr, slog.New(slog.DiscardHandler))
	s.T().Cleanup(stop)

	var response *http.Response
	s.Require().Eventually(func() bool {
		var err error
		response, err = s.scrape()
		return err == nil
	}, 5*time.Second, 10*time.Millisecond, "the metrics listener never answered")
	defer response.Body.Close()
	body, err := io.ReadAll(response.Body)
	s.Require().NoError(err)

	s.Equal(http.StatusOK, response.StatusCode)
	s.Contains(response.Header.Get("Content-Type"), "application/openmetrics-text")
	s.Contains(string(body), "go_goroutines")
}

func (s *MetricsSuite) TestMetricsStopBeingServedWhenStopped() {
	stop, _ := serveMetrics(s.addr, slog.New(slog.DiscardHandler))
	s.Require().Eventually(func() bool {
		response, err := s.scrape()
		if err != nil {
			return false
		}
		response.Body.Close()
		return true
	}, 5*time.Second, 10*time.Millisecond, "the metrics listener never answered")

	stop()

	_, err := s.scrape()
	s.Error(err, "the listener still answers after the router stopped it")
}

func (s *MetricsSuite) TestHandingOverFreesTheAddressAndSaysSoOnce() {
	lines := &logged{}
	stop, handOver := serveMetrics(s.addr, slog.New(slog.NewTextHandler(lines, nil)))
	s.Require().Eventually(func() bool {
		response, err := s.scrape()
		if err != nil {
			return false
		}
		response.Body.Close()
		return true
	}, 5*time.Second, 10*time.Millisecond, "the metrics listener never answered")

	handOver()
	handOver()
	stop()

	successor, err := net.Listen("tcp", s.addr)
	s.Require().NoError(err, "the address was still held after the hand-over")
	s.Require().NoError(successor.Close())
	s.Equal(1, strings.Count(lines.String(), "stopped serving metrics for the router taking over"))
	s.Contains(lines.String(), "level=INFO msg=\"stopped serving metrics for the router taking over\" address="+s.addr)
	s.NotContains(lines.String(), "level=WARN")
}

func (s *MetricsSuite) TestNoAddressOpensNoListener() {
	lines := &logged{}

	stop, handOver := serveMetrics("", slog.New(slog.NewTextHandler(lines, nil)))
	defer stop()
	handOver()

	// With an empty address the server would listen on ":http" and either log that it is
	// serving or fail to bind and log a warning.
	s.Never(func() bool { return lines.String() != "" }, 200*time.Millisecond, 10*time.Millisecond,
		"serveMetrics started a listener without an address")
}

func (s *MetricsSuite) TestStoppedMetricsDoNotWarnAboutNotServing() {
	lines := &logged{}
	stop, _ := serveMetrics(s.addr, slog.New(slog.NewTextHandler(lines, nil)))
	s.Require().Eventually(func() bool {
		response, err := s.scrape()
		if err != nil {
			return false
		}
		response.Body.Close()
		return true
	}, 5*time.Second, 10*time.Millisecond, "the metrics listener never answered")

	stop()

	s.Never(func() bool { return strings.Contains(lines.String(), "level=WARN") }, 300*time.Millisecond, 10*time.Millisecond,
		"a stopped listener logged a warning")
}
