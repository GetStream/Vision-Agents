package main

import (
	"io"
	"log/slog"
	"net"
	"net/http"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"
)

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
	stop := serveMetrics(s.addr, slog.New(slog.DiscardHandler))
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
	stop := serveMetrics(s.addr, slog.New(slog.DiscardHandler))
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
