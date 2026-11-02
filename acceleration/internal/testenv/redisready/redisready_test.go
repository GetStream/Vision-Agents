package redisready

import (
	"bufio"
	"context"
	"errors"
	"io"
	"net"
	"os"
	"runtime"
	"strconv"
	"sync/atomic"
	"testing"
	"time"

	"github.com/redis/rueidis"
	"github.com/stretchr/testify/suite"
)

type RedisSuite struct {
	suite.Suite
}

func TestRedisSuite(t *testing.T) {
	suite.Run(t, new(RedisSuite))
}

// peer listens like a published container port: it accepts every connection at once, and
// answers PING with reply from the time ready has passed, closing the connection after.
func (s *RedisSuite) peer(ready time.Time, reply string) string {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	s.Require().NoError(err)
	s.T().Cleanup(func() { listener.Close() })
	go func() {
		for {
			conn, err := listener.Accept()
			if err != nil {
				return
			}
			go func() {
				defer conn.Close()
				line, err := bufio.NewReader(conn).ReadString('\n')
				if err != nil || line != "PING\r\n" {
					return
				}
				time.Sleep(time.Until(ready))
				conn.Write([]byte(reply))
			}()
		}
	}()
	return listener.Addr().String()
}

func (s *RedisSuite) TestARedisThatAnswersLateIsWaitedFor() {
	// The connection is accepted at once and left unanswered, as a Redis container that has
	// just started leaves it.
	ready := time.Now().Add(2500 * time.Millisecond)
	address := s.peer(ready, "+PONG\r\n")

	s.Require().NoError(Wait(address, 10*time.Second))
	s.False(time.Now().Before(ready), "returned before the Redis answered")
}

func (s *RedisSuite) TestARedisThatNeverAnswersIsAnError() {
	address := s.peer(time.Now().Add(time.Hour), "+PONG\r\n")

	start := time.Now()
	err := Wait(address, 1500*time.Millisecond)

	s.Error(err)
	s.Contains(err.Error(), address)
	s.Less(time.Since(start), 5*time.Second, "the wait ran past its deadline")
}

func (s *RedisSuite) TestARedisStillLoadingIsNotReady() {
	address := s.peer(time.Now(), "-LOADING Redis is loading the dataset in memory\r\n")

	s.Error(Wait(address, 500*time.Millisecond))
}

func (s *RedisSuite) TestARedisThatRefusesWhileItStartsIsWaitedFor() {
	// Reserve a port, free it, and listen on it again after 1.5 s: until then a connection
	// is refused, as when a container's Redis is not up yet.
	reserve, err := net.Listen("tcp", "127.0.0.1:0")
	s.Require().NoError(err)
	address := reserve.Addr().String()
	s.Require().NoError(reserve.Close())
	opened := time.Now().Add(1500 * time.Millisecond)
	listening := make(chan net.Listener, 1)
	go func() {
		time.Sleep(time.Until(opened))
		listener, err := net.Listen("tcp", address)
		if err != nil {
			listening <- nil
			return
		}
		listening <- listener
		for {
			conn, err := listener.Accept()
			if err != nil {
				return
			}
			go func() {
				defer conn.Close()
				if _, err := bufio.NewReader(conn).ReadString('\n'); err == nil {
					conn.Write([]byte("+PONG\r\n"))
				}
			}()
		}
	}()
	s.T().Cleanup(func() {
		if listener := <-listening; listener != nil {
			listener.Close()
		}
	})

	s.Require().NoError(Wait(address, 10*time.Second))
	s.False(time.Now().Before(opened), "returned before the Redis was listening")
}

func (s *RedisSuite) TestNothingListeningIsAnErrorAtTheDeadline() {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	s.Require().NoError(err)
	address := listener.Addr().String()
	s.Require().NoError(listener.Close())

	start := time.Now()
	err = Wait(address, 1200*time.Millisecond)

	s.Error(err)
	s.Contains(err.Error(), address)
	s.GreaterOrEqual(time.Since(start), 1200*time.Millisecond, "a refused connection was given up on at once")
	s.Less(time.Since(start), 5*time.Second)
}

func (s *RedisSuite) TestAFailureIsNotWaitedOutAgainForTheSameAddress() {
	original := ready
	ready = 300 * time.Millisecond
	s.T().Cleanup(func() { ready = original })
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	s.Require().NoError(err)
	address := listener.Addr().String()
	s.Require().NoError(listener.Close())
	first := Ready(address)
	s.Require().Error(first)

	start := time.Now()
	second := Ready(address)

	s.Equal(first, second)
	s.Less(time.Since(start), 100*time.Millisecond, "the second call waited out the deadline again")
}

// forwarder stands for Docker's port forwarding on a loaded machine: a connection accepted
// before until is held unanswered, or closed at once when hold is false; a later one is
// forwarded to upstream. The first connection is always forwarded: it is Ready's PING, which
// has to pass for NewClient to dial at all. It returns the connections accepted so far.
func (s *RedisSuite) forwarder(upstream string, until time.Time, hold bool) (string, *atomic.Int64) {
	accepted := new(atomic.Int64)
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	s.Require().NoError(err)
	done := make(chan struct{})
	s.T().Cleanup(func() {
		close(done)
		listener.Close()
	})
	go func() {
		for {
			conn, err := listener.Accept()
			if err != nil {
				return
			}
			first := accepted.Add(1) == 1
			go func() {
				defer conn.Close()
				if !first && time.Now().Before(until) {
					if hold {
						select {
						case <-time.After(time.Until(until)):
						case <-done:
						}
					}
					return
				}
				redis, err := net.Dial("tcp", upstream)
				if err != nil {
					return
				}
				defer redis.Close()
				go io.Copy(redis, conn)
				io.Copy(conn, redis)
			}()
		}
	}()
	return listener.Addr().String(), accepted
}

// redis is the Redis the suites dial, or a skip without one.
func (s *RedisSuite) redis() string {
	address := os.Getenv("ROUTER_REDIS_ADDR")
	if address == "" || Wait(address, time.Second) != nil {
		s.T().Skip("no Redis answering at ROUTER_REDIS_ADDR")
	}
	return address
}

// clientThrough dials address with a short dial timeout, so a test sees several attempts.
func clientThrough(address string) (rueidis.Client, error) {
	return NewClient(rueidis.ClientOption{
		InitAddress: []string{address}, DisableCache: true, Dialer: net.Dialer{Timeout: 300 * time.Millisecond},
	})
}

func (s *RedisSuite) TestAClientWhoseConnectionIsHeldPastItsDialTimeoutIsDialledAgain() {
	until := time.Now().Add(time.Second)
	address, _ := s.forwarder(s.redis(), until, true)

	client, err := clientThrough(address)

	s.Require().NoError(err)
	s.T().Cleanup(client.Close)
	s.False(time.Now().Before(until), "a client was made while its connections were held")
	reply, err := client.Do(context.Background(), client.B().Ping().Build()).ToString()
	s.Require().NoError(err)
	s.Equal("PONG", reply)
}

func (s *RedisSuite) TestAClientWhoseConnectionIsClosedAtOnceIsDialledAgain() {
	until := time.Now().Add(time.Second)
	address, _ := s.forwarder(s.redis(), until, false)

	client, err := clientThrough(address)

	s.Require().NoError(err)
	s.T().Cleanup(client.Close)
	reply, err := client.Do(context.Background(), client.B().Ping().Build()).ToString()
	s.Require().NoError(err)
	s.Equal("PONG", reply)
}

func (s *RedisSuite) TestAClientThatNeverConnectsIsAnErrorAtTheDeadline() {
	original := ready
	ready = 1200 * time.Millisecond
	s.T().Cleanup(func() { ready = original })
	address, _ := s.forwarder(s.redis(), time.Now().Add(time.Hour), true)

	start := time.Now()
	client, err := clientThrough(address)

	s.Nil(client)
	s.Require().Error(err)
	s.Contains(err.Error(), address)
	s.GreaterOrEqual(time.Since(start), ready, "the client was given up on before the deadline")
	s.Less(time.Since(start), 5*time.Second, "the dial ran past its deadline")
}

func (s *RedisSuite) TestAClientIsRefusedAtOnceWhereNoRedisAnswered() {
	original := ready
	ready = 300 * time.Millisecond
	s.T().Cleanup(func() { ready = original })
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	s.Require().NoError(err)
	address := listener.Addr().String()
	s.Require().NoError(listener.Close())
	first := Ready(address)
	s.Require().Error(first)

	start := time.Now()
	client, err := clientThrough(address)

	s.Nil(client)
	s.Equal(first, err)
	s.Less(time.Since(start), 100*time.Millisecond, "NewClient waited out the deadline again")
}

func (s *RedisSuite) TestAClientUsesTheOneConnectionItProved() {
	if runtime.GOMAXPROCS(0) < 2 {
		s.T().Skip("rueidis dials one connection on one CPU anyway")
	}
	address, accepted := s.forwarder(s.redis(), time.Now(), false)

	client, err := NewClient(rueidis.ClientOption{InitAddress: []string{address}, DisableCache: true})
	s.Require().NoError(err)
	s.T().Cleanup(client.Close)
	s.pingKeys(client)

	s.EqualValues(2, accepted.Load(), "Ready's PING and the client's one connection")
}

func (s *RedisSuite) TestAPipelineMultiplexTheCallerSetIsKept() {
	address, accepted := s.forwarder(s.redis(), time.Now(), false)

	client, err := NewClient(rueidis.ClientOption{InitAddress: []string{address}, DisableCache: true, PipelineMultiplex: 2})
	s.Require().NoError(err)
	s.T().Cleanup(client.Close)
	s.pingKeys(client)

	s.Greater(accepted.Load(), int64(2), "the caller asked for four connections")
}

// pingKeys sends commands on many keys, which rueidis spreads over every connection it has.
func (s *RedisSuite) pingKeys(client rueidis.Client) {
	for i := range 64 {
		key := "redisready-test:" + strconv.Itoa(i)
		s.Require().NoError(client.Do(context.Background(), client.B().Exists().Key(key).Build()).Error())
	}
}

func (s *RedisSuite) TestABuildThatFailsIsBuiltAgainUntilItSucceeds() {
	address := s.peer(time.Now(), "+PONG\r\n")
	builds := 0

	built, err := Retry(address, func() (string, error) {
		builds++
		if builds < 3 {
			return "", errors.New("context deadline exceeded")
		}
		return "node", nil
	})

	s.Require().NoError(err)
	s.Equal("node", built)
	s.Equal(3, builds)
}

func (s *RedisSuite) TestABuildThatNeverSucceedsIsAnErrorAtTheDeadline() {
	original := ready
	ready = 1200 * time.Millisecond
	s.T().Cleanup(func() { ready = original })
	address := s.peer(time.Now(), "+PONG\r\n")
	failed := errors.New("context deadline exceeded")

	start := time.Now()
	built, err := Retry(address, func() (*int, error) { return new(int), failed })

	s.Nil(built, "what a failed build returned was handed on")
	s.Require().ErrorIs(err, failed)
	s.Contains(err.Error(), address)
	s.GreaterOrEqual(time.Since(start), ready, "the build was given up on before the deadline")
	s.Less(time.Since(start), 5*time.Second, "the builds ran past their deadline")
}

func (s *RedisSuite) TestNothingIsBuiltWhereNoRedisAnswered() {
	original := ready
	ready = 300 * time.Millisecond
	s.T().Cleanup(func() { ready = original })
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	s.Require().NoError(err)
	address := listener.Addr().String()
	s.Require().NoError(listener.Close())
	first := Ready(address)
	s.Require().Error(first)
	builds := 0

	start := time.Now()
	_, err = Retry(address, func() (int, error) {
		builds++
		return 1, nil
	})

	s.Equal(first, err)
	s.Zero(builds, "a build was tried against a Redis that never answered")
	s.Less(time.Since(start), 100*time.Millisecond, "Retry waited out the deadline again")
}
