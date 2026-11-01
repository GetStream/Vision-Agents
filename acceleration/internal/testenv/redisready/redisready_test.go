package redisready

import (
	"bufio"
	"context"
	"io"
	"net"
	"os"
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
// forwarded to upstream.
func (s *RedisSuite) forwarder(upstream string, until time.Time, hold bool) string {
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
			go func() {
				defer conn.Close()
				if time.Now().Before(until) {
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
	return listener.Addr().String()
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
	address := s.forwarder(s.redis(), until, true)

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
	address := s.forwarder(s.redis(), until, false)

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
	address := s.forwarder("", time.Now().Add(time.Hour), true)

	start := time.Now()
	client, err := clientThrough(address)

	s.Nil(client)
	s.Require().Error(err)
	s.Contains(err.Error(), address)
	s.GreaterOrEqual(time.Since(start), ready, "the client was given up on before the deadline")
	s.Less(time.Since(start), 5*time.Second, "the dial ran past its deadline")
}
