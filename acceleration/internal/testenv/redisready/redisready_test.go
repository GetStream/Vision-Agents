package redisready

import (
	"bufio"
	"net"
	"testing"
	"time"

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
