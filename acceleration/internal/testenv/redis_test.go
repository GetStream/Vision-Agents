package testenv

import (
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv/redisready"
)

type RedisSuite struct {
	suite.Suite
}

func TestRedisSuite(t *testing.T) {
	suite.Run(t, new(RedisSuite))
}

func (s *RedisSuite) TestTheConfiguredRedisIsAwaitedWhenTheProcessStarts() {
	address := os.Getenv(redisAddressVar)
	if address == "" || redisready.Wait(address, time.Second) != nil {
		s.T().Skip("no Redis answering at " + redisAddressVar)
	}

	s.Equal(address, redisAwaited, "init did not wait for the Redis the suites dial")
}
