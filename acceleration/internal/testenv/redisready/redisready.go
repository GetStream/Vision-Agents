// Package redisready waits for the Redis the tests dial to answer. It has no init, so a test
// that does not load the test settings can use it.
package redisready

import (
	"fmt"
	"io"
	"net"
	"sync"
	"time"

	"github.com/redis/rueidis"
)

// ready is how long Ready waits for an answer. A Redis container started just
// before the run answered 12.9 s after `docker run` returned, with the machine loaded;
// 60 s leaves room above that without hiding a Redis that is not coming. With no Redis at
// the address, Ready spends all of it once per process, then fails.
var ready = 60 * time.Second

// redisAttempt bounds one PING: a connection that is accepted and then left unanswered
// must not use up the whole wait.
const redisAttempt = time.Second

var (
	mu      sync.Mutex
	results = map[string]error{}
)

// Ready waits for the Redis at address to answer PING, for at most one minute, once per
// process: later calls for the same address return the first result at once.
func Ready(address string) error {
	mu.Lock()
	defer mu.Unlock()
	if err, ok := results[address]; ok {
		return err
	}
	err := Wait(address, ready)
	results[address] = err
	return err
}

// NewClient is rueidis.NewClient for a test: it tries again until Ready's minute is up, and
// at once with Ready's error when no Redis answered at the address.
//
// rueidis gives each new connection a 5 s timeout for the TCP connect, then the same again
// for its HELLO handshake. Through Docker's port forwarding on a loaded machine, a connection
// opened after Ready got its answer can still sit past that timeout, or be closed at once.
// NewClient proves one connection, so a caller that left PipelineMultiplex at 0 gets -1, one
// connection: rueidis' default of four dials the other three on first use, unproven.
func NewClient(options rueidis.ClientOption) (rueidis.Client, error) {
	if len(options.InitAddress) > 0 {
		if err := Ready(options.InitAddress[0]); err != nil {
			return nil, err
		}
	}
	if options.PipelineMultiplex == 0 {
		options.PipelineMultiplex = -1
	}
	deadline := time.Now().Add(ready)
	for {
		client, err := rueidis.NewClient(options)
		if err == nil {
			return client, nil
		}
		if time.Now().After(deadline) {
			return nil, fmt.Errorf("redisready: no client for %v within %s: %w", options.InitAddress, ready, err)
		}
		time.Sleep(100 * time.Millisecond)
	}
}

// Wait returns once the Redis at address answers PING, or an error after the given time.
//
// A Redis container that has only just started either refuses the connection or, with
// Docker's port forwarding, accepts it before Redis replies. A client dialled then sits in
// its handshake until its own 5 s dial timeout and fails. Both are waited for.
func Wait(address string, within time.Duration) error {
	deadline := time.Now().Add(within)
	for {
		err := ping(address)
		if err == nil {
			return nil
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("redis at %s did not answer PING within %s: %w", address, within, err)
		}
		time.Sleep(100 * time.Millisecond)
	}
}

// ping sends one inline PING and wants +PONG back. A Redis still loading its dataset
// answers with an error instead, which is not ready.
func ping(address string) error {
	conn, err := net.DialTimeout("tcp", address, redisAttempt)
	if err != nil {
		return err
	}
	defer conn.Close()
	if err := conn.SetDeadline(time.Now().Add(redisAttempt)); err != nil {
		return err
	}
	if _, err := conn.Write([]byte("PING\r\n")); err != nil {
		return err
	}
	const pong = "+PONG\r\n"
	reply := make([]byte, len(pong))
	if _, err := io.ReadFull(conn, reply); err != nil {
		return err
	}
	if string(reply) != pong {
		return fmt.Errorf("unexpected reply %q", reply)
	}
	return nil
}
