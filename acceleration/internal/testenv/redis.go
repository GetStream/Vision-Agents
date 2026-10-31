package testenv

import (
	"errors"
	"fmt"
	"io"
	"net"
	"syscall"
	"time"
)

// redisReady is how long awaitRedis waits for an answer. A Redis container started just
// before the run answered 12.9 s after `docker run` returned, with the machine loaded;
// 60 s leaves room above that without hiding a Redis that is not coming.
const redisReady = 60 * time.Second

// redisAttempt bounds one PING: a connection that is accepted and then left unanswered
// must not use up the whole wait.
const redisAttempt = time.Second

// awaitRedis returns once the Redis at address answers PING, or an error when it has
// accepted connections and still not answered within the given time.
//
// A Redis container that has only just started accepts the connection (Docker's port
// forwarding does) before Redis replies. A client dialled then sits in its handshake until
// its own 5 s dial timeout and fails. A connection that is refused is not that case: nothing
// is there to wait for, and a suite that needs the Redis says so itself, so this returns nil
// at once rather than hold every package that imports testenv while no Redis runs.
func awaitRedis(address string, within time.Duration) error {
	deadline := time.Now().Add(within)
	for {
		err := ping(address)
		if err == nil || errors.Is(err, syscall.ECONNREFUSED) {
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
