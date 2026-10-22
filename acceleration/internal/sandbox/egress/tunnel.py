"""The sandbox's TLS tunnel to the router's egress proxy.

Clients send their proxy traffic in plain HTTP to 127.0.0.1, and this carries each connection
over a TLS connection of its own to the proxy, verified with the router CA alone. So the proxy
credential in Proxy-Authorization never crosses the network in the clear.

One thread and asyncio: two threads on one SSL socket hang and crash the forwarder (live probe,
2026-10-22).

Usage: python3 tunnel.py <listen port> <proxy IPv4 address> <proxy port> <CA file> <pid file>
"""

import asyncio
import os
import ssl
import sys

# One read: 64 KiB, asyncio's own stream buffer limit (_DEFAULT_LIMIT, CPython Lib/asyncio/streams.py).
CHUNK = 2**16
# Seconds a TLS connection to the proxy may take to open, as the probed forwarder had; a refused
# or unreachable proxy fails sooner.
CONNECT_TIMEOUT = 10
# Connections waiting to be accepted: Linux's SOMAXCONN before 5.4 (listen(2)), above asyncio's
# default of 100, so a burst of parallel clients is not refused.
BACKLOG = 128
# Seconds the proxy may stay silent once a client has stopped sending: the proxy's own idle
# timeout for a tunnel, http.DefaultTransport's IdleConnTimeout (egress/proxy.go idleTimeout).
IDLE_AFTER_EOF = 90

LISTEN, HOST, PORT, CA, PID = sys.argv[1:6]
CONTEXT = ssl.create_default_context(cafile=CA)


async def pipe(reader, writer, read):
    """Copies reader to writer until reader ends, adding what it read to read[0]."""
    try:
        while True:
            data = await reader.read(CHUNK)
            if not data:
                break
            read[0] += len(data)
            writer.write(data)
            await writer.drain()
    except OSError:
        pass


async def handle(client_reader, client_writer):
    try:
        # A proxy's client speaks first, so a connection that sends nothing, such as the check
        # that the forwarder accepts, opens no connection to the proxy.
        first = await client_reader.read(CHUNK)
        if not first:
            client_writer.close()
            return
        upstream_reader, upstream_writer = await asyncio.wait_for(
            asyncio.open_connection(HOST, int(PORT), ssl=CONTEXT), CONNECT_TIMEOUT
        )
    except (OSError, asyncio.TimeoutError) as error:
        print(
            "the egress proxy cannot be reached:",
            repr(error),
            file=sys.stderr,
            flush=True,
        )
        client_writer.close()
        return
    upstream_writer.write(first)

    def close(_):
        client_writer.close()
        upstream_writer.close()

    # The proxy's end, or a write to a client that has gone, ends the connection both ways.
    received = [0]
    reply = asyncio.ensure_future(pipe(upstream_reader, client_writer, received))
    reply.add_done_callback(close)
    await pipe(client_reader, upstream_writer, [0])
    # A client that stops sending may still read the reply (a half-close). The TLS connection
    # cannot pass that on (asyncio's SSL transport has no write_eof), so it stays open while the
    # proxy still answers.
    while not reply.done():
        before = received[0]
        await asyncio.wait({reply}, timeout=IDLE_AFTER_EOF)
        if received[0] == before:
            close(reply)
            break


async def main():
    server = await asyncio.start_server(
        handle, "127.0.0.1", int(LISTEN), backlog=BACKLOG
    )
    # Written once the port is ours, so a live pid is a forwarder that holds it.
    with open(PID, "w") as pid:
        pid.write(str(os.getpid()))
    async with server:
        await server.serve_forever()


asyncio.run(main())
