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

LISTEN, HOST, PORT, CA, PID = sys.argv[1:6]
CONTEXT = ssl.create_default_context(cafile=CA)


async def pipe(reader, writer):
    try:
        while True:
            data = await reader.read(CHUNK)
            if not data:
                break
            writer.write(data)
            await writer.drain()
    except OSError:
        pass
    finally:
        writer.close()


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
        print("the egress proxy cannot be reached:", repr(error), file=sys.stderr, flush=True)
        client_writer.close()
        return
    upstream_writer.write(first)
    await asyncio.gather(pipe(client_reader, upstream_writer), pipe(upstream_reader, client_writer))


async def main():
    server = await asyncio.start_server(handle, "127.0.0.1", int(LISTEN), backlog=BACKLOG)
    # Written once the port is ours, so a live pid is a forwarder that holds it.
    with open(PID, "w") as pid:
        pid.write(str(os.getpid()))
    async with server:
        await server.serve_forever()


asyncio.run(main())
