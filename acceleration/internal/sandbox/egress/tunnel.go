package egress

import (
	_ "embed"
	"net/url"
)

// forwarder is the sandbox's TLS tunnel to an https proxy: most of a sandbox's clients cannot
// use a TLS proxy URL (curl, urllib, wget and node fail: live probe, 2026-10-22), so they send
// plain HTTP to it on the sandbox's loopback.
//
//go:embed tunnel.py
var forwarder []byte

// The forwarder's file, its pid and log, in /tmp, as caPath. It listens on 127.0.0.1:3128,
// Squid's default port (https://www.squid-cache.org/Doc/config/http_port/), as in the probes.
const (
	tunnelPath = "/tmp/va-egress-tunnel.py"
	tunnelPID  = "/tmp/va-egress-tunnel.pid"
	tunnelLog  = "/tmp/va-egress-tunnel.log"
	tunnelPort = "3128"
)

// startTunnel is the command that starts the forwarder to proxy unless one runs and accepts. A
// restart cuts every download in flight (live probe, 2026-10-22), so a live one is left alone.
// setsid nohup keeps it running once the command that started it ends; it starts in about
// 100 ms (same probe), so 50 tries 0.1 s apart leave room. The tries stop once the forwarder
// it started has exited, with one more for a forwarder another check started. An image
// without python3 fails it.
func startTunnel(proxy *url.URL) string {
	up := `python3 -c 'import os, socket, sys; os.kill(int(open(sys.argv[1]).read()), 0); ` +
		`socket.create_connection(("127.0.0.1", ` + tunnelPort + `), 1).close()' ` + tunnelPID + ` 2>/dev/null`
	return `command -v python3 >/dev/null || { echo "the sandbox image has no python3, which its TLS tunnel to the proxy needs" >&2; exit 1; }; ` +
		up + ` && exit 0; ` +
		`setsid nohup python3 ` + tunnelPath + ` ` + tunnelPort + ` ` + proxy.Hostname() + ` ` + proxy.Port() + ` ` + caPath + ` ` +
		tunnelPID + ` </dev/null >>` + tunnelLog + ` 2>&1 & ` +
		`i=0; while [ $i -lt 50 ] && kill -0 $! 2>/dev/null; do ` + up + ` && exit 0; sleep 0.1; i=$((i+1)); done; ` +
		up + ` && exit 0; ` +
		`echo "the TLS tunnel to the proxy did not start: $(tail -n 3 ` + tunnelLog + `)" >&2; exit 1`
}
