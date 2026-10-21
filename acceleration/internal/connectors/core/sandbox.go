package core

import (
	"fmt"
	"regexp"
)

// SandboxRule is the manifest's sandbox block: how code in a session's sandbox calls the
// provider's API with a connection's credential without holding it. The sandbox gets an env
// var whose value is a placeholder, and the sandbox's egress puts the credential in its place
// only in a request to one of Hosts. Daytona Secrets is such an egress
// (https://www.daytona.io/docs/en/secrets.md, opened October 20, 2026).
type SandboxRule struct {
	// Env is the name of the env var code in the sandbox reads the placeholder from, such as
	// GITHUB_TOKEN.
	Env string `yaml:"env" json:"env"`
	// Hosts are the only hosts a request may carry the credential to, each an exact host or
	// *. and a domain, which takes the domain and every name under it.
	Hosts []string `yaml:"hosts" json:"hosts"`
}

var (
	// sandboxEnv is a portable env var name: upper-case letters, digits and underscores, not
	// starting with a digit (POSIX.1-2024, Base Definitions, section 8.1,
	// https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/V1_chap08.html).
	sandboxEnv = regexp.MustCompile(`^[A-Z_][A-Z0-9_]*$`)
	// sandboxHost is a host as Daytona's allowlist takes one: «use hostnames; omit protocols,
	// paths, ports, or query strings», «exact hosts and *. wildcards» (secrets.md above). Its
	// labels are RFC 1123 section 2.1's, at least two as hostSuffix asks, so *.com cannot hand
	// a credential to a whole top-level domain; lowercase, so one host is written one way.
	sandboxHost = regexp.MustCompile(`^(\*\.)?[a-z0-9]([a-z0-9-]*[a-z0-9])?(\.[a-z0-9]([a-z0-9-]*[a-z0-9])?)+$`)
	// sandboxNumericTLD is a host whose last label is all digits. RFC 3696 section 2 says a
	// top-level domain is never all-numeric, so such a host is an IPv4 literal (1.2.3.4,
	// 0x7f.0.0.1 and *.0.0.1 included) or a name the resolver may read as one, and the
	// allowlist takes host names only. A label starting 0x is refused even when it is a real
	// name (api.0x.org): inet_aton reads 1.2.3.0x4 and 0x7f.0x0.0x0.0x1 as IPv4 literals, hex
	// parts included. List such a host only if one is ever needed.
	sandboxNumericTLD = regexp.MustCompile(`\.[0-9]+$|(^|\.)0x`)
)

func (m Manifest) checkSandbox(fail func(field, format string, args ...any)) {
	if !sandboxEnv.MatchString(m.Sandbox.Env) {
		fail("sandbox.env", "%q is not an env var name of upper-case letters, digits and underscores", m.Sandbox.Env)
	}
	if len(m.Sandbox.Hosts) == 0 {
		fail("sandbox.hosts", "is empty: a credential with no host would go to every host")
	}
	for i, host := range m.Sandbox.Hosts {
		if !sandboxHost.MatchString(host) || sandboxNumericTLD.MatchString(host) {
			fail(fmt.Sprintf("sandbox.hosts[%d]", i), "%q is not a lowercase host name or *. and a domain, with no scheme, port, path or query", host)
		}
	}
}
