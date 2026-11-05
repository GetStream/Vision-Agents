package core

import (
	"fmt"
	"path"
	"regexp"
	"slices"
	"strings"
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
	// Reads are the requests to Hosts that only read with a method other than GET and HEAD.
	// Under the egress proxy (code_sandbox.broker: proxy) a request to a host of Hosts with any
	// other method is refused, so code in the sandbox writes only through the connector's tools.
	// A session takes them from the connector's latest revision, whatever revision its
	// connection pins (ReadsAsConnection leaves them out), so a new one needs no reconnect.
	// Left out of the JSON when empty, so a block without any is stored as it was.
	Reads []SandboxRead `yaml:"reads,omitempty" json:"reads,omitempty"`
}

// SandboxRead is one kind of request that reads with another method: Method to a path of Host
// that Path matches.
type SandboxRead struct {
	// Host is one of the rule's Hosts, as it is written there.
	Host string `yaml:"host" json:"host"`
	// Method is the request's method as sent, in upper case: methods are case-sensitive (RFC 9110
	// section 9.1). A GraphQL read leaves it out.
	Method string `yaml:"method" json:"method"`
	// Path is a path.Match pattern over the request's path, from its leading slash: * matches
	// within one segment.
	Path string `yaml:"path" json:"path"`
	// GraphQL makes it a GraphQL endpoint, where a query and a mutation are both a POST
	// (GraphQL over HTTP, https://http-spec.graphql.org/draft/, 4.4 POST): the proxy lets a POST
	// through only when every operation in its body is a query. Its Method is left out, so a
	// router that predates this field, which reads the stored revision without it, matches the
	// read to no request and refuses the POST. Left out of the JSON when false, so a read without
	// it is stored as it was.
	GraphQL bool `yaml:"graphql,omitempty" json:"graphql,omitempty"`
	// SQL makes it a SQL endpoint, where a SELECT and a write are both a POST: the proxy lets a
	// POST through only when the host's own dry run of its body says the statement is one SELECT.
	// The one such endpoint the proxy reads is BigQuery's jobs.query
	// (https://cloud.google.com/bigquery/docs/reference/rest/v2/jobs/query). Its Method is left
	// out, as a GraphQL read's is, and it is left out of the JSON when false.
	SQL bool `yaml:"sql,omitempty" json:"sql,omitempty"`
	// LFS makes it a Git LFS batch endpoint, where a download and an upload are both a POST
	// (https://github.com/git-lfs/git-lfs/blob/main/docs/api/batch.md): the proxy lets a POST
	// through only when its body asks to download. Its Method is left out, as a GraphQL read's is,
	// and it is left out of the JSON when false.
	LFS bool `yaml:"lfs,omitempty" json:"lfs,omitempty"`
	// Downloads are the hosts an LFS read's answer sends the client to for the objects
	// (actions.download.href), which carry their own authorization. The proxy reaches them for
	// GET and HEAD alone and puts no token in a request to one, so none may be one of the rule's
	// Hosts.
	Downloads []string `yaml:"downloads,omitempty" json:"downloads,omitempty"`
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
	// sandboxMethod is a method as a client sends one that reads: upper-case letters, as every
	// method in RFC 9110 section 9 is.
	sandboxMethod = regexp.MustCompile(`^[A-Z]+$`)
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
	for i, read := range m.Sandbox.Reads {
		field := fmt.Sprintf("sandbox.reads[%d]", i)
		if !slices.Contains(m.Sandbox.Hosts, read.Host) {
			fail(field+".host", "%q is not one of sandbox.hosts", read.Host)
		}
		body := read.GraphQL || read.SQL || read.LFS
		switch {
		case read.GraphQL && read.SQL:
			fail(field+".sql", "is set beside graphql: a read is one kind of endpoint")
		case read.LFS && (read.GraphQL || read.SQL):
			fail(field+".lfs", "is set beside graphql or sql: a read is one kind of endpoint")
		case body && read.Method != "":
			fail(field+".method", "%q is set on a GraphQL, SQL or LFS read, which takes a POST and leaves its method out", read.Method)
		case !body && !sandboxMethod.MatchString(read.Method):
			fail(field+".method", "%q is not a method in upper-case letters", read.Method)
		}
		if len(read.Downloads) > 0 && !read.LFS {
			fail(field+".downloads", "is set on a read that is no LFS read")
		}
		for j, host := range read.Downloads {
			switch {
			case !sandboxHost.MatchString(host) || sandboxNumericTLD.MatchString(host):
				fail(fmt.Sprintf("%s.downloads[%d]", field, j), "%q is not a lowercase host name or *. and a domain, with no scheme, port, path or query", host)
			case slices.ContainsFunc(m.Sandbox.Hosts, func(entry string) bool { return hostCovers(entry, host) }):
				fail(fmt.Sprintf("%s.downloads[%d]", field, j), "%q is under one of sandbox.hosts, which get the credential", host)
			}
		}
		// path.Match reports a malformed pattern whatever it is matched against.
		if _, err := path.Match(read.Path, ""); err != nil || !strings.HasPrefix(read.Path, "/") {
			fail(field+".path", "%q is not a path pattern starting with /", read.Path)
		}
	}
}

// hostCovers says whether entry, an exact host or *. and a domain, takes host, as the egress
// matches one: a wildcard takes the domain and every name under it.
func hostCovers(entry, host string) bool {
	host = strings.TrimPrefix(host, "*.")
	if domain, wildcard := strings.CutPrefix(entry, "*."); wildcard {
		return host == domain || strings.HasSuffix(host, "."+domain)
	}
	return host == entry
}
