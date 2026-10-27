//go:build integration

package egress_test

import (
	"bytes"
	"compress/gzip"
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/redis/rueidis"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox/egress"
)

const (
	// queries is the SQL read of the tests' grant: BigQuery's jobs.query path on a test host.
	queries = "https://bigquery.github.test/bigquery/v2/projects/p/queries"
	// sqlRefused is the end of every refusal on a host with a SQL read.
	sqlRefused = "code in the sandbox reads BigQuery with one SELECT per request, sent as POST /bigquery/v2/projects/<project>/queries " +
		"(jobs.query, such as client.query_and_wait in Python); client.query() sends jobs.insert, which is refused, and so are scripts, " +
		"DML, DDL and EXPORT DATA; write with the connector's tools\n"
	// escapedQuery is the key query with its q as a JSON escape, built here: a \u escape written
	// in this file's text can end up as the letter itself.
	escapedQuery = string(rune(92)) + "u0071uery"
)

// bigQuery is a fake BigQuery: it answers a dry run of jobs.query with dryStatus and dryAnswer,
// after dryDelay, and any other request 200 with no rows. It keeps every request it got.
type bigQuery struct {
	mu        sync.Mutex
	dryStatus int
	dryAnswer string
	dryDelay  time.Duration
	got       []bigQueryRequest
}

type bigQueryRequest struct {
	method, uri, authorization, contentType, body string
}

func (b *bigQuery) serve(w http.ResponseWriter, r *http.Request) {
	body, _ := io.ReadAll(r.Body)
	var members struct {
		DryRun bool `json:"dryRun"`
	}
	_ = json.Unmarshal(body, &members)
	b.mu.Lock()
	b.got = append(b.got, bigQueryRequest{method: r.Method, uri: r.URL.RequestURI(), authorization: r.Header.Get("Authorization"),
		contentType: r.Header.Get("Content-Type"), body: string(body)})
	status, answer, delay := b.dryStatus, b.dryAnswer, b.dryDelay
	b.mu.Unlock()
	if !members.DryRun {
		_, _ = io.WriteString(w, `{"kind":"bigquery#queryResponse","jobComplete":true,"rows":[]}`)
		return
	}
	time.Sleep(delay)
	w.WriteHeader(status)
	_, _ = io.WriteString(w, answer)
}

// dryRunAnswers has the fake answer a dry run with status and answer from now on.
func (b *bigQuery) dryRunAnswers(status int, answer string) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.dryStatus, b.dryAnswer, b.got = status, answer, nil
}

func (b *bigQuery) requests() []bigQueryRequest {
	b.mu.Lock()
	defer b.mu.Unlock()
	return append([]bigQueryRequest(nil), b.got...)
}

func (b *bigQuery) forget() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.got = nil
}

// queryRequest is one request of code in the sandbox to the SQL read, and the status the proxy
// answers it with.
type queryRequest struct {
	name, method, target, contentType, encoding, body string
	status                                            int
}

// queryCase is a JSON POST of body to jobs.query, answered with status.
func queryCase(name, body string, status int) queryRequest {
	return queryRequest{name: name, method: http.MethodPost, target: queries, contentType: "application/json", body: body, status: status}
}

// readBigQuery gives the test's grant BigQuery's SQL read on bigquery.github.test, and has every
// host answered by a fake BigQuery whose dry run says SELECT.
func (s *ProxySuite) readBigQuery() *bigQuery {
	grant := s.grant(s.placeholder, "bigquery.github.test", "api.github.test")
	grant.Reads = []core.SandboxRead{{Host: "bigquery.github.test", Path: "/bigquery/v2/projects/*/queries", SQL: true}}
	s.put(s.credential, egress.Record{Session: s.session, Grants: []egress.Grant{grant}})
	fake := &bigQuery{dryStatus: http.StatusOK, dryAnswer: `{"kind":"bigquery#queryResponse","statementType":"SELECT"}`}
	s.answering(fake.serve)
	return fake
}

// sendQueries sends each request by the sandbox. One the proxy lets through reaches the host
// twice, as the dry run with the token and the members with dryRun true, then with the token
// and its own bytes. One the proxy refuses reaches the host at most as the dry run, and gets the
// SQL read's answer, prefixed by why when a check said it.
func (s *ProxySuite) sendQueries(fake *bigQuery, requests []queryRequest, dryRun bool) {
	for _, c := range requests {
		fake.forget()
		request := s.request(c.method, c.target, "Bearer "+s.placeholder, c.body)
		if c.contentType != "" {
			request.Header.Set("Content-Type", c.contentType)
		}
		if c.encoding != "" {
			request.Header.Set("Content-Encoding", c.encoding)
		}

		response, body, err := s.send(s.sandbox(s.credential), request)

		s.Require().NoError(err, c.name)
		s.Equal(c.status, response.StatusCode, c.name)
		got := fake.requests()
		if c.status != http.StatusOK {
			s.True(strings.HasSuffix(body, sqlRefused), "%s: %s", c.name, body)
			if !dryRun {
				s.Empty(got, c.name)
				continue
			}
			s.Require().Len(got, 1, c.name)
			s.Equal("Bearer gho_first-token-of-the-connection", got[0].authorization, c.name)
			continue
		}
		s.Require().Len(got, 2, c.name)
		dry, sent := got[0], got[1]
		s.Equal(http.MethodPost, dry.method, c.name)
		s.Equal(strings.TrimPrefix(c.target, "https://bigquery.github.test"), dry.uri, c.name)
		s.Equal("Bearer gho_first-token-of-the-connection", dry.authorization, c.name)
		s.Equal("application/json", dry.contentType, c.name)
		var want map[string]any
		s.Require().NoError(json.Unmarshal([]byte(c.body), &want), c.name)
		want["dryRun"] = true
		wanted, err := json.Marshal(want)
		s.Require().NoError(err)
		s.JSONEq(string(wanted), dry.body, c.name)
		s.Equal(dry.uri, sent.uri, c.name)
		s.Equal("Bearer gho_first-token-of-the-connection", sent.authorization, c.name)
		s.Equal(c.body, sent.body, c.name)
	}
}

// TestASQLReadPassesWhatTheDryRunSaysIsOneSELECT: the proxy reads no SQL; BigQuery's dry run
// does, so a WITH query, a legacy SQL SELECT, a request that is a dry run itself and the body
// google-cloud-bigquery's client.query_and_wait sends, with its URL query, each pass when it
// says SELECT. A key escaped is the key.
func (s *ProxySuite) TestASQLReadPassesWhatTheDryRunSaysIsOneSELECT() {
	fake := s.readBigQuery()
	// _job_helpers.py (3.41.0) _to_query_request and query_and_wait; google-cloud-core 2.5.1
	// adds ?prettyPrint=false.
	client := queryCase("client.query_and_wait", `{"useLegacySql": false, "formatOptions": {"useInt64Timestamp": true}, `+
		`"query": "SELECT word FROM t LIMIT 1", "maxResults": 10, "jobCreationMode": "JOB_CREATION_OPTIONAL", `+
		`"requestId": "a1b2", "timeoutMs": 9750, "location": "US"}`, http.StatusOK)
	client.target += "?prettyPrint=false"

	s.sendQueries(fake, []queryRequest{
		queryCase("a SELECT", `{"query":"SELECT word FROM t LIMIT 1"}`, http.StatusOK),
		queryCase("a WITH query", `{"query":"WITH a AS (SELECT word FROM t) SELECT * FROM a"}`, http.StatusOK),
		queryCase("a legacy SQL SELECT", `{"query":"SELECT word FROM [p:d.t] LIMIT 1","useLegacySql":true}`, http.StatusOK),
		queryCase("a dry run", `{"query":"SELECT 1","dryRun":true}`, http.StatusOK),
		queryCase("a dry run turned off", `{"query":"SELECT 1","dryRun":false}`, http.StatusOK),
		queryCase("every member the proxy reads", `{"kind":"bigquery#queryRequest","query":"SELECT @x","useLegacySql":false,`+
			`"maxResults":1,"timeoutMs":1000,"location":"EU","dryRun":false,"defaultDataset":{"projectId":"p","datasetId":"d"},`+
			`"parameterMode":"NAMED","queryParameters":[{"name":"x","parameterType":{"type":"INT64"},"parameterValue":{"value":"1"}}],`+
			`"maximumBytesBilled":"1000000","useQueryCache":true,"formatOptions":{},"requestId":"r","jobCreationMode":"JOB_CREATION_REQUIRED"}`,
			http.StatusOK),
		queryCase("a key escaped", `{"`+escapedQuery+`":"SELECT 1"}`, http.StatusOK),
		queryCase("HTML in the query", `{"query":"SELECT '<a>&' AS x"}`, http.StatusOK),
		client,
	}, true)
}

// TestASQLReadRefusesWhatTheDryRunSaysIsNoSELECT: a script, an EXPORT DATA, DDL or DML, the dry
// run's own refusal (BigQuery's 403 for DML without write rights), its failure, an answer that is
// no JSON or names no type, a redirect and no answer in time each get the 403, and the query
// reaches BigQuery as the dry run alone.
func (s *ProxySuite) TestASQLReadRefusesWhatTheDryRunSaysIsNoSELECT() {
	fake := s.readBigQuery()
	s.proxy = httptest.NewServer(s.newProxy(s.store, func(options *egress.Options) { options.DryRunTimeout = 300 * time.Millisecond }))
	s.T().Cleanup(s.proxy.Close)
	query := queryCase("", `{"query":"SELECT 1; DELETE FROM t WHERE true"}`, http.StatusForbidden)

	for answer, why := range map[string]string{
		`{"statementType":"SCRIPT"}`:                "(a jobs.query SCRIPT, not one SELECT)",
		`{"statementType":"EXPORT_DATA"}`:           "(a jobs.query EXPORT_DATA, not one SELECT)",
		`{"statementType":"CREATE_SCHEMA"}`:         "(a jobs.query CREATE_SCHEMA, not one SELECT)",
		`{"statementType":"DELETE"}`:                "(a jobs.query DELETE, not one SELECT)",
		`{"statementType":"select"}`:                "(a jobs.query dry run that names no statement type)",
		`{"statementType":"SELECT; DROP"}`:          "(a jobs.query dry run that names no statement type)",
		`{"statementType":1}`:                       "(a jobs.query dry run whose answer is not JSON)",
		`{"kind":"bigquery#queryResponse"}`:         "(a jobs.query dry run that names no statement type)",
		`{"statistics":{"statementType":"SELECT"}}`: "(a jobs.query dry run that names no statement type)",
		`SELECT`: "(a jobs.query dry run whose answer is not JSON)",
		``:       "(a jobs.query dry run whose answer is not JSON)",
	} {
		fake.dryRunAnswers(http.StatusOK, answer)
		query.name = answer
		s.sendQueries(fake, []queryRequest{query}, true)
		request := s.request(http.MethodPost, queries, "Bearer "+s.placeholder, query.body)
		request.Header.Set("Content-Type", "application/json")
		_, body, err := s.send(s.sandbox(s.credential), request)
		s.Require().NoError(err)
		s.Equal("egress: POST bigquery.github.test/bigquery/v2/projects/p/queries refused "+why+": "+sqlRefused, body, answer)
	}
	for status, why := range map[int]string{
		http.StatusForbidden:           "(a jobs.query dry run answered 403)",
		http.StatusBadRequest:          "(a jobs.query dry run answered 400)",
		http.StatusInternalServerError: "(a jobs.query dry run answered 500)",
		http.StatusFound:               "(a jobs.query dry run answered 302)",
	} {
		fake.dryRunAnswers(status, `{"statementType":"SELECT"}`)
		query.name = why
		s.sendQueries(fake, []queryRequest{query}, true)
		s.Contains(s.logs.String(), `why="`+strings.Trim(why, "()")+`"`)
	}

	fake.dryRunAnswers(http.StatusOK, `{"statementType":"SELECT"}`)
	fake.mu.Lock()
	fake.dryDelay = time.Second
	fake.mu.Unlock()
	query.name = "no answer in time"
	s.sendQueries(fake, []queryRequest{query}, true)
	s.Contains(s.logs.String(), `why="a jobs.query dry run that got no answer (`)
	s.NotContains(s.logs.String(), "DELETE FROM t", "the SQL is never logged")
}

// TestASQLReadTakesOneJSONRequestOfTheMembersItReads: the body is one JSON object in UTF-8 with
// a string query, the members of jobs.query the proxy passes and each once; anything else reaches
// no host, not even as a dry run, and asks for no token.
func (s *ProxySuite) TestASQLReadTakesOneJSONRequestOfTheMembersItReads() {
	fake := s.readBigQuery()
	asked := s.asked

	s.sendQueries(fake, []queryRequest{
		queryCase("an empty body", ``, http.StatusForbidden),
		queryCase("SQL as it is", `SELECT 1`, http.StatusForbidden),
		queryCase("null", `null`, http.StatusForbidden),
		queryCase("a string", `"SELECT 1"`, http.StatusForbidden),
		queryCase("an array", `[{"query":"SELECT 1"}]`, http.StatusForbidden),
		queryCase("an array of a key and a value", `["query","SELECT 1"]`, http.StatusForbidden),
		queryCase("an empty object", `{}`, http.StatusForbidden),
		queryCase("a JSON object cut short", `{"query":"SELECT 1"`, http.StatusForbidden),
		queryCase("a second JSON value", `{"query":"SELECT 1"}{"query":"DELETE FROM t WHERE true"}`, http.StatusForbidden),
		queryCase("invalid UTF-8", "{\"query\":\"SELECT '\xff'\"}", http.StatusForbidden),
		queryCase("query twice, a DELETE last", `{"query":"SELECT 1","query":"DELETE FROM t WHERE true"}`, http.StatusForbidden),
		queryCase("query twice, escaped", `{"query":"SELECT 1","`+escapedQuery+`":"DELETE FROM t WHERE true"}`, http.StatusForbidden),
		queryCase("dryRun twice", `{"query":"DELETE FROM t WHERE true","dryRun":true,"dryRun":false}`, http.StatusForbidden),
		queryCase("Query beside query", `{"Query":"DELETE FROM t WHERE true","query":"SELECT 1"}`, http.StatusForbidden),
		queryCase("createSession", `{"query":"SELECT 1","createSession":true}`, http.StatusForbidden),
		queryCase("connectionProperties", `{"query":"SELECT 1","connectionProperties":[{"key":"session_id","value":"s"}]}`, http.StatusForbidden),
		queryCase("labels", `{"query":"SELECT 1","labels":{"a":"b"}}`, http.StatusForbidden),
		queryCase("a member in snake case", `{"query":"SELECT 1","dry_run":false}`, http.StatusForbidden),
		queryCase("a null query", `{"query":null}`, http.StatusForbidden),
		queryCase("an empty query", `{"query":""}`, http.StatusForbidden),
		queryCase("a number for a query", `{"query":1}`, http.StatusForbidden),
		queryCase("an array for a query", `{"query":["SELECT 1"]}`, http.StatusForbidden),
		queryCase("a string for dryRun", `{"query":"SELECT 1","dryRun":"true"}`, http.StatusForbidden),
		queryCase("null for dryRun", `{"query":"SELECT 1","dryRun":null}`, http.StatusOK),
	}, false)
	s.Equal(asked+1, s.asked, "only the one that passed asked for a token, once for its dry run and itself")
	s.Contains(s.logs.String(), `why="a jobs.query request with a member the proxy does not read, or one twice"`)
	s.Contains(s.logs.String(), `why="a jobs.query request that is no JSON object"`)
	s.Contains(s.logs.String(), `why="a jobs.query POST whose body is not JSON in UTF-8"`)
	s.Contains(s.logs.String(), `why="a jobs.query request whose query is no string"`)
	s.Contains(s.logs.String(), `why="a jobs.query request whose dryRun is no bool"`)
}

// TestASQLReadTakesAJSONBodyOfAMiBAtMostWithNoOtherURLQuery: the proxy reads a body as
// application/json in UTF-8, of 1 MiB at most, sent with no content coding and with no URL query
// but google-cloud-core's prettyPrint=false, which a host could read members from too.
func (s *ProxySuite) TestASQLReadTakesAJSONBodyOfAMiBAtMostWithNoOtherURLQuery() {
	fake := s.readBigQuery()
	query := `{"query":"SELECT 1"}`
	// 1 MiB is egress.maxQueryBody; JSON takes whitespace after its value (RFC 8259 section 2).
	full := query + strings.Repeat(" ", 1<<20-len(query))
	withType := func(contentType string, status int) queryRequest {
		c := queryCase(contentType, query, status)
		c.contentType = contentType
		return c
	}
	withQuery := func(raw string, status int) queryRequest {
		c := queryCase(raw, query, status)
		c.target += raw
		return c
	}
	zipped := &bytes.Buffer{}
	writer := gzip.NewWriter(zipped)
	_, _ = writer.Write([]byte(query))
	s.Require().NoError(writer.Close())

	s.sendQueries(fake, []queryRequest{
		withType("application/json; charset=utf-8", http.StatusOK),
		withType("Application/JSON; charset=UTF-8", http.StatusOK),
		withType("application/json; charset=iso-8859-1", http.StatusForbidden),
		withType("text/plain", http.StatusForbidden),
		withType("application/x-www-form-urlencoded", http.StatusForbidden),
		withType("", http.StatusForbidden),
		withType("application/json; charset", http.StatusForbidden),
		queryCase("a body of 1 MiB", full, http.StatusOK),
		queryCase("a byte over 1 MiB", full+" ", http.StatusForbidden),
		queryCase("gzip", zipped.String(), http.StatusForbidden),
		{name: "JSON sent as gzip", method: http.MethodPost, target: queries, contentType: "application/json", encoding: "gzip",
			body: query, status: http.StatusForbidden},
		{name: "JSON sent as identity and gzip", method: http.MethodPost, target: queries, contentType: "application/json",
			encoding: "identity, gzip", body: query, status: http.StatusForbidden},
		{name: "JSON sent as identity", method: http.MethodPost, target: queries, contentType: "application/json", encoding: "Identity",
			body: query, status: http.StatusOK},
		withQuery("?prettyPrint=false", http.StatusOK),
		withQuery("?prettyPrint=true", http.StatusForbidden),
		withQuery("?alt=json", http.StatusForbidden),
		withQuery("?prettyPrint=false&dryRun=false", http.StatusForbidden),
		withQuery("?query=DELETE", http.StatusForbidden),
	}, false)
	s.Contains(s.logs.String(), `why="a jobs.query POST over the size the proxy reads"`)
	s.Contains(s.logs.String(), `why="a jobs.query POST with a content coding"`)
	s.Contains(s.logs.String(), `why="a jobs.query POST with a URL query"`)
	s.Contains(s.logs.String(), `why="a jobs.query POST that is not application/json in UTF-8"`)
}

// TestBigQueryIsReadWithGETsAndJobsQueryAlone: GETs read datasets, tables, rows and a query's
// results; jobs.insert, which client.query() sends, and any other POST or method are refused
// with the SQL read's answer, reaching no host and asking for no token. The query's path takes
// one project.
func (s *ProxySuite) TestBigQueryIsReadWithGETsAndJobsQueryAlone() {
	fake := s.readBigQuery()
	base := "https://bigquery.github.test/bigquery/v2/projects/p"
	for _, target := range []string{base + "/datasets", base + "/datasets/d/tables/t/data?maxResults=10", base + "/queries/job-1?location=US",
		base + "/jobs/job-1"} {
		fake.forget()
		response, _, err := s.send(s.sandbox(s.credential), s.request(http.MethodGet, target, "Bearer "+s.placeholder, ""))
		s.Require().NoError(err, target)
		s.Equal(http.StatusOK, response.StatusCode, target)
		s.Require().Len(fake.requests(), 1, target)
		s.Equal("Bearer gho_first-token-of-the-connection", fake.requests()[0].authorization, target)
	}
	query := `{"query":"SELECT 1"}`
	jobsInsert := `{"configuration":{"query":{"query":"SELECT 1","useLegacySql":false}}}`
	s.sendQueries(fake, []queryRequest{
		{name: "jobs.insert", method: http.MethodPost, target: base + "/jobs", contentType: "application/json", body: jobsInsert,
			status: http.StatusForbidden},
		{name: "jobs.cancel", method: http.MethodPost, target: base + "/jobs/job-1/cancel", status: http.StatusForbidden},
		{name: "tables.insert", method: http.MethodPost, target: base + "/datasets/d/tables", contentType: "application/json",
			body: `{}`, status: http.StatusForbidden},
		{name: "a PUT to queries", method: http.MethodPut, target: queries, contentType: "application/json", body: query,
			status: http.StatusForbidden},
		{name: "a POST under queries", method: http.MethodPost, target: queries + "/job-1", contentType: "application/json",
			body: query, status: http.StatusForbidden},
		{name: "a POST of two projects", method: http.MethodPost, target: base + "/x/queries", contentType: "application/json",
			body: query, status: http.StatusForbidden},
		{name: "a POST with an escape", method: http.MethodPost, target: base + "/%71ueries", contentType: "application/json",
			body: query, status: http.StatusForbidden},
	}, false)
	_, body, err := s.send(s.sandbox(s.credential), s.request(http.MethodPost, base+"/jobs", "Bearer "+s.placeholder, jobsInsert))
	s.Require().NoError(err)
	s.Equal("egress: POST bigquery.github.test/bigquery/v2/projects/p/jobs refused: "+sqlRefused, body)
}

// TestARequestWithoutThePlaceholderIsDryRunAsItIsSent: a query with no placeholder goes to the
// dry run with the Authorization it carries, or none, as it goes on itself.
func (s *ProxySuite) TestARequestWithoutThePlaceholderIsDryRunAsItIsSent() {
	fake := s.readBigQuery()
	asked := s.asked
	for _, authorization := range []string{"", "Bearer ya29.the-sandboxs-own"} {
		fake.forget()
		request := s.request(http.MethodPost, queries, authorization, `{"query":"SELECT 1"}`)
		request.Header.Set("Content-Type", "application/json")

		response, _, err := s.send(s.sandbox(s.credential), request)

		s.Require().NoError(err)
		s.Equal(http.StatusOK, response.StatusCode)
		s.Require().Len(fake.requests(), 2)
		s.Equal(authorization, fake.requests()[0].authorization)
		s.Equal(authorization, fake.requests()[1].authorization)
	}
	s.Equal(asked, s.asked)
}

// TestASQLReadIsCountedAsTheOtherReadsAre: a query that passes is counted as a request with the
// token, one refused as denied_method, and the dry run is timed as a request to the host.
func (s *ProxySuite) TestASQLReadIsCountedAsTheOtherReadsAre() {
	fake := s.readBigQuery()
	passed := s.requestCount("bigquery.github.test", "POST", "token", "2xx")
	refused := s.requestCount("bigquery.github.test", "POST", "denied_method", "4xx")
	timed := s.metric("router_sandbox_egress_upstream_seconds", map[string]string{"host": "bigquery.github.test"})

	s.sendQueries(fake, []queryRequest{queryCase("a SELECT", `{"query":"SELECT 1"}`, http.StatusOK)}, true)
	fake.dryRunAnswers(http.StatusOK, `{"statementType":"SCRIPT"}`)
	s.sendQueries(fake, []queryRequest{queryCase("a script", `{"query":"SELECT 1; SELECT 2"}`, http.StatusForbidden)}, true)

	s.Equal(passed+1, s.requestCount("bigquery.github.test", "POST", "token", "2xx"))
	s.Equal(refused+1, s.requestCount("bigquery.github.test", "POST", "denied_method", "4xx"))
	s.Equal(timed+3, s.metric("router_sandbox_egress_upstream_seconds", map[string]string{"host": "bigquery.github.test"}))
}

// TestAnotherGrantedHostKeepsReadOnlysAnswer: beside a SQL read, a POST to another host of the
// grant, one with no read and one with a GraphQL read, gets read-only's answer as before (on the
// base, e1a71dcc4, every refused POST got it).
func (s *ProxySuite) TestAnotherGrantedHostKeepsReadOnlysAnswer() {
	fake := s.readBigQuery()

	response, body, err := s.send(s.sandbox(s.credential), s.request(http.MethodPost, "https://api.github.test/bigquery/v2/projects/p/queries",
		"Bearer "+s.placeholder, `{"query":"SELECT 1"}`))

	s.Require().NoError(err)
	s.Equal(http.StatusForbidden, response.StatusCode)
	s.Equal("egress: POST api.github.test/bigquery/v2/projects/p/queries refused: code in the sandbox only reads through a connector; "+
		"write with the connector's tools\n", body)
	s.Empty(fake.requests())

	grant := s.grant(s.placeholder, "bigquery.github.test", "api.github.test")
	grant.Reads = []core.SandboxRead{{Host: "bigquery.github.test", Path: "/bigquery/v2/projects/*/queries", SQL: true},
		{Host: "api.github.test", Path: "/graphql", GraphQL: true}}
	s.put(s.credential, egress.Record{Session: s.session, Grants: []egress.Grant{grant}})
	request := s.request(http.MethodPost, "https://api.github.test/graphql", "Bearer "+s.placeholder, `{"query":"mutation { a }"}`)
	request.Header.Set("Content-Type", "application/json")

	response, body, err = s.send(s.sandbox(s.credential), request)

	s.Require().NoError(err)
	s.Equal(http.StatusForbidden, response.StatusCode)
	s.Equal("egress: POST api.github.test/graphql refused: code in the sandbox only reads through a connector; "+
		"write with the connector's tools\n", body)
	s.Empty(fake.requests())
}

// TestASQLReadHoldsBesideAnotherReadOfItsPath: a SQL read beside a read of the same method and
// path still takes what the dry run says is one SELECT.
func (s *ProxySuite) TestASQLReadHoldsBesideAnotherReadOfItsPath() {
	fake := s.readBigQuery()
	grant := s.grant(s.placeholder, "bigquery.github.test")
	grant.Reads = []core.SandboxRead{{Host: "bigquery.github.test", Path: "/bigquery/v2/projects/*/queries", SQL: true},
		{Host: "bigquery.github.test", Method: http.MethodPost, Path: "/bigquery/v2/projects/*/*"}}
	s.put(s.credential, egress.Record{Session: s.session, Grants: []egress.Grant{grant}})
	fake.dryRunAnswers(http.StatusOK, `{"statementType":"SCRIPT"}`)

	s.sendQueries(fake, []queryRequest{queryCase("a script", `{"query":"SELECT 1; SELECT 2"}`, http.StatusForbidden)}, true)
}

// TestQueriesAtOnceEachGetTheirOwnDryRun: queries sent at the same time each pass on their own
// dry run's answer, with their own bytes.
func (s *ProxySuite) TestQueriesAtOnceEachGetTheirOwnDryRun() {
	fake := s.readBigQuery()
	var wg sync.WaitGroup
	statuses := make([]int, 8)
	for i := range statuses {
		wg.Go(func() {
			request := s.request(http.MethodPost, queries, "Bearer "+s.placeholder, `{"query":"SELECT `+strings.Repeat("1", i+1)+`"}`)
			request.Header.Set("Content-Type", "application/json")
			response, err := s.sandbox(s.credential).Do(request)
			if err == nil {
				statuses[i] = response.StatusCode
				_ = response.Body.Close()
			}
		})
	}
	wg.Wait()

	s.Equal([]int{200, 200, 200, 200, 200, 200, 200, 200}, statuses)
	sent := map[string]int{}
	for _, got := range fake.requests() {
		sent[got.body]++
	}
	for i := range statuses {
		s.Equal(1, sent[`{"query":"SELECT `+strings.Repeat("1", i+1)+`"}`], i)
	}
	s.Len(fake.requests(), 16)
}

// TestARefusedDryRunTellsTheModelWhatBigQuerySaid: the 403 of a dry run BigQuery refused carries
// its error.message, the token and its echoes replaced by the placeholder, cut to 300 bytes at a
// character's start; the log names the status alone, since the message can quote the SQL.
func (s *ProxySuite) TestARefusedDryRunTellsTheModelWhatBigQuerySaid() {
	fake := s.readBigQuery()
	token := "gho_first-token-of-the-connection"
	inBase64 := base64.StdEncoding.EncodeToString
	long := "a" + strings.Repeat("é", 200)

	for _, c := range []struct{ name, message, said string }{
		{"a syntax error", `Syntax error: Unexpected identifier \"SELEC\" at [1:1]`, `Syntax error: Unexpected identifier "SELEC" at [1:1]`},
		{"the token echoed", "Not found: " + token + " " + inBase64([]byte(token)) + " " + s.placeholder,
			"Not found: " + s.placeholder + " " + inBase64([]byte(s.placeholder)) + " " + s.placeholder},
		{"a long message", long, "a" + strings.Repeat("é", 149)},
	} {
		fake.dryRunAnswers(http.StatusBadRequest, `{"error":{"code":400,"message":"`+c.message+`","status":"INVALID_ARGUMENT"}}`)
		request := s.request(http.MethodPost, queries, "Bearer "+s.placeholder, `{"query":"SELEC 1"}`)
		request.Header.Set("Content-Type", "application/json")

		response, body, err := s.send(s.sandbox(s.credential), request)

		s.Require().NoError(err, c.name)
		s.Equal(http.StatusForbidden, response.StatusCode, c.name)
		s.Equal("egress: POST bigquery.github.test/bigquery/v2/projects/p/queries refused (a jobs.query dry run answered 400: "+c.said+
			"): "+sqlRefused, body, c.name)
		s.Len(fake.requests(), 1, c.name)
	}
	s.Contains(s.logs.String(), `why="a jobs.query dry run answered 400"`)
	s.NotContains(s.logs.String(), "Unexpected identifier")
	s.NotContains(s.logs.String(), "Not found")
}

// TestASQLReadTakesOneContentType: the dry run carries one Content-Type, so a query sent with
// two, the second one BigQuery could read the body by, reaches no host.
func (s *ProxySuite) TestASQLReadTakesOneContentType() {
	fake := s.readBigQuery()
	request := s.request(http.MethodPost, queries, "Bearer "+s.placeholder, `{"query":"SELECT 1"}`)
	request.Header.Add("Content-Type", "application/json")
	request.Header.Add("Content-Type", "application/x-www-form-urlencoded")

	response, body, err := s.send(s.sandbox(s.credential), request)

	s.Require().NoError(err)
	s.Equal(http.StatusForbidden, response.StatusCode)
	s.True(strings.HasSuffix(body, sqlRefused), body)
	s.Empty(fake.requests())
	s.Contains(s.logs.String(), `why="a jobs.query POST that is not application/json in UTF-8"`)
}

// TestAGrantThatEndsDuringTheDryRunSendsNothingOn: the grants are read again after the dry run,
// so a grant deleted meanwhile gets the 403 of a grant that ended, and grants that cannot be read
// then a 503; the query reaches BigQuery as the dry run alone.
func (s *ProxySuite) TestAGrantThatEndsDuringTheDryRunSendsNothingOn() {
	fake := s.readBigQuery()
	lost, err := rueidis.NewClient(rueidis.ClientOption{InitAddress: []string{os.Getenv("ROUTER_REDIS_ADDR")}, DisableCache: true})
	s.Require().NoError(err)
	s.proxy = httptest.NewServer(s.newProxy(egress.NewStore(lost)))
	s.T().Cleanup(s.proxy.Close)
	var during func()
	s.answering(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		if strings.Contains(string(body), `"dryRun":true`) {
			during()
		}
		r.Body = io.NopCloser(bytes.NewReader(body))
		fake.serve(w, r)
	})
	query := func() (int, string) {
		fake.forget()
		request := s.request(http.MethodPost, queries, "Bearer "+s.placeholder, `{"query":"SELECT 1"}`)
		request.Header.Set("Content-Type", "application/json")
		response, body, err := s.send(s.sandbox(s.credential), request)
		s.Require().NoError(err)
		return response.StatusCode, body
	}

	during = func() { s.NoError(s.store.Delete(context.Background(), s.credential)) }
	status, body := query()
	s.Equal(http.StatusForbidden, status)
	s.Equal("egress: the sandbox's grant for this host has ended\n", body)
	s.Len(fake.requests(), 1)

	grant := s.grant(s.placeholder, "bigquery.github.test")
	grant.Reads = []core.SandboxRead{{Host: "bigquery.github.test", Path: "/bigquery/v2/projects/*/queries", SQL: true}}
	s.put(s.credential, egress.Record{Session: s.session, Grants: []egress.Grant{grant}})
	during = lost.Close
	status, body = query()
	s.Equal(http.StatusServiceUnavailable, status)
	s.Equal("egress: the sandbox's grants cannot be read just now\n", body)
	s.Len(fake.requests(), 1)
}

// TestADryRunsAnswerIsReadToAMiBOnItsOwnConnection: an answer over 1 MiB is no JSON the proxy
// reads, and each dry run has a connection of its own, closed after its answer, so bytes a host
// sends after it reach no log.
func (s *ProxySuite) TestADryRunsAnswerIsReadToAMiBOnItsOwnConnection() {
	fake := s.readBigQuery()
	s.mu.Lock()
	s.addresses = nil
	s.mu.Unlock()

	s.sendQueries(fake, []queryRequest{queryCase("a SELECT", `{"query":"SELECT 1"}`, http.StatusOK)}, true)

	s.mu.Lock()
	s.Equal([]string{"bigquery.github.test:443", "bigquery.github.test:443"}, s.addresses, "the dry run and the query dial apart")
	s.mu.Unlock()
	answer := `{"statementType":"SELECT","schema":"` + strings.Repeat("x", 1<<20) + `"}`
	fake.dryRunAnswers(http.StatusOK, answer)
	s.sendQueries(fake, []queryRequest{queryCase("an answer over 1 MiB", `{"query":"SELECT 1"}`, http.StatusForbidden)}, true)
	s.Contains(s.logs.String(), `why="a jobs.query dry run whose answer is not JSON"`)
}
