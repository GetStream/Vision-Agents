package egress

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"mime"
	"net/http"
	"strings"

	"github.com/vektah/gqlparser/v2/ast"
	"github.com/vektah/gqlparser/v2/parser"
)

// maxGraphQLBody is the most of a GraphQL request's body the proxy reads to see its operations,
// 1 MiB (AI-2033): it bounds what one request holds in the router's memory, and a query that
// reads is a few KiB. A larger body is refused. The parser bounds the document's nesting itself
// (parser.MaxNestingDepth, 1000).
const maxGraphQLBody = 1 << 20

// graphQLKeys are the members of a GraphQL request the proxy reads (GraphQL over HTTP,
// https://http-spec.graphql.org/draft/, 4.1 GraphQL-over-HTTP Request), extensions left out: a
// member it does not read, such as a persisted document's id or hash, could name an operation
// it cannot see.
var graphQLKeys = map[string]bool{"query": true, "operationName": true, "variables": true}

// onlyQueries says whether r is one GraphQL request whose every operation is a query, or why
// not. It reads the body and puts back the same bytes, so the host gets them as
// the sandbox sent them. Each check refuses a request whose operation the host could read
// another way than the proxy: from the URL, another media type or charset, a member the proxy
// does not read, or a key sent twice.
func onlyQueries(r *http.Request) error {
	if r.URL.RawQuery != "" {
		return errors.New("a GraphQL POST with a URL query")
	}
	media, params, err := mime.ParseMediaType(r.Header.Get("Content-Type"))
	// A POST's body is UTF-8 JSON, application/json (4.4 POST).
	if err != nil || media != "application/json" || (params["charset"] != "" && !strings.EqualFold(params["charset"], "utf-8")) {
		return errors.New("a GraphQL POST that is not application/json in UTF-8")
	}
	// A body cut short is checked as far as it came, and the host gets those bytes.
	body, _ := io.ReadAll(io.LimitReader(r.Body, maxGraphQLBody+1))
	r.Body = io.NopCloser(bytes.NewReader(body))
	switch {
	case len(body) > maxGraphQLBody:
		return errors.New("a GraphQL POST over the size the proxy reads")
	case !json.Valid(body):
		return errors.New("a GraphQL POST whose body is not JSON")
	}
	// One request object: GitHub answers a batch (an array) 400 «Body should be a JSON object»
	// (live probe 2026-10-27, AI-2033 review), so the proxy refuses one too.
	return queries(body)
}

// queries says whether raw is one GraphQL request whose every operation is a query, or why not.
func queries(raw json.RawMessage) error {
	members, err := object(raw)
	if err != nil {
		return err
	}
	// A query that is missing or no string leaves query empty, a document with no operation.
	var query string
	_ = json.Unmarshal(members["query"], &query)
	// The parser's error may quote the document, so it is not returned.
	document, err := parser.ParseQuery(&ast.Source{Input: query})
	if err != nil || len(document.Operations) == 0 {
		return errors.New("a GraphQL request whose query is no document with an operation")
	}
	for _, operation := range document.Operations {
		if operation.Operation != ast.Query {
			return errors.New("a GraphQL " + string(operation.Operation))
		}
	}
	return nil
}

// object is raw's members, when it is a JSON object whose every key is one of graphQLKeys, once.
// It reads raw a token at a time: decoding into a map keeps the last of two equal keys, and a
// host could keep the first. raw is valid JSON (onlyQueries), so reading it fails nowhere.
func object(raw json.RawMessage) (map[string]json.RawMessage, error) {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	if open, _ := decoder.Token(); open != json.Delim('{') {
		return nil, errors.New("a GraphQL request that is no JSON object")
	}
	members := map[string]json.RawMessage{}
	for decoder.More() {
		key, _ := decoder.Token()
		var value json.RawMessage
		_ = decoder.Decode(&value)
		name, _ := key.(string)
		if _, twice := members[name]; twice || !graphQLKeys[name] {
			return nil, errors.New("a GraphQL request with a member the proxy does not read, or one twice")
		}
		members[name] = value
	}
	return members, nil
}
