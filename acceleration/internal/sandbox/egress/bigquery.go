package egress

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"mime"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"
)

const (
	// maxQueryBody is the most of a jobs.query body the proxy reads, 1 MiB, as maxGraphQLBody: an
	// unresolved GoogleSQL query «can be up to 1 MB long»
	// (https://docs.cloud.google.com/bigquery/quotas, opened 2026-10-27). A larger body is refused.
	maxQueryBody = 1 << 20
	// maxDryRunAnswer is the most of a dry run's answer the proxy reads for its statementType: the
	// answer of a SELECT's dry run is its schema and statistics, no rows.
	maxDryRunAnswer = 1 << 20
	// dryRunTimeout bounds a dry run, its answer read whole: half of the 60 s google-cloud-core
	// gives a request (google/cloud/_http/__init__.py, _DEFAULT_TIMEOUT, 2.5.1), so the query it
	// lets through has the other half.
	dryRunTimeout = 30 * time.Second
	// prettyPrintOff is the one URL query a jobs.query POST may carry: google-cloud-core adds it to
	// every request (google/cloud/_http/__init__.py:270, 2.5.1), client.query_and_wait's included.
	prettyPrintOff = "prettyPrint=false"
)

// queryKeys are the members of a jobs.query request (QueryRequest,
// https://docs.cloud.google.com/bigquery/docs/reference/rest/v2/jobs/query, «Last updated
// 2026-06-30») the proxy passes: those that pick and shape what one SELECT reads. Any other is
// refused, createSession and connectionProperties (a session, whose later queries read its
// state) and labels, reservation or jobTimeoutMs among them, as is a member added later.
var queryKeys = map[string]bool{
	"query": true, "useLegacySql": true, "maxResults": true, "timeoutMs": true, "location": true,
	"dryRun": true, "defaultDataset": true, "parameterMode": true, "queryParameters": true,
	"maximumBytesBilled": true, "useQueryCache": true, "formatOptions": true, "requestId": true,
	"jobCreationMode": true, "kind": true,
}

// statementName is a statementType as the proxy repeats it to the model: BigQuery's are upper-case
// words joined by underscores (SELECT, SCRIPT, EXPORT_DATA, CREATE_SCHEMA; live probe
// 2026-10-27T19:55Z).
var statementName = regexp.MustCompile(`^[A-Z_]{1,32}$`)

// onlyOneQuery is r's jobs.query request, when it is one the proxy reads as the host does, or why
// not. It reads the body and puts back the same bytes, so the host gets them as the sandbox sent
// them. Each check refuses a request the host could read another way than the proxy: from the
// URL, another media type or charset, a content coding, a member the proxy does not read, or a
// key sent twice.
func onlyOneQuery(r *http.Request) (map[string]json.RawMessage, error) {
	if r.URL.RawQuery != "" && r.URL.RawQuery != prettyPrintOff {
		return nil, errors.New("a jobs.query POST with a URL query")
	}
	media, params, err := mime.ParseMediaType(r.Header.Get("Content-Type"))
	if err != nil || media != "application/json" || (params["charset"] != "" && !strings.EqualFold(params["charset"], "utf-8")) {
		return nil, errors.New("a jobs.query POST that is not application/json in UTF-8")
	}
	for _, coding := range r.Header.Values("Content-Encoding") {
		for _, token := range strings.Split(coding, ",") {
			if token = strings.TrimSpace(token); token != "" && !strings.EqualFold(token, "identity") {
				return nil, errors.New("a jobs.query POST with a content coding")
			}
		}
	}
	// A body cut short is checked as far as it came, and the host gets those bytes.
	body, _ := io.ReadAll(io.LimitReader(r.Body, maxQueryBody+1))
	r.Body = io.NopCloser(bytes.NewReader(body))
	switch {
	case len(body) > maxQueryBody:
		return nil, errors.New("a jobs.query POST over the size the proxy reads")
	// The dry run sends the members on re-encoded, and invalid UTF-8 could be read two ways.
	case !json.Valid(body) || !utf8.Valid(body):
		return nil, errors.New("a jobs.query POST whose body is not JSON in UTF-8")
	}
	members, err := queryObject(body)
	if err != nil {
		return nil, err
	}
	var query string
	var dryRun bool
	if json.Unmarshal(members["query"], &query) != nil || query == "" {
		return nil, errors.New("a jobs.query request whose query is no string")
	}
	if raw, set := members["dryRun"]; set && json.Unmarshal(raw, &dryRun) != nil {
		return nil, errors.New("a jobs.query request whose dryRun is no bool")
	}
	return members, nil
}

// queryObject is raw's members, when it is a JSON object whose every key is one of queryKeys,
// once. It reads raw a token at a time, as object does: decoding into a map keeps the last of
// two equal keys, and a host could keep the first. raw is valid JSON (onlyOneQuery).
func queryObject(raw []byte) (map[string]json.RawMessage, error) {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	if open, _ := decoder.Token(); open != json.Delim('{') {
		return nil, errors.New("a jobs.query request that is no JSON object")
	}
	members := map[string]json.RawMessage{}
	for decoder.More() {
		key, _ := decoder.Token()
		var value json.RawMessage
		_ = decoder.Decode(&value)
		name, _ := key.(string)
		if _, twice := members[name]; twice || !queryKeys[name] {
			return nil, errors.New("a jobs.query request with a member the proxy does not read, or one twice")
		}
		members[name] = value
	}
	return members, nil
}

// selects says whether the host's dry run of the request r carries, members with dryRun true,
// says it is one SELECT, or why not. The dry run goes to r's URL with authorization, as the
// request itself would, and the host's own parser reads the SQL: BigQuery answers a dry run
// with the statement's type, SCRIPT for more than one statement, EXPORT_DATA, CREATE_SCHEMA,
// and a 403 for DML on a table the token cannot write (live probe 2026-10-27T19:55Z). Its error
// names the answer's status or type alone, never its body.
func (p *Proxy) selects(ctx context.Context, r *http.Request, t tunnel, members map[string]json.RawMessage, authorization string) error {
	members["dryRun"] = json.RawMessage("true")
	body := &bytes.Buffer{}
	encoder := json.NewEncoder(body)
	// The members go on as the sandbox wrote them: an escaped < is the same JSON string, and
	// none is needed.
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(members); err != nil {
		return errors.New("a jobs.query request the proxy cannot encode again")
	}
	ctx, cancel := context.WithTimeout(ctx, p.dryRunTimeout)
	defer cancel()
	dry, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://"+t.host+r.URL.RequestURI(), body)
	if err != nil {
		return errors.New("a jobs.query request the proxy cannot send again")
	}
	dry.Header.Set("Content-Type", "application/json")
	dry.Header.Set("Accept-Encoding", "identity")
	if authorization != "" {
		dry.Header.Set("Authorization", authorization)
	}
	// A connection reused after a host sent more than its answer logs those bytes, as for the
	// request itself (inner).
	dry.Close = true
	response, err := timed{next: p.transport, host: t.label}.RoundTrip(dry)
	if err != nil {
		return errors.New("a jobs.query dry run that got no answer (" + errorClass(err) + ")")
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return errors.New("a jobs.query dry run answered " + strconv.Itoa(response.StatusCode))
	}
	var answer struct {
		StatementType string `json:"statementType"`
	}
	raw, err := io.ReadAll(io.LimitReader(response.Body, maxDryRunAnswer))
	if err != nil || json.Unmarshal(raw, &answer) != nil {
		return errors.New("a jobs.query dry run whose answer is not JSON")
	}
	if answer.StatementType != "SELECT" {
		if !statementName.MatchString(answer.StatementType) {
			return errors.New("a jobs.query dry run that names no statement type")
		}
		return errors.New("a jobs.query " + answer.StatementType + ", not one SELECT")
	}
	return nil
}
