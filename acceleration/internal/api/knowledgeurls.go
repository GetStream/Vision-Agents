package api

import (
	"context"
	"net/http"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge/urls"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/danielgtaylor/huma/v2"
)

// errNoKnowledgeURLs is what these paths say on a deployment that cannot honour a
// subscription: it takes a database to remember one and a reader to fetch the page.
var errNoKnowledgeURLs = notConfigured("knowledge urls are not available: no database or no way to read a page configured")

// errUnknownKnowledgeURL is what a caller is told about a page that is not theirs, which is
// the same thing they are told about one that never existed.
var errUnknownKnowledgeURL = APIError{
	Type: ErrorTypeNotFound, Code: codeKnowledgeURLNotFound,
	Message: "no such knowledge url",
}

// listKnowledgeUrls returns the pages the calling customer's knowledge bases are filled
// from, newest first.
func (s *Server) listKnowledgeUrls(ctx context.Context, request *listKnowledgeUrlsRequest) (*listKnowledgeUrlsResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if s.pages == nil {
		return nil, errNoKnowledgeURLs
	}

	namespace := ""
	if request.Namespace.ptr() != nil {
		namespace = *request.Namespace.ptr()
	}

	stored, err := s.pages.List(ctx, customerID, namespace)
	if err != nil {
		return nil, err
	}

	listed := make([]KnowledgeUrl, 0, len(stored))
	for _, page := range stored {
		listed = append(listed, knowledgeURLOf(page))
	}
	return &listKnowledgeUrlsResponse{Body: listed}, nil
}

// addKnowledgeUrl subscribes a knowledge base to a page and reads it.
//
// A page that could not be read is a 201 with the row in the failed state rather than an
// error: the subscription was made, and what went wrong reading it is on the row where the
// caller can see it and try again.
func (s *Server) addKnowledgeUrl(ctx context.Context, request *addKnowledgeUrlRequest) (*addKnowledgeUrlResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if s.pages == nil {
		return nil, errNoKnowledgeURLs
	}
	if request.Body == nil {
		return nil, invalidRequest("a request body is required")
	}

	wanted := urls.Subscription{Namespace: request.Body.Namespace, URL: request.Body.Url}
	if request.Body.Title != nil {
		wanted.Title = *request.Body.Title
	}
	if request.Body.Description != nil {
		wanted.Description = *request.Body.Description
	}
	if request.Body.RefreshHours != nil {
		wanted.RefreshHours = *request.Body.RefreshHours
	}

	page, err := s.pages.Add(ctx, customerID, wanted)
	if err != nil {
		return nil, invalidRequest(err.Error())
	}
	return &addKnowledgeUrlResponse{Body: knowledgeURLOf(page)}, nil
}

// getKnowledgeUrl returns one page.
func (s *Server) getKnowledgeUrl(ctx context.Context, request *getKnowledgeUrlRequest) (*getKnowledgeUrlResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if s.pages == nil {
		return nil, errNoKnowledgeURLs
	}

	page, err := s.pages.Get(ctx, customerID, request.Id)
	if err != nil {
		return nil, errUnknownKnowledgeURL
	}
	return &getKnowledgeUrlResponse{Body: knowledgeURLOf(page)}, nil
}

// listKnowledgeUrlPassages reads back what a page was last read into.
func (s *Server) listKnowledgeUrlPassages(ctx context.Context, request *listKnowledgeUrlPassagesRequest) (*listKnowledgeUrlPassagesResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if s.pages == nil || s.knowledge == nil {
		return nil, errNoKnowledgeURLs
	}

	page, err := s.pages.Get(ctx, customerID, request.Id)
	if err != nil {
		return nil, errUnknownKnowledgeURL
	}
	passages, err := s.knowledgePassages(ctx, customerID, page.Namespace, page.URL, page.Passages)
	if err != nil {
		return nil, err
	}
	return &listKnowledgeUrlPassagesResponse{Body: passages}, nil
}

// deleteKnowledgeUrl stops filling a knowledge base from a page, and removes the passages
// it wrote.
func (s *Server) deleteKnowledgeUrl(ctx context.Context, request *deleteKnowledgeUrlRequest) (*struct{}, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if s.pages == nil {
		return nil, errNoKnowledgeURLs
	}

	if err := s.pages.Remove(ctx, customerID, request.Id); err != nil {
		return nil, errUnknownKnowledgeURL
	}
	return nil, nil
}

// indexKnowledgeUrl reads a page again.
func (s *Server) indexKnowledgeUrl(ctx context.Context, request *indexKnowledgeUrlRequest) (*indexKnowledgeUrlResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if s.pages == nil {
		return nil, errNoKnowledgeURLs
	}

	page, err := s.pages.Reindex(ctx, customerID, request.Id)
	if err != nil {
		return nil, errUnknownKnowledgeURL
	}
	return &indexKnowledgeUrlResponse{Body: knowledgeURLOf(page)}, nil
}

// knowledgeURLOf is the stored row as the API describes it.
func knowledgeURLOf(page store.KnowledgeURL) KnowledgeUrl {
	described := KnowledgeUrl{
		Id:            page.ID,
		Namespace:     page.Namespace,
		Url:           page.URL,
		State:         KnowledgeUrlState(page.State),
		Passages:      page.Passages,
		LastIndexedAt: page.LastIndexedAt,
		CreatedAt:     page.CreatedAt,
		UpdatedAt:     page.UpdatedAt,
	}
	// What the subscription calls the page wins over what the page calls itself: a caller
	// who named it said what they wanted it filed under.
	title := page.DeclaredTitle
	if title == "" {
		title = page.Title
	}
	if title != "" {
		described.Title = &title
	}
	if page.Description != "" {
		described.Description = &page.Description
	}
	if page.Error != "" {
		described.Error = &page.Error
	}
	if page.RefreshHours > 0 {
		described.RefreshHours = &page.RefreshHours
	}
	return described
}

// registerKnowledgeurls declares the operations served in knowledgeurls.go.
func (s *Server) registerKnowledgeurls(api huma.API) {
	huma.Register(api, huma.Operation{
		OperationID: "listKnowledgeUrls",
		Method:      http.MethodGet,
		Path:        "/v1/agents/knowledge/urls",
		Summary:     "The pages a knowledge base is kept filled from",
		Responses: map[string]*huma.Response{
			"200": {Description: "The customer's pages, newest first"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden},
	}, s.listKnowledgeUrls)
	huma.Register(api, huma.Operation{
		OperationID: "addKnowledgeUrl",
		Method:      http.MethodPost,
		Path:        "/v1/agents/knowledge/urls",
		Summary:     "Keep a knowledge base filled from a page",
		Description: "Posting a document is a thing that happens once; a url is a subscription, because the " +
			"page behind it changes and nobody re-posts it. The page is fetched, turned into " +
			"markdown, cut into passages the same way a document is, and written under the url so a " +
			"later read replaces it rather than adding a second copy.\n" +
			"The fetch is queued rather than done before this answers, since a live crawl takes " +
			"seconds: the page comes back pending, and indexed or failed once it has been read. A " +
			"read that fails is tried again a few times first. A page that could not be read is " +
			"still stored, in the failed state with the reason on it, rather than refused and " +
			"forgotten.\n" +
			"Adding a page a knowledge base already has is a re-read of it rather than a second " +
			"copy: the subscription is the url, so a declaration of what an agent reads can be " +
			"applied again without being diffed first.\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an end " +
			"user's device.",
		DefaultStatus: http.StatusCreated,
		Responses: map[string]*huma.Response{
			"201": {Description: "The page was stored and its read queued"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden},
	}, s.addKnowledgeUrl)
	huma.Register(api, huma.Operation{
		OperationID: "getKnowledgeUrl",
		Method:      http.MethodGet,
		Path:        "/v1/agents/knowledge/urls/{id}",
		Summary:     "One page, and when it was last read",
		Responses: map[string]*huma.Response{
			"200": {Description: "The page"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.getKnowledgeUrl)
	huma.Register(api, huma.Operation{
		OperationID: "deleteKnowledgeUrl",
		Method:      http.MethodDelete,
		Path:        "/v1/agents/knowledge/urls/{id}",
		Summary:     "Stop filling a knowledge base from a page",
		Description: "The passages the page wrote are removed too. Leaving them would have the agent go on " +
			"answering out of a page nobody subscribes to any more, which is worse than it saying it " +
			"does not know.\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an end " +
			"user's device.",
		DefaultStatus: http.StatusNoContent,
		Responses: map[string]*huma.Response{
			"204": {Description: "The page and its passages are gone"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.deleteKnowledgeUrl)
	huma.Register(api, huma.Operation{
		OperationID: "listKnowledgeUrlPassages",
		Method:      http.MethodGet,
		Path:        "/v1/agents/knowledge/urls/{id}/passages",
		Summary:     "What a page was last read into, in order",
		Description: "Empty until the page has been read.\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an end " +
			"user's device.",
		Responses: map[string]*huma.Response{
			"200": {Description: "The passages"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.listKnowledgeUrlPassages)
	huma.Register(api, huma.Operation{
		OperationID: "indexKnowledgeUrl",
		Method:      http.MethodPost,
		Path:        "/v1/agents/knowledge/urls/{id}/index",
		Summary:     "Read a page again",
		Description: "Nothing re-reads a page on its own, so this is what a caller with its own schedule " +
			"calls. The read is queued, the same as adding the page; last_indexed_at moves once it " +
			"has happened. Passages past the end of the new version are removed, so a page that got " +
			"shorter does not leave its old tail behind.\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an end " +
			"user's device.",
		Responses: map[string]*huma.Response{
			"200": {Description: "The page as it is while the read is queued"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.indexKnowledgeUrl)
}

type listKnowledgeUrlsRequest struct {
	Namespace optionalParam[string] `query:"namespace" doc:"One knowledge base. Omit to list every page the customer has."`
}

type listKnowledgeUrlsResponse struct {
	Body []KnowledgeUrl `nullable:"false"`
}

type addKnowledgeUrlRequest struct {
	Body *KnowledgeUrlRequest `required:"true"`
}

type addKnowledgeUrlResponse struct {
	Body KnowledgeUrl
}

type getKnowledgeUrlRequest struct {
	Id string `path:"id" doc:"The resource, as returned when it was created."`
}

type getKnowledgeUrlResponse struct {
	Body KnowledgeUrl
}

type deleteKnowledgeUrlRequest struct {
	Id string `path:"id" doc:"The resource, as returned when it was created."`
}

type listKnowledgeUrlPassagesRequest struct {
	Id string `path:"id" doc:"The resource, as returned when it was created."`
}

type listKnowledgeUrlPassagesResponse struct {
	Body []KnowledgePassage `nullable:"false"`
}

type indexKnowledgeUrlRequest struct {
	Id string `path:"id" doc:"The resource, as returned when it was created."`
}

type indexKnowledgeUrlResponse struct {
	Body KnowledgeUrl
}

// KnowledgeUrl is the KnowledgeUrl schema.
type KnowledgeUrl struct {
	CreatedAt     time.Time         `json:"created_at"`
	Description   *string           `json:"description,omitempty" doc:"What the page was subscribed as being. Empty unless it was given one."`
	Error         *string           `json:"error,omitempty" doc:"Why the last read failed. Empty otherwise."`
	Id            string            `json:"id"`
	LastIndexedAt *time.Time        `json:"last_indexed_at,omitempty" doc:"When it was last read successfully. Absent means never, which is what separates a page that has never worked from one that worked and has since broken." nullable:"true"`
	Namespace     string            `json:"namespace"`
	Passages      int               `json:"passages" doc:"How many passages the page was last cut into."`
	RefreshHours  *int              `json:"refresh_hours,omitempty" doc:"How often the page is read again on its own, in hours. Absent means never."`
	State         KnowledgeUrlState `json:"state"`
	Title         *string           `json:"title,omitempty" doc:"What the page is called: the title it was subscribed with, or what it called itself when it was last read."`
	UpdatedAt     time.Time         `json:"updated_at"`
	Url           string            `json:"url"`
}

func (*KnowledgeUrl) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Properties["passages"].Format = ""
	schema.Properties["refresh_hours"].Format = ""
	return schema
}

// KnowledgeUrlRequest is the KnowledgeUrlRequest schema.
type KnowledgeUrlRequest struct {
	Description  *string `json:"description,omitempty" doc:"What the page is, for a reader of the subscription. Optional, and kept as written: it says why this page is subscribed to, which a crawler cannot know." example:"What each plan includes and where the limits are."`
	Namespace    string  `json:"namespace" doc:"The knowledge base to fill, which is what a config's knowledge_namespace names." example:"docs"`
	RefreshHours *int    `json:"refresh_hours,omitempty" doc:"How often the page is read again on its own, in hours. Omit it, or send zero, and the page is read when it is added and when it is re-indexed, never on a schedule. Adding the page again replaces it." minimum:"0" example:"24"`
	Title        *string `json:"title,omitempty" doc:"What to call the page, for a reader of the subscription. Optional: a page that is not named here is named by what it called itself when it was last read." example:"Pricing"`
	Url          string  `json:"url" doc:"The page to read. It must be http or https: this is handed to a crawler and then used to key the passages it becomes." example:"https://example.com/pricing"`
}

func (*KnowledgeUrlRequest) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Properties["refresh_hours"].Format = ""
	return schema
}

// KnowledgeUrlState Where the page has got to. Pending means it has been added and its first read is queued or being retried; failed means every attempt failed.
type KnowledgeUrlState string

// Defines values for KnowledgeUrlState.
const (
	KnowledgeUrlStateFailed  KnowledgeUrlState = "failed"
	KnowledgeUrlStateIndexed KnowledgeUrlState = "indexed"
	KnowledgeUrlStatePending KnowledgeUrlState = "pending"
)

// Valid indicates whether the value is a known member of the KnowledgeUrlState enum.
func (e KnowledgeUrlState) Valid() bool {
	switch e {
	case KnowledgeUrlStateFailed:
		return true
	case KnowledgeUrlStateIndexed:
		return true
	case KnowledgeUrlStatePending:
		return true
	default:
		return false
	}
}

func (KnowledgeUrlState) Schema(registry huma.Registry) *huma.Schema {
	return namedEnum(registry, "KnowledgeUrlState", "Where the page has got to. Pending means it has been added and its first read is queued or being retried; failed means every attempt failed.", "pending", "indexed", "failed")
}
