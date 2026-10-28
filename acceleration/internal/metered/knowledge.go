package metered

import (
	"context"
	"slices"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// The models a knowledge row is recorded as, each priced on its own.
const (
	// KnowledgeLookup is a search or a fetch.
	KnowledgeLookup = "lookup"
	// KnowledgeWrite is an upsert or a delete.
	KnowledgeWrite = "write"
	// KnowledgeStorage is an hour of a knowledge base's size.
	KnowledgeStorage = "storage"
)

// KnowledgeStorageKind names an hour of storage in its usage id.
const KnowledgeStorageKind = "knowledge_storage"

// knowledgeFailed is the error code of a knowledge call that failed.
const knowledgeFailed = "knowledge_failed"

// Knowledge holds the knowledge backend. A Store or a Writer comes only from For, which needs
// an owner, so nothing can be looked up or written that nobody is charged for.
type Knowledge struct {
	backend  knowledge.Backend
	recorder *routing.Recorder
	gate     routing.Gate
}

// NewKnowledge meters a backend, recording through recorder as knowledge rows, and admits each
// write through gate as a model call is admitted. A nil recorder records nothing; a nil gate
// admits every write.
func NewKnowledge(backend knowledge.Backend, recorder *routing.Recorder, gate routing.Gate) *Knowledge {
	if recorder != nil {
		recorder = recorder.For(routing.Knowledge)
	}
	return &Knowledge{backend: backend, recorder: recorder, gate: gate}
}

// For is the knowledge base as owner reads and writes it: every call is a row of owner's.
func (k *Knowledge) For(owner routing.Owner) OwnedKnowledge {
	return OwnedKnowledge{knowledge: k, owner: owner}
}

// Provider is the backend's name.
func (k *Knowledge) Provider() string { return k.backend.Provider() }

// Close closes the backend.
func (k *Knowledge) Close() error { return k.backend.Close() }

// Namespaces lists the knowledge bases whose name starts with prefix. Nothing is recorded: a
// listing is nobody's work, and the provider bills none.
func (k *Knowledge) Namespaces(ctx context.Context, prefix string) ([]string, error) {
	return k.backend.Namespaces(ctx, prefix)
}

// Storage records the hour that starts at hour of a knowledge base's size for owner, at its
// size now: the provider keeps no past sizes. The row is dated to the hour it bills, and its
// usage id is the namespace and the hour, so sampling the same hour again bills it once. A
// knowledge base that holds nothing writes no row. It answers what the sample found.
func (k *Knowledge) Storage(ctx context.Context, owner routing.Owner, namespace string, hour time.Time) (knowledge.Billed, error) {
	hour = hour.UTC().Truncate(time.Hour)
	meter := k.meter(owner, KnowledgeStorage)
	meter.UsageID = routing.UsageID(KnowledgeStorageKind, namespace, hour.Format(time.RFC3339))
	meter.BilledAt = hour
	// The sample is the router's own measure of the hour, not something a vendor reported.
	meter.Attributes = map[string]string{routing.AttributeMeasuredBy: routing.MeasuredByRouter}
	meter.Omit = func(used pricing.Quantities) bool { return used[pricing.ByteHours] == 0 }
	return routing.Call(ctx, meter, func(ctx context.Context) (knowledge.Billed, error) {
		return k.backend.Stored(ctx, namespace)
	})
}

func (k *Knowledge) meter(owner routing.Owner, model string) routing.Meter {
	return routing.Meter{
		Recorder: k.recorder,
		Config:   routing.ProviderConfig{Provider: k.backend.Provider(), Model: model},
		Owner:    owner,
		Code:     func(error) string { return knowledgeFailed },
		Estimate: pricing.Quantities{pricing.Requests: 1},
	}
}

// OwnedKnowledge is a knowledge.Store and a knowledge.Writer for one owner. Each call is a
// routing.Call: one row, a lookup for a search or a fetch, a write for an upsert or a delete,
// with what the provider billed it. Callers name a knowledge base as the customer does, and
// each call scopes it to the owner's customer.
type OwnedKnowledge struct {
	knowledge *Knowledge
	owner     routing.Owner
}

var (
	_ knowledge.Store  = OwnedKnowledge{}
	_ knowledge.Writer = OwnedKnowledge{}
)

// Search looks a question up.
func (o OwnedKnowledge) Search(ctx context.Context, query knowledge.Query) ([]knowledge.Document, error) {
	query.Namespace = o.scoped(query.Namespace)
	found, err := routing.Call(ctx, o.knowledge.meter(o.owner, KnowledgeLookup), func(ctx context.Context) (knowledge.Found, error) {
		return o.knowledge.backend.Search(ctx, query)
	})
	return found.Documents, err
}

// Fetch reads passages back by id. Asking for none asks the provider nothing.
func (o OwnedKnowledge) Fetch(ctx context.Context, namespace string, ids []string) ([]knowledge.Document, error) {
	namespace = o.scoped(namespace)
	if len(ids) == 0 {
		return nil, nil
	}
	found, err := routing.Call(ctx, o.knowledge.meter(o.owner, KnowledgeLookup), func(ctx context.Context) (knowledge.Found, error) {
		return o.knowledge.backend.Fetch(ctx, namespace, ids)
	})
	return found.Documents, err
}

// Upsert writes passages, a row per request of knowledge.WriteBatch, so when one fails the
// requests before it, which the provider billed, are billed too. Writing none asks the
// provider nothing.
func (o OwnedKnowledge) Upsert(ctx context.Context, namespace string, documents []knowledge.Document) error {
	if err := o.Admit(ctx); err != nil {
		return err
	}
	namespace = o.scoped(namespace)
	for batch := range slices.Chunk(documents, knowledge.WriteBatch) {
		_, err := routing.Call(ctx, o.knowledge.meter(o.owner, KnowledgeWrite), func(ctx context.Context) (knowledge.Billed, error) {
			return o.knowledge.backend.Upsert(ctx, namespace, batch)
		})
		if err != nil {
			return err
		}
	}
	return nil
}

// Delete removes passages by id, a row per request as Upsert writes them. Removing none asks
// the provider nothing.
func (o OwnedKnowledge) Delete(ctx context.Context, namespace string, ids []string) error {
	namespace = o.scoped(namespace)
	for batch := range slices.Chunk(ids, knowledge.WriteBatch) {
		_, err := routing.Call(ctx, o.knowledge.meter(o.owner, KnowledgeWrite), func(ctx context.Context) (knowledge.Billed, error) {
			return o.knowledge.backend.Delete(ctx, namespace, batch)
		})
		if err != nil {
			return err
		}
	}
	return nil
}

// Admit refuses a write the owner may not pay for: writing is most of what knowledge costs,
// and its rows are only recorded once the provider has billed them. Deletes are not admitted,
// so data can always be removed. Upsert admits itself; a caller that pays for work before
// the write, such as reading the page it is for, admits that work first.
func (o OwnedKnowledge) Admit(ctx context.Context) error {
	if o.knowledge.gate == nil {
		return nil
	}
	admission, err := o.knowledge.gate.Admit(ctx, o.owner)
	if err != nil {
		return err
	}
	return admission.Spend
}

// scoped is the provider's name for the owner's knowledge base namespace. An empty name
// stays empty so the backend refuses it: with the customer prefix it would be a base of
// its own that nothing meant to write to.
func (o OwnedKnowledge) scoped(namespace string) string {
	if strings.TrimSpace(namespace) == "" {
		return namespace
	}
	return knowledge.Scoped(o.owner.CustomerID, namespace)
}

// Provider is the backend's name.
func (o OwnedKnowledge) Provider() string { return o.knowledge.Provider() }

// Close releases nothing: the backend is the process's, and closes with it.
func (o OwnedKnowledge) Close() error { return nil }
