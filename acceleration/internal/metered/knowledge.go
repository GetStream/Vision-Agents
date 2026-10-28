package metered

import (
	"context"
	"fmt"
	"maps"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
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

// NamespaceRegions is where the region each knowledge base is kept in is recorded.
// *store.Store is one.
type NamespaceRegions interface {
	// KnowledgeRegion is where the base is kept, and false for one never written.
	KnowledgeRegion(ctx context.Context, customerID, namespace string) (string, bool, error)
	// ClaimKnowledgeRegion keeps the base in region unless it is kept somewhere already, and
	// answers where it is kept.
	ClaimKnowledgeRegion(ctx context.Context, customerID, namespace, region string) (string, error)
}

// Knowledge holds the knowledge backends, one per region. A Store or a Writer comes only from
// For, which needs an owner, so nothing can be looked up or written that nobody is charged for.
//
// Each call goes to the region the owner's policies keep their work in (routing.Regioner, the
// default region when the gate is none), and a region with no backend is refused, never served
// by another. A knowledge base stays in the region it was first written in: its reads and
// writes from an owner whose policies now name another are refused rather than moved, and its
// deletes go where it is.
type Knowledge struct {
	backends map[string]knowledge.Backend
	recorder *routing.Recorder
	gate     routing.Gate
	regions  NamespaceRegions
	// kept is the region of each base whose region is recorded, by customer and namespace. A
	// recorded region never changes, so it is kept for the life of the process.
	kept sync.Map
}

// NewKnowledge meters a backend in the default region, recording through recorder as knowledge
// rows, and admits each write through gate as a model call is admitted. A nil recorder records
// nothing; a nil gate admits every write.
func NewKnowledge(backend knowledge.Backend, recorder *routing.Recorder, gate routing.Gate) *Knowledge {
	return NewRegionalKnowledge(map[string]knowledge.Backend{pricing.DefaultRegion: backend}, recorder, gate, nil)
}

// NewRegionalKnowledge is NewKnowledge with a backend per region, the region each knowledge base
// is kept in recorded in regions. Nil regions records none, and every call goes to the owner's.
func NewRegionalKnowledge(backends map[string]knowledge.Backend, recorder *routing.Recorder, gate routing.Gate,
	regions NamespaceRegions) *Knowledge {
	if recorder != nil {
		recorder = recorder.For(routing.Knowledge)
	}
	return &Knowledge{backends: maps.Clone(backends), recorder: recorder, gate: gate, regions: regions}
}

// For is the knowledge base as owner reads and writes it: every call is a row of owner's.
func (k *Knowledge) For(owner routing.Owner) OwnedKnowledge {
	return OwnedKnowledge{knowledge: k, owner: owner}
}

// Provider is the backends' name.
func (k *Knowledge) Provider() string {
	for _, region := range k.Regions() {
		return k.backends[region].Provider()
	}
	return ""
}

// Regions is every region there is a backend in, sorted.
func (k *Knowledge) Regions() []string { return slices.Sorted(maps.Keys(k.backends)) }

// Close closes every backend.
func (k *Knowledge) Close() error {
	var first error
	for _, region := range k.Regions() {
		if err := k.backends[region].Close(); err != nil && first == nil {
			first = err
		}
	}
	return first
}

// Namespaces lists the knowledge bases in region whose name starts with prefix. Nothing is
// recorded: a listing is nobody's work, and the provider bills none.
func (k *Knowledge) Namespaces(ctx context.Context, region, prefix string) ([]string, error) {
	backend, err := k.in(region)
	if err != nil {
		return nil, err
	}
	return backend.Namespaces(ctx, prefix)
}

// Storage records the hour that starts at hour of the size of a knowledge base in region for
// owner, at its size now: the provider keeps no past sizes. The row is dated to the hour it
// bills, and its usage id is the namespace and the hour (and the region, outside the default
// one), so sampling the same hour again bills it once. A knowledge base that holds nothing
// writes no row. It answers what the sample found.
func (k *Knowledge) Storage(ctx context.Context, owner routing.Owner, region, namespace string, hour time.Time) (knowledge.Billed, error) {
	backend, err := k.in(region)
	if err != nil {
		return knowledge.Billed{}, err
	}
	hour = hour.UTC().Truncate(time.Hour)
	meter := k.meter(owner, backend, region, KnowledgeStorage)
	key := []string{namespace, hour.Format(time.RFC3339)}
	if region != pricing.DefaultRegion {
		key = append(key, region)
	}
	meter.UsageID = routing.UsageID(KnowledgeStorageKind, key...)
	meter.BilledAt = hour
	// The sample is the router's own measure of the hour, not something a vendor reported.
	meter.Attributes = map[string]string{routing.AttributeMeasuredBy: routing.MeasuredByRouter}
	meter.Omit = func(used pricing.Quantities) bool { return used[pricing.ByteHours] == 0 }
	return routing.Call(ctx, meter, func(ctx context.Context) (knowledge.Billed, error) {
		return backend.Stored(ctx, namespace)
	})
}

func (k *Knowledge) meter(owner routing.Owner, backend knowledge.Backend, region, model string) routing.Meter {
	return routing.Meter{
		Recorder: k.recorder,
		Config:   routing.ProviderConfig{Provider: backend.Provider(), Model: model, Region: region},
		Owner:    owner,
		Code:     func(error) string { return knowledgeFailed },
		Estimate: pricing.Quantities{pricing.Requests: 1},
	}
}

// in is the backend of region. A region with none is refused, naming it.
func (k *Knowledge) in(region string) (knowledge.Backend, error) {
	backend, ok := k.backends[region]
	if !ok {
		return nil, routing.Refused(stack.Wrap(fmt.Errorf("metered: this deployment keeps no knowledge in region %s", region)))
	}
	return backend, nil
}

// keptIn is the region a customer's knowledge base is kept in, and false for one nothing has
// been written to or with no regions recorded.
func (k *Knowledge) keptIn(ctx context.Context, customerID, namespace string) (string, bool, error) {
	if k.regions == nil || strings.TrimSpace(namespace) == "" {
		return "", false, nil
	}
	key := customerID + "\x00" + namespace
	if region, ok := k.kept.Load(key); ok {
		return region.(string), true, nil
	}
	region, ok, err := k.regions.KnowledgeRegion(ctx, customerID, namespace)
	if err != nil || !ok {
		return "", false, err
	}
	k.kept.Store(key, region)
	return region, true, nil
}

// OwnedKnowledge is a knowledge.Store and a knowledge.Writer for one owner. Each call is a
// routing.Call: one row, a lookup for a search or a fetch, a write for an upsert or a delete,
// with what the provider billed it, in the region it was done in. Callers name a knowledge base
// as the customer does, and each call scopes it to the owner's customer.
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
	backend, region, err := o.reading(ctx, query.Namespace)
	if err != nil {
		return nil, err
	}
	query.Namespace = o.scoped(query.Namespace)
	found, err := routing.Call(ctx, o.knowledge.meter(o.owner, backend, region, KnowledgeLookup), func(ctx context.Context) (knowledge.Found, error) {
		return backend.Search(ctx, query)
	})
	return found.Documents, err
}

// Fetch reads passages back by id. Asking for none asks the provider nothing.
func (o OwnedKnowledge) Fetch(ctx context.Context, namespace string, ids []string) ([]knowledge.Document, error) {
	if len(ids) == 0 {
		return nil, nil
	}
	backend, region, err := o.reading(ctx, namespace)
	if err != nil {
		return nil, err
	}
	scoped := o.scoped(namespace)
	found, err := routing.Call(ctx, o.knowledge.meter(o.owner, backend, region, KnowledgeLookup), func(ctx context.Context) (knowledge.Found, error) {
		return backend.Fetch(ctx, scoped, ids)
	})
	return found.Documents, err
}

// Upsert writes passages, a row per request of knowledge.WriteBatch, so when one fails the
// requests before it, which the provider billed, are billed too. Writing none asks the
// provider nothing. The first write to a knowledge base keeps it in the owner's region.
func (o OwnedKnowledge) Upsert(ctx context.Context, namespace string, documents []knowledge.Document) error {
	if err := o.Admit(ctx); err != nil {
		return err
	}
	if len(documents) == 0 {
		return nil
	}
	backend, region, err := o.writing(ctx, namespace)
	if err != nil {
		return err
	}
	scoped := o.scoped(namespace)
	for batch := range slices.Chunk(documents, knowledge.WriteBatch) {
		_, err := routing.Call(ctx, o.knowledge.meter(o.owner, backend, region, KnowledgeWrite), func(ctx context.Context) (knowledge.Billed, error) {
			return backend.Upsert(ctx, scoped, batch)
		})
		if err != nil {
			return err
		}
	}
	return nil
}

// Delete removes passages by id, a row per request as Upsert writes them, in the region the
// knowledge base is kept in, whatever the owner's policies say now: data can always be removed
// where it is. Removing none asks the provider nothing.
func (o OwnedKnowledge) Delete(ctx context.Context, namespace string, ids []string) error {
	if len(ids) == 0 {
		return nil
	}
	region, kept, err := o.knowledge.keptIn(ctx, o.owner.CustomerID, namespace)
	if err != nil {
		return err
	}
	if !kept {
		if region, err = o.home(ctx); err != nil {
			return err
		}
	}
	backend, err := o.knowledge.in(region)
	if err != nil {
		return err
	}
	scoped := o.scoped(namespace)
	for batch := range slices.Chunk(ids, knowledge.WriteBatch) {
		_, err := routing.Call(ctx, o.knowledge.meter(o.owner, backend, region, KnowledgeWrite), func(ctx context.Context) (knowledge.Billed, error) {
			return backend.Delete(ctx, scoped, batch)
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

// home is the region the owner's policies keep their work in.
func (o OwnedKnowledge) home(ctx context.Context) (string, error) {
	regions, ok := o.knowledge.gate.(routing.Regioner)
	if !ok {
		return pricing.DefaultRegion, nil
	}
	return regions.Region(ctx, o.owner.CustomerID)
}

// reading is the backend a knowledge base is read from, and its region: the owner's, which a
// base kept elsewhere is refused for.
func (o OwnedKnowledge) reading(ctx context.Context, namespace string) (knowledge.Backend, string, error) {
	home, err := o.home(ctx)
	if err != nil {
		return nil, "", err
	}
	kept, recorded, err := o.knowledge.keptIn(ctx, o.owner.CustomerID, namespace)
	if err != nil {
		return nil, "", err
	}
	if recorded && kept != home {
		return nil, "", o.elsewhere(namespace, kept, home)
	}
	backend, err := o.knowledge.in(home)
	return backend, home, err
}

// writing is reading, and keeps a knowledge base nothing has been written to in the owner's
// region.
func (o OwnedKnowledge) writing(ctx context.Context, namespace string) (knowledge.Backend, string, error) {
	backend, home, err := o.reading(ctx, namespace)
	if err != nil || o.knowledge.regions == nil || strings.TrimSpace(namespace) == "" {
		return backend, home, err
	}
	kept, err := o.knowledge.regions.ClaimKnowledgeRegion(ctx, o.owner.CustomerID, namespace, home)
	if err != nil {
		return nil, "", err
	}
	o.knowledge.kept.Store(o.owner.CustomerID+"\x00"+namespace, kept)
	// Another router's first write in another region landed first.
	if kept != home {
		return nil, "", o.elsewhere(namespace, kept, home)
	}
	return backend, home, nil
}

// elsewhere refuses the owner a knowledge base kept in another region than theirs.
func (o OwnedKnowledge) elsewhere(namespace, kept, home string) error {
	return routing.Refused(stack.Wrap(fmt.Errorf(
		"metered: the knowledge base %q is kept in %s and your policies keep your work in %s: it is not read, written or moved there",
		namespace, kept, home)))
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
