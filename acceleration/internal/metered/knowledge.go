package metered

import (
	"cmp"
	"context"
	"errors"
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
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
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

// ErrNoAccountKey says a knowledge base is kept in its owner's own account at the vendor, and
// they keep no key for it there any more.
var ErrNoAccountKey = errors.New("metered: the knowledge base is kept in your own account, and you keep no key for it any more")

// NamespacePlaces is where the place each knowledge base is kept is recorded: its region, and
// the account in it. *store.Store is one.
type NamespacePlaces interface {
	// KnowledgePlace is where the base is kept, and false for one never written.
	KnowledgePlace(ctx context.Context, customerID, namespace string) (store.KnowledgePlace, bool, error)
	// ClaimKnowledgePlace keeps the base at place unless it is kept somewhere already, and
	// answers where it is kept.
	ClaimKnowledgePlace(ctx context.Context, customerID, namespace string, place store.KnowledgePlace) (store.KnowledgePlace, error)
}

// AccountBackend is the backend of a customer's own account at the knowledge vendor in region,
// on the key they keep for it there.
type AccountBackend func(region, key string) (knowledge.Backend, error)

// Knowledge holds the knowledge backends, one per region. A Store or a Writer comes only from
// For, which needs an owner, so nothing can be looked up or written that nobody is charged for.
//
// A knowledge base stays in the region it was first written in. Work naming a region
// (routing.Owner.Region, an agent's) reads and writes a base kept in it, makes one nothing was
// written to there, and is refused one kept in another rather than having it moved. Work naming
// none, as the knowledge API's, goes where the base is kept, and makes a new one in the default
// region. A region with no backend is refused, never served by another, and a delete goes where
// the base is whatever the work names.
//
// With customer accounts (WithCustomerAccounts), a base an owner first writes while they keep a
// key for the vendor in the work's region is kept in their own account there, on that key, and its
// work is priced at the vendor's BYOK entries. A base stays in the account it was first written
// in: one in the deployment's stays there when its owner puts a key, and one in theirs is
// refused once its key is deleted, never reached in the deployment's.
type Knowledge struct {
	backends map[string]knowledge.Backend
	recorder *routing.Recorder
	gate     routing.Gate
	places   NamespacePlaces
	// keys and account reach a customer's own account. Nil keeps every base in the deployment's.
	keys    routing.CustomerKeys
	account AccountBackend
	// kept is the place of each base whose place is recorded, by customer and namespace. A
	// recorded place never changes, so it is kept for the life of the process.
	kept sync.Map
}

// NewKnowledge meters a backend in the default region, recording through recorder as knowledge
// rows, and admits each write through gate as a model call is admitted. A nil recorder records
// nothing; a nil gate admits every write.
func NewKnowledge(backend knowledge.Backend, recorder *routing.Recorder, gate routing.Gate) *Knowledge {
	return NewRegionalKnowledge(map[string]knowledge.Backend{pricing.DefaultRegion: backend}, recorder, gate, nil)
}

// NewRegionalKnowledge is NewKnowledge with a backend per region, the place each knowledge base
// is kept recorded in places. Nil places records none, and every call goes to the work's region.
func NewRegionalKnowledge(backends map[string]knowledge.Backend, recorder *routing.Recorder, gate routing.Gate,
	places NamespacePlaces) *Knowledge {
	if recorder != nil {
		recorder = recorder.For(routing.Knowledge)
	}
	return &Knowledge{backends: maps.Clone(backends), recorder: recorder, gate: gate, places: places}
}

// WithCustomerAccounts keeps the bases of an owner who keeps a key for the vendor in keys in
// their own account, reached with account. It is set before the first call, and needs places:
// without them a base's account could not be remembered.
func (k *Knowledge) WithCustomerAccounts(keys routing.CustomerKeys, account AccountBackend) *Knowledge {
	if k.places != nil {
		k.keys, k.account = keys, account
	}
	return k
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

// Namespaces lists the knowledge bases in the deployment's account in region whose name starts
// with prefix. Nothing is recorded: a listing is nobody's work, and the provider bills none.
func (k *Knowledge) Namespaces(ctx context.Context, region, prefix string) ([]string, error) {
	backend, err := k.in(region)
	if err != nil {
		return nil, err
	}
	return backend.Namespaces(ctx, prefix)
}

// Storage records the hour that starts at hour of the size of a knowledge base in the
// deployment's account in region for owner, at its size now: the provider keeps no past sizes.
// The row is dated to the hour it bills, and its usage id is the namespace and the hour (and the
// region, outside the default one), so sampling the same hour again bills it once. A knowledge
// base that holds nothing writes no row. It answers what the sample found.
func (k *Knowledge) Storage(ctx context.Context, owner routing.Owner, region, namespace string, hour time.Time) (knowledge.Billed, error) {
	return k.StorageAt(ctx, owner, store.KnowledgePlace{Region: region, Account: store.KnowledgeAccountStream}, namespace, hour)
}

// StorageAt is Storage for a base kept at place, which in the customer's account is read on
// the key they keep there and priced at the vendor's BYOK entry.
func (k *Knowledge) StorageAt(ctx context.Context, owner routing.Owner, place store.KnowledgePlace, namespace string, hour time.Time) (knowledge.Billed, error) {
	backend, err := k.at(ctx, owner.CustomerID, namespace, place)
	if err != nil {
		return knowledge.Billed{}, err
	}
	hour = hour.UTC().Truncate(time.Hour)
	meter := k.meter(owner, backend, place, KnowledgeStorage)
	key := []string{namespace, hour.Format(time.RFC3339)}
	if place.Region != pricing.DefaultRegion {
		key = append(key, place.Region)
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

func (k *Knowledge) meter(owner routing.Owner, backend knowledge.Backend, place store.KnowledgePlace, model string) routing.Meter {
	return routing.Meter{
		Recorder: k.recorder,
		Config: routing.ProviderConfig{Provider: backend.Provider(), Model: model, Region: place.Region,
			BYOK: place.Account == store.KnowledgeAccountCustomer},
		Owner:    owner,
		Code:     func(error) string { return knowledgeFailed },
		Estimate: pricing.Quantities{pricing.Requests: 1},
	}
}

// in is the deployment's backend of region. A region with none is refused, naming it.
func (k *Knowledge) in(region string) (knowledge.Backend, error) {
	backend, ok := k.backends[region]
	if !ok {
		return nil, routing.Refused(stack.Wrap(fmt.Errorf("metered: this deployment keeps no knowledge in region %s", region)))
	}
	return backend, nil
}

// at is the backend of a customer's knowledge base kept at place: the deployment's in its
// region, or the customer's own account there on the key they keep for it. A base in their
// account whose key is gone is refused, and never looked for in the deployment's.
func (k *Knowledge) at(ctx context.Context, customerID, namespace string, place store.KnowledgePlace) (knowledge.Backend, error) {
	backend, err := k.in(place.Region)
	if err != nil || place.Account != store.KnowledgeAccountCustomer {
		return backend, err
	}
	if k.keys == nil {
		return nil, stack.Wrap(fmt.Errorf("metered: the knowledge base %q is kept in its owner's own %s account in %s, "+
			"and this process reaches no customer's account: it has no keyring to open their keys with", namespace, backend.Provider(), place.Region))
	}
	key, kept, err := k.keys.CustomerKey(ctx, customerID, backend.Provider(), place.Region)
	if err != nil {
		return nil, err
	}
	if !kept {
		return nil, routing.Refused(stack.Wrap(fmt.Errorf(
			"%w: %q is in your %s account in %s; put a %s key for %s back to reach it, it is never read from or written to the deployment's account",
			ErrNoAccountKey, namespace, backend.Provider(), place.Region, backend.Provider(), place.Region)))
	}
	return k.account(place.Region, key)
}

// customerKey is the key a customer keeps for the vendor in region, and false when they keep
// none or this deployment reaches no customer's account.
func (k *Knowledge) customerKey(ctx context.Context, customerID, region string) (string, bool, error) {
	if k.keys == nil {
		return "", false, nil
	}
	return k.keys.CustomerKey(ctx, customerID, k.Provider(), region)
}

// Place is where a customer's knowledge base is kept, and false for one nothing has been
// written to or with no places recorded.
func (k *Knowledge) Place(ctx context.Context, customerID, namespace string) (store.KnowledgePlace, bool, error) {
	return k.keptAt(ctx, customerID, namespace)
}

// keptAt is where a customer's knowledge base is kept, and false for one nothing has been
// written to or with no places recorded.
func (k *Knowledge) keptAt(ctx context.Context, customerID, namespace string) (store.KnowledgePlace, bool, error) {
	if k.places == nil || strings.TrimSpace(namespace) == "" {
		return store.KnowledgePlace{}, false, nil
	}
	key := customerID + "\x00" + namespace
	if place, ok := k.kept.Load(key); ok {
		return place.(store.KnowledgePlace), true, nil
	}
	place, ok, err := k.places.KnowledgePlace(ctx, customerID, namespace)
	if err != nil || !ok {
		return store.KnowledgePlace{}, false, err
	}
	k.kept.Store(key, place)
	return place, true, nil
}

// OwnedKnowledge is a knowledge.Store and a knowledge.Writer for one owner. Each call is a
// routing.Call: one row, a lookup for a search or a fetch, a write for an upsert or a delete,
// with what the provider billed it, in the region and account it was done in. Callers name a
// knowledge base as the customer does, and each call scopes it to the owner's customer.
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
	backend, place, err := o.reading(ctx, query.Namespace)
	if err != nil {
		return nil, err
	}
	query.Namespace = o.scoped(query.Namespace)
	found, err := routing.Call(ctx, o.knowledge.meter(o.owner, backend, place, KnowledgeLookup), func(ctx context.Context) (knowledge.Found, error) {
		return backend.Search(ctx, query)
	})
	return found.Documents, err
}

// Fetch reads passages back by id. Asking for none asks the provider nothing.
func (o OwnedKnowledge) Fetch(ctx context.Context, namespace string, ids []string) ([]knowledge.Document, error) {
	if len(ids) == 0 {
		return nil, nil
	}
	backend, place, err := o.reading(ctx, namespace)
	if err != nil {
		return nil, err
	}
	scoped := o.scoped(namespace)
	found, err := routing.Call(ctx, o.knowledge.meter(o.owner, backend, place, KnowledgeLookup), func(ctx context.Context) (knowledge.Found, error) {
		return backend.Fetch(ctx, scoped, ids)
	})
	return found.Documents, err
}

// Upsert writes passages, a row per request of knowledge.WriteBatch, so when one fails the
// requests before it, which the provider billed, are billed too. Writing none asks the
// provider nothing. The first write to a knowledge base keeps it in the work's region, in the
// owner's own account when they keep a key for the vendor there.
func (o OwnedKnowledge) Upsert(ctx context.Context, namespace string, documents []knowledge.Document) error {
	if err := o.Admit(ctx); err != nil {
		return err
	}
	if len(documents) == 0 {
		return nil
	}
	backend, place, err := o.writing(ctx, namespace)
	if err != nil {
		return err
	}
	scoped := o.scoped(namespace)
	for batch := range slices.Chunk(documents, knowledge.WriteBatch) {
		_, err := routing.Call(ctx, o.knowledge.meter(o.owner, backend, place, KnowledgeWrite), func(ctx context.Context) (knowledge.Billed, error) {
			return backend.Upsert(ctx, scoped, batch)
		})
		if err != nil {
			return err
		}
	}
	return nil
}

// Delete removes passages by id, a row per request as Upsert writes them, where the knowledge
// base is kept, whatever region the work names: data can always be removed where it is.
// Removing none asks the provider nothing.
func (o OwnedKnowledge) Delete(ctx context.Context, namespace string, ids []string) error {
	if len(ids) == 0 {
		return nil
	}
	place, kept, err := o.knowledge.keptAt(ctx, o.owner.CustomerID, namespace)
	if err != nil {
		return err
	}
	if !kept {
		if place, err = o.unwritten(ctx, o.region()); err != nil {
			return err
		}
	}
	backend, err := o.knowledge.at(ctx, o.owner.CustomerID, namespace, place)
	if err != nil {
		return err
	}
	scoped := o.scoped(namespace)
	for batch := range slices.Chunk(ids, knowledge.WriteBatch) {
		_, err := routing.Call(ctx, o.knowledge.meter(o.owner, backend, place, KnowledgeWrite), func(ctx context.Context) (knowledge.Billed, error) {
			return backend.Delete(ctx, scoped, batch)
		})
		if err != nil {
			return err
		}
	}
	return nil
}

// Keep keeps a knowledge base nothing has been written to where the owner's first write would,
// so later work naming no region, such as a page's refresh, finds it there. A base kept in
// another region than the one the work names is refused, as a write is. The vendor is asked
// nothing.
func (o OwnedKnowledge) Keep(ctx context.Context, namespace string) error {
	_, _, err := o.writing(ctx, namespace)
	return err
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

// region is where a knowledge base the owner's work makes is kept: the region it names, else
// the default one.
func (o OwnedKnowledge) region() string {
	return cmp.Or(o.owner.Region, pricing.DefaultRegion)
}

// unwritten is where a base nothing has been written to would be kept in region: the owner's
// own account when they keep a key for the vendor there, else the deployment's.
func (o OwnedKnowledge) unwritten(ctx context.Context, region string) (store.KnowledgePlace, error) {
	_, theirs, err := o.knowledge.customerKey(ctx, o.owner.CustomerID, region)
	if err != nil {
		return store.KnowledgePlace{}, err
	}
	place := store.KnowledgePlace{Region: region, Account: store.KnowledgeAccountStream}
	if theirs {
		place.Account = store.KnowledgeAccountCustomer
	}
	return place, nil
}

// reading is the backend a knowledge base is read from, and where it is kept: where it is,
// unless the work names another region, which it is refused for; in the work's region for one
// nothing was written to. It is in the account the base is kept in.
func (o OwnedKnowledge) reading(ctx context.Context, namespace string) (knowledge.Backend, store.KnowledgePlace, error) {
	place, recorded, err := o.knowledge.keptAt(ctx, o.owner.CustomerID, namespace)
	if err != nil {
		return nil, store.KnowledgePlace{}, err
	}
	if recorded && o.owner.Region != "" && place.Region != o.owner.Region {
		return nil, store.KnowledgePlace{}, o.elsewhere(namespace, place.Region, o.owner.Region)
	}
	if !recorded {
		if place, err = o.unwritten(ctx, o.region()); err != nil {
			return nil, store.KnowledgePlace{}, err
		}
	}
	backend, err := o.knowledge.at(ctx, o.owner.CustomerID, namespace, place)
	return backend, place, err
}

// writing is reading, and keeps a knowledge base nothing has been written to where reading
// found it would be.
func (o OwnedKnowledge) writing(ctx context.Context, namespace string) (knowledge.Backend, store.KnowledgePlace, error) {
	backend, place, err := o.reading(ctx, namespace)
	if err != nil || o.knowledge.places == nil || strings.TrimSpace(namespace) == "" {
		return backend, place, err
	}
	kept, err := o.knowledge.places.ClaimKnowledgePlace(ctx, o.owner.CustomerID, namespace, place)
	if err != nil {
		return nil, store.KnowledgePlace{}, err
	}
	o.knowledge.kept.Store(o.owner.CustomerID+"\x00"+namespace, kept)
	switch {
	// Another router's first write in another region landed first.
	case kept.Region != place.Region:
		return nil, store.KnowledgePlace{}, o.elsewhere(namespace, kept.Region, place.Region)
	// Or in another account: a key put or deleted while the two were written.
	case kept.Account != place.Account:
		backend, err = o.knowledge.at(ctx, o.owner.CustomerID, namespace, kept)
		return backend, kept, err
	}
	return backend, place, nil
}

// elsewhere refuses work done in one region a knowledge base kept in another.
func (o OwnedKnowledge) elsewhere(namespace, kept, region string) error {
	return routing.Refused(stack.Wrap(fmt.Errorf(
		"metered: the knowledge base %q is kept in %s and this work is done in %s: it is not read, written or moved there",
		namespace, kept, region)))
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
