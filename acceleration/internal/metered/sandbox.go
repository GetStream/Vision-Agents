package metered

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"strconv"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox/daytona"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// The models a sandbox row is recorded as: the state Daytona bills it in.
const (
	// SandboxRunning is a sandbox being made, running or stopping: its vCPUs, memory and
	// disk bill.
	SandboxRunning = "running"
	// SandboxStopped is a sandbox stopped for idling: its disk alone bills.
	SandboxStopped = "stopped"
)

// SandboxKind names a sandbox's slices in their usage ids, and SandboxReconciledKind the rows
// the daily reconciliation records from Daytona's own figures.
const (
	SandboxKind           = "sandbox"
	SandboxReconciledKind = "sandbox_daytona"
)

// The attributes a sandbox row carries.
const (
	AttributeSandboxID = "sandbox_id"
	// AttributeSandboxClass is the sort of sandbox: container or linux-vm.
	AttributeSandboxClass = "class"
	// AttributeSandboxCPU, AttributeSandboxMemoryGB and AttributeSandboxDiskGB are what it
	// reserves.
	AttributeSandboxCPU      = "cpu"
	AttributeSandboxMemoryGB = "memory_gb"
	AttributeSandboxDiskGB   = "disk_gb"
	// AttributeOrphan marks a sandbox's time after its session closed: it could not be
	// deleted, or nothing was left to delete it. It is billed like the rest of its life.
	AttributeOrphan = "orphan"
	// AttributeSource is daytona on a row recorded from Daytona's own figures rather than
	// the router's clock.
	AttributeSource = "source"
	// AttributeDeployment is the deployment whose sandbox it was, as its LabelDeployment says.
	// The sandbox job reads only its own deployment's rows.
	AttributeDeployment = "deployment"
	// AttributeSandboxRegion is the region the sandbox ran in, as its LabelRegion says, which
	// the daily reconciliation prices what it records of it in.
	AttributeSandboxRegion = "region"
)

// The labels every sandbox is made with: which deployment and shard made it, so the hourly
// sweep finds its own, and whose it is, so the daily reconciliation can bill what Daytona
// billed. Their values are ids.
const (
	// LabelDeployment is the code_sandbox.deployment setting of the routers that made it. The
	// sandbox job lists, deletes and bills only sandboxes carrying its own.
	LabelDeployment   = "stream.deployment"
	LabelEnv          = "stream.env"
	LabelRouter       = "stream.router"
	LabelOrganization = "stream.organization"
	LabelApp          = "stream.app"
	LabelSession      = "stream.session"
	// LabelRegion is the region the sandbox runs in, such as eu: the owner's policies' region.
	LabelRegion = "stream.region"
)

// SandboxSecretsBroker is the code_sandbox.broker setting that keeps a sandbox's connector
// credentials with Daytona (SandboxSecrets).
const SandboxSecretsBroker = daytona.BrokerName

// SandboxesOptions are where sandboxes are made and how their time is billed.
type SandboxesOptions struct {
	// Recorder is where sandbox rows go, as the sandbox modality. Nil records nothing.
	Recorder *routing.Recorder
	// Deployment, Environment and Router are on every sandbox's labels: the deployment that
	// made it, the environment it reports as, and which of its shards. Deployment is also on
	// every row, and an empty one is left off both.
	Deployment, Environment, Router string
	// Lifecycle is when Daytona stops an idle sandbox and deletes a stopped one; the meter
	// keeps the same clock.
	Lifecycle sandbox.Lifecycle
	// APIKey and APIURL reach Daytona. Empty are DAYTONA_API_KEY and Daytona's own API.
	APIKey, APIURL string
	// Targets maps each region sandboxes are made in to the Daytona target they are made on.
	// The default region is always served: on its target here, else on DAYTONA_TARGET or
	// Daytona's own default. A region with no target is refused.
	Targets map[string]string
	// Regions says which region an owner's sandbox runs in. Nil is the default region.
	Regions routing.Regioner
	// Every is how long a slice of a sandbox's time runs. Zero is routing.SliceEvery.
	Every  time.Duration
	Logger *slog.Logger
}

// Sandboxes makes the sandboxes sessions run code in. A sandbox comes only from Open, which
// needs an owner, so none is made that nobody is billed for: from when Daytona makes it until
// it is deleted, by the state it is in, a slice at a time, priced in the region it runs in.
type Sandboxes struct {
	options SandboxesOptions
}

// NewSandboxes makes sandboxes as options say.
func NewSandboxes(options SandboxesOptions) *Sandboxes {
	if options.Recorder != nil {
		options.Recorder = options.Recorder.For(routing.Sandbox)
	}
	if options.Logger == nil {
		options.Logger = slog.Default()
	}
	return &Sandboxes{options: options}
}

// SandboxSpec is the sandbox a session asks for.
type SandboxSpec struct {
	Config sandbox.Config
	// Grants are the connections code in it may call with, through Broker.
	Grants []sandbox.Grant
	Broker sandbox.CredentialBroker
	// Slots bound how many of its organization's sandboxes run at once. Nil bounds nothing.
	Slots sandbox.Slots
}

// Open is the sandbox owner's session runs code in, labelled with whose it is and the region it
// runs in. Nothing is made until code first runs; from then on its time is billed to owner, the
// session the owner's attribution names.
//
// An owner with no app or no session is refused: every customer's sandboxes share one Daytona
// organization, so one labelled with no app is nobody's to bill or tell apart, and one with no
// session is deleted by the sweep while it runs. So is an owner whose policies keep their work
// in a region no Daytona target serves: their code is never run in another.
func (s *Sandboxes) Open(ctx context.Context, owner routing.Owner, spec SandboxSpec) (sandbox.Sandbox, error) {
	switch {
	case owner.CustomerID == "":
		return nil, stack.Wrap(errors.New("metered: a sandbox needs the app it is for, sandboxes are never shared"))
	case owner.SessionID == "":
		return nil, stack.Wrap(errors.New("metered: a sandbox needs the session it runs for"))
	}
	region := pricing.DefaultRegion
	if s.options.Regions != nil {
		var err error
		if region, err = s.options.Regions.Region(ctx, owner.CustomerID); err != nil {
			return nil, err
		}
	}
	target, ok := s.options.Targets[region]
	if !ok && region != pricing.DefaultRegion {
		return nil, routing.Refused(stack.Wrap(fmt.Errorf("metered: this deployment runs no sandboxes in region %s", region)))
	}
	labels := s.Labels(owner)
	labels[LabelRegion] = region
	return daytona.New(daytona.Options{
		APIKey: s.options.APIKey, APIURL: s.options.APIURL, Target: target, Config: spec.Config, Grants: spec.Grants, Broker: spec.Broker,
		Labels: labels, Lifecycle: s.options.Lifecycle, Logger: s.options.Logger, Slots: spec.Slots,
		Meter: &sandboxMeter{recorder: s.options.Recorder, owner: owner, every: s.options.Every, deployment: s.options.Deployment, region: region},
	})
}

// Labels are what a sandbox of owner's is made with. An id owner does not have is left off.
func (s *Sandboxes) Labels(owner routing.Owner) map[string]string {
	labels := map[string]string{}
	for name, value := range map[string]string{
		LabelDeployment: s.options.Deployment, LabelEnv: s.options.Environment, LabelRouter: s.options.Router,
		LabelOrganization: owner.OrganizationID,
		LabelApp:          owner.CustomerID, LabelSession: owner.SessionID,
	} {
		if value != "" {
			labels[name] = value
		}
	}
	return labels
}

// SandboxSecrets is the broker that keeps a session's connector credentials with Daytona as
// secrets named after prefix and the session, binding a late grant with bind. Daytona bills
// nothing for them.
func SandboxSecrets(prefix, session string, tokens sandbox.Tokens, bind sandbox.Bind, logger *slog.Logger) (sandbox.CredentialBroker, error) {
	return daytona.NewSecrets(daytona.SecretsOptions{Prefix: prefix, Session: session, Tokens: tokens, Bind: bind, Logger: logger})
}

// SandboxUsed is what seconds of a sandbox holding reserved used, billed as model: its vCPU,
// memory and disk seconds while it runs, its disk seconds alone once it has stopped.
func SandboxUsed(reserved daytona.Reserved, model string, seconds int64) pricing.Quantities {
	used := pricing.Quantities{pricing.DiskGBSeconds: times(seconds, reserved.DiskGB)}
	if model != SandboxStopped {
		used[pricing.VCPUSeconds] = times(seconds, reserved.CPU)
		used[pricing.MemoryGBSeconds] = times(seconds, reserved.MemoryGB)
	}
	return used
}

// times is seconds of a size Daytona reports as a fraction, to the nearest whole unit.
func times(seconds int64, size float64) int64 {
	return int64(math.Round(float64(seconds) * size))
}

// sandboxMeter bills one session's sandbox: a run of slices for each sandbox Daytona makes for
// it, switched between running and stopped as Daytona's timers move it, and stopped once it
// is gone.
type sandboxMeter struct {
	recorder   *routing.Recorder
	owner      routing.Owner
	every      time.Duration
	deployment string
	region     string

	mu     sync.Mutex
	slices *routing.Slices
	orphan bool
}

var _ daytona.Meter = (*sandboxMeter)(nil)

func (m *sandboxMeter) Created(reserved daytona.Reserved) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.slices.Stop()
	attributes := map[string]string{
		AttributeSandboxID: reserved.ID, AttributeSandboxClass: reserved.Class,
		AttributeSandboxCPU:      strconv.FormatFloat(reserved.CPU, 'f', -1, 64),
		AttributeSandboxMemoryGB: strconv.FormatFloat(reserved.MemoryGB, 'f', -1, 64),
		AttributeSandboxDiskGB:   strconv.FormatFloat(reserved.DiskGB, 'f', -1, 64),
	}
	if m.orphan {
		attributes[AttributeOrphan] = "true"
	}
	if m.deployment != "" {
		attributes[AttributeDeployment] = m.deployment
	}
	if m.region != "" {
		attributes[AttributeSandboxRegion] = m.region
	}
	m.slices = routing.StartSlices(routing.SliceOptions{
		Recorder: m.recorder,
		Config:   routing.ProviderConfig{Provider: daytona.Provider, Model: SandboxRunning, Region: m.region},
		Owner:    m.owner,
		Kind:     SandboxKind,
		ID:       reserved.ID,
		Quantities: func(model string, seconds int64) pricing.Quantities {
			return SandboxUsed(reserved, model, seconds)
		},
		Attributes: attributes,
		Every:      m.every,
	})
}

func (m *sandboxMeter) Stopped() { m.switchTo(SandboxStopped) }

func (m *sandboxMeter) Started() { m.switchTo(SandboxRunning) }

func (m *sandboxMeter) switchTo(model string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.slices.Switch(model)
}

func (m *sandboxMeter) Orphaned() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.orphan = true
	m.slices.Attribute(AttributeOrphan, "true")
}

func (m *sandboxMeter) Gone() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.slices.Stop()
	m.slices = nil
}

// SandboxGap is time Daytona billed a sandbox that the router recorded none of, such as the
// minutes after a router died mid-slice, or a whole sandbox a killed router never got to
// meter.
type SandboxGap struct {
	// Owner is whose the sandbox was, from its labels or the rows recorded for it.
	Owner     routing.Owner
	SandboxID string
	// Key tells the gap from the sandbox's others: the start of the Daytona period it is, or
	// the day it makes up.
	Key string
	// At dates the row.
	At    time.Time
	Model string
	Used  pricing.Quantities
	// Class is the sort of sandbox it was, which its price depends on. Empty is unknown, and
	// leaves the gap unpriced.
	Class string
	// Region is where the sandbox ran, which prices the gap. Empty, as for a sandbox made
	// before regions, is the default region.
	Region string
	// Orphan is time after the sandbox's session closed.
	Orphan bool
}

// Reconcile records a gap from Daytona's figures, as Daytona billed it, owned as the gap says
// and marked as the options' deployment's. Its usage id is the sandbox and the gap's key, so
// recording it again bills it once.
func (s *Sandboxes) Reconcile(gap SandboxGap) error {
	attributes := map[string]string{AttributeSandboxID: gap.SandboxID, AttributeSource: daytona.Provider}
	if gap.Orphan {
		attributes[AttributeOrphan] = "true"
	}
	if gap.Class != "" {
		attributes[AttributeSandboxClass] = gap.Class
	}
	if gap.Region != "" {
		attributes[AttributeSandboxRegion] = gap.Region
	}
	if s.options.Deployment != "" {
		attributes[AttributeDeployment] = s.options.Deployment
	}
	return routing.Settle(routing.Meter{
		Recorder:    s.options.Recorder,
		Config:      routing.ProviderConfig{Provider: daytona.Provider, Model: gap.Model, Region: gap.Region},
		Owner:       gap.Owner,
		UsageID:     routing.UsageID(SandboxReconciledKind, gap.SandboxID, gap.Key),
		BilledAt:    gap.At,
		Attributes:  attributes,
		OperationID: gap.SandboxID,
		Unmeasured:  true,
	}, gap.Used)
}
