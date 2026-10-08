package phone

import (
	"context"
	"errors"
	"fmt"
	"net/netip"
	"regexp"
	"strings"
	"time"

	"github.com/google/uuid"

	"github.com/GetStream/Vision-Agents/acceleration/internal/egress"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone/sipbridge"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// SIPTrunkVendor is the vendor every number on a customer's own SIP trunk is recorded
// under. It is not in the registry: the registry is what every customer is offered to buy
// from, and nobody buys from this one.
const SIPTrunkVendor = "sip_trunk"

// ErrSIPTrunksDisabled says this deployment has no key to seal a trunk's password with.
var ErrSIPTrunksDisabled = errors.New("phone: sip trunks are off on this deployment: it has no key " +
	"encryption key to seal their passwords with")

// ErrInvalidSIPTrunk starts every reason a trunk or a number on one is refused for, the way
// dlc.ErrInvalid does for 10DLC, so a caller can show what follows it to a person.
var ErrInvalidSIPTrunk = errors.New("sip_trunk: invalid")

// invalid is one ErrInvalidSIPTrunk naming every reason, nil when there is none.
func invalid(reasons []string) error {
	if len(reasons) == 0 {
		return nil
	}
	return stack.Wrap(fmt.Errorf("%w: %s", ErrInvalidSIPTrunk, strings.Join(reasons, "; ")))
}

const (
	defaultSIPPort      = 5060
	defaultSIPTransport = "tcp"
)

var (
	e164Pattern    = regexp.MustCompile(`^\+[1-9][0-9]{6,14}$`)
	countryPattern = regexp.MustCompile(`^[A-Z]{2}$`)
)

// SIPTrunk is a customer's own SIP trunk as a call is dialled through it, password open.
type SIPTrunk struct {
	Host      string
	Port      int
	Transport string
	Username  string
	Password  string
	LateOffer bool
	Codecs    []string
}

// SIPTrunkSettings is what a customer says about a trunk.
type SIPTrunkSettings struct {
	Name      string
	Host      string
	Port      int
	Transport string
	Username  string
	// Password is nil on an update that keeps the stored one.
	Password  *string
	LateOffer bool
	Codecs    []string
}

// SettingsOf is what a stored trunk says, without its password, for an update to change
// some of it.
func SettingsOf(trunk store.SIPTrunk) SIPTrunkSettings {
	return SIPTrunkSettings{
		Name: trunk.Name, Host: trunk.Host, Port: trunk.Port, Transport: trunk.Transport,
		Username: trunk.Username, LateOffer: trunk.LateOffer, Codecs: trunk.Codecs,
	}
}

func (settings SIPTrunkSettings) normalized() SIPTrunkSettings {
	settings.Host = strings.TrimSpace(settings.Host)
	if settings.Port == 0 {
		settings.Port = defaultSIPPort
	}
	if settings.Transport == "" {
		settings.Transport = defaultSIPTransport
	}
	settings.Transport = strings.ToLower(settings.Transport)
	if len(settings.Codecs) == 0 {
		settings.Codecs = []string{"PCMU", "PCMA"}
	}
	codecs := make([]string, len(settings.Codecs))
	for i, codec := range settings.Codecs {
		codecs[i] = strings.ToUpper(codec)
	}
	settings.Codecs = codecs
	return settings
}

// nonPublicIP says host is an IP address calls may not go to. A name is only checked when a
// call is placed, since what it resolves to can change after it is saved.
func nonPublicIP(host string) bool {
	ip, err := netip.ParseAddr(host)
	return err == nil && !egress.IsPublic(ip)
}

// validate reports every problem at once, so a form is fixed in one pass. creating says the
// password has to be there; an update without one keeps the stored password.
func (settings SIPTrunkSettings) validate(creating bool) error {
	var reasons []string
	if settings.Name == "" {
		reasons = append(reasons, "name is required")
	}
	switch {
	case settings.Host == "":
		reasons = append(reasons, "host is required")
	case strings.ContainsAny(settings.Host, ":@/; "):
		reasons = append(reasons, fmt.Sprintf("host %q must be a bare hostname, without sip: or a port", settings.Host))
	case nonPublicIP(settings.Host):
		reasons = append(reasons, fmt.Sprintf("host %q is not a public address", settings.Host))
	}
	if settings.Port < 1 || settings.Port > 65535 {
		reasons = append(reasons, fmt.Sprintf("port %d must be 1..65535", settings.Port))
	}
	if settings.Transport != "udp" && settings.Transport != "tcp" && settings.Transport != "tls" {
		reasons = append(reasons, fmt.Sprintf("transport %q must be udp, tcp or tls", settings.Transport))
	}
	if settings.Username == "" {
		reasons = append(reasons, "username is required")
	}
	switch {
	case settings.Password == nil && creating:
		reasons = append(reasons, "password is required")
	case settings.Password != nil && *settings.Password == "":
		reasons = append(reasons, "password cannot be empty")
	}
	for _, codec := range settings.Codecs {
		if sipbridge.CheckCodecs([]string{codec}) != nil {
			reasons = append(reasons, fmt.Sprintf("codec %q is not supported, use PCMU, PCMA or G722", codec))
		}
	}
	return invalid(reasons)
}

// TrunkNumber is a number a customer says is on one of their own trunks.
type TrunkNumber struct {
	Owner   routing.Owner
	TrunkID string
	E164    string
	// Country is the ISO 3166-1 alpha-2 code. It is asked for rather than worked out,
	// because nothing here parses numbers.
	Country string
}

func (s *Service) sipTrunksReady() error {
	if s.sealer == nil || s.sipTrunks == nil {
		return stack.Wrap(ErrSIPTrunksDisabled)
	}
	if s.store == nil {
		return stack.Wrap(errors.New("phone: sip trunks need a database"))
	}
	return nil
}

// sipTrunkAAD binds a sealed password to its trunk, so ciphertext copied onto another row
// does not open.
func sipTrunkAAD(customerID, id string) []byte {
	return []byte("sip_trunks/" + customerID + "/" + id)
}

// CreateSIPTrunk stores a customer's trunk with its password sealed.
func (s *Service) CreateSIPTrunk(ctx context.Context, customerID string, settings SIPTrunkSettings) (store.SIPTrunk, error) {
	if err := s.sipTrunksReady(); err != nil {
		return store.SIPTrunk{}, err
	}
	if customerID == "" {
		return store.SIPTrunk{}, stack.Wrap(errors.New("phone: a sip trunk must belong to a customer"))
	}
	settings = settings.normalized()
	if err := settings.validate(true); err != nil {
		return store.SIPTrunk{}, err
	}
	trunk := store.SIPTrunk{ID: uuid.NewString(), CustomerID: customerID}
	if err := s.applySIPTrunk(&trunk, settings); err != nil {
		return store.SIPTrunk{}, err
	}
	if err := s.store.CreateSIPTrunk(ctx, &trunk); err != nil {
		return store.SIPTrunk{}, err
	}
	return trunk, nil
}

// SIPTrunks returns a customer's trunks.
func (s *Service) SIPTrunks(ctx context.Context, customerID string) ([]store.SIPTrunk, error) {
	if err := s.sipTrunksReady(); err != nil {
		return nil, err
	}
	return s.store.SIPTrunks(ctx, customerID)
}

// SIPTrunk returns one of a customer's trunks.
func (s *Service) SIPTrunk(ctx context.Context, customerID, id string) (store.SIPTrunk, error) {
	if err := s.sipTrunksReady(); err != nil {
		return store.SIPTrunk{}, err
	}
	return s.store.SIPTrunk(ctx, customerID, id)
}

// UpdateSIPTrunk replaces what a trunk says. A nil password keeps the stored one.
func (s *Service) UpdateSIPTrunk(ctx context.Context, customerID, id string, settings SIPTrunkSettings) (store.SIPTrunk, error) {
	if err := s.sipTrunksReady(); err != nil {
		return store.SIPTrunk{}, err
	}
	trunk, err := s.store.SIPTrunk(ctx, customerID, id)
	if err != nil {
		return store.SIPTrunk{}, err
	}
	settings = settings.normalized()
	if err := settings.validate(false); err != nil {
		return store.SIPTrunk{}, err
	}
	if err := s.applySIPTrunk(&trunk, settings); err != nil {
		return store.SIPTrunk{}, err
	}
	if err := s.store.UpdateSIPTrunk(ctx, &trunk); err != nil {
		return store.SIPTrunk{}, err
	}
	return trunk, nil
}

// DeleteSIPTrunk removes a trunk that has no numbers left on it.
func (s *Service) DeleteSIPTrunk(ctx context.Context, customerID, id string) error {
	if err := s.sipTrunksReady(); err != nil {
		return err
	}
	return s.store.DeleteSIPTrunk(ctx, customerID, id)
}

// AddTrunkNumber records a number on one of the customer's trunks. Nothing is bought and
// nothing is checked with a carrier: whether the number is really the customer's is the
// trunk's carrier's to decide when it is dialled from.
func (s *Service) AddTrunkNumber(ctx context.Context, request TrunkNumber) (store.PhoneNumber, error) {
	if err := s.sipTrunksReady(); err != nil {
		return store.PhoneNumber{}, err
	}
	country := strings.ToUpper(request.Country)
	var reasons []string
	if !e164Pattern.MatchString(request.E164) {
		reasons = append(reasons, fmt.Sprintf("e164 %q is not an E.164 number, e.g. +15551234567", request.E164))
	}
	if !countryPattern.MatchString(country) {
		reasons = append(reasons, fmt.Sprintf("country %q must be a two-letter code, e.g. US", request.Country))
	}
	if err := request.Owner.Tags.Validate(); err != nil {
		reasons = append(reasons, strings.TrimPrefix(err.Error(), "routing: "))
	}
	if err := invalid(reasons); err != nil {
		return store.PhoneNumber{}, err
	}
	number := store.PhoneNumber{
		E164:         request.E164,
		Vendor:       SIPTrunkVendor,
		Country:      country,
		Capabilities: []string{string(Voice)},
		CustomerID:   request.Owner.CustomerID,
		Tags:         request.Owner.Tags,
		SIPTrunkID:   request.TrunkID,
		PurchasedAt:  time.Now().UTC(),
	}
	if err := s.store.AddTrunkNumber(ctx, &number); err != nil {
		return store.PhoneNumber{}, err
	}
	return number, nil
}

// applySIPTrunk copies settings onto a trunk row, sealing a new password when one was given.
func (s *Service) applySIPTrunk(trunk *store.SIPTrunk, settings SIPTrunkSettings) error {
	trunk.Name = settings.Name
	trunk.Host = settings.Host
	trunk.Port = settings.Port
	trunk.Transport = settings.Transport
	trunk.Username = settings.Username
	trunk.LateOffer = settings.LateOffer
	trunk.Codecs = settings.Codecs
	if settings.Password == nil {
		return nil
	}
	sealed, err := s.sealer.SealWithAAD(*settings.Password, sipTrunkAAD(trunk.CustomerID, trunk.ID))
	if err != nil {
		return stack.Wrap(fmt.Errorf("phone: seal sip trunk password: %w", err))
	}
	trunk.PasswordSealed = sealed
	trunk.PasswordKEKVersion = s.sealer.CurrentVersion()
	return nil
}

// openSIPTrunk is a trunk as a call is dialled through it.
func (s *Service) openSIPTrunk(ctx context.Context, customerID, id string) (*SIPTrunk, error) {
	trunk, err := s.store.SIPTrunk(ctx, customerID, id)
	if err != nil {
		return nil, err
	}
	if !trunk.HasPassword() {
		return nil, stack.Wrap(fmt.Errorf("phone: sip trunk %s has no password; set one before calling through it", id))
	}
	password, err := s.sealer.OpenWithAADVersion(trunk.PasswordSealed, sipTrunkAAD(customerID, id), trunk.PasswordKEKVersion)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("phone: open sip trunk password: %w", err))
	}
	return &SIPTrunk{
		Host: trunk.Host, Port: trunk.Port, Transport: trunk.Transport, Username: trunk.Username,
		Password: password, LateOffer: trunk.LateOffer, Codecs: trunk.Codecs,
	}, nil
}

// Close hangs up the calls this node holds through customers' own trunks. Calls placed
// at a vendor are the vendor's to hold and are not touched.
func (s *Service) Close(ctx context.Context) error {
	if closer, ok := s.sipTrunks.(interface{ Close(context.Context) error }); ok {
		return closer.Close(ctx)
	}
	return nil
}
