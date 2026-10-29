//go:build integration

package llmrouter

import (
	"context"
	"os"
	"testing"

	"github.com/google/uuid"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/appconfig"
	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/anthropic"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing/pricingtest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/providerkeys"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// BYOKSuite routes on keys customers keep, put sealed through the configuration cache a router
// reads them through, and reads back the rows and the live health the work recorded.
type BYOKSuite struct {
	suite.Suite
	ctx        context.Context
	store      *store.Store
	live       *live.Client
	configs    *appconfig.Store
	sealer     *auth.Sealer
	customerID string
}

func TestBYOKSuite(t *testing.T) {
	suite.Run(t, new(BYOKSuite))
}

func (s *BYOKSuite) SetupSuite() {
	dsn, address := os.Getenv("ROUTER_POSTGRES_DSN"), os.Getenv("ROUTER_REDIS_ADDR")
	if dsn == "" || address == "" {
		s.T().Skip("ROUTER_POSTGRES_DSN and ROUTER_REDIS_ADDR must be set")
	}
	s.ctx = context.Background()
	opened, err := store.Open(dsn)
	s.Require().NoError(err)
	s.Require().NoError(opened.Migrate(s.ctx))
	s.store = opened
	s.T().Cleanup(func() { s.Require().NoError(opened.Close()) })
	s.live, err = live.New(live.Options{Address: address})
	s.Require().NoError(err)
	s.T().Cleanup(s.live.Close)
	s.configs, err = appconfig.New(appconfig.Options{Store: opened, Address: address})
	s.Require().NoError(err)
	s.T().Cleanup(s.configs.Close)
	s.sealer, err = auth.NewSealer("byok suite")
	s.Require().NoError(err)
}

func (s *BYOKSuite) SetupTest() {
	s.customerID = "byok-" + uuid.NewString()
}

// put keeps key as the customer's for provider in region, as the API puts one.
func (s *BYOKSuite) put(customerID, provider, region, key string) {
	sealed, err := providerkeys.Seal(s.sealer, customerID, provider, region, key)
	s.Require().NoError(err)
	_, _, err = s.configs.PutProviderKey(s.ctx, &sealed)
	s.Require().NoError(err)
}

// router routes config on the keys the suite's customers keep, recording into the suite's
// store and live counters, priced by prices (nil is the config's own list).
func (s *BYOKSuite) router(config routing.ModalityConfig, prices pricing.Pricer) *Router {
	router, err := New(Options{
		Registry: DefaultRegistry(), Config: config, Store: s.store, Live: s.live, Pricer: prices,
		Keys: providerkeys.NewKeyring(s.configs, s.sealer),
	})
	s.Require().NoError(err)
	s.T().Cleanup(router.Close)
	return router
}

// rows are the customer's rows, in the order they were recorded.
func (s *BYOKSuite) rows() []store.Request {
	var rows []store.Request
	s.Require().NoError(s.store.DB().NewSelect().Model(&rows).Where("customer_id = ?", s.customerID).Order("id ASC").Scan(s.ctx))
	return rows
}

func (s *BYOKSuite) TestTheCustomersKeyIsTriedFirstAndItsRowIsBilledAtTheBYOKFeeCostingTheOperatorNothing() {
	at := startVendor(s.T())
	s.put(s.customerID, anthropic.ProviderName, pricing.RegionUS, customerKey)
	list := pricingtest.List(s.T(), "byok")
	router := s.router(keyedConfig(s.T(), at), list)

	answeredFor(s.ctx, s.T(), router, s.customerID)
	router.Close()

	s.Equal([]string{customerKey}, at.asked())
	rows := s.rows()
	s.Require().Len(rows, 1)
	s.True(rows[0].Success)
	s.Equal("llm-anthropic-claude-haiku-5-5-us-byok", rows[0].SKU)
	s.Zero(rows[0].BaseCostMicros, "the vendor bills the customer")
	s.Equal(pricingtest.Micros(s.T(), list, pricingtest.LLM(pricing.RegionUS, true), rows[0].Quantities()), rows[0].CostMicros)
}

func (s *BYOKSuite) TestARefusedKeyFallsBackToTheDeploymentsInstanceAndLeavesTheModelsHealthAlone() {
	at := startVendor(s.T(), customerKey)
	s.put(s.customerID, anthropic.ProviderName, pricing.RegionUS, customerKey)
	config := keyedConfig(s.T(), at)
	// A model of the test's own, so nothing else that ran on this Redis is in its health.
	model := "claude-haiku-byok-" + uuid.NewString()
	config.Providers[0].Model = model
	router := s.router(config, nil)

	response := answeredFor(s.ctx, s.T(), router, s.customerID)
	router.Close()

	s.Equal("hello", response.OutputText)
	s.Equal([]string{customerKey, deploymentKey}, at.asked())
	rows := s.rows()
	s.Require().Len(rows, 2)
	s.False(rows[0].Success)
	s.Equal(routing.ErrorKeyRefused, rows[0].ErrorCode, "the customer is told their key was refused")
	s.Contains(rows[0].ErrorMessage, "anthropic refused the key this app keeps for it in us")
	s.NotContains(rows[0].ErrorMessage, customerKey)
	s.True(rows[1].Success)
	health, err := s.live.Health(s.ctx, string(routing.LLM), anthropic.ProviderName, model)
	s.Require().NoError(err)
	s.EqualValues(1, health.Requests, "the deployment's request alone")
	s.Zero(health.Errors, "one customer's refused key is not the model failing")
	s.True(health.Available)
}

func (s *BYOKSuite) TestADeletedKeyIsNoLongerRoutedOn() {
	at := startVendor(s.T())
	s.put(s.customerID, anthropic.ProviderName, pricing.RegionUS, customerKey)
	router := s.router(keyedConfig(s.T(), at), nil)
	answeredFor(s.ctx, s.T(), router, s.customerID)

	_, err := s.configs.DeleteProviderKey(s.ctx, s.customerID, anthropic.ProviderName, pricing.RegionUS)
	s.Require().NoError(err)
	answeredFor(s.ctx, s.T(), router, s.customerID)

	s.Equal([]string{customerKey, deploymentKey}, at.asked(), "the first on the customer's key, the next on the deployment's")
}

func (s *BYOKSuite) TestASealedKeyCopiedToAnotherRowDoesNotOpen() {
	s.put(s.customerID, anthropic.ProviderName, pricing.RegionUS, customerKey)
	kept, err := s.store.ProviderKeys(s.ctx, s.customerID)
	s.Require().NoError(err)
	s.Require().Len(kept, 1)
	keyring := providerkeys.NewKeyring(s.configs, s.sealer)

	toAnotherCustomer := kept[0]
	toAnotherCustomer.CustomerID = "byok-" + uuid.NewString()
	_, _, err = s.configs.PutProviderKey(s.ctx, &toAnotherCustomer)
	s.Require().NoError(err)
	_, _, otherCustomer := keyring.CustomerKey(s.ctx, toAnotherCustomer.CustomerID, anthropic.ProviderName, pricing.RegionUS)
	toAnotherRegion := kept[0]
	toAnotherRegion.Region = pricing.RegionEU
	_, _, err = s.configs.PutProviderKey(s.ctx, &toAnotherRegion)
	s.Require().NoError(err)
	_, _, otherRegion := keyring.CustomerKey(s.ctx, s.customerID, anthropic.ProviderName, pricing.RegionEU)

	for _, err := range []error{otherCustomer, otherRegion} {
		s.Require().Error(err)
		s.ErrorContains(err, "does not open")
		s.NotContains(err.Error(), customerKey)
	}
	at := startVendor(s.T())
	answeredFor(s.ctx, s.T(), s.router(keyedConfig(s.T(), at), nil), toAnotherCustomer.CustomerID)
	s.Equal([]string{deploymentKey}, at.asked(), "a key that does not open leaves the deployment's instance to serve")
}
