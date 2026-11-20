package egress

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"math/big"
	"reflect"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"
)

type DefaultsSuite struct {
	suite.Suite
}

func TestDefaultsSuite(t *testing.T) {
	suite.Run(t, new(DefaultsSuite))
}

// TestTheDefaultLimitsAreTheDocumentedOnes: the blind tunnel's idle limit is 90 s (Go's
// http.DefaultTransport IdleConnTimeout), a kept connection's is 4 min (twice libcurl's 118 s,
// see keepAliveTimeout). A changed default must change this test too.
func (s *DefaultsSuite) TestTheDefaultLimitsAreTheDocumentedOnes() {
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	s.Require().NoError(err)
	template := &x509.Certificate{SerialNumber: big.NewInt(1), Subject: pkix.Name{CommonName: "defaults"},
		NotBefore: time.Now().Add(-time.Hour), NotAfter: time.Now().Add(time.Hour), IsCA: true, BasicConstraintsValid: true,
		KeyUsage: x509.KeyUsageCertSign}
	der, err := x509.CreateCertificate(rand.Reader, template, template, &key.PublicKey, key)
	s.Require().NoError(err)
	keyDER, err := x509.MarshalECPrivateKey(key)
	s.Require().NoError(err)

	proxy, err := New(Options{
		CACert: pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der}),
		CAKey:  pem.EncodeToMemory(&pem.Block{Type: "EC PRIVATE KEY", Bytes: keyDER}),
	})
	s.Require().NoError(err)

	s.Equal(90*time.Second, proxy.idleTimeout)
	s.Equal(4*time.Minute, proxy.keepAliveTimeout)
}

// the proxy's default transport is NewUnpooledClient's, whose unpooled route closes
// each HTTP/1.1 connection after one request.
func (s *DefaultsSuite) TestTheDefaultTransportSendsWhatItCannotReplayOnANewConnection() {
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	s.Require().NoError(err)
	template := &x509.Certificate{SerialNumber: big.NewInt(1), Subject: pkix.Name{CommonName: "defaults"},
		NotBefore: time.Now().Add(-time.Hour), NotAfter: time.Now().Add(time.Hour), IsCA: true, BasicConstraintsValid: true,
		KeyUsage: x509.KeyUsageCertSign}
	der, err := x509.CreateCertificate(rand.Reader, template, template, &key.PublicKey, key)
	s.Require().NoError(err)
	keyDER, err := x509.MarshalECPrivateKey(key)
	s.Require().NoError(err)
	proxy, err := New(Options{
		CACert: pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der}),
		CAKey:  pem.EncodeToMemory(&pem.Block{Type: "EC PRIVATE KEY", Bytes: keyDER}),
	})
	s.Require().NoError(err)

	unpooled := reflect.ValueOf(proxy.transport.(outbound).next).FieldByName("unpooled")

	s.Require().True(unpooled.IsValid(), "internal/egress's client transport has no unpooled route")
	s.Require().False(unpooled.IsNil(), "the default transport is NewClient's, pooled for every request")
	s.Equal(int64(-1), unpooled.Elem().FieldByName("MaxIdleConnsPerHost").Int())
}
