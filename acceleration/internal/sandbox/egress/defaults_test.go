package egress

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"math/big"
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
