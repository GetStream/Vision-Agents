package twiliosig

import (
	"maps"
	"net/url"
	"testing"

	"github.com/stretchr/testify/suite"
)

// The vectors are Twilio's own, from the request-validator tests of its SDKs, all signed with
// the token "12345":
//   - twilio-python tests/unit/test_request_validator.py, test_compute_signature_duplicate_query_dict:
//     the URL below and "Sid=CA123&SidAccount=AC123&Digits=5678&Digits=1234&Digits=1234".
//   - twilio-go client/request_validator_test.go, "returns true when there's no other parameters
//     and the signature is right": a URL carrying bodySHA256 and no parameters.
const (
	sdkToken = "12345"
	sdkURL   = "https://mycompany.com/myapp.php?foo=1&bar=2"
	sdkBody  = "Sid=CA123&SidAccount=AC123&Digits=5678&Digits=1234&Digits=1234"
	sdkSig   = "IK+Dwps556ElfBT0I3Rgjkr1wJU="

	sdkBodyHashURL = "https://mycompany.com/myapp.php?bodySHA256=" +
		"0a1ff7634d9ab3b95db5c9a2dfe9416e41502b283a80c7cf19632632f96e6620"
	sdkBodyHashSig = "y77kIzt2vzLz71DgmJGsen2scGs="
)

type SignatureSuite struct {
	suite.Suite
	params url.Values
}

func TestSignature(t *testing.T) { suite.Run(t, new(SignatureSuite)) }

func (s *SignatureSuite) SetupTest() {
	var err error
	s.params, err = url.ParseQuery(sdkBody)
	s.Require().NoError(err)
}

// The duplicate vector catches names sorted after joining them to their values (SidAccount
// would come first), a repeated value signed twice, and repeated values signed unsorted.
func (s *SignatureSuite) TestTheSDKVectorsVerify() {
	s.NoError(Verify(sdkToken, sdkURL, s.params, sdkSig))
	s.NoError(Verify(sdkToken, sdkBodyHashURL, nil, sdkBodyHashSig))
}

func (s *SignatureSuite) TestTheOrderParametersArriveInDoesNotMatter() {
	for _, body := range []string{
		"Digits=1234&SidAccount=AC123&Digits=1234&Sid=CA123&Digits=5678",
		"SidAccount=AC123&Digits=1234&Digits=5678&Sid=CA123",
	} {
		params, err := url.ParseQuery(body)
		s.Require().NoError(err)

		s.NoError(Verify(sdkToken, sdkURL, params, sdkSig), body)
	}
}

func (s *SignatureSuite) TestOneChangedByteFails() {
	changed := func(name string, values ...string) url.Values {
		params := maps.Clone(s.params)
		params[name] = values
		return params
	}
	cases := map[string]struct {
		url       string
		params    url.Values
		signature string
	}{
		"url":            {"https://mycompany.com/myapp.php?foo=1&bar=3", s.params, sdkSig},
		"value":          {sdkURL, changed("Sid", "CA124"), sdkSig},
		"repeated value": {sdkURL, changed("Digits", "5678", "1235"), sdkSig},
		"name":           {sdkURL, changed("Side", "CA123"), sdkSig},
		"signature":      {sdkURL, s.params, "IK+Dwps556ElfBT0I3Rgjkr1wJV="},
	}
	for name, c := range cases {
		s.Error(Verify(sdkToken, c.url, c.params, c.signature), name)
	}
}

// Dropping the parameters from what is signed would let a captured signature carry any body.
func (s *SignatureSuite) TestTheParametersAreSigned() {
	s.Error(Verify(sdkToken, sdkBodyHashURL, url.Values{"Sid": {"CA123"}}, sdkBodyHashSig))
}

// A parameter sent empty still signs its name. The signature is computed outside this package:
// HMAC-SHA1 with "12345" over the URL followed by "BodySidCA123".
func (s *SignatureSuite) TestAnEmptyValueSignsItsName() {
	params := url.Values{"Body": {""}, "Sid": {"CA123"}}

	s.NoError(Verify(sdkToken, sdkURL, params, "hgUYSJCJjLlX4ZzJCMRw9lN7Hb8="))
}

func (s *SignatureSuite) TestAnotherTokensSignatureFails() {
	s.Error(Verify(sdkToken, sdkURL, s.params, Sign("54321", sdkURL, s.params)))
}

func (s *SignatureSuite) TestAMissingSignatureFails() {
	s.Error(Verify(sdkToken, sdkURL, s.params, ""))
}

func (s *SignatureSuite) TestASignatureIsComparedAsSent() {
	for _, signature := range []string{" " + sdkSig, sdkSig + "\n", "IK-Dwps556ElfBT0I3Rgjkr1wJU="} {
		s.Error(Verify(sdkToken, sdkURL, s.params, signature), signature)
	}
}

// Anyone can sign with an empty key, so an unset token must verify nothing.
func (s *SignatureSuite) TestWithoutATokenNothingPasses() {
	s.Error(Verify("", sdkURL, s.params, Sign("", sdkURL, s.params)))
}

func (s *SignatureSuite) TestATokenWithWhitespaceIsRefused() {
	s.NoError(CheckAuthToken("0123456789abcdef0123456789abcdef"))
	for _, token := range []string{
		"0123456789abcdef0123456789abcdef\n",
		"0123456789abcdef0123456789abcdef\r\n",
		" 0123456789abcdef0123456789abcdef",
		"0123456789abcdef\t0123456789abcdef",
	} {
		s.Error(CheckAuthToken(token), "%q", token)
	}
}
