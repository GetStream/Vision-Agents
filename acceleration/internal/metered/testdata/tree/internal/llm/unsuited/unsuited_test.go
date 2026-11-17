package unsuited

import "testing"

func TestTheNameIsTheVendors(t *testing.T) {
	if ProviderName != "unsuited" {
		t.Fail()
	}
}
