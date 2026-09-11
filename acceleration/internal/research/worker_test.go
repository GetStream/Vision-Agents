package research

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestWorkerRejectsUnauthenticatedAndUnscoped(t *testing.T) {
	_, repos := fixture(t)
	w := &Worker{Profile: Profile{Repositories: repos}, Token: "test-secret"}
	for _, tc := range []struct {
		auth, body string
		code       int
	}{{"", `{}`, 401}, {"Bearer wrong", `{}`, 401}, {"Bearer test-secret", `{"product":"video","sdk":"react","question":"where is the view?"}`, 400}} {
		r := httptest.NewRequest(http.MethodPost, "/research", strings.NewReader(tc.body))
		r.Header.Set("Authorization", tc.auth)
		out := httptest.NewRecorder()
		w.ServeHTTP(out, r)
		if out.Code != tc.code {
			t.Fatal(out.Code, tc.code)
		}
	}
}
