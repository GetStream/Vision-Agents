package scenario

import "testing"

func TestAPlainPluralIsTheSameValue(t *testing.T) {
	for _, c := range []struct {
		got, want string
		same      bool
	}{
		{"peanuts", "peanut", true},
		{"Peanut", "peanuts", true},
		{"allergies", "allergy", true},
		{"glass", "glas", false},
		{"ABC123456s", "ABC123456", false},
		{"walnut", "peanut", false},
	} {
		if got := MatchStructuredValue(c.got, c.want); got != c.same {
			t.Errorf("MatchStructuredValue(%q, %q) = %v, want %v", c.got, c.want, got, c.same)
		}
	}
}

func TestAnAllergenSaidWithItsAllergyIsTheAllergen(t *testing.T) {
	for _, c := range []struct {
		got, want string
		same      bool
	}{
		{"peanut allergy", "peanut", true},
		{"severe peanut allergy", "peanut", true},
		{"gluten free", "gluten", true},
		{"Gluten-free", "gluten", true},
		{"tree nut allergy", "peanut", false},
		{"allergy", "peanut", false},
		{"peanut allergy", "none", false},
		{"no allergies", "none", true},
	} {
		if got := MatchStructuredValue(c.got, c.want); got != c.same {
			t.Errorf("MatchStructuredValue(%q, %q) = %v, want %v", c.got, c.want, got, c.same)
		}
	}
}
