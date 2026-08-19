package world

import (
	"fmt"
	"strings"
)

func registerRestaurant(s *Server) {
	s.handle("check_availability", func(sess *Session, args map[string]any) (any, error) {
		wantTime := normalizeTime(strArg(args, "time"))
		party := int(numArg(args, "party_size"))
		patio := boolArg(args, "patio")
		slots, _ := sess.State["slots"].([]any)
		var matches []any
		for _, raw := range slots {
			slot, _ := raw.(map[string]any)
			if slot == nil {
				continue
			}
			if wantTime != "" && normalizeTime(asString(slot["time"])) != wantTime {
				continue
			}
			if patio && !truthy(slot["patio"]) {
				continue
			}
			cap := int(asFloat(slot["capacity"]))
			if cap > 0 && party > cap {
				continue
			}
			if !truthy(slot["available"]) {
				continue
			}
			matches = append(matches, cloneMap(slot))
		}
		alternates := []any{}
		if len(matches) == 0 {
			for _, raw := range slots {
				slot, _ := raw.(map[string]any)
				if slot != nil && truthy(slot["available"]) {
					alternates = append(alternates, cloneMap(slot))
				}
			}
		}
		return map[string]any{"available": matches, "alternates": alternates}, nil
	})

	s.handle("create_reservation", func(sess *Session, args map[string]any) (any, error) {
		slots, _ := sess.State["slots"].([]any)
		timeStr := normalizeTime(strArg(args, "time"))
		party := int(numArg(args, "party_size"))
		patio := boolArg(args, "patio")
		allergen := strArg(args, "allergen")
		if allergen == "" {
			return nil, fmt.Errorf("allergen is required")
		}
		var chosen map[string]any
		for _, raw := range slots {
			slot, _ := raw.(map[string]any)
			if slot == nil {
				continue
			}
			if normalizeTime(asString(slot["time"])) == timeStr && truthy(slot["available"]) {
				if patio && !truthy(slot["patio"]) {
					continue
				}
				chosen = slot
				break
			}
		}
		if chosen == nil {
			return nil, fmt.Errorf("no table at %s", timeStr)
		}
		chosen["available"] = false
		res := map[string]any{
			"name":       strArg(args, "name"),
			"phone":      strArg(args, "phone"),
			"time":       timeStr,
			"party_size": party,
			"patio":      patio,
			"high_chair": boolArg(args, "high_chair"),
			"allergen":   allergen,
			"notes":      strArg(args, "notes"),
		}
		sess.State["reservation"] = res
		return res, nil
	})

	s.handle("update_reservation", func(sess *Session, args map[string]any) (any, error) {
		res := getMap(sess.State, "reservation")
		if len(res) == 0 {
			return nil, fmt.Errorf("no reservation")
		}
		for _, key := range []string{"name", "phone", "allergen", "notes"} {
			if v, ok := args[key]; ok {
				res[key] = v
			}
		}
		if v, ok := args["time"]; ok {
			res["time"] = normalizeTime(asString(v))
		}
		if _, ok := args["party_size"]; ok {
			res["party_size"] = int(numArg(args, "party_size"))
		}
		if _, ok := args["patio"]; ok {
			res["patio"] = boolArg(args, "patio")
		}
		if _, ok := args["high_chair"]; ok {
			res["high_chair"] = boolArg(args, "high_chair")
		}
		sess.State["reservation"] = res
		return res, nil
	})

	s.handle("lookup_menu", func(sess *Session, args map[string]any) (any, error) {
		return map[string]any{"items": getList(sess.State, "menu")}, nil
	})

	s.handle("create_order", func(sess *Session, args map[string]any) (any, error) {
		items, _ := args["items"].([]any)
		menu := getList(sess.State, "menu")
		var lines []any
		var total float64
		for _, raw := range items {
			line, _ := raw.(map[string]any)
			if line == nil {
				continue
			}
			name := asString(line["name"])
			item := findMenu(menu, name)
			if item == nil {
				return nil, fmt.Errorf("unknown item %s", name)
			}
			if !truthy(item["available"]) {
				return map[string]any{
					"error":      "86",
					"item":       name,
					"substitute": item["substitute"],
					"message":    asString(item["name"]) + " is 86'd",
				}, nil
			}
			price := asFloat(item["price"])
			total += price
			line["price"] = price
			lines = append(lines, line)
		}
		order := map[string]any{
			"name":          strArg(args, "name"),
			"items":         lines,
			"total":         total,
			"pickup_window": strArg(args, "pickup_window"),
			"allergen":      strArg(args, "allergen"),
			"modifiers":     args["modifiers"],
		}
		if order["allergen"] == "" {
			return nil, fmt.Errorf("allergen is required")
		}
		sess.State["order"] = order
		return order, nil
	})
}

func findMenu(menu []any, name string) map[string]any {
	want := strings.ToLower(name)
	for _, raw := range menu {
		item, _ := raw.(map[string]any)
		if item != nil && strings.ToLower(asString(item["name"])) == want {
			return item
		}
	}
	return nil
}

func truthy(v any) bool {
	switch t := v.(type) {
	case bool:
		return t
	case string:
		return t == "true" || t == "1"
	default:
		return false
	}
}

func asFloat(v any) float64 {
	switch t := v.(type) {
	case float64:
		return t
	case int:
		return float64(t)
	case int64:
		return float64(t)
	default:
		return 0
	}
}
