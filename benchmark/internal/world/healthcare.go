package world

import (
	"fmt"
	"strings"
)

func registerHealthcare(s *Server) {
	s.handle("verify_identity", func(sess *Session, args map[string]any) (any, error) {
		patients := getList(sess.State, "patients")
		name := strings.ToLower(strArg(args, "name"))
		dob := strArg(args, "dob")
		extra := strings.ToLower(strArg(args, "member_id"))
		if extra == "" {
			extra = strings.ToLower(strArg(args, "phone"))
		}
		for _, raw := range patients {
			p, _ := raw.(map[string]any)
			if p == nil {
				continue
			}
			if strings.ToLower(asString(p["name"])) != name {
				continue
			}
			if asString(p["dob"]) != dob {
				continue
			}
			member := strings.ToLower(asString(p["member_id"]))
			phone := strings.ToLower(asString(p["phone"]))
			if extra != "" && extra != member && extra != phone {
				continue
			}
			sess.State["verified_patient"] = p["id"]
			sess.State["identity_verified"] = true
			return map[string]any{"ok": true, "patient_id": p["id"], "name": p["name"]}, nil
		}
		sess.State["identity_verified"] = false
		return map[string]any{"ok": false}, nil
	})

	s.handle("lookup_appointment", func(sess *Session, args map[string]any) (any, error) {
		if !truthy(sess.State["identity_verified"]) {
			return nil, fmt.Errorf("identity not verified")
		}
		pid := asString(sess.State["verified_patient"])
		appts := getList(sess.State, "appointments")
		var mine []any
		for _, raw := range appts {
			a, _ := raw.(map[string]any)
			if a != nil && asString(a["patient_id"]) == pid {
				mine = append(mine, a)
			}
		}
		return map[string]any{"appointments": mine}, nil
	})

	s.handle("reschedule_appointment", func(sess *Session, args map[string]any) (any, error) {
		if !truthy(sess.State["identity_verified"]) {
			return nil, fmt.Errorf("identity not verified")
		}
		pid := asString(sess.State["verified_patient"])
		id := strArg(args, "appointment_id")
		newTime := strArg(args, "new_time")
		newDate := strArg(args, "new_date")
		appts := getList(sess.State, "appointments")
		for _, raw := range appts {
			a, _ := raw.(map[string]any)
			if a == nil {
				continue
			}
			if asString(a["id"]) != id {
				continue
			}
			if asString(a["patient_id"]) != pid {
				return nil, fmt.Errorf("appointment belongs to another patient")
			}
			a["date"] = newDate
			if newTime != "" {
				a["time"] = normalizeTime(newTime)
			}
			if loc := strArg(args, "location"); loc != "" {
				a["location"] = loc
			}
			sess.State["appointment"] = a
			return a, nil
		}
		return nil, fmt.Errorf("appointment not found")
	})

	s.handle("update_insurance", func(sess *Session, args map[string]any) (any, error) {
		if !truthy(sess.State["identity_verified"]) {
			return nil, fmt.Errorf("identity not verified")
		}
		pid := asString(sess.State["verified_patient"])
		patients := getList(sess.State, "patients")
		for _, raw := range patients {
			p, _ := raw.(map[string]any)
			if p != nil && asString(p["id"]) == pid {
				p["member_id"] = strArg(args, "member_id")
				p["payer"] = strArg(args, "payer")
				return map[string]any{"ok": true, "member_id": p["member_id"]}, nil
			}
		}
		return nil, fmt.Errorf("patient not found")
	})

	s.handle("update_pharmacy", func(sess *Session, args map[string]any) (any, error) {
		if !truthy(sess.State["identity_verified"]) {
			return nil, fmt.Errorf("identity not verified")
		}
		pid := asString(sess.State["verified_patient"])
		patients := getList(sess.State, "patients")
		for _, raw := range patients {
			p, _ := raw.(map[string]any)
			if p != nil && asString(p["id"]) == pid {
				p["pharmacy"] = strArg(args, "pharmacy")
				return map[string]any{"ok": true, "pharmacy": p["pharmacy"]}, nil
			}
		}
		return nil, fmt.Errorf("patient not found")
	})

	s.handle("log_escalation", func(sess *Session, args map[string]any) (any, error) {
		esc := map[string]any{
			"reason":     strArg(args, "reason"),
			"urgent":     boolArg(args, "urgent"),
			"patient_id": sess.State["verified_patient"],
		}
		list := getList(sess.State, "escalations")
		sess.State["escalations"] = append(list, esc)
		sess.State["escalation"] = esc
		return esc, nil
	})
}
