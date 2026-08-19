package world

import (
	"fmt"
	"strings"
)

func registerTelecom(s *Server) {
	s.handle("verify_account", func(sess *Session, args map[string]any) (any, error) {
		accounts := getList(sess.State, "accounts")
		pin := strArg(args, "pin")
		last4 := strArg(args, "last4")
		address := strArg(args, "address")
		for _, raw := range accounts {
			a, _ := raw.(map[string]any)
			if a == nil {
				continue
			}
			if asString(a["pin"]) != pin {
				continue
			}
			if last4 != "" && asString(a["last4"]) != last4 {
				continue
			}
			if address != "" && !strings.Contains(strings.ToLower(asString(a["address"])), strings.ToLower(address)) {
				continue
			}
			sess.State["identity_verified"] = true
			sess.State["account"] = a
			return map[string]any{"ok": true, "account_id": a["id"]}, nil
		}
		sess.State["identity_verified"] = false
		return map[string]any{"ok": false}, nil
	})

	s.handle("check_outage", func(sess *Session, args map[string]any) (any, error) {
		outage := truthy(sess.State["outage"])
		return map[string]any{
			"outage":  outage,
			"message": sess.State["outage_message"],
		}, nil
	})

	s.handle("walk_reboot", func(sess *Session, args map[string]any) (any, error) {
		ok := truthy(sess.State["reboot_succeeds"])
		sess.State["reboot_attempted"] = true
		sess.State["reboot_ok"] = ok
		return map[string]any{"ok": ok, "error_code": sess.State["gateway_error"]}, nil
	})

	s.handle("create_ticket", func(sess *Session, args map[string]any) (any, error) {
		if !truthy(sess.State["identity_verified"]) {
			return nil, fmt.Errorf("identity not verified")
		}
		ticket := map[string]any{
			"id":      "T-" + asString(sess.State["ticket_seq"]),
			"reason":  strArg(args, "reason"),
			"address": strArg(args, "address"),
		}
		if asString(ticket["id"]) == "T-" {
			ticket["id"] = "T-1842"
		}
		list := getList(sess.State, "tickets")
		sess.State["tickets"] = append(list, ticket)
		sess.State["ticket"] = ticket
		return ticket, nil
	})

	s.handle("dispatch_tech", func(sess *Session, args map[string]any) (any, error) {
		if !truthy(sess.State["reboot_attempted"]) {
			return nil, fmt.Errorf("reboot the gateway before dispatching a truck")
		}
		if truthy(sess.State["reboot_ok"]) {
			return nil, fmt.Errorf("service restored after reboot; no truck needed")
		}
		if !truthy(sess.State["identity_verified"]) {
			return nil, fmt.Errorf("identity not verified")
		}
		dispatch := map[string]any{
			"window": strArg(args, "window"),
			"ticket": strArg(args, "ticket_id"),
		}
		sess.State["dispatch"] = dispatch
		return dispatch, nil
	})

	s.handle("apply_credit", func(sess *Session, args map[string]any) (any, error) {
		if !truthy(sess.State["identity_verified"]) {
			return nil, fmt.Errorf("identity not verified")
		}
		if !truthy(sess.State["credit_eligible"]) {
			return nil, fmt.Errorf("account is not eligible for a bill credit")
		}
		credit := map[string]any{"amount": numArg(args, "amount")}
		sess.State["credit"] = credit
		return credit, nil
	})

	s.handle("change_plan", func(sess *Session, args map[string]any) (any, error) {
		if !truthy(sess.State["identity_verified"]) {
			return nil, fmt.Errorf("identity not verified")
		}
		plan := strArg(args, "plan")
		acct := getMap(sess.State, "account")
		acct["plan"] = plan
		sess.State["account"] = acct
		return acct, nil
	})

	s.handle("create_transfer_summary", func(sess *Session, args map[string]any) (any, error) {
		summary := strArg(args, "summary")
		sess.State["transfer_summary"] = summary
		return map[string]any{"summary": summary}, nil
	})
}
