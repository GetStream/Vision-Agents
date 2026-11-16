package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"

	getstream "github.com/GetStream/getstream-go/v5"
)

func main() {
	client, err := getstream.NewClientFromEnvVars()
	if err != nil {
		fmt.Println("client:", err)
		os.Exit(1)
	}
	ctx := context.Background()

	listed, err := client.Video().ListSIPTrunks(ctx, &getstream.ListSIPTrunksRequest{})
	if err != nil {
		fmt.Println("list:", err)
		os.Exit(1)
	}
	for _, trunk := range listed.Data.SipTrunks {
		dumped, _ := json.MarshalIndent(trunk, "", "  ")
		fmt.Println(string(dumped))
	}

	rules, err := client.Video().ListSIPInboundRoutingRule(ctx, &getstream.ListSIPInboundRoutingRuleRequest{})
	if err != nil {
		fmt.Println("rules:", err)
		os.Exit(1)
	}
	for _, rule := range rules.Data.SipInboundRoutingRules {
		dumped, _ := json.MarshalIndent(rule, "", "  ")
		fmt.Println(string(dumped))
	}

	for _, id := range os.Args[1:] {
		_, ruleErr := client.Video().DeleteSIPInboundRoutingRule(ctx, id, &getstream.DeleteSIPInboundRoutingRuleRequest{})
		_, trunkErr := client.Video().DeleteSIPTrunk(ctx, id, &getstream.DeleteSIPTrunkRequest{})
		fmt.Println("deleted", id, "rule:", ruleErr, "trunk:", trunkErr)
	}
}
