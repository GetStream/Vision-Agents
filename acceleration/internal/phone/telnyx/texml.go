package telnyx

import (
	"encoding/xml"
	"fmt"

	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// Telnyx's <Dial> rings for between these many seconds.
const (
	minRingSeconds = 5
	maxRingSeconds = 120
)

// dialTrunk is the TeXML that joins a call to the Stream trunk. ringSeconds bounds how long
// the trunk is rung, and zero leaves Telnyx's default.
func dialTrunk(bridge phone.Bridge, ringSeconds int) ([]byte, error) {
	return renderTexml(bridge, ringSeconds, "")
}

// renderTexml is dialTrunk with digits played first, so a menu on the answered leg is
// gone through before the agent joins.
func renderTexml(bridge phone.Bridge, ringSeconds int, digits string) ([]byte, error) {
	if err := bridge.Validate(); err != nil {
		return nil, err
	}
	if ringSeconds != 0 && (ringSeconds < minRingSeconds || ringSeconds > maxRingSeconds) {
		return nil, stack.Wrap(fmt.Errorf("telnyx: a trunk rings for %d to %d seconds, not %d",
			minRingSeconds, maxRingSeconds, ringSeconds))
	}

	response := texmlResponse{
		Dial: texmlDial{
			Timeout: ringSeconds,
			Sip: texmlSip{
				URI:      bridge.URI,
				Username: bridge.Username,
				Password: bridge.Password,
			},
		},
	}
	if digits != "" {
		response.Play = &texmlPlay{Digits: digits}
	}

	encoded, err := xml.Marshal(response)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("telnyx: encode texml: %w", err))
	}
	return append([]byte(xml.Header), encoded...), nil
}

type texmlResponse struct {
	XMLName xml.Name   `xml:"Response"`
	Play    *texmlPlay `xml:"Play,omitempty"`
	Dial    texmlDial  `xml:"Dial"`
}

type texmlPlay struct {
	Digits string `xml:"digits,attr"`
}

type texmlDial struct {
	Timeout int      `xml:"timeout,attr,omitempty"`
	Sip     texmlSip `xml:"Sip"`
}

type texmlSip struct {
	URI      string `xml:",chardata"`
	Username string `xml:"username,attr,omitempty"`
	Password string `xml:"password,attr,omitempty"`
}
