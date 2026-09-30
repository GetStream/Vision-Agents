package streamedge

import (
	"strings"

	rtc "github.com/GetStream/getstream-go-webrtc"
)

// regionAirports is the airport nearest each GCP and AWS region. The coordinator places a
// client by an airport code, and picks the SFUs nearest that airport.
var regionAirports = map[string]string{
	// GCP
	"us-east1":                "CHS",
	"us-east4":                "IAD",
	"us-east5":                "CMH",
	"us-central1":             "OMA",
	"us-south1":               "DFW",
	"us-west1":                "PDX",
	"us-west2":                "LAX",
	"us-west3":                "SLC",
	"us-west4":                "LAS",
	"northamerica-northeast1": "YUL",
	"northamerica-northeast2": "YYZ",
	"northamerica-south1":     "QRO",
	"southamerica-east1":      "GRU",
	"southamerica-west1":      "SCL",
	"europe-west1":            "BRU",
	"europe-west2":            "LHR",
	"europe-west3":            "FRA",
	"europe-west4":            "AMS",
	"europe-west6":            "ZRH",
	"europe-west8":            "MXP",
	"europe-west9":            "CDG",
	"europe-west10":           "BER",
	"europe-west12":           "TRN",
	"europe-north1":           "HEL",
	"europe-north2":           "ARN",
	"europe-central2":         "WAW",
	"europe-southwest1":       "MAD",
	"asia-east1":              "TPE",
	"asia-east2":              "HKG",
	"asia-northeast1":         "NRT",
	"asia-northeast2":         "KIX",
	"asia-northeast3":         "ICN",
	"asia-south1":             "BOM",
	"asia-south2":             "DEL",
	"asia-southeast1":         "SIN",
	"asia-southeast2":         "CGK",
	"australia-southeast1":    "SYD",
	"australia-southeast2":    "MEL",
	"me-west1":                "TLV",
	"me-central1":             "DOH",
	"me-central2":             "DMM",
	"africa-south1":           "JNB",
	// AWS
	"us-east-1":      "IAD",
	"us-east-2":      "CMH",
	"us-west-1":      "SFO",
	"us-west-2":      "PDX",
	"ca-central-1":   "YUL",
	"ca-west-1":      "YYC",
	"mx-central-1":   "QRO",
	"sa-east-1":      "GRU",
	"eu-west-1":      "DUB",
	"eu-west-2":      "LHR",
	"eu-west-3":      "CDG",
	"eu-central-1":   "FRA",
	"eu-central-2":   "ZRH",
	"eu-north-1":     "ARN",
	"eu-south-1":     "MXP",
	"eu-south-2":     "MAD",
	"ap-east-1":      "HKG",
	"ap-northeast-1": "NRT",
	"ap-northeast-2": "ICN",
	"ap-northeast-3": "KIX",
	"ap-south-1":     "BOM",
	"ap-south-2":     "HYD",
	"ap-southeast-1": "SIN",
	"ap-southeast-2": "SYD",
	"ap-southeast-3": "CGK",
	"ap-southeast-4": "MEL",
	"ap-southeast-5": "KUL",
	"ap-southeast-7": "BKK",
	"me-south-1":     "BAH",
	"me-central-1":   "DXB",
	"il-central-1":   "TLV",
	"af-south-1":     "CPT",
}

// locationFor is the location the agent joins with: the airport nearest region, region
// itself when it is already an airport code, and rtc.LocationAuto (the coordinator's GeoIP
// on the agent's address) when there is no region or it is not one this knows.
func locationFor(region string) (location string, known bool) {
	region = strings.ToLower(strings.TrimSpace(region))
	if region == "" || region == rtc.LocationAuto {
		return rtc.LocationAuto, true
	}
	if airport, ok := regionAirports[region]; ok {
		return airport, true
	}
	if len(region) == 3 && strings.Trim(region, "abcdefghijklmnopqrstuvwxyz") == "" {
		return strings.ToUpper(region), true
	}
	return rtc.LocationAuto, false
}
