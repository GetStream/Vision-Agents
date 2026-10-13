# GCP / GKE security audit

Copied from the `gcp-security-audit` skill in Stream's private infra repo (`GetStream/chat`),
which is where the cluster list, the `accelerate` chart and the credentials live (see the
`deploy` skill). This repo is public: name clusters and projects in the report you write, not
in this file.

Read-only sweep across every GKE cluster + namespace the router runs in. Proves each of the top mistakes below is **not** present. No `apply` / `patch` / `delete` — surface findings, don't fix them.

Cluster list comes from the `gcp-fleet` skill in the infra repo (snapshot, no re-querying GCP). For the router alone, audit the clusters and namespaces the `accelerate` chart is deployed to.

---

## Conventions

```bash
PROJECT=$(gcloud config get-value project)
# clusters: from the gcp-fleet snapshot in the infra repo
# namespaces: the ones the accelerate chart (infra/charts/accelerate) is released into
```

Permissions needed (read-only is enough):

- `roles/container.clusterViewer` + `roles/container.viewer` on each cluster
- `roles/compute.networkViewer` on the project (firewall + NAT + LB inspection)
- `roles/iam.securityReviewer` on the project (SA + WI inspection)
- `roles/binaryauthorization.policyViewer` + `roles/containeranalysis.notes.viewer`

---

## Top mistakes (the audit matrix)

Each mistake below = one section of the audit + one row in the report. Mistake text is verbatim. Pass criteria + verification commands follow.

### 1. Workload identity misuse

> Workloads authenticate as the node service account or ship long-lived JSON keys. Risk: one Pod compromise can become project-level API access. GCP example: Pods use node metadata credentials from a Compute Engine default service account. Fix: enable Workload Identity Federation for GKE; use a custom node service account with only minimum required node roles and avoid service-account keys.

```bash
# 1a. Workload Identity must be enabled on every cluster.
gcloud container clusters describe "$CL" --location="$LOC" \
  --format='value(workloadIdentityConfig.workloadPool)'
# Pass: prints "<PROJECT>.svc.id.goog". FAIL if empty.

# 1b. Every node pool must run GKE Metadata Server (blocks pod access to node SA).
gcloud container node-pools list --cluster="$CL" --location="$LOC" \
  --format='value(name,config.workloadMetadataConfig.mode)'
# Pass: every row shows GKE_METADATA. FAIL on GCE_METADATA or empty.

# 1c. Nodes must NOT run as the Compute Engine default SA.
gcloud container node-pools list --cluster="$CL" --location="$LOC" \
  --format='value(name,config.serviceAccount)'
# FAIL on "default" or "<projectnum>-compute@developer.gserviceaccount.com".

# 1d. No long-lived SA keys anywhere in the project.
gcloud iam service-accounts list --format='value(email)' \
  | xargs -I{} gcloud iam service-accounts keys list --iam-account={} \
       --managed-by=user --format='value(name,validAfterTime)'
# Pass: empty. Any user-managed key is a FAIL — pivot to WI Federation.

# 1e. KSAs that claim to use WI must have the matching annotation.
kubectl get ksa -A -o json | jq -r '
  .items[]
  | select(.metadata.annotations["iam.gke.io/gcp-service-account"])
  | "\(.metadata.namespace)/\(.metadata.name) -> \(.metadata.annotations["iam.gke.io/gcp-service-account"])"'
# Cross-check each GSA's IAM policy has roles/iam.workloadIdentityUser bound to that KSA.
```

### 2. Over-privileged RBAC and token sprawl

> cluster-admin, wildcard verbs, or bindings to system:* groups remain common. Risk: easy privilege escalation and Secret disclosure; auto-mounted tokens widen blast radius. GCP example: an app ServiceAccount gets a cluster-wide binding, or system:authenticated is bound to a role. Fix: prefer namespace-scoped Roles and RoleBindings; disable token automount by default and avoid system:masters, system:authenticated, and system:unauthenticated bindings.

```bash
# 2a. No non-system principal should hold cluster-admin.
kubectl get clusterrolebindings -o json | jq -r '
  .items[]
  | select(.roleRef.name=="cluster-admin")
  | "\(.metadata.name): " + ((.subjects // []) | map("\(.kind)/\(.namespace // "-")/\(.name)") | join(", "))'
# FAIL on any app KSA, human user that isn't on the break-glass list, or any
# binding that includes system:authenticated / system:unauthenticated / system:masters.

# 2b. Wildcard verbs/resources in any Role or ClusterRole.
kubectl get clusterroles,roles -A -o json | jq -r '
  .items[]
  | . as $r | $r.rules[]?
  | select((.verbs[]? == "*") or (.resources[]? == "*") or (.apiGroups[]? == "*"))
  | "\($r.kind)/\($r.metadata.namespace // "-")/\($r.metadata.name): verbs=\(.verbs) resources=\(.resources)"'
# Investigate every hit. Built-in roles are OK; project-defined ones almost never need *.

# 2c. ServiceAccounts auto-mount tokens unless explicitly opted out.
kubectl get sa -A -o json | jq -r '
  .items[]
  | select(.automountServiceAccountToken != false)
  | "\(.metadata.namespace)/\(.metadata.name)"' | wc -l
# Goal: app SAs set automountServiceAccountToken: false. Quantify, then list outliers
# (any SA in a namespace that talks to the K8s API needs it true — call those out).

# 2d. Bindings that name system:* groups as subjects.
kubectl get clusterrolebindings,rolebindings -A -o json | jq -r '
  .items[]
  | . as $b | (.subjects // [])[]
  | select(.name | startswith("system:"))
  | "\($b.kind)/\($b.metadata.namespace // "-")/\($b.metadata.name) -> \(.kind)/\(.name)"'
# FAIL on any binding to system:authenticated, system:unauthenticated, system:masters.
```

### 3. Exposed control plane or public nodes

> Broadly reachable API endpoints and node external IPs increase attack surface. Risk: more internet-facing paths to the kube API and worker nodes. GCP example: IP-based endpoint left open, no authorized networks, private nodes disabled. Fix: prefer DNS-only or private control-plane access; otherwise enforce authorized networks, enable private nodes, and keep legacy auth methods and ABAC disabled.

```bash
gcloud container clusters describe "$CL" --location="$LOC" --format=yaml \
  | grep -E 'enablePrivateNodes|enablePrivateEndpoint|publicEndpoint|privateEndpoint|authorizedNetworksConfig|legacyAbac|masterAuth' -A2
```

Pass criteria:

- `privateClusterConfig.enablePrivateNodes: true` — **nodes have no external IPs**.
- `privateClusterConfig.enablePrivateEndpoint: true` OR `controlPlaneEndpointsConfig.dnsEndpointConfig.allowExternalTraffic: false`.
- `masterAuthorizedNetworksConfig.enabled: true` AND `cidrBlocks` is non-empty and contains no `0.0.0.0/0`.
- `legacyAbac.enabled` is absent / false.
- `masterAuth.clientCertificateConfig.issueClientCertificate` is absent / false.
- `masterAuth.username` / `password` are absent (legacy basic auth disabled).

```bash
# Per-node check: no external IPs (final defense if a node pool was created public).
for row in $(gcloud container clusters list --format='value(name,location)' | tr '\t' '|'); do
  CL="${row%|*}"; LOC="${row#*|}"
  gcloud compute instances list \
    --filter="metadata.items.key=cluster-name AND metadata.items.value=${CL}" \
    --format='table(name,networkInterfaces[0].accessConfigs[0].natIP:label=EXTERNAL_IP)'
done
# Pass: every EXTERNAL_IP column is empty.
```

### 4. Ingress / egress / load balancers / NAT and NodePort

The user-called-out section. Audit every path packets can take in or out of a node.

```bash
# 4a. Cloud NAT — egress should leave via NAT, not via node external IPs.
gcloud compute routers list --format='value(name,region,network)'
gcloud compute routers nats list --router=<router> --region=<region> \
  --format='value(name,natIpAllocateOption,sourceSubnetworkIpRangesToNat,enableEndpointIndependentMapping,logConfig.enable)'
# Pass: at least one NAT per region the cluster lives in; logging enabled
# (logConfig.enable=true); endpoint-independent mapping OFF unless required.

# 4b. NodePort services — should not exist in production namespaces.
kubectl get svc -A -o json | jq -r '
  .items[]
  | select(.spec.type=="NodePort" or .spec.type=="LoadBalancer")
  | "\(.metadata.namespace)/\(.metadata.name): type=\(.spec.type) extIP=\(.status.loadBalancer.ingress // []) ports=\([.spec.ports[].port] | tostring)"'
# FAIL on NodePort in app namespaces. LoadBalancer is OK only when matched
# against the firewall + LB checks below.

# 4c. LoadBalancer Services without source range restrictions.
kubectl get svc -A -o json | jq -r '
  .items[]
  | select(.spec.type=="LoadBalancer")
  | "\(.metadata.namespace)/\(.metadata.name): loadBalancerSourceRanges=\(.spec.loadBalancerSourceRanges // [])"'
# Pass: either internal LB (annotation networking.gke.io/load-balancer-type: Internal)
# OR loadBalancerSourceRanges is set to a known allow-list.
# FAIL: external LB with no source range filter, or 0.0.0.0/0.

# 4d. GKE-managed forwarding rules — what is actually internet-facing.
gcloud compute forwarding-rules list \
  --format='table(name,region,IPAddress,IPProtocol,target,loadBalancingScheme,ports)'
# Cross-check every EXTERNAL scheme against an expected ingress; investigate
# anything that maps to a backend service nobody owns.

# 4e. Firewall rules — must not allow 0.0.0.0/0 to node ports or to the
# NodePort range 30000-32767 except through the GKE-managed health-check
# rule and the LB rule that has a matching backend.
gcloud compute firewall-rules list --filter='disabled=false AND direction=INGRESS' \
  --format='table(name,sourceRanges.list():label=SRC,allowed[].ports.flatten():label=PORTS,targetTags.list():label=TARGETS)'
# FAIL on any rule with SRC=0.0.0.0/0 that opens 22, 3389, 30000-32767, etc.
# to node tags. The standard gke-<cluster>-<hash>-vms / -all rules are fine
# if they ONLY restrict to known LB ranges.

# 4f. Egress firewall — by default GCP has implicit allow-all egress.
# Confirm there is at least one EGRESS DENY rule constraining workloads
# that must not reach the public internet (Secret Manager / Artifact
# Registry should go via Private Google Access).
gcloud compute firewall-rules list --filter='direction=EGRESS' \
  --format='table(name,priority,destinationRanges.list():label=DST,denied[].IPProtocol:label=DENY)'

# 4g. Private Google Access on every node subnet — otherwise nodes need
# external IPs OR Cloud NAT to reach googleapis.com.
gcloud compute networks subnets list \
  --filter="name~^gke- OR purpose=PRIVATE_RFC_1918 OR purpose=PRIVATE" \
  --format='table(name,region,privateIpGoogleAccess)'
# Pass: privateIpGoogleAccess=True on every node subnet.

# 4h. Hostport pods — equivalent to NodePort, bypasses Services entirely.
kubectl get pods -A -o json | jq -r '
  .items[]
  | . as $p
  | (.spec.containers // [])[] | (.ports // [])[]
  | select(.hostPort != null)
  | "\($p.metadata.namespace)/\($p.metadata.name): hostPort=\(.hostPort) containerPort=\(.containerPort)"'
# FAIL on any hit outside system namespaces (kube-system, gmp-system, etc.).
```

### 5. Missing east-west segmentation

> Without NetworkPolicy, Pod-to-Pod traffic is effectively open. Risk: fast lateral movement after a single workload compromise. GCP example: GKE network policy enforcement is off, or namespaces have no default-deny policy. Fix: enable GKE network policy enforcement; apply namespace default-deny ingress and egress rules, then add explicit allows.

```bash
# 5a. NetworkPolicy enforcement on the cluster.
gcloud container clusters describe "$CL" --location="$LOC" \
  --format='value(networkPolicy.enabled,networkConfig.datapathProvider)'
# Pass: networkPolicy.enabled=True OR datapathProvider=ADVANCED_DATAPATH (Dataplane v2,
# which enforces policy natively without the addon).

# 5b. Every app namespace has at least one default-deny ingress + egress policy.
for NS in $(kubectl get ns -o name | cut -d/ -f2); do
  POLS=$(kubectl -n "$NS" get networkpolicy -o json | jq -r '.items | length')
  DENY=$(kubectl -n "$NS" get networkpolicy -o json | jq -r '
    .items[]
    | select(((.spec.podSelector // {}) | length) == 0)
    | select((.spec.policyTypes // []) | (index("Ingress") and index("Egress")))
    | .metadata.name')
  printf "%s policies=%s default_deny=%s\n" "$NS" "$POLS" "${DENY:-NONE}"
done
# FAIL on any app namespace where default_deny is NONE.
```

### 6. Missing Pod Security enforcement

> Pods run as root, privileged, with host namespaces, extra capabilities, or no seccomp. Risk: container escape or host compromise. GCP example: no PodSecurity labels and no runAsNonRoot, allowPrivilegeEscalation: false, or seccompProfile: RuntimeDefault.

```bash
# 6a. Pod Security admission labels on every namespace.
kubectl get ns -o json | jq -r '
  .items[]
  | "\(.metadata.name): enforce=\(.metadata.labels["pod-security.kubernetes.io/enforce"] // "MISSING") audit=\(.metadata.labels["pod-security.kubernetes.io/audit"] // "MISSING")"'
# Pass: every app namespace has enforce=restricted (or baseline + audit=restricted).

# 6b. Privileged / host-namespace pods, anywhere.
kubectl get pods -A -o json | jq -r '
  .items[]
  | . as $p
  | (.spec.containers + (.spec.initContainers // []))[]
  | select(
      ((.securityContext // {}).privileged == true)
      or ($p.spec.hostNetwork == true)
      or ($p.spec.hostPID == true)
      or ($p.spec.hostIPC == true)
      or (((.securityContext // {}).capabilities.add // []) | length > 0)
    )
  | "\($p.metadata.namespace)/\($p.metadata.name): privileged=\((.securityContext // {}).privileged) hostNet=\($p.spec.hostNetwork) caps=\((.securityContext // {}).capabilities.add // [])"'
# Investigate every hit. Whitelist legitimate ones (CNI, csi-driver, node-exporter)
# in the report; everything else is a FAIL.

# 6c. Pods missing runAsNonRoot, seccompProfile, or allowPrivilegeEscalation=false.
kubectl get pods -A -o json | jq -r '
  .items[]
  | . as $p
  | (.spec.containers // [])[]
  | select(
      ((.securityContext // {}).runAsNonRoot != true)
      or ((.securityContext // {}).allowPrivilegeEscalation == true)
      or ((($p.spec.securityContext // {}).seccompProfile // .securityContext.seccompProfile // {}).type != "RuntimeDefault")
    )
  | "\($p.metadata.namespace)/\($p.metadata.name)/\(.name)"' | sort -u | wc -l
# Quantify, then list top-10 namespaces by count.
```

### 7. Poor secrets handling

> Sensitive values live in Kubernetes Secrets or service-account keys. Risk: anyone who can read API objects or create Pods in a namespace may obtain them; Kubernetes Secrets are unencrypted in etcd unless configured. GCP example: database passwords or Google service-account keys stored as cluster Secrets. Fix: prefer Secret Manager with Workload Identity or the Secret Manager add-on; if Secrets must stay in-cluster, enable application-layer encryption with Cloud KMS.

```bash
# 7a. Application-layer secrets encryption (Cloud KMS) is enabled on the cluster.
gcloud container clusters describe "$CL" --location="$LOC" \
  --format='value(databaseEncryption.state,databaseEncryption.keyName)'
# Pass: state=ENCRYPTED, keyName points at a Cloud KMS key. FAIL on DECRYPTED.

# 7b. Secret Manager CSI / add-on present (preferred over raw Secrets).
gcloud container clusters describe "$CL" --location="$LOC" \
  --format='value(addonsConfig.gcpSecretManagerCsiDriverConfig.enabled,addonsConfig.gcsFuseCsiDriverConfig.enabled)'
# Pass: gcpSecretManagerCsiDriverConfig.enabled=True.

# 7c. Any cluster Secret that holds a GCP SA key (a long-lived JSON credential).
kubectl get secrets -A -o json | jq -r '
  .items[]
  | select(.data["key.json"] or .data["credentials.json"] or .data["service-account.json"])
  | "\(.metadata.namespace)/\(.metadata.name)"'
# FAIL on any hit — those should be Workload Identity Federation instead.

# 7d. Plaintext-looking keys in Secret data.
kubectl get secrets -A -o json | jq -r '
  .items[]
  | select(.type=="Opaque")
  | .data // {} | to_entries[]
  | select(.key | test("(?i)pass|secret|token|key|cred"))
  | .key' | sort | uniq -c | sort -rn | head
# Sanity check — every value should still be base64-encoded ciphertext when KMS
# encryption is on. Use this to spot field names that suggest poor hygiene.
```

### 8. Weak ingress and edge protection

> Public apps are exposed without hardened TLS, WAF, or access controls, or still rely on self-managed ingress-nginx. Risk: web attacks hit workloads directly; ingress-controller flaws can expose Secrets. GCP example: public Ingress with no Cloud Armor, HTTPS redirect, SSL policy, or IAP. Fix: use GKE Gateway or Ingress with Cloud Armor, HTTPS redirect, SSL policy, and IAP where appropriate; migrate off ingress-nginx.

```bash
# 8a. Every public-facing Ingress / Gateway is backed by a BackendConfig
#     that references a Cloud Armor policy.
kubectl get ingress,gateway -A -o json | jq -r '
  .items[]
  | "\(.kind) \(.metadata.namespace)/\(.metadata.name): class=\(.spec.ingressClassName // .spec.gatewayClassName // .metadata.annotations["kubernetes.io/ingress.class"])"'
kubectl get backendconfig -A -o json | jq -r '
  .items[]
  | "\(.metadata.namespace)/\(.metadata.name): cloudArmor=\(.spec.securityPolicy.name // "NONE") iap=\(.spec.iap.enabled // false)"'
# FAIL on any externally-routed Ingress whose BackendConfig has cloudArmor=NONE.

# 8b. Cloud Armor policies exist and have at least the OWASP preconfigured rules
#     + a default-deny + a rate-limit rule.
gcloud compute security-policies list --format='value(name,type)'
for P in $(gcloud compute security-policies list --format='value(name)'); do
  echo "=== $P ==="
  gcloud compute security-policies describe "$P" \
    --format='table(rules.priority,rules.action,rules.match.expr.expression,rules.description)'
done
# Pass: every policy has a rate-limit rule and at least one OWASP CRS rule
# (e.g. evaluatePreconfiguredExpr('sqli-stable')).

# 8c. HTTPS redirect + SSL policy on the LB.
gcloud compute target-https-proxies list \
  --format='table(name,sslPolicy,urlMap)'
gcloud compute ssl-policies list \
  --format='table(name,minTlsVersion,profile)'
# Pass: every public LB has an SSL policy attached, minTlsVersion=TLS_1_2 or higher,
# profile=MODERN or RESTRICTED. FAIL on COMPATIBLE / no policy.

# 8d. IAP on internal admin UIs (Grafana, Prometheus, anything that's a UI
#     rather than an API).
gcloud iap web list --format=json | jq -r '.[] | "\(.name) \(.iapSettings.applicationSettings.csmSettings.rctokenAud // "-")"'

# 8e. No self-managed ingress-nginx in production namespaces.
kubectl get deploy,ds -A -l app.kubernetes.io/name=ingress-nginx 2>/dev/null
kubectl get pods -A -l app.kubernetes.io/name=ingress-nginx 2>/dev/null
# FAIL on any hit in app namespaces. Allowed: gke-ingress, networking.gke.io
# managed controllers.
```

### 9. Weak image and deployment supply chain

> Teams deploy mutable tags from unscanned repositories and leave admission policy permissive. Risk: the image you scanned might not be the image you ran. GCP example: deploying :latest, Artifact Analysis scanning not enabled on the repo, and Binary Authorization left at ALLOW_ALL. Fix: pin images by digest; enable Artifact Analysis scanning and require Binary Authorization attestations or signatures in production.

```bash
# 9a. Binary Authorization is enabled on the cluster AND has a non-default policy.
gcloud container clusters describe "$CL" --location="$LOC" \
  --format='value(binaryAuthorization.evaluationMode,binaryAuthorization.enabled)'
# Pass: evaluationMode=PROJECT_SINGLETON_POLICY_ENFORCE (or POLICY_BINDINGS for
# multi-cluster). FAIL on DISABLED or unset.

gcloud container binauthz policy export \
  | grep -E 'defaultAdmissionRule|evaluationMode|enforcementMode|requireAttestationsBy'
# Pass: defaultAdmissionRule.evaluationMode=REQUIRE_ATTESTATION,
# enforcementMode=ENFORCED_BLOCK_AND_AUDIT_LOG, requireAttestationsBy is non-empty.
# FAIL on ALWAYS_ALLOW.

# 9b. Artifact Analysis vulnerability scanning is on for every repo we use.
gcloud artifacts repositories list \
  --format='value(name,format,mode)'
gcloud services list --enabled --filter='config.name=containeranalysis.googleapis.com OR config.name=containerscanning.googleapis.com' \
  --format='value(config.name)'
# Pass: containerscanning.googleapis.com is enabled.

# 9c. No mutable tags in production. Every running pod's image must be a
#     @sha256:... digest reference.
kubectl get pods -A -o json | jq -r '
  .items[]
  | .spec.containers[]
  | select((.image | contains("@sha256:")) | not)
  | "\(.image)"' | sort -u | head -50
# FAIL on :latest, :main, :master, untagged. Tagged-by-version (e.g. :v1.2.3) is
# acceptable only if the policy requires a matching attestation; otherwise also FAIL.
```

### 10. Patch and node-hardening lag

> Clusters drift behind current patch levels or Standard nodes disable hardening features. Risk: published GKE bulletins continue to include privilege-escalation fixes for node and kernel components. GCP example: long maintenance exclusions, no release channel, or Shielded GKE Nodes and secure boot turned off.

```bash
# 10a. Release channel + version skew.
gcloud container clusters describe "$CL" --location="$LOC" \
  --format='value(releaseChannel.channel,currentMasterVersion,currentNodeVersion)'
# Pass: releaseChannel in {RAPID, REGULAR, STABLE}. FAIL on UNSPECIFIED / static.
# Surface clusters where currentNodeVersion != currentMasterVersion (node-pool drift).

# 10b. Maintenance exclusions that are still active and longer than 30 days.
gcloud container clusters describe "$CL" --location="$LOC" \
  --format='value(maintenancePolicy.window.maintenanceExclusions)'
# FAIL on any exclusion window > 30 days that's still in effect.

# 10c. Shielded GKE Nodes + Secure Boot + Integrity Monitoring on every node pool.
gcloud container node-pools list --cluster="$CL" --location="$LOC" \
  --format='value(name,config.shieldedInstanceConfig.enableSecureBoot,config.shieldedInstanceConfig.enableIntegrityMonitoring)'
# Pass: every row shows True / True. FAIL on any False.

gcloud container clusters describe "$CL" --location="$LOC" \
  --format='value(shieldedNodes.enabled)'
# Pass: True.

# 10d. Auto-upgrade + auto-repair on every node pool.
gcloud container node-pools list --cluster="$CL" --location="$LOC" \
  --format='value(name,management.autoUpgrade,management.autoRepair)'
# Pass: every row True / True.
```

---

## Workflow

1. **Enumerate.** Cluster list from the `gcp-fleet` skill in the infra repo; namespaces from the `accelerate` chart's releases.
2. **Run sections 1-10 per cluster** in parallel where possible. Sections 4b/4h/5b/6a-c/8a iterate over the router's namespaces; the rest are per-cluster.
3. **Write the report** using the [template](#report-template). Right-most column = worst per-cluster verdict. Findings sorted by severity.
4. **Surface, don't fix.** Every FAIL is a report line, never a `kubectl edit`.

---

## Report template

```markdown
# GCP / GKE security audit

Run: <UTC timestamp>
Project: <project>
Clusters audited: <list>
Namespaces audited: <count>

## Per-cluster matrix

| Cluster | Region | 1 WI | 2 RBAC | 3 Expose | 4 Net | 5 NetPol | 6 PodSec | 7 Secrets | 8 Ingress | 9 Supply | 10 Patch | Worst |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| <cluster-a> | <region> | OK | OK | OK | FAIL | OK | OK | OK | OK | OK | OK | **FAIL** |
| <cluster-b> | <region> | OK | OK | OK | OK | OK | WARN | OK | OK | OK | OK | WARN |

Legend: OK = pass criteria met; WARN = minor / known exception; FAIL = must-fix; BLOCKED = could not evaluate.

## Findings (sorted by severity)

### CRITICAL

1. **<cluster-a> — LoadBalancer Service `<namespace>/router-public` is internet-facing with no `loadBalancerSourceRanges`** (mistake #4).
   - Forwarding rule `a1234...` scheme EXTERNAL, no Cloud Armor policy on backend (mistake #8 too).
   - Fix: add `loadBalancerSourceRanges`, attach a BackendConfig with `securityPolicy.name` pointing at the Cloud Armor policy for the router.

### HIGH

2. **<cluster-b> — 12 pods in `<namespace>` run without `seccompProfile: RuntimeDefault`** (mistake #6).
   - Top offenders: router (8), router fetch initContainer (4).
   - Fix: set `securityContext.seccompProfile.type: RuntimeDefault` in the `accelerate` chart's values.

### MEDIUM

3. **<cluster-a> — `releaseChannel` is UNSPECIFIED** (mistake #10).
   - Fix: enroll cluster in REGULAR channel, enable auto-upgrade on every node pool.

### LOW / INFO

4. **All clusters — Binary Authorization policy `defaultAdmissionRule.enforcementMode=DRYRUN`** (mistake #9).
   - Reduces guarantee to audit-log only. Flip to `ENFORCED_BLOCK_AND_AUDIT_LOG` once the deny-rate drops below 0.1%.
```

---

## Common pitfalls

- **Cluster context drift.** Every `kubectl` call must be preceded by `gcloud container clusters get-credentials "$CL" --location="$LOC"`. Mixing contexts produces ghost findings.
- **Whitelist legitimate hostPath / hostNetwork / privileged pods.** CNI, csi-driver, node-problem-detector, gke-metadata-server, gke-metrics-agent, calico-node, falco etc. all need it. Hard-code the whitelist in the report so the same legitimate pods don't show up as findings every run.
- **`automountServiceAccountToken: false` isn't always right.** Anything that calls the K8s API (operators, controllers, kubectl-as-a-pod) needs it true. Report the count, then list the outliers — don't blanket-flag.
- **NodePort vs LoadBalancer.** A NodePort Service is still reachable on every node's external IP unless private nodes is on AND there's no LB in front. Both are FAILs in different ways — keep them separate in the report.
- **Cloud Armor without rate-limiting.** A policy with only OWASP rules and no rate-limit rule will not stop credential-stuffing. Treat "no `rate-based-ban` action" as WARN.
- **Binary Authorization "enabled" != "enforcing".** `binaryAuthorization.enabled=true` with `evaluationMode=PROJECT_SINGLETON_POLICY_ENFORCE` and `defaultAdmissionRule.evaluationMode=ALWAYS_ALLOW` is the most common silent-failure mode. Always cross-check the policy itself, not just the cluster flag.
- **The router fetches its binary at start.** The image is fixed and a fetch initContainer downloads `router-linux-<arch>` from a version registry (see the `deploy` skill), so a pinned image digest (mistake #9) does not pin the code that runs. Check the initContainer verifies the binary's checksum against the registry, fetches over HTTPS only, and that the registry is writable only by the release workflow. Report it under #9.
- **The router needs egress to model providers.** A default-deny egress policy (mistake #5) has to allow it out to the internet over 443; check it allows that and nothing wider (no egress to the metadata server, other namespaces, or arbitrary ports).
- **Workload Identity Federation vs Workload Identity for GKE.** Same name pool (`<project>.svc.id.goog`), different binding flows. The audit only requires the pool exists + GKE_METADATA on node pools + the iam.gke.io/gcp-service-account annotation on KSAs.

---

## See also

- The `deploy` skill here — how the router reaches the clusters, and where the runbook is.
- In the infra repo: `gcp-fleet` (cluster list, auth source of truth), `gcp-security-audit` (the original of this file), `infra/charts/accelerate/values.yaml` (the router's `podSecurityContext`, `securityContext`, BackendConfig, NetworkPolicy).
