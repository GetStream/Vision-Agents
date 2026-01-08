#!/bin/bash
set -e

# Add nebius to PATH
export PATH="$HOME/.nebius/bin:$PATH"

# Configuration
PROJECT_ID="project-e01zw2jzpr000vckjm7t7n"
SUBNET_ID="vpcsubnet-e01jfyqqs0hfzpp2c3"
CLUSTER_NAME="vision-agent-cluster"
REGISTRY_NAME="vision-agent-registry"
IMAGE_NAME="vision-agent-deploy"

echo "=== Nebius Vision Agent Deployment ==="

# Check if registry exists, create if not
echo "Checking registry..."
REGISTRY=$(nebius registry list --format json 2>/dev/null | jq -r '.items[]? | select(.metadata.name=="'$REGISTRY_NAME'") | .metadata.id' 2>/dev/null || echo "")

if [ -z "$REGISTRY" ]; then
    echo "Creating registry..."
    nebius registry create \
        --parent-id "$PROJECT_ID" \
        --name "$REGISTRY_NAME"
    sleep 5
    REGISTRY=$(nebius registry list --format json | jq -r '.items[] | select(.metadata.name=="'$REGISTRY_NAME'") | .metadata.id')
fi
echo "Registry ID: $REGISTRY"

# Get registry endpoint (strip "registry-" prefix for the path)
REGISTRY_FQDN=$(nebius registry get --id "$REGISTRY" --format json | jq -r '.status.registry_fqdn // empty')
REGISTRY_ID_SHORT="${REGISTRY#registry-}"
REGISTRY_ENDPOINT="$REGISTRY_FQDN/$REGISTRY_ID_SHORT"
echo "Registry endpoint: $REGISTRY_ENDPOINT"

# Check if cluster exists, create if not
echo "Checking k8s cluster..."
CLUSTER=$(nebius mk8s cluster list --format json 2>/dev/null | jq -r '.items[]? | select(.metadata.name=="'$CLUSTER_NAME'") | .metadata.id' 2>/dev/null || echo "")

if [ -z "$CLUSTER" ]; then
    echo "Creating k8s cluster (this may take 5-10 minutes)..."
    nebius mk8s cluster create \
        --parent-id "$PROJECT_ID" \
        --name "$CLUSTER_NAME" \
        --control-plane-subnet-id "$SUBNET_ID" \
        --control-plane-version "1.31" \
        --control-plane-endpoints-public-endpoint
    
    echo "Waiting for cluster to be ready..."
    while true; do
        STATE=$(nebius mk8s cluster list --format json | jq -r '.items[] | select(.metadata.name=="'$CLUSTER_NAME'") | .status.state' 2>/dev/null || echo "")
        echo "Cluster state: $STATE"
        if [ "$STATE" = "RUNNING" ]; then
            break
        fi
        sleep 30
    done
    CLUSTER=$(nebius mk8s cluster list --format json | jq -r '.items[] | select(.metadata.name=="'$CLUSTER_NAME'") | .metadata.id')
fi
echo "Cluster ID: $CLUSTER"

# Create node group if needed
echo "Checking node group..."
NODEGROUP=$(nebius mk8s node-group list --parent-id "$CLUSTER" --format json 2>/dev/null | jq -r '.items[0]?.metadata.id' 2>/dev/null || echo "")

if [ -z "$NODEGROUP" ] || [ "$NODEGROUP" = "null" ]; then
    echo "Creating node group..."
    nebius mk8s node-group create \
        --parent-id "$CLUSTER" \
        --name "default-nodes" \
        --fixed-node-count 1 \
        --template-resources-platform "cpu-d3" \
        --template-resources-preset "4vcpu-16gb" \
        --template-boot-disk-type "network_ssd" \
        --template-boot-disk-size-gibibytes 64
    
    echo "Waiting for node group to be ready..."
    while true; do
        NG_STATE=$(nebius mk8s node-group list --parent-id "$CLUSTER" --format json | jq -r '.items[0]?.status.state // empty' 2>/dev/null || echo "")
        echo "Node group state: $NG_STATE"
        if [ "$NG_STATE" = "RUNNING" ]; then
            break
        fi
        sleep 30
    done
fi

# Get kubeconfig
echo "Getting kubeconfig..."
nebius mk8s cluster get-credentials --id "$CLUSTER" --external

# Build and push image
echo "Building Docker image..."
cd "$(dirname "$0")/../.."
docker build -t "$IMAGE_NAME" -f examples/05_deploy_example/Dockerfile .

if [ -n "$REGISTRY_FQDN" ]; then
    echo "Configuring Docker credentials..."
    nebius registry configure-helper
    
    echo "Tagging and pushing to registry..."
    FULL_IMAGE="$REGISTRY_ENDPOINT/$IMAGE_NAME:latest"
    docker tag "$IMAGE_NAME" "$FULL_IMAGE"
    docker push "$FULL_IMAGE"
    
    # Deploy with Helm
    echo "Deploying with Helm..."
    cd examples/05_deploy_example
    helm upgrade --install vision-agent ./helm \
        --set image.repository="$REGISTRY_ENDPOINT/$IMAGE_NAME" \
        --set image.tag=latest \
        --set image.pullPolicy=Always
else
    echo "Warning: No registry endpoint available. Skipping image push."
    echo "Deploy manually after registry is ready."
fi

echo ""
echo "=== Deployment Complete ==="
echo "Check status with: kubectl get pods"

