#!/bin/sh

# Wait for Grafana to be ready
echo "Waiting for Grafana to be ready..."
until curl -s http://grafana:3000/api/health > /dev/null 2>&1; do
    echo "Grafana not ready yet, waiting..."
    sleep 2
done

echo "Grafana is ready!"
sleep 5  # Give it a bit more time for provisioning to complete

# Get the dashboard UID
DASHBOARD_UID="stream-agents-metrics"

# Set the home dashboard for the organization
echo "Setting org home dashboard to Stream Agents - Performance Metrics..."
curl -X PUT \
  -H "Content-Type: application/json" \
  -d "{\"homeDashboardUID\":\"${DASHBOARD_UID}\"}" \
  http://grafana:3000/api/org/preferences

# Also set it as the default home dashboard for admin user (for when they log in)
echo ""
echo "Setting admin user home dashboard..."
curl -X PUT \
  -u "admin:admin" \
  -H "Content-Type: application/json" \
  -d "{\"homeDashboardUID\":\"${DASHBOARD_UID}\"}" \
  http://grafana:3000/api/user/preferences

echo ""
echo "Home dashboard configured successfully!"
