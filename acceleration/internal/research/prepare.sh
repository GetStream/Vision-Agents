#!/bin/sh
set -eu
find /opt/repositories -type l -delete
find /opt/repositories -mindepth 2 -type d -name '.*' -prune -exec rm -rf '{}' +
find /opt/repositories -type f -name '.*' -delete
find /opt/repositories -type f \( -name AGENTS.md -o -name CLAUDE.md -o -name SKILL.md \) -delete
mkdir -p /opt/repositories/.cursor
printf '%s' '{"permissions":{"allow":["Read(/opt/repositories/**)"],"deny":["Write(**)","Shell(*)","WebFetch(*)","Mcp(*:*)","Read(/home/**)","Read(/proc/**)","Read(/opt/support/**)"]}}' > /opt/repositories/.cursor/cli.json
chown -R root:root /opt/repositories /opt/support /opt/cursor
chmod -R a+rX,go-w,u-w /opt/repositories /opt/cursor
chmod 600 /opt/support/cursor-key /opt/support/worker-token
setpriv --reuid=10001 --regid=10001 --init-groups --no-new-privs sh -c 'test ! -w /opt/repositories && test ! -w /opt/repositories/.cursor/cli.json && test ! -r /opt/support/cursor-key'
