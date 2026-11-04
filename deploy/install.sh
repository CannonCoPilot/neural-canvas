#!/bin/bash
# Install / remove a neural-canvas launchd user agent.
#   deploy/install.sh [service|imify]            install (or reinstall) and start (default: service)
#   deploy/install.sh [service|imify] --remove   stop and uninstall
#     service  the watch-folder runner (com.aion.neural-canvas)
#     imify    the local reverse-image search API on 127.0.0.1:8890 (com.aion.imify)
case "$1" in
    imify) LABEL=com.aion.imify; shift ;;
    service) LABEL=com.aion.neural-canvas; shift ;;
    *) LABEL=com.aion.neural-canvas ;;
esac
SRC="$(cd "$(dirname "$0")" && pwd)/$LABEL.plist"
DST="$HOME/Library/LaunchAgents/$LABEL.plist"
DOMAIN="gui/$(id -u)"
launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null
if [ "$1" = "--remove" ]; then
    rm -f "$DST"
    echo "removed $LABEL"
    exit 0
fi
mkdir -p "$(dirname "$SRC")/../var/logs"
cp "$SRC" "$DST"
launchctl bootstrap "$DOMAIN" "$DST" && echo "installed $LABEL"
launchctl print "$DOMAIN/$LABEL" 2>/dev/null | grep -E "state =|pid =" | head -2
