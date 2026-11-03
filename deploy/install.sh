#!/bin/bash
# Install / remove the neural-canvas watch-folder service (launchd user agent).
#   deploy/install.sh            install (or reinstall) and start
#   deploy/install.sh --remove   stop and uninstall
LABEL=com.aion.neural-canvas
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
launchctl bootstrap "$DOMAIN" "$DST" && echo "installed $LABEL — watching $(dirname "$SRC")/../var/inbox"
launchctl print "$DOMAIN/$LABEL" 2>/dev/null | grep -E "state =|pid =" | head -2
