#!/bin/bash
# Mount / unmount an SD card partition at the Eventide recordings mountpoint.
#
# Not run by hand: config/99-eventide-sd.rules (udev) selects SD cards by their
# kernel "mmc" type — never USB storage, never eMMC — and starts
# eventide-sd-mount@<partition>.service, which calls this with mount on
# insertion and umount when that service stops (i.e. when the card goes away).
#
# Usage: eventide-sd-mount mount|umount <kernel name, e.g. mmcblk1p1>
# install.sh installs this to /usr/local/sbin/eventide-sd-mount.

set -u

MOUNTPOINT="${EVENTIDE_SD_MOUNTPOINT:-/media/eventide}"
action="${1:-}"
name="${2:-}"
dev="/dev/$name"

log() { echo "[eventide-sd] $*"; }

[ -b "$dev" ] || { log "$dev is not a block device"; exit 1; }

case "$action" in
mount)
    # Never take over the disk the OS is running from: on a board that boots
    # from its SD slot, that card is an SD card too and matches the udev rule.
    parent=$(lsblk -no PKNAME "$dev" 2>/dev/null | head -n1)
    disk="/dev/${parent:-$name}"
    if lsblk -nro MOUNTPOINT "$disk" 2>/dev/null | grep -Eq '^(/|/boot(/.*)?)$'; then
        log "$dev is on the boot/root disk — not mounting it"
        exit 0
    fi

    # Single-card design: one fixed mountpoint, so a second card is left alone.
    if findmnt -n "$MOUNTPOINT" > /dev/null 2>&1; then
        log "$MOUNTPOINT is already mounted — leaving $dev unmounted"
        exit 0
    fi

    fstype=$(blkid -o value -s TYPE "$dev" 2>/dev/null)
    if [ -z "$fstype" ]; then
        log "no recognisable filesystem on $dev — not mounting it"
        exit 0
    fi

    opts="noatime"
    case "$fstype" in
        # No unix ownership on these: make files belong to whoever owns the
        # eventide install so non-root users (e.g. the kiosk user) can use them.
        vfat|exfat|ntfs|ntfs3)
            owner=$(stat -c '%u,%g' /usr/local/eventide 2>/dev/null || echo "0,0")
            opts="$opts,uid=${owner%,*},gid=${owner#*,},umask=002"
            ;;
    esac

    mkdir -p "$MOUNTPOINT"
    if mount -o "$opts" "$dev" "$MOUNTPOINT"; then
        log "mounted $dev ($fstype) at $MOUNTPOINT"
    else
        log "mounting $dev at $MOUNTPOINT failed"
        exit 1
    fi
    ;;

umount)
    # Only unmount our own card — never something else that's since been
    # mounted at the same mountpoint.
    [ "$(findmnt -n -o SOURCE "$MOUNTPOINT" 2>/dev/null)" = "$dev" ] || exit 0
    # Try a normal unmount first so an orderly stop (shutdown) flushes writes;
    # fall back to lazy for a pulled card or a recorder still holding files open.
    umount "$MOUNTPOINT" 2> /dev/null || umount -l "$MOUNTPOINT"
    log "unmounted $MOUNTPOINT"
    ;;

*)
    echo "usage: $0 mount|umount <partition, e.g. mmcblk1p1>" >&2
    exit 2
    ;;
esac
