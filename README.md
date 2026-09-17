# Eventide

Eventide turns a Single Board Computer into a self-contained camera
recording payload: a Flask backend manages cameras, gimbals, and other
hardware as installable **modules**, a web dashboard gives you a live,
customisable control panel and a recordings browser, and everything runs
under supervisord so it survives reboots and crashes unattended.

## Modules

The base platform is intentionally minimal — cameras, gimbal control, and
every other hardware component are **modules**, installed from GitHub
repositories (or a zip upload) via the dashboard's **MODULES** tab. Each
module declares what it needs and how it runs in an `eventide-module.json`
manifest; the backend clones it, builds it, and runs it as one or more
supervisord programs, all without touching the base system or any other
installed module. Modules can be installed, edited, and uninstalled entirely
from the dashboard — no SSH required for day-to-day use.

- Full module system documentation, including the manifest format, the
  dashboard `ui` widget types, and the backend API: [docs/MODULES.md](docs/MODULES.md)
- Template for writing your own module: [module-template/](module-template/)
- Upgrading from a pre-module install: see "Migrating from a pre-module
  install" in the docs

## Installation

Run the following on a clean Raspberry Pi OS Lite (64-bit) or Armbian/Orange
Pi install:

```bash
sudo apt update && sudo apt install -y git && git clone https://github.com/Y2Kmeltdown/eventide.git && cd eventide && sudo chmod +x install.sh && ./install.sh
```

This installs the base platform only: OS configuration, the dashboard
backend, supervisord, nginx, the watchdog/RTC services, the playback server,
and the Rust toolchain (so Rust modules can build on-device). It aborts with
a clear error (logged to `/tmp/eventide-install.log`) if any step fails, and
reboots automatically once it finishes. Cameras, gimbal control, and other
hardware are installed afterwards as modules from the dashboard — see
[Modules](#modules) above.

By default recordings are written to `~/recordings` on the device's main
storage, and the timezone is set to `Australia/Sydney`. The recording
directory can also be overridden with a positional argument
(`./install.sh /path/to/dir`), and can be changed later at any time from the
dashboard's SETTINGS tab.

A few extra install steps are optional and off by default, controlled by
environment variables set before running the script:

- **Recordings directory / SD card** — pass a directory as the first
  argument, or set `RECORDINGS_SD_LABEL` to have `install.sh` set up a
  dedicated recordings SD card instead, mounted by filesystem label (not
  `/dev/mmcblkN`, which isn't stable across reboots or reader swaps). Format
  and label the card yourself first (`sudo mkfs.ext4 -L mylabel /dev/...`),
  then run:

  ```bash
  RECORDINGS_SD_LABEL=mylabel ./install.sh
  ```

  This writes a `nofail` systemd `.mount` unit so a missing/removed card
  never blocks boot. It only sets up the mount — you still need to point the
  dashboard's SETTINGS tab at the mountpoint (`/media/eventide` by default,
  override with `RECORDINGS_SD_MOUNTPOINT`) to actually record there.
- **Touchscreen kiosk display** — set `SETUP_KIOSK_DISPLAY=1` to configure
  the device to boot straight into a full-screen Chromium kiosk (either the
  full dashboard or the touchscreen-optimised `/kiosk` UI) instead of a login
  prompt, via `config/setup-display.sh`. Override any of its display/touch
  settings (`KIOSK_URL`, `HDMI_OUTPUT`, `ROTATION`, `SCALE_FACTOR`,
  `WINDOW_SIZE`, `TOUCH_DEVICE`, `CHROMIUM_BIN`, `FORCE_MODELINE`,
  `FORCE_KMSDEV`, `KIOSK_WAIT_SECS`) the same way, e.g.:

  ```bash
  SETUP_KIOSK_DISPLAY=1 KIOSK_URL=http://localhost/kiosk TOUCH_DEVICE="wch.cn USB2IIC_CTP_CONTROL" ./install.sh
  ```

  Leave it unset for a headless install or one only ever accessed remotely —
  `config/setup-display.sh` can always be run on its own later. Before
  starting X, the kiosk waits up to `KIOSK_WAIT_SECS` (default 60, `0` to
  disable) for `KIOSK_URL` to actually respond, so a slow-starting backend
  (a slow DHCP source, a large module registry, ...) shows as a plain wait
  message on the console instead of a broken page in Chromium. The
  installer also disables `NetworkManager`/`systemd-networkd`'s
  wait-online units, since `eventide.service` only needs loopback and
  otherwise ends up waiting on DHCP for no reason.
- **Timezone** — set `INSTALL_TIMEZONE` (an IANA name, e.g.
  `America/New_York`) to override the `Australia/Sydney` default. Like the
  recordings directory, this can also be changed later from the dashboard
  without re-running the installer.
- **USB-C host mode (Raspberry Pi 5 only)** — set `ENABLE_USBC_HOST=1` to
  configure the Pi 5's USB-C port as a functioning USB 2.0 host port
  (`dtoverlay=dwc2,dr_mode=host`), alongside its normal role as power input —
  the two coexist since power negotiation goes through a separate PMIC, not
  the dwc2 data role. Off by default.

## The modular control panel

The dashboard's **MAIN** tab is a customisable workspace rather than a fixed
layout: a left sidebar, a right sidebar, and a tabbed centre area that you
build up out of **components** — live camera feeds, control forms,
telemetry readouts, maps, and more — contributed by whatever modules are
installed. Open the **＋ COMPONENTS** palette to see every component every
installed module offers, add or remove them, and drag them between regions;
the layout is saved per backend host so it persists across reloads. A module
can mark a component `default: true` so it appears automatically as soon as
the module is installed, but nothing is fixed — the same evk-datalogger
install might show a live feed and bias controls on one payload and be
hidden entirely on another.

Because the panel is driven entirely by module manifests, a new camera or
sensor module lights up its own live view and controls the moment it's
installed, with no dashboard code changes required. See "Dashboard UI
components (`ui`)" in [docs/MODULES.md](docs/MODULES.md#dashboard-ui-components-ui)
for the full list of widget types (`mjpeg`, `form`, `telemetry`, `features`,
`recording`, `joystick`, `table`, `map`, `orientation3d`, and the built-in
`master-record`/`schedule-table` components) if you're building a module of
your own.

## Playback

The **PLAYBACK** tab is the recordings browser. It gets one inner tab per
**recording source** — automatically, for every installed module that
declares a `recordings_subdir` (and one more per copy, for modules with
copyable per-instance programs, e.g. multiple serial devices) — with no
manual configuration. Each tab lists that source's recorded files, showing
size and file type, with per-file controls to favourite, download, or
permanently delete a recording.

Selecting a playable recording (video/raw formats the base platform's
playback server can decode) streams it back as MJPEG through that same
server, with a playback speed control and live-adjustable stream quality,
resolution, and frame rate — the same playback pipeline used for live camera
feeds elsewhere in the dashboard, just pointed at a file instead of a
camera. Favourited recordings are flagged across the dashboard (including
being exempt from the optional automatic retention policy in SETTINGS) so
you can protect specific clips from cleanup or an SD card swap.
