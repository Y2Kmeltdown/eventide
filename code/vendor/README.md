# Vendored dashboard dependencies

Third-party JS/CSS the dashboard (`dashboard.html`) loads, vendored here
instead of pulled from a CDN at page-load time — so the dashboard has no
internet dependency once these files are in place. Served at `/vendor/...`
by both `eventide.py` and `frontend.py` (see each script's `vendor_file`
route and `--vendor-dir` flag).

| Directory   | Library                                    | Pinned version |
| ----------- | ------------------------------------------- | -------------- |
| `leaflet/`  | [Leaflet](https://leafletjs.com/)           | 1.9.4          |
| `gridstack/`| [GridStack](https://gridstackjs.com/)       | 13.0.2         |
| `litegraph/`| [litegraph.js](https://github.com/jagenjo/litegraph.js) | 0.7.18 |

`leaflet/images/` holds the three PNGs `leaflet.css` references
(`layers.png`, `layers-2x.png`, `marker-icon.png`) — the dashboard only
ever uses custom `L.divIcon` markers and no layers control, so none of
these actually render today, but they're vendored anyway so nothing 404s
if that ever changes. `gridstack.min.css` and `litegraph.css` have no
external asset references (only inline data-URIs), so nothing else to
vendor for those two.

## Updating a version

1. Download the new version's files from unpkg (or wherever upstream
   publishes it) into the matching directory here, overwriting the old
   ones — e.g. for litegraph.js `X.Y.Z`:
   ```
   curl -o litegraph/litegraph.min.js https://unpkg.com/litegraph.js@X.Y.Z/build/litegraph.min.js
   curl -o litegraph/litegraph.css    https://unpkg.com/litegraph.js@X.Y.Z/css/litegraph.css
   ```
2. Update the version in the table above.
3. Sanity-check: `node --check <file>.js` for JS, and skim the CSS for any
   new `url(...)` asset references that would need vendoring too (see
   `leaflet/images/` above for how that's handled).
