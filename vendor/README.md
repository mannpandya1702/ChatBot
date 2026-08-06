# vendor

Third-party binaries that cannot be installed from PyPI.

## LibreHardwareMonitorLib.dll

CPU temperatures and fan speeds, read by the elevated helper (`src/jarvis/helper/lhm.py`).
These are the two readings a non-elevated process genuinely cannot get on Windows, which is
why the helper exists at all.

| | |
|---|---|
| Version | 0.9.4 |
| Source | NuGet package `LibreHardwareMonitorLib`, `lib/net472/` |
| Licence | MPL-2.0 (§1 of `CLAUDE.md` permits it) |
| SHA-256 | `a0f2728f1734c236a9d02d9e25a88bc4f8cb7bd1faff1770726beb7af06bf8dc` |
| Upstream | https://github.com/LibreHardwareMonitor/LibreHardwareMonitor |

Re-fetch or upgrade with `scripts/fetch_vendor.ps1`, which verifies the hash after download.

### Why net472 and not netstandard2.0

The helper loads the assembly through pythonnet, which on a stock Windows 11 box resolves
against .NET Framework 4.8. The `net472` build is the one that path has been exercised with.
If you switch pythonnet to the CoreCLR runtime, take `lib/net8.0/` from the same package
instead and update the hash above.

### The HidSharp dependency

The package declares a dependency on HidSharp, used only for fan controller hardware.
`lhm.py` leaves `IsControllerEnabled` at its default of false, so that code path is never
entered and HidSharp is not vendored. If controller sensors are ever enabled, add
`HidSharp.dll` (Apache-2.0) beside this file: `Assembly.LoadFrom` probes the directory of the
assembly that requested it, so no loader changes are needed.

### Licence obligations

MPL-2.0 is file-level copyleft. The DLL is loaded at runtime through pythonnet and is never
linked into or modified by this project, so no JARVIS source falls under it. If you modify
LibreHardwareMonitor itself, you must publish those changes under MPL-2.0.
