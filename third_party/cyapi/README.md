# Cypress CyAPI — Windows x64

Minimal Windows USB build dependency from
[Infineon FX3 SDK 1.3.5](https://softwaretools.infineon.com/tools/com.ifx.tb.tool.ezusbfx3sdk):
two headers and the prebuilt AMD64 `CyAPI.lib`, unchanged from the SDK.
[SHA256SUMS](SHA256SUMS) records the vendor file checksums.

Copyright Cypress Semiconductor Corporation, 2009–2018. All rights reserved.
The vendor files use the original [component license](license/license.rtf) and
[SDK license](license/license.txt), not this project's Apache license.

Actions uses this directory directly. Windows wheels retain both licenses and
this notice. USB execution requires the CyUSB3 device driver and MSVC x64 runtime.
