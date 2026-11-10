# Cypress CyUSB — Linux x86-64

Minimal Linux USB build dependency from [Infineon FX3 SDK 1.3.5](https://softwaretools.infineon.com/tools/com.ifx.tb.tool.ezusbfx3sdk): the `cyusb.h` header, prebuilt x86-64 `libcyusb.so`, its original `libcyusb.so.1` compatibility link, and the corresponding `libcyusb.cpp` source, unchanged from the SDK.
[SHA256SUMS](SHA256SUMS) records the vendor file checksums.

Copyright Cypress Semiconductors Inc. / ATR-LABS. All rights reserved. The vendor files use the original [LGPL 2.1 license](license/license.txt), not this project's Apache license.

Actions uses this directory directly. Linux wheels retain the runtime library and license; source distributions retain the corresponding source. USB execution still requires the system `libusb-1.0` runtime, device permissions, and the board's FX3 firmware. The SDK installer, GUI, udev/configuration files, firmware images, examples, and `cyusb_linux` utility are excluded.
