"""Parse the USB fields shared by local Web execution tasks."""

from cim22.runtime import CIMTimingConfig, UsbConfig


def parse_usb_config(fields: object) -> UsbConfig:
    """Validate JSON transport/timing fields without opening the board."""
    if not isinstance(fields, dict):
        raise ValueError("usb_config 必须是 JSON 对象")
    values = dict(fields)
    timing = values.get("timing")
    if timing is not None and not isinstance(timing, dict):
        raise ValueError("usb_config.timing 必须是 JSON 对象")
    # Web forms use the flat pll_port/clock_frequency_mhz interface.
    if values.get("pll") is not None:
        raise ValueError("请使用 pll_port 和 clock_frequency_mhz 配置板卡时钟")
    try:
        if timing is not None:
            values["timing"] = CIMTimingConfig(**timing)
        return UsbConfig(**values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"板卡配置无效：{error}") from error
