from pathlib import Path


DEVICE_CONFIG_PATH = Path(__file__).resolve().parents[1] / "easyedit_device.txt"
DEFAULT_DEVICE = "mps"


def get_easyedit_device() -> str:
    if not DEVICE_CONFIG_PATH.exists():
        return DEFAULT_DEVICE

    for line in DEVICE_CONFIG_PATH.read_text(encoding="utf-8").splitlines():
        value = line.split("#", 1)[0].strip()
        if value:
            return value

    return DEFAULT_DEVICE


EASYEDIT_DEVICE = get_easyedit_device()
