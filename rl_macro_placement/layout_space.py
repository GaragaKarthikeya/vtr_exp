from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LayoutAction:
    aspect_ratio: float
    dsp_startx: int
    bram_startx: int

    def tag(self) -> str:
        return (
            f"a{self.aspect_ratio:.4f}"
            f"_d{self.dsp_startx}"
            f"_b{self.bram_startx}"
        )


def clamp_action(action: LayoutAction, cfg) -> LayoutAction:
    aspect_ratio = min(cfg.aspect_ratio_max, max(cfg.aspect_ratio_min, action.aspect_ratio))
    dsp_startx = min(cfg.dsp_startx_max, max(cfg.dsp_startx_min, int(round(action.dsp_startx))))
    bram_startx = min(cfg.bram_startx_max, max(cfg.bram_startx_min, int(round(action.bram_startx))))
    return LayoutAction(
        aspect_ratio=aspect_ratio,
        dsp_startx=dsp_startx,
        bram_startx=bram_startx,
    )


def write_arch_variant(base_arch: Path, action: LayoutAction, out_path: Path) -> None:
    tree = ET.parse(base_arch)
    root = tree.getroot()

    auto_layout = root.find("./layout/auto_layout")
    if auto_layout is None:
        raise ValueError("Expected <layout><auto_layout ...> in architecture XML")

    auto_layout.set("aspect_ratio", f"{action.aspect_ratio:.4f}")

    dsp_cols = [c for c in auto_layout.findall("col") if c.get("type") == "mult_36"]
    mem_cols = [c for c in auto_layout.findall("col") if c.get("type") == "memory"]

    if not dsp_cols:
        raise ValueError("No <col type=\"mult_36\"> found in architecture XML")
    if not mem_cols:
        raise ValueError("No <col type=\"memory\"> found in architecture XML")

    dsp_cols[0].set("startx", str(action.dsp_startx))
    mem_cols[0].set("startx", str(action.bram_startx))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_path, encoding="utf-8", xml_declaration=False)
