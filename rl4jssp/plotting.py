from __future__ import annotations

from pathlib import Path
from typing import Dict, List


def _scale(vals, lo, hi, out_lo, out_hi):
    if hi - lo < 1e-9:
        return [(out_lo + out_hi) / 2 for _ in vals]
    return [out_lo + (v - lo) * (out_hi - out_lo) / (hi - lo) for v in vals]


def save_line_svg(values: List[float], title: str, path: Path, width=800, height=420):
    if not values:
        values = [0.0]
    xpad, ypad = 50, 40
    xs = _scale(list(range(len(values))), 0, max(1, len(values) - 1), xpad, width - xpad)
    ys = _scale(values, min(values), max(values), height - ypad, ypad)
    points = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
    <rect width="100%" height="100%" fill="white"/>
    <text x="{width/2}" y="22" text-anchor="middle" font-size="18">{title}</text>
    <line x1="{xpad}" y1="{height-ypad}" x2="{width-xpad}" y2="{height-ypad}" stroke="black"/>
    <line x1="{xpad}" y1="{ypad}" x2="{xpad}" y2="{height-ypad}" stroke="black"/>
    <polyline fill="none" stroke="#2b6cb0" stroke-width="2" points="{points}"/>
    </svg>'''
    path.write_text(svg, encoding="utf-8")


def save_bar_svg(summary: Dict[str, float], metric: str, path: Path, width=900, height=460):
    keys = list(summary.keys())
    vals = [summary[k] for k in keys]
    xpad, ypad = 60, 50
    usable_w = width - 2 * xpad
    bar_w = usable_w / max(1, len(vals)) * 0.7
    gap = usable_w / max(1, len(vals))
    ys = _scale(vals, 0, max(vals) if vals else 1, height - ypad, ypad)

    bars = []
    labels = []
    for i, (k, v, y) in enumerate(zip(keys, vals, ys)):
        x = xpad + i * gap + (gap - bar_w) / 2
        h = (height - ypad) - y
        bars.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="#4a5568"/>')
        labels.append(f'<text x="{x+bar_w/2:.1f}" y="{height-20}" text-anchor="middle" font-size="10">{k}</text>')
        labels.append(f'<text x="{x+bar_w/2:.1f}" y="{max(18, y-4):.1f}" text-anchor="middle" font-size="10">{v:.1f}</text>')

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
    <rect width="100%" height="100%" fill="white"/>
    <text x="{width/2}" y="24" text-anchor="middle" font-size="18">Method comparison - {metric}</text>
    <line x1="{xpad}" y1="{height-ypad}" x2="{width-xpad}" y2="{height-ypad}" stroke="black"/>
    <line x1="{xpad}" y1="{ypad}" x2="{xpad}" y2="{height-ypad}" stroke="black"/>
    {''.join(bars)}
    {''.join(labels)}
    </svg>'''
    path.write_text(svg, encoding="utf-8")
