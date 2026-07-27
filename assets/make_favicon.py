"""Regenerates assets/favicon.png — a black circular badge (matches the app's #030202 dark theme)
with a bold white/cream ring outline, a dense red/green random-walk price line (hundreds of
points, real granular noise, not a handful of stylized candles), and a gold-gradient "P" with
subtle depth. Used both as the browser favicon AND at ~100px in the sidebar, so it's rendered at
high supersample for quality at both scales. Run from the repo root: python assets/make_favicon.py
"""
from pathlib import Path
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

SS = 6                 # supersample factor for anti-aliasing
BASE = 512
SIZE = BASE * SS
cx = cy = SIZE / 2
r = SIZE / 2 - 12 * SS

# ── Black fill with a faint vignette (pure flat black reads dead at small sizes — a hair of
# center-to-edge falloff keeps it looking intentional) ──
ys, xs = np.mgrid[0:SIZE, 0:SIZE]
dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2) / r
t = np.clip(dist, 0, 1)
shade = (18 - 14 * t).astype(np.uint8)                    # 18 -> 4, near-black throughout
rgb = np.stack([shade, shade, shade], axis=-1)
alpha = np.where(dist <= 1.0, 255, 0).astype(np.uint8)
img = Image.fromarray(np.dstack([rgb, alpha]), mode="RGBA")
draw = ImageDraw.Draw(img)

coin_mask = Image.new("L", (SIZE, SIZE), 0)
ImageDraw.Draw(coin_mask).ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)


def _masked(layer):
    layer.putalpha(Image.composite(layer.split()[3], Image.new("L", (SIZE, SIZE), 0), coin_mask))
    return layer


# ── Fonts ──
font = None
for path, scale in (
    ("C:/Windows/Fonts/segoescb.ttf", 0.62),
    ("C:/Windows/Fonts/segoesc.ttf", 0.62),
    ("C:/Windows/Fonts/verdanab.ttf", 0.5),
):
    try:
        font = ImageFont.truetype(path, int(SIZE * scale))
        break
    except Exception:
        continue
if font is None:
    font = ImageFont.load_default()

letter = "P"
probe = ImageDraw.Draw(Image.new("L", (1, 1)))
bbox = probe.textbbox((0, 0), letter, font=font)
tx = cx - (bbox[2] - bbox[0]) / 2 - bbox[0]
ty = cy - (bbox[3] - bbox[1]) / 2 - bbox[1] - SIZE * 0.06
letter_bottom = ty + bbox[3] + SIZE * 0.015

# ── Dense random-walk price line — hundreds of points, real granular noise, biased up overall,
# steep breakout near the end, using nearly the full coin height. Legibility against the letter
# comes from the dark halo drawn afterward, not from carving out geometry here. ──
chart = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
cdraw = ImageDraw.Draw(chart)
n = 260
wrng = np.random.default_rng(4)
steps = wrng.normal(loc=0.014, scale=0.10, size=n)
steps[int(n * 0.82):] += 0.05
walk = np.cumsum(steps)
walk -= walk.min()
walk /= walk.max()

xs_frac = np.linspace(-0.88, 0.88, n)
by0 = cy + r * 0.86             # near the bottom rim
top_all = cy - r * 0.86         # near the top rim — full height for everyone, letter handled by the halo
GREEN, RED = (34, 197, 94, 255), (239, 68, 68, 255)
pts = [(cx + r * xf, by0 - wv * (by0 - top_all)) for xf, wv in zip(xs_frac, walk)]

for i in range(len(pts) - 1):
    up = walk[i + 1] >= walk[i]
    col = GREEN if up else RED
    cdraw.line([pts[i], pts[i + 1]], fill=col, width=max(3, int(SS * 2.0)), joint="curve")
chart = chart.filter(ImageFilter.GaussianBlur(SS * 0.1))
img.alpha_composite(_masked(chart))

# ── Ring rendered as an engraved herringbone/chevron band, like a bicolor coin's milled edge —
# short angled strokes alternating tilt every few steps, alternating light/dark for real engraved
# depth, instead of a flat stroke or plain concentric grooves. ──
draw = ImageDraw.Draw(img)
glow = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
ImageDraw.Draw(glow).ellipse([cx - r, cy - r, cx + r, cy + r], outline=(245, 240, 228, 255), width=int(SS * 3))
glow = glow.filter(ImageFilter.GaussianBlur(SIZE * 0.006))
img.alpha_composite(glow)

band_lo, band_hi = r * 0.905, r * 1.0
mid_r = (band_lo + band_hi) / 2
band_w = (band_hi - band_lo)
n_seg = 140
light_col, dark_col = (232, 205, 150, 255), (120, 84, 30, 255)
for i in range(n_seg):
    ang = 2 * math.pi * i / n_seg
    rad = (math.cos(ang), math.sin(ang))
    tang = (-math.sin(ang), math.cos(ang))
    tilt = 1 if (i // 3) % 2 == 0 else -1
    cx0, cy0 = cx + rad[0] * mid_r, cy + rad[1] * mid_r
    half = band_w * 0.62
    dx = (rad[0] * 0.35 + tang[0] * 0.85 * tilt) * half
    dy = (rad[1] * 0.35 + tang[1] * 0.85 * tilt) * half
    col = light_col if tilt > 0 else dark_col
    draw.line([(cx0 - dx, cy0 - dy), (cx0 + dx, cy0 + dy)], fill=col, width=max(1, int(SS * 0.85)))
# thin crisp inner/outer edge lines to frame the band
for rr, w in ((band_lo, SS * 0.6), (band_hi, SS * 0.6)):
    draw.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], outline=(210, 178, 120, 220), width=max(1, int(w)))

# ── Dark halo behind the letter — a real clear zone so the dense chart line never fuses with it.
# Drawn twice (tight + wide blur) for a solid near-letter core plus a soft falloff. ──
for blur, alpha in ((SIZE * 0.010, 255), (SIZE * 0.026, 190)):
    halo = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    ImageDraw.Draw(halo).text((tx, ty), letter, font=font, fill=(2, 2, 2, alpha))
    halo = halo.filter(ImageFilter.GaussianBlur(blur))
    img.alpha_composite(_masked(halo))

# ── Gold-gradient "P" with real surface texture — grain + patina blotches (verdigris teal, rust
# orange), like the aged coin reference — clipped tightly to the letter so it stays legible. ──
letter_mask = Image.new("L", (SIZE, SIZE), 0)
ImageDraw.Draw(letter_mask).text((tx, ty), letter, font=font, fill=255)

gold = np.stack([
    np.interp(t, [0, 1], [252, 176]), np.interp(t, [0, 1], [214, 122]), np.interp(t, [0, 1], [110, 30]),
], axis=-1).astype(np.float64)

grng = np.random.default_rng(21)
grain = grng.normal(0, 1, (SIZE, SIZE)).astype(np.float32)
grain_img = Image.fromarray(((grain - grain.min()) / (grain.max() - grain.min()) * 255).astype(np.uint8))
grain_img = grain_img.filter(ImageFilter.GaussianBlur(SS * 2.2))   # feature size must survive the /SS downscale
grain_n = (np.asarray(grain_img).astype(np.float64) - 127.5) / 127.5
gold += grain_n[..., None] * 42

def _field(seed, blur):
    """Smooth 0..1 noise field, no thresholding — gives organic, full-coverage color drift
    instead of sparse gated blobs (which read as barely-there at this letter's size)."""
    bn = np.random.default_rng(seed).normal(0, 1, (SIZE, SIZE)).astype(np.float32)
    bi = Image.fromarray(((bn - bn.min()) / (bn.max() - bn.min()) * 255).astype(np.uint8))
    bi = bi.filter(ImageFilter.GaussianBlur(blur))
    return np.asarray(bi).astype(np.float64) / 255.0

teal = np.array([60, 150, 138])
rust = np.array([168, 80, 32])
teal_f = _field(31, SS * 8)[..., None] ** 1.4          # push toward extremes for more contrast
rust_f = _field(47, SS * 6)[..., None] ** 1.4
gold = gold * (1 - teal_f * 0.5) + teal * (teal_f * 0.5)
gold = gold * (1 - rust_f * 0.42) + rust * (rust_f * 0.42)
gold = np.clip(gold, 0, 255).astype(np.uint8)

gold_img = Image.fromarray(np.dstack([gold, np.full((SIZE, SIZE), 255, np.uint8)]), "RGBA")
gold_img.putalpha(letter_mask)

shadow = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
off = SIZE * 0.009
ImageDraw.Draw(shadow).text((tx + off * 1.6, ty + off * 1.6), letter, font=font, fill=(0, 0, 0, 200))
shadow = shadow.filter(ImageFilter.GaussianBlur(SIZE * 0.006))
img.alpha_composite(shadow)
img.alpha_composite(gold_img)

highlight = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
ImageDraw.Draw(highlight).text((tx - off * 1.2, ty - off * 1.2), letter, font=font, fill=(255, 244, 214, 45))
highlight = highlight.filter(ImageFilter.GaussianBlur(SIZE * 0.005))
hi_mask = Image.new("L", (SIZE, SIZE), 0)
ImageDraw.Draw(hi_mask).text((tx, ty), letter, font=font, fill=255)
highlight.putalpha(Image.composite(highlight.split()[3], Image.new("L", (SIZE, SIZE), 0), hi_mask))
img.alpha_composite(highlight)

# ── Downscale with high-quality resampling for crisp anti-aliased edges ──
img = img.resize((BASE, BASE), Image.LANCZOS)

out = Path(__file__).parent / "favicon.png"
img.save(out)
print(f"saved {out}")
