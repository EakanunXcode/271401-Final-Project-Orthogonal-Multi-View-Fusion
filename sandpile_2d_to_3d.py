#!/usr/bin/env python3
"""
Sandpile 2D -> 3D  |  Orthogonal Multi-View Fusion (Silhouette + Distance Transform)
Final Project 271401

Pipeline
  1. Calibration        : ruler photo -> px/cm  (top view);  base-width matching (side view)
  2. Top segmentation   : Lab/Otsu + morphology + largest contour -> binary mask, per-angle radius R(theta)
  3. Side profiling     : saturation mask -> top edge -> height profile P(d), d = distance from base edge
  4. 3D fusion          : Euclidean Distance Transform of the top mask  x  side profile  -> 2.5D height map
                          + physical constraints (monotonic profile, angle of repose) + grain micro-relief
  5. Layer slicing      : height split into 3 EQUAL parts (H/3, 2H/3, H) -> volume per layer
  6. Outputs            : interactive 3D HTML (real photo texture), diagnostic PNG, layer contours CSV, JSON

Usage (sample data)
  python sandpile_2d_to_3d.py --top IMG_2020.jpg --side IMG_2017.jpg \--ruler-img IMG_2021.jpg --ruler-pts 292,100,875,100 --ruler-cm 17.78 --height-cm 5.04
"""
import argparse, csv, json, os
import numpy as np
import cv2
from scipy import ndimage as ndi
from scipy.signal import savgol_filter
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ------------------------------------------------------------------ helpers
def load(path):
    im = cv2.imread(path)
    if im is None:
        raise FileNotFoundError(path)
    return im


def largest_component(mask, min_y=None):
    n, lab, stats, cen = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    best, best_a = 0, 0
    for i in range(1, n):
        if min_y is not None and cen[i][1] < min_y:      # ignore objects in upper part (hand, ruler)
            continue
        if stats[i, cv2.CC_STAT_AREA] > best_a:
            best, best_a = i, stats[i, cv2.CC_STAT_AREA]
    return (lab == best).astype(np.uint8)


def fill_holes(mask):
    return ndi.binary_fill_holes(mask).astype(np.uint8)


# ------------------------------------------------------------ 1. calibration
def calibrate_px_per_cm(pts, real_cm):
    x1, y1, x2, y2 = pts
    return float(np.hypot(x2 - x1, y2 - y1) / real_cm)


# --------------------------------------------------- 2. top-view segmentation
def segment_top(img):
    """Sand (dark, slightly saturated) vs. white tile. Returns clean binary mask."""
    blur = cv2.GaussianBlur(img, (0, 0), 2)
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Sand is brownish (chroma > 0) while white tile, grid lines, ruler and dark vignette are neutral grey.
    # Chroma-based score is therefore immune to dark image corners / shadows.
    chroma = np.hypot(lab[..., 1].astype(np.float32) - 128, lab[..., 2].astype(np.float32) - 128)
    chroma[lab[..., 0] < 25] = 0                                  # pure-black areas carry no colour information
    dist8 = cv2.normalize(chroma, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, m = cv2.threshold(dist8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # reject coloured rulers (blue/green/cyan, hue 40-140) - sand is orange-brown (hue ~10-30)
    ruler = ((hsv[..., 0] >= 40) & (hsv[..., 0] <= 140) & (hsv[..., 1] > 70)).astype(np.uint8)
    ruler = cv2.dilate(ruler, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    m[ruler > 0] = 0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=2)      # removes scattered grains
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    m = fill_holes(largest_component(m))
    # large opening (~2.5% of image size) cuts thin attachments such as transparent ruler edges
    ks = max(9, int(0.025 * min(m.shape))) | 1
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks)))
    return fill_holes(largest_component(m))


def radial_profile(mask, n=360):
    """R(theta): distance from centroid to contour for every angle -> shape irregularity."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)[:, 0, :].astype(float)
    M = cv2.moments(mask, True)
    cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
    ang = np.arctan2(c[:, 1] - cy, c[:, 0] - cx)
    rad = np.hypot(c[:, 0] - cx, c[:, 1] - cy)
    bins = np.linspace(-np.pi, np.pi, n + 1)
    idx = np.clip(np.digitize(ang, bins) - 1, 0, n - 1)
    R = np.array([rad[idx == i].mean() if np.any(idx == i) else np.nan for i in range(n)])
    R = np.interp(np.arange(n), np.where(~np.isnan(R))[0], R[~np.isnan(R)])
    return (cx, cy), R


# ----------------------------------------------------- 3. side-view profiling
def segment_side(img):
    hsv = cv2.cvtColor(cv2.GaussianBlur(img, (0, 0), 2), cv2.COLOR_BGR2HSV)
    s, v = hsv[..., 1].astype(np.uint8), hsv[..., 2].astype(np.uint8)
    # sand = more saturated than grey wall/tile and not blown-out white
    thr, _ = cv2.threshold(s[img.shape[0] // 2:], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = ((s > thr) & (v < 235)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=3)
    return fill_holes(largest_component(m, min_y=img.shape[0] * 0.6))


def side_profile(mask):
    """Top edge of silhouette -> height-vs-distance-from-base-edge P(d) in pixels."""
    cols = np.where(mask.any(0))[0]
    xl, xr = cols.min(), cols.max()
    top = np.array([np.argmax(mask[:, x]) if mask[:, x].any() else np.nan for x in range(xl, xr + 1)])
    top = np.interp(np.arange(len(top)), np.where(~np.isnan(top))[0], top[~np.isnan(top)])
    top = savgol_filter(top, 31, 3)
    edge = max(3, int(0.02 * len(top)))
    base_y = 0.5 * (top[:edge].mean() + top[-edge:].mean())     # ground level at pile extremes
    h = np.clip(base_y - top, 0, None)
    # fold left & right halves onto the same axis (distance from nearest base edge)
    half = len(h) // 2
    left, right = h[:half], h[::-1][:half]
    P = savgol_filter(0.5 * (left + right), 21, 3)
    P = np.maximum.accumulate(np.clip(P, 0, None))             # physics: height never decreases inward
    return P, (xr - xl + 1), h


# ------------------------------------------------------------ 4. 3D fusion
def fuse(top_img, mask, P, H_cm, px_cm_top, grain_cm=0.02, seed=0):
    """z(x,y) = H * g(u),  u = EDT(mask)/max(EDT)  (0 at base edge, 1 at the ridge/centre)."""
    edt = cv2.distanceTransform(mask, cv2.DIST_L2, 5).astype(np.float32)
    edt = cv2.GaussianBlur(edt, (0, 0), 3)                      # soften medial-axis ridge
    u = np.clip(edt / edt.max(), 0, 1)
    g = P / P.max()                                              # normalised side profile (0..1)
    g_u = np.interp(u * (len(g) - 1), np.arange(len(g)), g)
    z = H_cm * g_u
    # grain micro-relief from top-view luminance (high-pass) -> realism, volume-neutral
    gray = cv2.cvtColor(top_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hp = gray - cv2.GaussianBlur(gray, (0, 0), 4)
    hp = hp / (np.abs(hp).max() + 1e-6)
    z = z + grain_cm * hp * (z > 0.05) * np.clip(z / (0.2 * H_cm), 0, 1)
    z = np.where(mask > 0, np.clip(z, 0, H_cm), 0)
    return z.astype(np.float32), u


# ------------------------------------------------------------ 5. layers
def layer_stats(z, px_cm, H):
    cell = 1.0 / px_cm ** 2
    b = [0, H / 3, 2 * H / 3, H]
    vols = [float(np.clip(z, b[i], b[i + 1]).__sub__(b[i]).sum() * cell) for i in range(3)]
    # (z-b[i]) clipped to layer thickness
    vols = [float((np.clip(z, b[i], b[i + 1]) - b[i]).sum() * cell) for i in range(3)]
    return b, vols


def layer_contours(z, mask, px_cm, c0, b, out_csv):
    rows = []
    for i in range(3):
        lvl = b[i]
        m = ((z > lvl + 1e-3) if i else (mask > 0)).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in cnts:
            if cv2.contourArea(c) < 30:
                continue
            for p in cv2.approxPolyDP(c, 1.0, True)[:, 0, :]:
                rows.append([i + 1, round(lvl, 3), round((p[0] - c0[0]) / px_cm, 3),
                             round(-(p[1] - c0[1]) / px_cm, 3)])
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["layer", "z_cm", "x_cm", "y_cm"]); w.writerows(rows)


# ------------------------------------------------------------ 6. outputs
def build_html(z, top_img, mask, px_cm, c0, b, vols, H, D, out_html, res=130):
    ys, xs = np.where(mask > 0)
    pad = 40
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, z.shape[0] - 1)
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, z.shape[1] - 1)
    gx = np.linspace(x0, x1, res).astype(int)
    gy = np.linspace(y0, y1, res).astype(int)
    X, Y = np.meshgrid(gx, gy)
    Z = z[Y, X]
    rgb = cv2.cvtColor(top_img, cv2.COLOR_BGR2RGB)[Y, X]
    xv, yv = (X - c0[0]) / px_cm, -(Y - c0[1]) / px_cm
    vx, vy = xv.ravel().round(3), yv.ravel().round(3)
    ii, jj = np.meshgrid(np.arange(res - 1), np.arange(res - 1))
    a = (ii * res + jj).ravel(); bb = a + 1; c = a + res; d = c + 1
    I = np.concatenate([a, bb]); J = np.concatenate([bb, d]); K = np.concatenate([c, c])
    cols = [f"rgb({r},{g},{bl})" for r, g, bl in rgb.reshape(-1, 3)]
    lc = [[0, "#10b981"], [1 / 3, "#10b981"], [1 / 3, "#eab308"], [2 / 3, "#eab308"], [2 / 3, "#ef4444"], [1, "#ef4444"]]
    light = dict(ambient=0.55, diffuse=0.9, specular=0.15, roughness=0.9, fresnel=0.1)
    traces, states = [], []
    for mode in ("photo", "layer"):
        for lim, name in ((b[1], "L1"), (b[2], "L1+L2"), (H + 1, "ALL")):
            zc = np.minimum(Z, lim).ravel().round(3)
            kw = dict(vertexcolor=cols) if mode == "photo" else dict(
                intensity=zc, cmin=0, cmax=H, colorscale=lc, showscale=False)
            traces.append(go.Mesh3d(x=vx, y=vy, z=zc, i=I, j=J, k=K, flatshading=False,
                                    lighting=light, visible=False, name=f"{mode}-{name}", **kw))
            states.append((mode, name))
    traces[2].visible = True
    btn = [dict(label=f"{'Photo' if m == 'photo' else 'Layers'}: {n}", method="update",
                args=[{"visible": [k == idx for k in range(6)]}]) for idx, (m, n) in enumerate(states)]
    fig = go.Figure(traces)
    fig.update_layout(
        title=f"Sandpile 3D  |  H={H:.2f} cm  D={D:.2f} cm  |  "
              f"L1 {vols[0]:.1f} / L2 {vols[1]:.1f} / L3 {vols[2]:.1f} cm³  (total {sum(vols):.1f})",
        scene=dict(aspectmode="data", xaxis_title="X (cm)", yaxis_title="Y (cm)", zaxis_title="Z (cm)"),
        updatemenus=[dict(type="buttons", direction="right", x=0.0, y=1.08, buttons=btn)],
        margin=dict(l=0, r=0, t=70, b=0))
    fig.write_html(out_html, include_plotlyjs="cdn")


def diagnostics(top, mask, side, smask, P_cm, g_ideal, z, b, H, out_png):
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    t = cv2.cvtColor(top, cv2.COLOR_BGR2RGB).copy()
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(t, cnts, -1, (255, 0, 0), 4)
    ax[0, 0].imshow(t); ax[0, 0].set_title("1) Top view + contour")
    ax[0, 1].imshow(smask, cmap="gray"); ax[0, 1].set_title("2) Side silhouette")
    s = cv2.cvtColor(side, cv2.COLOR_BGR2RGB); ax[0, 2].imshow(s)
    ax[0, 2].contour(smask, [0.5], colors="lime"); ax[0, 2].set_title("3) Side view + edge")
    d = np.arange(len(P_cm))
    ax[1, 0].plot(P_cm[:, 0], P_cm[:, 1], label="measured profile")
    ax[1, 0].plot(P_cm[:, 0], g_ideal, "--", label="ideal cone (angle of repose)")
    ax[1, 0].set_xlabel("distance from base edge (cm)"); ax[1, 0].set_ylabel("height (cm)")
    for h in b[1:3]: ax[1, 0].axhline(h, color="gray", lw=.8, ls=":")
    ax[1, 0].legend(); ax[1, 0].set_title("4) Side profile P(d), layer lines = H/3, 2H/3")
    im = ax[1, 1].imshow(z, cmap="terrain"); plt.colorbar(im, ax=ax[1, 1], label="cm")
    ax[1, 1].set_title("5) Fused height map z(x,y)")
    lay = np.digitize(z, [b[1], b[2]]) + 1; lay[z <= 0.01] = 0
    ax[1, 2].imshow(lay, cmap=matplotlib.colors.ListedColormap(["#ddd", "#10b981", "#eab308", "#ef4444"]))
    ax[1, 2].set_title("6) 3 equal-height layers")
    for a_ in ax.ravel(): a_.axis("off") if a_ is not ax[1, 0] else None
    plt.tight_layout(); plt.savefig(out_png, dpi=110); plt.close()


def _b64(img, w, q=80):
    import base64
    h = int(img.shape[0] * w / img.shape[1])
    ok, buf = cv2.imencode(".jpg", cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, q])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def export_dashboard(z, top, side, mask, px_cm, c0, b, vols, H, D, angle, diag_png, out_html, prof, N=200):
    """Embed height map + masked photo texture + profile into dashboard_template_v2.html"""
    here = os.path.dirname(os.path.abspath(__file__))
    tpl = next((os.path.join(here, f) for f in ("dashboard_template_v2.html", "dashboard_template.html")
                if os.path.exists(os.path.join(here, f))), None)
    if not tpl:
        print("[warn] dashboard template not found next to the script -> dashboard skipped"); return
    half = int(0.5 * D * px_cm * 1.3)
    cx, cy = int(round(c0[0])) + half, int(round(c0[1])) + half
    pad = lambda im, mode, **kw: cv2.copyMakeBorder(im, half, half, half, half, mode, **kw)
    crop = lambda im: im[cy - half:cy + half, cx - half:cx + half]
    zc = crop(pad(z, cv2.BORDER_CONSTANT, value=0))
    tc = crop(pad(top, cv2.BORDER_REPLICATE))
    mc = crop(pad(mask * 255, cv2.BORDER_CONSTANT, value=0))
    zg = cv2.resize(zc, (N, N), interpolation=cv2.INTER_AREA)
    d_cm, P = prof; sel = np.linspace(0, len(d_cm) - 1, 90).astype(int)
    data = dict(H=float(H), D=float(D), angle=float(angle), bounds=[float(v) for v in b], vols=vols, N=N,
                side_cm=float(2 * half / px_cm), z=[round(float(v), 3) for v in zg.ravel()],
                tex=_b64(tc, 1536, 90), mask=_b64(mc, 1536, 92), top_img=_b64(top, 900, 82), side_img=_b64(side, 420),
                diag_img=_b64(cv2.imread(diag_png), 1400, 78), px_cm=float(px_cm), top_w=int(top.shape[1]),
                prof_d=[round(float(d_cm[i]), 3) for i in sel], prof_p=[round(float(P[i]), 3) for i in sel])
    html = open(tpl, encoding="utf-8").read().replace("/*DATA*/null", json.dumps(data))
    open(out_html, "w", encoding="utf-8").write(html)


# ------------------------------------------------------------------ main
def pick_points(path):
    """Show the ruler photo; user clicks 2 ruler marks. Returns pixel coords in the ORIGINAL image."""
    img = load(path)
    sc = min(1.0, 1100 / img.shape[1], 800 / img.shape[0])
    view = cv2.resize(img, None, fx=sc, fy=sc)
    pts, win = [], "Click 2 ruler marks (ESC = cancel)"
    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < 2:
            pts.append((x, y)); cv2.circle(view, (x, y), 5, (0, 0, 255), -1)
            if len(pts) == 2:
                cv2.line(view, pts[0], pts[1], (0, 255, 255), 2)
    cv2.namedWindow(win); cv2.setMouseCallback(win, on_mouse)
    while True:
        cv2.imshow(win, view); k = cv2.waitKey(30) & 0xFF
        if k == 27: cv2.destroyAllWindows(); raise SystemExit("cancelled")
        if len(pts) == 2:
            cv2.imshow(win, view); cv2.waitKey(600); break
    cv2.destroyAllWindows()
    return [pts[0][0] / sc, pts[0][1] / sc, pts[1][0] / sc, pts[1][1] / sc]


def run(top_p, side_p, ruler_img=None, ruler_pts=None, ruler_cm=None, px_per_cm=None,
        diameter_cm=None, height_cm=None, repose=34.0, res=130, out="output"):
    os.makedirs(out, exist_ok=True)
    top, side = load(top_p), load(side_p)
    mask = segment_top(top)
    (cx, cy), Rth = radial_profile(mask)
    D_px = 2 * np.sqrt(mask.sum() / np.pi)                       # equivalent-circle diameter

    # --- 1. scale
    if px_per_cm:
        px_cm = px_per_cm
    elif ruler_img and ruler_cm:
        pts = ruler_pts or pick_points(ruler_img)               # click on the ruler if no coords given
        rimg = load(ruler_img)
        px_cm_r = calibrate_px_per_cm(pts, ruler_cm)
        D_cm_r = 2 * np.sqrt(segment_top(rimg).sum() / np.pi) / px_cm_r
        px_cm = D_px / D_cm_r
        print(f"[calib] ruler photo: {px_cm_r:.2f} px/cm -> base diameter {D_cm_r:.2f} cm")
    elif diameter_cm:
        px_cm = D_px / diameter_cm
    else:
        raise ValueError("ต้องระบุสเกล: ภาพไม้บรรทัด+ระยะจริง, หรือเส้นผ่านศูนย์กลางฐาน (cm)")
    D_cm = D_px / px_cm
    print(f"[top ] scale {px_cm:.2f} px/cm | equivalent base diameter {D_cm:.2f} cm | "
          f"radius spread {100 * np.std(Rth) / np.mean(Rth):.1f}% (non-circularity)")

    # --- 2. side profile (scale by matching base width = orthogonal-view consistency)
    smask = segment_side(side)
    P_px, width_px, _ = side_profile(smask)
    side_px_cm = width_px / D_cm
    H_meas = float(P_px.max() / side_px_cm)
    H = height_cm or min(H_meas, 0.5 * D_cm * np.tan(np.radians(repose)) * 1.15)
    print(f"[side] silhouette height from image {H_meas:.2f} cm | using H = {H:.2f} cm "
          f"({'user-measured' if height_cm else 'physically bounded'})")
    P = P_px / P_px.max() * H
    d_cm = np.arange(len(P)) / side_px_cm
    angle = np.degrees(np.arctan(H / (D_cm / 2)))
    ideal = np.minimum(H, d_cm * np.tan(np.radians(angle)))

    # --- 3/4. fusion + layers
    z, _ = fuse(top, mask, P, H, px_cm)
    b, vols = layer_stats(z, px_cm, H)
    V_cone = np.pi * (D_cm / 2) ** 2 * H / 3
    print(f"[3D  ] angle of repose ~ {angle:.1f} deg | volume {sum(vols):.1f} cm3 (ideal cone {V_cone:.1f})")
    for i in range(3):
        print(f"       Layer {i + 1}: {b[i]:.2f}-{b[i + 1]:.2f} cm  ->  {vols[i]:.1f} cm3")

    # --- 5. outputs
    build_html(z, top, mask, px_cm, (cx, cy), b, vols, H, D_cm, f"{out}/sandpile_3d.html", res)
    diagnostics(top, mask, side, smask, np.c_[d_cm, P], ideal, z, b, H, f"{out}/diagnostics.png")
    export_dashboard(z, top, side, mask, px_cm, (cx, cy), b, vols, H, D_cm, angle,
                     f"{out}/diagnostics.png", f"{out}/dashboard.html", (d_cm, P))
    layer_contours(z, mask, px_cm, (cx, cy), b, f"{out}/layer_contours.csv")
    json.dump(dict(peak_height_cm=H, base_diameter_cm=D_cm, angle_of_repose_deg=angle,
                   layer_bounds_cm=b, layer_volumes_cm3=vols, total_volume_cm3=sum(vols),
                   ideal_cone_volume_cm3=V_cone, px_per_cm_top=px_cm),
              open(f"{out}/summary.json", "w"), indent=2)
    print(f"[done] -> {out}/ (dashboard.html, sandpile_3d.html, diagnostics.png, layer_contours.csv, summary.json)")
    return os.path.abspath(f"{out}/dashboard.html")


def launch_gui():
    """Popup window: choose Top / Side / Ruler images, enter numbers, press Run."""
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    import webbrowser
    root = tk.Tk(); root.title("Sandpile 2D → 3D  |  Final Project 271401"); root.geometry("640x470")
    root.configure(padx=22, pady=16)
    ttk.Label(root, text="อัปโหลดรูปเพื่อสร้างโมเดล 3D", font=("Segoe UI", 15, "bold")).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 12))
    vars_ = {k: tk.StringVar() for k in ("top", "side", "ruler", "ruler_cm", "diam", "height")}
    vars_["ruler_cm"].set("17.78")

    def row(r, label, key, browse=True, hint=""):
        ttk.Label(root, text=label).grid(row=r, column=0, sticky="w", pady=5)
        ttk.Entry(root, textvariable=vars_[key], width=46).grid(row=r, column=1, padx=8)
        if browse:
            ttk.Button(root, text="Browse...", command=lambda: vars_[key].set(filedialog.askopenfilename(
                filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp")]) or vars_[key].get())).grid(row=r, column=2)
        if hint:
            ttk.Label(root, text=hint, foreground="#666").grid(row=r + 1, column=1, sticky="w", padx=8)

    row(1, "1) รูปมุมบน (Top View)", "top", hint="ถ่ายตรงจากด้านบน เห็นกองทรายเป็นวงกลม")
    row(3, "2) รูปมุมข้าง (Side View)", "side", hint="ถ่ายระดับพื้น เห็นกองทรายเป็นสามเหลี่ยมทั้งกอง")
    row(5, "3) รูปมีไม้บรรทัด (Top View)", "ruler", hint="ใช้ตั้งสเกล – หลังกด Run จะให้คลิก 2 ขีดบนไม้บรรทัด")
    row(7, "    ระยะจริงของ 2 ขีดนั้น (cm)", "ruler_cm", browse=False)
    row(8, "    หรือ เส้นผ่านศูนย์กลางฐานจริง (cm)", "diam", browse=False, hint="ใช้แทนข้อ 3 ถ้าไม่มีรูปไม้บรรทัด")
    row(10, "4) ความสูงยอดกองจริง (cm)", "height", browse=False, hint="ไม่บังคับ แต่ใส่แล้วแม่นกว่า")
    status = ttk.Label(root, text="", foreground="#0a7"); status.grid(row=12, column=0, columnspan=3, pady=6)

    def go():
        try:
            if not vars_["top"].get() or not vars_["side"].get():
                raise ValueError("กรุณาเลือกรูป Top View และ Side View")
            f = lambda k: float(vars_[k].get()) if vars_[k].get().strip() else None
            status.config(text="กำลังประมวลผล... (ถ้ามีหน้าต่างรูป ให้คลิก 2 ขีดบนไม้บรรทัด)"); root.update()
            path = run(vars_["top"].get(), vars_["side"].get(),
                       ruler_img=vars_["ruler"].get() or None, ruler_cm=f("ruler_cm") if vars_["ruler"].get() else None,
                       diameter_cm=f("diam"), height_cm=f("height"))
            status.config(text="เสร็จแล้ว! กำลังเปิด dashboard"); webbrowser.open("file:///" + path.replace("\\", "/"))
        except SystemExit:
            status.config(text="ยกเลิก")
        except Exception as e:
            messagebox.showerror("Error", str(e)); status.config(text="")

    ttk.Button(root, text="▶  Run", command=go).grid(row=11, column=1, sticky="we", pady=10)
    root.mainloop()


def main():
    ap = argparse.ArgumentParser(description="Run with no arguments to open the upload window.")
    ap.add_argument("--top"); ap.add_argument("--side")
    ap.add_argument("--ruler-img"); ap.add_argument("--ruler-pts", help="x1,y1,x2,y2 (omit to click on the image)")
    ap.add_argument("--ruler-cm", type=float, help="real distance between the two ruler points (cm)")
    ap.add_argument("--px-per-cm", type=float); ap.add_argument("--diameter-cm", type=float)
    ap.add_argument("--height-cm", type=float); ap.add_argument("--repose-deg", type=float, default=34.0)
    ap.add_argument("--res", type=int, default=130); ap.add_argument("--out", default="output")
    a = ap.parse_args()
    if not (a.top and a.side):
        return launch_gui()
    pts = [float(v) for v in a.ruler_pts.split(",")] if a.ruler_pts else None
    run(a.top, a.side, a.ruler_img, pts, a.ruler_cm, a.px_per_cm, a.diameter_cm, a.height_cm,
        a.repose_deg, a.res, a.out)


if __name__ == "__main__":
    main()
