# this script will download clips from youtube and package them in such a way to upload to my colab script, run propogation, colour code, and isomap

"""
local stage for CV pipeline (minimal version)
1: download YT clip section via yt-dlp with descriptive name
2: re-encode to OpenCV-friendly mp4
3: contact-sheet preview to choose frames to label
4: interactive clicking for selected frames
5: save JSON + contact sheet + fixed video into zip

usage:
  python3 A_download_and_click.py config.json
  python3 A_download_and_click.py config.json --mode drift --frames 70 78 82
  python3 A_download_and_click.py config.json --force --preview-stride 10
"""

import argparse, json, os, re, shutil, subprocess, sys, textwrap, zipfile
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_OUT_DIR = SCRIPT_DIR / "isomap_pipeline" / "outputs"
CONTACT_SHEET_NAME = "contact_sheet.jpg"


# helper funcs
def req_tool(name: str):
    if shutil.which(name) is None:  
        sys.exit(f"tool not found in path {name}")


def timestr_to_tag(t: str) -> str:
    m = re.match(r"^(\d{2}):(\d{2})(?::\d{2})?$", t) or re.match(r"^(\d{2}):(\d{2}):\d{2}$", t)
    if not m:
        parts = t.split(":")
        if len(parts) >= 2 and len(parts[0]) == 2 and len(parts[1]) == 2:
            return f"{parts[0]}-{parts[1]}"
        return t.replace(":", "-")
    hh, mm = m.group(1), m.group(2)
    return f"{hh}-{mm}"


def safe_slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", s.strip())


def run(cmd: list, cwd: Path | None = None):
    subprocess.run(cmd, check=True, cwd=cwd)


def make_contact_sheet(frames: list[np.ndarray], stride: int, out_path: Path, max_cols: int = 12):
    sampled = list(range(0, len(frames), max(1, stride)))
    if not sampled:
        sampled = [0]
    thumbs = [frames[i] for i in sampled]

    n = len(thumbs)
    cols = min(max_cols, n)
    rows = int(np.ceil(n / cols))
    h, w, _ = thumbs[0].shape
    gap = 4
    sheet = np.ones(((h + gap) * rows - gap, (w + gap) * cols - gap, 3), dtype=np.uint8) * 255

    for idx, img in enumerate(thumbs):
        r, c = divmod(idx, cols)
        y0 = r * (h + gap)
        x0 = c * (w + gap)
        sheet[y0:y0 + h, x0:x0 + w] = img
        disp_idx = sampled[idx]
        cv2.putText(sheet, str(disp_idx), (x0 + 8, y0 + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)

    Image.fromarray(sheet).save(out_path)
    return sampled


def load_video_frames(video_path: Path, resize: tuple[int, int] | None):
    cap = cv2.VideoCapture(str(video_path))
    frames, indices = [], []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if resize:
            frame = cv2.resize(frame, resize)
        frames.append(frame)
        indices.append(i)
        i += 1
    cap.release()
    return frames, indices


def interactive_click(frames: list[np.ndarray], frame_indices: list[int]) -> dict:
    plt.ion()
    all_points = {}
    for frame_idx in frame_indices:
        fig, ax = plt.subplots(figsize=(10, 10))
        img = frames[frame_idx].copy()
        ax.imshow(img)
        ax.set_title(f"Frame {frame_idx}: left=positive, right=negative. hit X when done")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        pos, neg = [], []

        def onclick(event):
            if event.xdata is None or event.ydata is None:
                return
            x, y = float(event.xdata), float(event.ydata)
            if event.button == 1:
                pos.append([x, y])
                ax.scatter(x, y, c="lime", s=80, edgecolor="black", linewidth=1.2)
            elif event.button == 3:
                neg.append([x, y])
                ax.scatter(x, y, c="red", s=80, edgecolor="black", linewidth=1.2)
            fig.canvas.draw_idle()

        cid = fig.canvas.mpl_connect("button_press_event", onclick)
        plt.show(block=True)
        plt.close(fig)
        fig.canvas.mpl_disconnect(cid)
        all_points[str(frame_idx)] = {"pos": pos, "neg": neg}
    return all_points


def merge_json(existing_path: Path, new_points: dict) -> dict:
    if existing_path.exists():
        with open(existing_path, "r") as f:
            base = json.load(f)
    else:
        base = {}
    base.update(new_points)
    return base


# main script
def main():
    parser = argparse.ArgumentParser(
        description="download clip, transcode, contact sheet, click prompts, zip outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Modes:
          initial (default) : normal first labeling
          drift             : reuse existing JSON; only label the frames you pass with --frames

        Examples:
          python3 A_download_and_click.py config.json
          python3 A_download_and_click.py config.json --mode drift --frames 70 78 82
          python3 A_download_and_click.py config.json --force --preview-stride 8
        """)
    )
    parser.add_argument("config", type=Path, help="Path to config.json")
    parser.add_argument("--mode", choices=["initial", "drift"], default="initial")
    parser.add_argument("--frames", type=int, nargs="*", default=None)
    parser.add_argument("--resize", type=int, nargs=2, metavar=("W", "H"), default=None)
    parser.add_argument("--preview-stride", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    req_tool("yt-dlp")
    req_tool("ffmpeg")

    if not args.config.exists():
        sys.exit(f"Config not found: {args.config}")
    with open(args.config, "r") as f:
        cfg = json.load(f)

    athlete = safe_slug(cfg.get("athlete", "Unknown"))
    team = safe_slug(cfg.get("team", "Unknown"))
    date = safe_slug(cfg.get("date", "YYYY-MM-DD"))
    url = cfg["youtube_url"]
    t_start = cfg.get("clip_start", "00:00:00")
    t_end = cfg.get("clip_end", "00:00:05")

    start_tag = timestr_to_tag(t_start)
    prefix = f"{athlete}_{team}_{date}_{start_tag}"

    out_root = BASE_OUT_DIR / prefix
    out_root.mkdir(parents=True, exist_ok=True)

    raw_mp4 = out_root / f"{prefix}.mp4"
    fixed_mp4 = out_root / f"{prefix}_fixed.mp4"
    manual_json = out_root / f"manual_points_{prefix}.json"
    upload_zip = out_root / f"{prefix}_upload_ready.zip"
    contact_sheet_path = out_root / CONTACT_SHEET_NAME

# 1: download clip via yt-dlp
    if not raw_mp4.exists() or args.force:
        print("downloading clip via yt-dlp...")
        dl_tmpl = str(raw_mp4.with_suffix(".%(ext)s"))
        section = f"*{t_start}-{t_end}"
        cmd = [
            "yt-dlp",
            "-f", "bestvideo+bestaudio",
            "--merge-output-format", "mp4",
            "--download-sections", section,
            "-o", dl_tmpl,
            url,
        ]
        run(cmd)
        if not raw_mp4.exists():
            candidates = list(out_root.glob(f"{prefix}.*"))
            if candidates:
                candidates[0].rename(raw_mp4)
        if not raw_mp4.exists():
            sys.exit("download did not produce expected mp4")
    else:
        print("download skipped (exists), use --force to redo")

# convert to opencv compatible mp4 type
    if not fixed_mp4.exists() or args.force:
        print("transcoding to OpenCV-friendly mp4")
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(raw_mp4),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            str(fixed_mp4),
        ]
        run(cmd)
    else:
        print("use --force to redo - this will allow adjustment of the time but still redownloaded accurately")

    # delete orig yt-dlp video after successful transcode
    if raw_mp4.exists():
        raw_mp4.unlink()

# load frames for manual click prompting
    print("loading frames...")
    resize_tuple = tuple(args.resize) if args.resize is not None else None
    frames, indices = load_video_frames(fixed_mp4, resize_tuple)


# print a contact sheet to inspect every 10th frame
    print("creating contact sheet")
    sampled = make_contact_sheet(frames, args.preview_stride, contact_sheet_path)
    print(f"contact sheet saved → {contact_sheet_path}")
    print(f"sampled indices shown: {sampled[:min(10,len(sampled))]}{' ...' if len(sampled)>10 else ''}")

# label specific frames
    if args.mode == "drift":
        if args.frames is None or len(args.frames) == 0:
            raw = input("enter frames to relabel (space/comma-separated): ").strip()
            chosen = [int(x) for x in re.split(r"[,\s]+", raw) if x.strip().isdigit()]
        else:
            chosen = args.frames
    else:
        if args.frames is None or len(args.frames) == 0:
            raw = input("enter frames to label (space/comma-separated) [default first 3]: ").strip()
            if raw:
                chosen = [int(x) for x in re.split(r"[,\s]+", raw) if x.strip().isdigit()]
            else:
                chosen = indices[:3]
        else:
            chosen = args.frames

    chosen = sorted(set([i for i in chosen if 0 <= i < len(frames)]))
    if not chosen:
        sys.exit("no valid frames selected")

    print(f"labeling frames: {chosen}")

# pop clicker open
    new_points = interactive_click(frames, chosen)

# save in JSON
    merged = merge_json(manual_json, new_points)
    with open(manual_json, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"saved manual points in {manual_json}")

# compose zip file which will be uploaded to colab 
    if upload_zip.exists():
        upload_zip.unlink()
    print("zipping outputs")
    with zipfile.ZipFile(upload_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.write(fixed_mp4, arcname=fixed_mp4.name)
        z.write(manual_json, arcname=manual_json.name)
        if contact_sheet_path.exists():
            z.write(contact_sheet_path, arcname=contact_sheet_path.name)

    print(f"Upload this to Colab:\n   {upload_zip}")


if __name__ == "__main__":
    main()