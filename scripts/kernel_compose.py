"""ffmpeg side-by-side hstack with text labels."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="demo/baseline.mp4")
    ap.add_argument("--dpo", default="demo/dpo_qa.mp4")
    ap.add_argument("--out", default="demo/sxs.mp4")
    args = ap.parse_args()

    cmd = [
        "ffmpeg", "-y", "-i", args.base, "-i", args.dpo,
        "-filter_complex",
        (
            "[0:v]scale=640:-2,drawtext=text='BASELINE Northstar':fontcolor=white:fontsize=22:"
            "box=1:boxcolor=black@0.6:boxborderw=8:x=20:y=20[a];"
            "[1:v]scale=640:-2,drawtext=text='DPO-QA finetuned':fontcolor=white:fontsize=22:"
            "box=1:boxcolor=black@0.6:boxborderw=8:x=20:y=20[b];"
            "[a][b]hstack=inputs=2"
        ),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", args.out,
    ]
    subprocess.run(cmd, check=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
