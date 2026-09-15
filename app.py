import os
import json
import uuid
import subprocess
import threading

import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, render_template

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TMP_DIR = os.path.join(BASE_DIR, "tmp")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(__name__)

progress = {}


def odd(n, lo=3, hi=501):
    n = max(lo, min(hi, int(n)))
    return n if n % 2 else n + 1


def make_mask(img, ranges, settings):
    height, width = img.shape[:2]

    white_min = int(settings.get("whiteMin", 180))
    white_max = int(settings.get("whiteMax", 230))
    color_diff = int(settings.get("colorDiff", 35))
    mask_dilate = max(1, int(settings.get("maskDilate", 20)))

    if white_min > white_max:
        white_min, white_max = white_max, white_min

    b, g, r = cv2.split(img)

    lowest = np.minimum(np.minimum(r, g), b)
    highest = np.maximum(np.maximum(r, g), b)

    mask = (
        (lowest >= white_min) &
        (highest <= white_max) &
        ((highest - lowest) <= color_diff)
    ).astype(np.uint8) * 255

    # マスク拡張
    pin = np.ones((mask_dilate, mask_dilate), np.uint8)
    mask = cv2.dilate(mask, pin, iterations=1)

    exclude = np.zeros((height, width), np.uint8)
    force = np.zeros((height, width), np.uint8)

    for a in ranges:
        x1 = max(0, min(width, int(a.get("x", 0))))
        y1 = max(0, min(height, int(a.get("y", 0))))
        x2 = max(0, min(width, int(a.get("x", 0) + int(a.get("w", 0)))))
        y2 = max(0, min(height, int(a.get("y", 0) + int(a.get("h", 0)))))

        if x2 <= x1 or y2 <= y1:
            continue

        if a.get("type") == "exclude":
            exclude[y1:y2, x1:x2] = 255

        elif a.get("type") == "force":
            force[y1:y2, x1:x2] = 255

    mask[exclude > 0] = 0
    mask[force > 0] = 255
    mask[exclude > 0] = 0

    return mask


def replace_mask_with_median(img, mask, median_size):
    result = img.copy()

    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return result

    # メディアンフィルタの半径
    radius = median_size // 2

    # マスク全体を囲む矩形
    x1 = max(0, xs.min() - radius)
    y1 = max(0, ys.min() - radius)
    x2 = min(img.shape[1], xs.max() + radius + 1)
    y2 = min(img.shape[0], ys.max() + radius + 1)

    # マスク周辺だけ切り出す
    roi = img[y1:y2, x1:x2]

    # ROIだけにメディアンフィルタ
    median_roi = cv2.medianBlur(
        roi,
        median_size
    )

    # ROI内のマスク
    mask_roi = mask[y1:y2, x1:x2]

    # マスク部分だけ置き換える
    result_roi = result[y1:y2, x1:x2]

    result_roi[mask_roi > 0] = \
        median_roi[mask_roi > 0]

    return result


def process_video(input_path, temp_path, ranges, settings, job):
    probe = subprocess.run([
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate:format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        input_path
    ], capture_output=True, text=True, check=True)

    lines = probe.stdout.strip().splitlines()

    if len(lines) < 4:
        raise RuntimeError("動画情報を取得できませんでした")

    width = int(lines[0])
    height = int(lines[1])

    num, den = lines[2].split("/")
    fps = float(num) / float(den)

    duration = float(lines[3])
    total_frames = max(1, int(duration * fps))

    median_size = odd(
        settings.get("medianSize", 21),
        3,
        51
    )

    fin = subprocess.Popen(
        [
            "ffmpeg",
            "-i", input_path,
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=10**8
    )

    fout = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            temp_path
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=10**8
    )

    frame_size = width * height * 3
    frame_count = 0

    try:
        while True:
            raw = fin.stdout.read(frame_size)

            if len(raw) != frame_size:
                break

            frame_count += 1
            progress[job] = int(frame_count / total_frames * 90)

            img = np.frombuffer(
                raw,
                np.uint8
            ).reshape((height, width, 3))

            mask = make_mask(img, ranges, settings)

            result = replace_mask_with_median(
                img,
                mask,
                median_size
            )

            fout.stdin.write(result.tobytes())

    finally:
        if fin.stdout:
            fin.stdout.close()

        fin.wait()

        if fout.stdin:
            fout.stdin.close()

        fout.wait()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/preview", methods=["POST"])
def preview():
    if "image" not in request.files:
        return jsonify(error="画像がありません"), 400

    try:
        ranges = json.loads(request.form.get("ranges", "[]"))
        settings = json.loads(request.form.get("settings", "{}"))
    except Exception:
        return jsonify(error="範囲または設定が不正です"), 400

    file = request.files["image"]

    data = np.frombuffer(
        file.read(),
        np.uint8
    )

    img = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify(error="画像を読み込めませんでした"), 400

    median_size = odd(
        settings.get("medianSize", 21),
        3,
        51
    )

    mask = make_mask(
        img,
        ranges,
        settings
    )

    result = replace_mask_with_median(
        img,
        mask,
        median_size
    )

    ok, encoded = cv2.imencode(
        ".jpg",
        result,
        [cv2.IMWRITE_JPEG_QUALITY, 95]
    )

    if not ok:
        return jsonify(error="プレビュー画像の生成に失敗しました"), 500

    return encoded.tobytes(), 200, {
        "Content-Type": "image/jpeg"
    }

@app.route("/preview_mask", methods=["POST"])
def preview_mask():
    if "image" not in request.files:
        return jsonify(error="画像がありません"), 400

    try:
        ranges = json.loads(
            request.form.get("ranges", "[]")
        )

        settings = json.loads(
            request.form.get("settings", "{}")
        )

    except Exception:
        return jsonify(error="範囲または設定が不正です"), 400

    file = request.files["image"]

    data = np.frombuffer(
        file.read(),
        np.uint8
    )

    img = cv2.imdecode(
        data,
        cv2.IMREAD_COLOR
    )

    if img is None:
        return jsonify(error="画像を読み込めませんでした"), 400

    # 既存のmake_mask()でフィルタ対象を生成
    mask = make_mask(
        img,
        ranges,
        settings
    )

    # マスクを白黒画像として返す
    ok, encoded = cv2.imencode(
        ".png",
        mask
    )

    if not ok:
        return jsonify(error="マスク画像の生成に失敗しました"), 500

    return encoded.tobytes(), 200, {
        "Content-Type": "image/png"
    }

@app.route("/process", methods=["POST"])
def process():
    if "video" not in request.files:
        return jsonify(error="動画がありません"), 400

    f = request.files["video"]

    try:
        ranges = json.loads(request.form.get("ranges", "[]"))
        settings = json.loads(request.form.get("settings", "{}"))
    except Exception:
        return jsonify(error="範囲または設定が不正です"), 400

    job = uuid.uuid4().hex

    ext = os.path.splitext(f.filename)[1] or ".mp4"

    inp = os.path.join(
        TMP_DIR,
        job + "_input" + ext
    )

    tmp = os.path.join(
        OUTPUT_DIR,
        job + "_video.mp4"
    )

    out = os.path.join(
        OUTPUT_DIR,
        job + "_clean.mp4"
    )

    f.save(inp)

    progress[job] = 0

    def run_job():
        try:
            process_video(
                inp,
                tmp,
                ranges,
                settings,
                job
            )

            progress[job] = 95

            subprocess.run([
                "ffmpeg",
                "-y",
                "-i", tmp,
                "-i", inp,
                "-map", "0:v:0",
                "-map", "1:a?",
                "-c:v", "copy",
                "-c:a", "copy",
                "-shortest",
                out
            ], check=True)

            progress[job] = 100

            for p in (inp, tmp):
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except PermissionError:
                        pass

        except Exception as e:
            progress[job] = -1
            print(f"処理エラー: {e}")

            for p in (inp, tmp):
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except PermissionError:
                        pass

    threading.Thread(
        target=run_job,
        daemon=True
    ).start()

    return jsonify(job=job)


@app.route("/progress/<job>")
def get_progress(job):
    return jsonify(
        progress=progress.get(job, 0)
    )


@app.route("/download/<filename>")
def download(filename):
    return send_from_directory(
        OUTPUT_DIR,
        filename
    )


if __name__ == "__main__":
    print("ブラウザで http://127.0.0.1:5000 を開いてください")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )