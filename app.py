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

@app.route("/")
def index():
    return render_template("index.html")

def odd(n, lo=3, hi=501):
    n = max(lo, min(hi, int(n)))
    return n if n % 2 else n + 1

def process_video(input_path, temp_path, ranges, settings, job):
    probe = subprocess.run([
        "ffprobe","-v","error","-select_streams","v:0",
        "-show_entries","stream=width,height,r_frame_rate,nb_frames,duration",
        "-of","default=noprint_wrappers=1:nokey=1",input_path
    ], capture_output=True, text=True, check=True)

    lines=probe.stdout.strip().splitlines()
    if len(lines)<4: raise RuntimeError("動画情報を取得できませんでした")

    width,height=int(lines[0]),int(lines[1])
    num,den=lines[2].split("/")
    fps=float(num)/float(den)

    duration=float(lines[3])
    total_frames=max(1,int(duration*fps))

    white_min=int(settings.get("whiteMin",180))
    white_max=int(settings.get("whiteMax",230))
    if white_min > white_max:
        white_min, white_max = white_max, white_min
    cd=int(settings.get("colorDiff",35))
    blur_type=str(settings.get("blurType","mosaic"))
    md=max(1,int(settings.get("maskDilate",20)))
    bs=odd(settings.get("blurSize",151),3,501)
    mosaic_block=max(2,int(settings.get("mosaicBlock",12)))
    eb=odd(settings.get("edgeBlur",31),3,101)

    exclude=np.zeros((height,width),np.uint8)
    force=np.zeros((height,width),np.uint8)
    for a in ranges:
        x1=max(0,min(width,int(a.get("x",0))))
        y1=max(0,min(height,int(a.get("y",0))))
        x2=max(0,min(width,int(a.get("x",0)+int(a.get("w",0)))))
        y2=max(0,min(height,int(a.get("y",0)+int(a.get("h",0)))))
        if x2<=x1 or y2<=y1: continue
        if a.get("type")=="exclude": exclude[y1:y2,x1:x2]=255
        elif a.get("type")=="force": force[y1:y2,x1:x2]=255

    pin=np.ones((md,md),np.uint8)
    fbool=force>0

    fin=subprocess.Popen(["ffmpeg","-i",input_path,"-f","rawvideo","-pix_fmt","bgr24","-"],
                         stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,bufsize=10**8)
    fout=subprocess.Popen(["ffmpeg","-y","-f","rawvideo","-pix_fmt","bgr24",
                           "-s",f"{width}x{height}","-r",str(fps),"-i","-",
                           "-c:v","libx264","-preset","fast","-crf","18",
                           "-pix_fmt","yuv420p",temp_path],
                          stdin=subprocess.PIPE,stderr=subprocess.DEVNULL,bufsize=10**8)
    frame_size=width*height*3
    frame_count = 0

    while True:
        raw=fin.stdout.read(frame_size)
        if len(raw)!=frame_size: break

        frame_count += 1
        progress[job] = int(frame_count / total_frames * 90)
        
        img=np.frombuffer(raw,np.uint8).reshape((height,width,3))
        b,g,r=cv2.split(img)
        r16=r.astype(np.int16);g16=g.astype(np.int16);b16=b.astype(np.int16)
        lowest=np.minimum(np.minimum(r,g),b)
        highest=np.maximum(np.maximum(r,g),b)
        mask=((lowest>=white_min)&(highest<=white_max)&
              ((highest-lowest)<=cd)).astype(np.uint8)*255
        mask[exclude>0]=0
        mask=cv2.dilate(mask,pin,iterations=1)
        mask[fbool]=255
        mask[exclude>0]=0
        mask=cv2.GaussianBlur(mask,(eb,eb),0)

        if blur_type=="mosaic":
            small_w=max(1,width//mosaic_block)
            small_h=max(1,height//mosaic_block)
            small=cv2.resize(img,(small_w,small_h),interpolation=cv2.INTER_AREA)
            blurred=cv2.resize(small,(width,height),interpolation=cv2.INTER_NEAREST)
        else:
            blurred=cv2.GaussianBlur(img,(bs,bs),0)

        alpha=(mask.astype(np.float32)/255.0)[:,:,None]
        result=(img.astype(np.float32)*(1-alpha)+blurred.astype(np.float32)*alpha).clip(0,255).astype(np.uint8)
        fout.stdin.write(result.tobytes())
    fin.stdout.close();fin.wait()
    fout.stdin.close();fout.wait()

@app.route("/process",methods=["POST"])
def process():
    if "video" not in request.files:
        return jsonify(error="動画がありません"),400

    f=request.files["video"]

    try:
        ranges=json.loads(request.form.get("ranges","[]"))
        settings=json.loads(request.form.get("settings","{}"))
    except Exception:
        return jsonify(error="範囲または設定が不正です"),400

    job=uuid.uuid4().hex
    ext=os.path.splitext(f.filename)[1] or ".mp4"

    inp=os.path.join(TMP_DIR,job+"_input"+ext)
    tmp=os.path.join(OUTPUT_DIR,job+"_video.mp4")
    out=os.path.join(OUTPUT_DIR,job+"_clean.mp4")

    f.save(inp)

    progress[job]=0

    def run_job():
        try:
            process_video(inp,tmp,ranges,settings,job)

            progress[job]=95

            subprocess.run([
                "ffmpeg","-y",
                "-i",tmp,
                "-i",inp,
                "-map","0:v:0",
                "-map","1:a?",
                "-c:v","copy",
                "-c:a","copy",
                "-shortest",
                out
            ],check=True)

            progress[job]=100

            for p in (inp,tmp):
                if os.path.exists(p):
                    os.remove(p)

        except Exception as e:
            progress[job]=-1
            print(f"処理エラー: {e}")

            for p in (inp,tmp):
                if os.path.exists(p):
                    os.remove(p)

    threading.Thread(target=run_job,daemon=True).start()

    return jsonify(job=job)

@app.route("/progress/<job>")
def get_progress(job):
    return jsonify(progress=progress.get(job, 0))

@app.route("/download/<filename>")
def download(filename):
    return send_from_directory(OUTPUT_DIR,filename)

if __name__=="__main__":
    print("ブラウザで http://127.0.0.1:5000 を開いてください")
    app.run(host="127.0.0.1",port=5000,debug=False)
