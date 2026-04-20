from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageFilter, ImageStat
from openai import OpenAI
from waitress import serve
from dotenv import load_dotenv
import os, json, base64, requests, subprocess, uuid, logging, glob, time, threading, tempfile, shutil

# 1. Load environment variables
load_dotenv()

TEMP_DIR = tempfile.gettempdir()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("video_framer")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)
app.config['CORS_HEADERS'] = 'Content-Type'

# Global log storage for frontend access
log_messages = []

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response

@app.errorhandler(Exception)
def handle_global_exception(error):
    logger.exception("Unhandled exception")
    response = jsonify({"status": "error", "message": str(error)})
    response.status_code = 500
    return response

class FrontendLogger(logging.Handler):
    def emit(self, record):
        log_entry = {
            'timestamp': record.created,
            'level': record.levelname,
            'message': record.getMessage(),
            'source': 'main.py'
        }
        log_messages.append(log_entry)
        if len(log_messages) > 100:
            log_messages.pop(0)

frontend_handler = FrontendLogger()
logger.addHandler(frontend_handler)

# --- CONFIG ---
conf = {
    "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
    "SUPABASE_URL": os.environ.get("SUPABASE_URL"),
    "SUPABASE_KEY": os.environ.get("SUPABASE_KEY"),
    "DINO_ENDPOINT": os.environ.get("DINO_ENDPOINT")
}
client = OpenAI(api_key=conf["OPENAI_API_KEY"])

# --- UTILS ---
def fire_and_forget_dino(endpoint, payload):
    """Pushes to DINO server with high-patience timeout."""
    try:
        response = requests.post(endpoint, json=payload, timeout=60)
        if response.status_code == 200:
            logger.info(f"SUCCESS: DINOv2 generated vectors for {payload['image_url'].split('/')[-1]}")
        else:
            logger.error(f"DINO Error {response.status_code}: {response.text}")
    except Exception as e:
        logger.warning(f"DINO Note: {e}")

def get_ffmpeg_path():
    """Detects if we are on Leapcell (Linux) or Local (Windows)"""
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    
    local_exe = os.path.join(os.getcwd(), "ffmpeg-8.1-essentials_build", "bin", "ffmpeg.exe")
    if os.path.exists(local_exe):
        return local_exe
        
    raise FileNotFoundError("FFmpeg not found in system PATH or local directory.")

def extract_frames(video_path, output_dir):
    """Uses ffmpeg to extract exactly 3 spread-out frames (Hook, Middle, End)."""
    ffmpeg_exe = get_ffmpeg_path()
    
    cmd = [
        ffmpeg_exe, "-i", video_path,
        "-vf", "thumbnail,select='eq(n,0)+eq(n,int(N/2))+eq(n,N-1)'",
        "-vsync", "vfr",
        "-q:v", "2",
        os.path.join(output_dir, "frame_%03d.jpg"),
        "-y"
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    frames = glob.glob(os.path.join(output_dir, "*.jpg"))
    frames.sort()
    return frames

def download_carousel_image(url, output_path):
    """Directly downloads images for TikTok carousels."""
    try:
        response = requests.get(url, timeout=20)
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            return True
    except Exception as e:
        logger.error(f"Carousel download failed: {e}")
    return False

# --- AI SEQUENTIAL FUNCTIONS ---
def get_coordinates(frames_data):
    """Task A: Analyzes all frames at HIGH-RES to find products and return coordinates."""
    logger.info(f"Sending {len(frames_data)} frames for HIGH-RES coordinate detection...")
    
    payload_imgs = [
        {"type": "image_url", "image_url": {"url": f["b64"], "detail": "high"}}
        for f in frames_data
    ]
    
    prompt = """
    You are a precision computer vision assistant.
    Analyze these frames. Identify the primary product being showcased.
    
    CRITICAL INSTRUCTION: Act as an intelligent filter. If a frame DOES NOT clearly contain the main product, return an empty array [] for the coordinates. Provide a tight crop but ensure no part of the product is cut off.
    
    Return JSON strictly in this format:
    {
      "frames": [
        {
          "id": "frame_id_from_image eg f_1", 
          "has_product": true/false, 
          "coords": [ymin, xmin, ymax, xmax] // Bounding box values mapped from 0 to 1000.
        }
      ]
    }
    Make sure the 'id' matches the order provided exactly.
    """
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": prompt}, {"role": "user", "content": payload_imgs}]
        )
        return json.loads(res.choices[0].message.content).get("frames", [])
    except Exception as e:
        logger.error(f"Coordinate AI failed: {e}")
        return []

def get_video_setting(frames_data):
    """Task B: Describes setting using the 3 spread-out frames."""
    logger.info("Sending frames for video setting analysis...")
    
    payload_imgs = [
        {"type": "image_url", "image_url": {"url": f["b64"], "detail": "low"}}
        for f in frames_data
    ]
    
    prompt = """
    You are an AI analyzing e-commerce short-form videos.
    Analyze these frames. 
    1. Describe the environment and background briefly to summarize the video.
    2. Extract ALL readable text (especially prices, sizes, or brand names).
    Return JSON strictly in this format: {"video_setting": "Your detailed description here"}
    """
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": prompt}, {"role": "user", "content": payload_imgs}]
        )
        return json.loads(res.choices[0].message.content).get("video_setting", "Unknown setting")
    except Exception as e:
        logger.error(f"Setting AI failed: {e}")
        return "Unknown setting"

@app.route('/')
def health():
    """Explicit health check for Leapcell proxy"""
    return jsonify({"status": "alive", "service": "video-framer"}), 200

@app.route('/favicon.ico')
def favicon():
    """Prevents 404 errors from browsers/proxies looking for an icon"""
    return "", 204
    
# --- MAIN PIPELINE ENDPOINT ---
@app.route('/process_video', methods=['POST'])
def process_video():
    run_id = uuid.uuid4().hex[:6]
    video_run_dir = os.path.join(TEMP_DIR, f"run_{run_id}")
    os.makedirs(video_run_dir, exist_ok=True)
    video_path = None
    
    try:
        # 1. Handle Input (Logic for both file uploads and Deno JSON carousels)
        image_urls = []
        video_id = str(uuid.uuid4())[:8]

        if request.is_json:
            data = request.get_json()
            image_urls = data.get('image_urls', [])
            video_id = data.get('video_id', video_id)
            logger.info(f"Received JSON request for video_id: {video_id}")
        elif 'video' in request.files:
            video_file = request.files['video']
            video_id = request.form.get('video_id', video_id)
            video_path = os.path.join(video_run_dir, f"vid_{video_id}.mp4")
            video_file.save(video_path)
            logger.info(f"Processing uploaded video file: {video_file.filename}")
        else:
            return jsonify({"status": "error", "message": "No input provided"}), 400

        # 2. Acquire Frames
        frame_paths = []
        if image_urls:
            logger.info(f"Processing Carousel ({len(image_urls)} images)...")
            for i, url in enumerate(image_urls[:5]): # Limits to 5 frames
                p = os.path.join(video_run_dir, f"frame_{i:03d}.jpg")
                if download_carousel_image(url, p):
                    frame_paths.append(p)
        else:
            logger.info(f"Extracting 3 frames using FFmpeg...")
            frame_paths = extract_frames(video_path, video_run_dir)
        
        if not frame_paths:
            return jsonify({"status": "error", "message": "Failed to acquire frames."}), 500

        # Convert to Base64
        clean_frames = []
        for i, path in enumerate(frame_paths):
            with open(path, "rb") as f:
                b64_data = base64.b64encode(f.read()).decode('utf-8')
            clean_frames.append({
                "id": f"f_{i+1}",
                "path": path,
                "b64": f"data:image/jpeg;base64,{b64_data}"
            })
            
        logger.info(f"Successfully prepared {len(clean_frames)} frames.")

        # 3. AI Execution
        video_setting = get_video_setting(clean_frames)
        time.sleep(1) 
        coords_data = get_coordinates(clean_frames)
        
        logger.info(f"RAW AI COORDINATE OUTPUT: {json.dumps(coords_data)}")

        # 4. Crop & Fire to Supabase and DINO
        processed_count = 0
        for f_info in coords_data:
            c = f_info.get("coords", [])
            
            if not f_info.get("has_product") or not c or len(c) != 4:
                continue
                
            local_frame = next((f for f in clean_frames if f["id"] == f_info["id"]), None)
            if not local_frame: continue

            img = Image.open(local_frame["path"])
            w, h = img.size
            
            left, top, right, bottom = int(c[1]*w/1000), int(c[0]*h/1000), int(c[3]*w/1000), int(c[2]*h/1000)
            
            # 10% Padding logic
            padding_w = int((right - left) * 0.10)
            padding_h = int((bottom - top) * 0.10)
            
            left = max(0, left - padding_w)
            top = max(0, top - padding_h)
            right = min(w, right + padding_w)
            bottom = min(h, bottom + padding_h)
            
            if left >= right or top >= bottom:
                continue

            cropped_img = img.crop((left, top, right, bottom))
            crop_path = os.path.join(video_run_dir, f"crop_{f_info['id']}.jpg")
            cropped_img.save(crop_path, format="JPEG")

            # Upload to Supabase Storage
            storage_path = f"tiktok_crops/{video_id}_{f_info['id']}.jpg"
            storage_url = f"{conf['SUPABASE_URL']}/storage/v1/object/shoe-crops/{storage_path}"
            headers = {"Authorization": f"Bearer {conf['SUPABASE_KEY']}", "Content-Type": "image/jpeg"}
            
            with open(crop_path, "rb") as f_up:
                upload_result = requests.post(storage_url, headers=headers, data=f_up)
                if upload_result.status_code not in [200, 201]:
                    logger.error(f"Upload failed for {storage_path}: {upload_result.text}")
                    continue

            pub_url = f"{conf['SUPABASE_URL']}/storage/v1/object/public/shoe-crops/{storage_path}"

            # Push to DINO
            dino_endpoint = f"{conf['DINO_ENDPOINT']}/index-video-frame"
            dino_payload = {
                "video_id": video_id,
                "image_url": pub_url,
                "video_setting": video_setting 
            }
            
            threading.Thread(target=fire_and_forget_dino, args=(dino_endpoint, dino_payload)).start()
            processed_count += 1
            time.sleep(1.0)
            
        logger.info(f"Successfully processed {processed_count} crops for video {video_id}")
        
        return jsonify({
            "status": "SUCCESS",
            "video_id": video_id,
            "video_setting": video_setting,
            "crops_pushed_to_index": processed_count,
            "message": f"Processed {len(clean_frames)} frames, uploaded {processed_count} crops"
        })

    except Exception as e:
        logger.exception(f"Processing failed")
        return jsonify({"status": "error", "message": str(e)}), 500

    finally:
        if os.path.exists(video_run_dir):
            shutil.rmtree(video_run_dir, ignore_errors=True)

# --- SERVER STARTUP ---
if __name__ == "__main__":
    logger.info("Server starting on http://0.0.0.0:8080")
    serve(app, host='0.0.0.0', port=8080)
