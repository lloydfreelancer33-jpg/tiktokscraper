import os, json, base64, requests, subprocess, uuid, logging, glob, time, threading
from flask import Flask, request, jsonify
from PIL import Image, ImageFilter, ImageStat
from openai import OpenAI
from waitress import serve

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("video_framer")

app = Flask(__name__)

# --- CONFIG ---
conf = {
    "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
    "SUPABASE_URL": os.environ.get("SUPABASE_URL"),
    "SUPABASE_KEY": os.environ.get("SUPABASE_KEY"),
    "DINO_ENDPOINT": os.environ.get("DINO_ENDPOINT")
}
client = OpenAI(api_key=conf["OPENAI_API_KEY"])

# --- UTILS ---
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def is_sharp(image_path, threshold=50):
    try:
        img = Image.open(image_path).convert('L') 
        edges = img.filter(ImageFilter.FIND_EDGES) 
        stat = ImageStat.Stat(edges)
        variance = stat.var[0] 
        return variance > threshold
    except Exception:
        return False

def fire_and_forget_dino(endpoint, payload):
    """Pushes to DINO server in the background without waiting for a response."""
    try:
        # Short timeout just to ensure the connection opens, but we don't care about the long response
        requests.post(endpoint, json=payload, timeout=3)
    except requests.exceptions.RequestException as e:
        logger.warning(f"Dino background push recorded an expected timeout/error: {e}")

# --- AI SEQUENTIAL FUNCTIONS ---
def get_coordinates(frames_data):
    """Task A: Analyzes all frames to find products and intelligently filter empty ones."""
    logger.info(f"Sending all {len(frames_data)} frames for coordinate detection at low-res...")
    
    payload_imgs = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{f['b64']}", "detail": "low"}}
        for f in frames_data
    ]
    
    prompt = """
    You are a highly accurate computer vision assistant.
    Analyze these frames. Identify the primary product being showcased.
    
    CRITICAL INSTRUCTION: Act as an intelligent filter. If a frame DOES NOT clearly contain the main product, you MUST return an empty array [] for the coordinates.
    
    Return JSON strictly in this format:
    {
      "frames": [
        {
          "id": "frame_id_from_image", 
          "has_product": true/false, 
          "coords": [ymin, xmin, ymax, xmax] // Bounding box values mapped from 0 to 1000. Provide empty array [] if no clear product is visible.
        }
      ]
    }
    Make sure the 'id' matches the order provided exactly. Prioritize accuracy of the bounding box.
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
    """Task B: Describes setting using exactly 3 spread-out frames."""
    logger.info("Sending 3 spread-out frames for video setting analysis...")
    
    # Grab 3 frames: Start (hook), Middle (50%), End (95%)
    if len(frames_data) >= 3:
        sample_frames = [frames_data[0], frames_data[len(frames_data)//2], frames_data[-1]]
    else:
        sample_frames = frames_data

    payload_imgs = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{f['b64']}", "detail": "low"}}
        for f in sample_frames
    ]
    
    prompt = """
    You are an AI analyzing e-commerce short-form videos.
    Analyze these 3 spread-out frames (hook, middle, and end). 
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
        return json.loads(res.choices[0].message.content).get("video_setting", "")
    except Exception as e:
        logger.error(f"Setting AI failed: {e}")
        return "Unknown setting"

@app.route('/')
def health():
    return "Worker is Online and Ready!"
    
# --- MAIN ENDPOINT ---
@app.route('/process_video', methods=['POST'])
def process_video():
    data = request.get_json()
    video_url = data['url']
    video_id = data.get('video_id', str(uuid.uuid4())[:8])
    run_id = uuid.uuid4().hex[:6]
    
    try:
        # 1. Snap 2 frames per second
        logger.info("Running FFmpeg stream...")
        output_pattern = f"/tmp/frame_{run_id}_%04d.jpg"
        subprocess.run([
            'ffmpeg', '-i', video_url, 
            '-vf', 'fps=2', 
            '-q:v', '2',
            output_pattern, '-y'
        ], capture_output=True, timeout=60)

        # 2. Filter Blurry Frames & Encode Clean Ones
        all_files = sorted(glob.glob(f"/tmp/frame_{run_id}_*.jpg"))
        clean_frames = []
        
        for idx, file_path in enumerate(all_files):
            if is_sharp(file_path, threshold=50):
                frame_id = f"f_{idx}"
                clean_frames.append({
                    "id": frame_id, 
                    "path": file_path, 
                    "b64": encode_image(file_path)
                })
            else:
                os.remove(file_path)

        if not clean_frames:
            return jsonify({"status": "skipped", "message": "No clear frames found"}), 200

        # 3. Sequential AI Execution with Cooldown
        logger.info("Processing setting AI (3 frames)...")
        video_setting = get_video_setting(clean_frames)
        
        logger.info("Cooling down for 2.5 seconds to prevent rate limits...")
        time.sleep(2.5)
        
        logger.info(f"Processing coordinate AI for all {len(clean_frames)} frames...")
        coords_data = get_coordinates(clean_frames)

        # 4. Crop & Fire to DINO (Phase 2: Proactive Indexing)
        processed_count = 0
        for f_info in coords_data:
            c = f_info.get("coords", [])
            
            # Filter: Skip if no product detected
            if not f_info.get("has_product") or not c or len(c) != 4:
                continue
                
            local_frame = next((f for f in clean_frames if f["id"] == f_info["id"]), None)
            if not local_frame: continue

            # Crop using PIL
            img = Image.open(local_frame["path"])
            w, h = img.size
            
            # Convert normalized 0-1000 coords to pixel values
            left, top, right, bottom = int(c[1]*w/1000), int(c[0]*h/1000), int(c[3]*w/1000), int(c[2]*h/1000)
            
            if left >= right or top >= bottom:
                continue

            cropped_img = img.crop((left, top, right, bottom))
            crop_path = f"/tmp/crop_{run_id}_{f_info['id']}.jpg"
            cropped_img.save(crop_path, format="JPEG")

            # Upload to Supabase Storage
            storage_path = f"tiktok_crops/{video_id}_{f_info['id']}.jpg"
            storage_url = f"{conf['SUPABASE_URL']}/storage/v1/object/shoe-crops/{storage_path}"
            headers = {"Authorization": f"Bearer {conf['SUPABASE_KEY']}", "Content-Type": "image/jpeg"}
            
            with open(crop_path, "rb") as f_up:
                requests.post(storage_url, headers=headers, data=f_up)
                
            pub_url = f"{conf['SUPABASE_URL']}/storage/v1/object/public/shoe-crops/{storage_path}"

            # --- POINTING TO NEW ENDPOINT ---
            dino_endpoint = f"{conf['DINO_ENDPOINT']}/index-video-frame"
            
            # Payload updated for Proactive matching & Lead tracking
            payload = {
                "video_id": video_id,
                "image_url": pub_url,
                "video_setting": video_setting 
            }
            
            threading.Thread(target=fire_and_forget_dino, args=(dino_endpoint, payload)).start()

            processed_count += 1
            if os.path.exists(crop_path): os.remove(crop_path)

        return jsonify({
            "status": "SUCCESS",
            "video_id": video_id,
            "video_setting": video_setting,
            "crops_pushed_to_index": processed_count
        })

    finally:
        # 5. Total Cleanup
        for f in glob.glob(f"/tmp/*{run_id}*"):
            if os.path.exists(f): os.remove(f)

if __name__ == "__main__":
    logger.info("Starting Waitress server on port 8080...")
    serve(app, host='0.0.0.0', port=8080)
