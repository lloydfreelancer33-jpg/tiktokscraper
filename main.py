import os, json, base64, requests, subprocess, uuid, logging, glob
import concurrent.futures
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
    """Uses PIL to detect blur. Returns True if sharp, False if blurry."""
    try:
        img = Image.open(image_path).convert('L') # Convert to grayscale
        edges = img.filter(ImageFilter.FIND_EDGES) # Highlight edges
        stat = ImageStat.Stat(edges)
        variance = stat.var[0] # Calculate variance of edges
        return variance > threshold
    except Exception:
        return False

# --- AI PARALLEL FUNCTIONS ---
def get_coordinates(frames_data):
    """Task A: Finds products and returns coordinates."""
    logger.info(f"Sending {len(frames_data)} frames for coordinate detection...")
    payload_imgs = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{f['b64']}"}}
        for f in frames_data
    ]
    
    prompt = """
    Analyze these frames. Return JSON strictly in this format:
    {
      "frames": [
        {
          "id": "frame_id_from_image", 
          "has_product": true/false, 
          "coords": [ymin, xmin, ymax, xmax] // Values 0-1000. Empty array if no product.
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
    """Task B: Describes the general setting and OCR text."""
    logger.info("Sending sample frames for video setting analysis...")
    # We only need 2-3 frames to understand the setting, saving tokens.
    sample_frames = [frames_data[0], frames_data[len(frames_data)//2], frames_data[-1]]
    payload_imgs = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{f['b64']}"}}
        for f in sample_frames
    ]
    
    prompt = """
    Analyze these sample frames from a TikTok video. 
    Describe the environment, background, and extract ALL readable text (like prices, sizes, or brand names).
    Return JSON: {"video_setting": "Your detailed description here"}
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
        # 1. Snap 2 frames per second (every 0.5s) directly from URL
        logger.info("Running FFmpeg stream...")
        output_pattern = f"/tmp/frame_{run_id}_%04d.jpg"
        subprocess.run([
            'ffmpeg', '-i', video_url, 
            '-vf', 'fps=2', # 2 frames per second
            '-q:v', '2',    # High quality
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
                os.remove(file_path) # Toss blurry frame immediately

        # Failsafe if video is too dark/blurry entirely
        if not clean_frames:
            return jsonify({"status": "skipped", "message": "No clear frames found"}), 200

        # Optional: Limit to top 15 frames so we don't overload GPT tokens
        clean_frames = clean_frames[:15] 

        # 3. Parallel AI Execution
        logger.info(f"Triggering parallel AI for {len(clean_frames)} clean frames...")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_coords = executor.submit(get_coordinates, clean_frames)
            future_setting = executor.submit(get_video_setting, clean_frames)
            
            coords_data = future_coords.result()
            video_setting = future_setting.result()

        # 4. Crop & Upload Valid Products
        processed_results = []
        for f_info in coords_data:
            if not f_info.get("has_product"):
                continue
                
            local_frame = next((f for f in clean_frames if f["id"] == f_info["id"]), None)
            if not local_frame: continue

            # Crop using PIL
            img = Image.open(local_frame["path"])
            w, h = img.size
            c = f_info["coords"]
            
            left, top, right, bottom = int(c[1]*w/1000), int(c[0]*h/1000), int(c[3]*w/1000), int(c[2]*h/1000)
            cropped_img = img.crop((left, top, right, bottom))
            
            crop_path = f"/tmp/crop_{run_id}_{f_info['id']}.jpg"
            cropped_img.save(crop_path, format="JPEG")

            # Upload to Supabase
            storage_path = f"tiktok_crops/{video_id}_{f_info['id']}.jpg"
            storage_url = f"{conf['SUPABASE_URL']}/storage/v1/object/shoe-crops/{storage_path}"
            headers = {"Authorization": f"Bearer {conf['SUPABASE_KEY']}", "Content-Type": "image/jpeg"}
            
            with open(crop_path, "rb") as f_up:
                requests.post(storage_url, headers=headers, data=f_up)
                
            pub_url = f"{conf['SUPABASE_URL']}/storage/v1/object/public/shoe-crops/{storage_path}"

            # Ping DINO Worker
            dino_res = requests.post(f"{conf['DINO_ENDPOINT']}/match", json={"image": pub_url})
            matches = dino_res.json().get("matches", []) if dino_res.status_code == 200 else []

            processed_results.append({
                "frame_id": f_info["id"],
                "image_url": pub_url,
                "matches": matches
            })
            
            if os.path.exists(crop_path): os.remove(crop_path)

        return jsonify({
            "status": "success",
            "video_id": video_id,
            "video_setting": video_setting,
            "processed_crops": processed_results
        })

    finally:
        # 5. Total Cleanup
        for f in glob.glob(f"/tmp/*{run_id}*"):
            if os.path.exists(f): os.remove(f)

if __name__ == "__main__":
    logger.info("Starting Waitress server on port 8080...")
    serve(app, host='0.0.0.0', port=8080)
