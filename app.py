from flask import Flask, render_template, request, Response, jsonify, send_file
import cv2
import os
from ultralytics import YOLO
import threading
import time
import imageio
import sqlite3
from datetime import datetime
import csv
import io

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/outputs"
DB_PATH = "traffic_detections.db"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

model = YOLO("yolov8m.pt")

counts = {"car":0,"bus":0,"truck":0,"motorcycle":0}
camera_counts = {"car":0,"bus":0,"truck":0,"motorcycle":0}
processing = {"status": False, "progress": 0, "current_file": None}
stop_event = threading.Event()
camera_stop_event = threading.Event()

# Initialize database
def init_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            filename TEXT NOT NULL,
            file_type TEXT NOT NULL,
            cars INTEGER DEFAULT 0,
            buses INTEGER DEFAULT 0,
            trucks INTEGER DEFAULT 0,
            motorcycles INTEGER DEFAULT 0,
            total_vehicles INTEGER DEFAULT 0
        )
    ''')
    conn.commit()
    conn.close()

def save_detection(filename, file_type, cars, buses, trucks, motorcycles):
    """Save detection records to database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    total = cars + buses + trucks + motorcycles
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        INSERT INTO detections (timestamp, filename, file_type, cars, buses, trucks, motorcycles, total_vehicles)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (timestamp, filename, file_type, cars, buses, trucks, motorcycles, total))
    conn.commit()
    conn.close()

def get_all_detections():
    """Get all detection history from database"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM detections ORDER BY timestamp DESC')
    detections = cursor.fetchall()
    conn.close()
    return detections

# Initialize database on startup
init_database()


@app.route("/")
def home():
    global counts, processing
    # Reset processing state and counts on home page load
    counts = {"car":0,"bus":0,"truck":0,"motorcycle":0}
    processing = {"status": False, "progress": 0, "current_file": None}
    return render_template("index.html", counts=counts, before=None, after=None, video=None, camera=False, processing=False)


# IMAGE DETECTION
@app.route("/detect_image", methods=["POST"])
def detect_image():

    file = request.files["image"]
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    img = cv2.imread(path)
    results = model(img)

    global counts
    counts = {"car":0,"bus":0,"truck":0,"motorcycle":0}

    for r in results:
        for box in r.boxes:

            cls = int(box.cls[0])
            name = model.names[cls].lower()

            if name in counts:
                counts[name]+=1

        img = r.plot()

    out_path = os.path.join(OUTPUT_FOLDER,"detected.jpg")
    cv2.imwrite(out_path,img)

    # Save to database
    save_detection(file.filename, "image", counts["car"], counts["bus"], counts["truck"], counts["motorcycle"])

    return render_template(
        "index.html",
        before="uploads/"+file.filename,
        after="outputs/detected.jpg",
        counts=counts,
        processing=False,
        video=None,
        camera=False
    )


# BACKGROUND VIDEO PROCESSING
def process_video_background(video_path):
    global counts, processing, stop_event
    
    processing["status"] = True
    processing["progress"] = 0
    processing["current_file"] = video_path
    stop_event.clear()
    
    try:
        cap = cv2.VideoCapture(video_path)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps == 0:
            fps = 25

        output_path = os.path.join(OUTPUT_FOLDER, "detected_video.mp4")

        counts = {"car":0,"bus":0,"truck":0,"motorcycle":0}

        # Collect all frames first
        frames = []
        frame_count = 0
        
        while True:
            
            # Check if stop was requested
            if stop_event.is_set():
                print("Video processing stopped by user")
                cap.release()
                processing["status"] = False
                return

            ret, frame = cap.read()

            if not ret:
                break

            results = model(frame)

            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    name = model.names[cls].lower()
                    if name in counts:
                        counts[name]+=1
                frame = r.plot()

            frames.append(frame)
            frame_count += 1
            processing["progress"] = int((frame_count / total_frames) * 100)

        cap.release()

        # Write all frames to MP4 using imageio
        if frames:
            print(f"Writing {len(frames)} frames to MP4...")
            print(f"Frame size: {frames[0].shape}")
            print(f"FPS: {fps}")
            try:
                imageio.mimwrite(output_path, frames, fps=fps, codec='libx264', pixelformat='yuv420p')
                # Verify file was created
                if os.path.exists(output_path):
                    file_size = os.path.getsize(output_path)
                    print(f"✅ Video successfully saved: {output_path} ({file_size} bytes)")
                else:
                    print(f"❌ ERROR: Video file was not created at {output_path}")
            except Exception as write_error:
                print(f"❌ Error writing MP4: {write_error}")
                raise
        
        # Save to database
        video_filename = os.path.basename(video_path)
        save_detection(video_filename, "video", counts["car"], counts["bus"], counts["truck"], counts["motorcycle"])
        
    except Exception as e:
        print(f"Error during video processing: {e}")
        import traceback
        traceback.print_exc()

    processing["status"] = False


# VIDEO DETECTION
@app.route("/detect_video", methods=["POST"])
def detect_video():

    file = request.files["video"]

    video_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(video_path)

    # Start background processing
    thread = threading.Thread(target=process_video_background, args=(video_path,))
    thread.start()

    return render_template(
        "index.html",
        video="outputs/detected_video.mp4",
        processing=True,
        counts=counts,
        before=None,
        after=None,
        camera=False
    )


@app.route("/processing_status")
def processing_status():
    return jsonify({
        "status": processing["status"],
        "progress": processing["progress"],
        "car": counts["car"],
        "bus": counts["bus"],
        "truck": counts["truck"],
        "motorcycle": counts["motorcycle"]
    })


@app.route("/stop_processing", methods=["POST"])
def stop_processing():
    global stop_event
    stop_event.set()
    return jsonify({"message": "Processing stopped"})


# CAMERA STREAM
def generate_camera():
    global camera_counts, camera_stop_event

    cap = cv2.VideoCapture(0)
    
    # Check if camera is available
    if not cap.isOpened():
        print("❌ ERROR: Camera not found or not accessible")
        return

    print("✅ Camera opened successfully")
    camera_counts = {"car":0,"bus":0,"truck":0,"motorcycle":0}
    camera_stop_event.clear()
    
    try:
        while True:
            # Check if stop was requested
            if camera_stop_event.is_set():
                print("Camera stream stopped by user")
                break
                
            success, frame = cap.read()

            if not success:
                print("❌ Failed to read frame from camera")
                break
            
            # Resize frame for faster processing
            frame = cv2.resize(frame, (640, 480))

            results = model(frame)
            
            # Count vehicles in this frame
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    name = model.names[cls].lower()
                    if name in camera_counts:
                        camera_counts[name] += 1
                frame = r.plot()

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    except Exception as e:
        print(f"❌ Camera stream error: {e}")
    finally:
        cap.release()
        camera_counts = {"car":0,"bus":0,"truck":0,"motorcycle":0}


@app.route("/camera_feed")
def camera_feed():
    return Response(generate_camera(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/check_camera")
def check_camera():
    """Check if camera is available"""
    cap = cv2.VideoCapture(0)
    is_available = cap.isOpened()
    cap.release()
    return jsonify({"camera_available": is_available})


@app.route("/camera_counts")
def get_camera_counts():
    """Get live camera counts"""
    return jsonify({
        "car": camera_counts["car"],
        "bus": camera_counts["bus"],
        "truck": camera_counts["truck"],
        "motorcycle": camera_counts["motorcycle"]
    })


@app.route("/stop_camera", methods=["POST"])
def stop_camera():
    """Stop camera stream"""
    global camera_stop_event
    camera_stop_event.set()
    return jsonify({"message": "Camera stopped"})


@app.route("/capture_frame")
def capture_frame():
    """Capture a single frame from camera"""
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return jsonify({"error": "Camera not available"}), 400
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return jsonify({"error": "Failed to capture frame"}), 400
        
        # Run detection on frame
        results = model(frame)
        for r in results:
            frame = r.plot()
        
        # Save frame temporarily
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        frame_path = os.path.join(OUTPUT_FOLDER, f"camera_frame_{timestamp}.jpg")
        cv2.imwrite(frame_path, frame)
        
        return send_file(
            frame_path,
            mimetype='image/jpeg',
            as_attachment=True,
            download_name=f'captured-frame-{timestamp}.jpg'
        )
    except Exception as e:
        print(f"Error capturing frame: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/camera")
def camera():
    global counts
    return render_template("index.html", camera=True, counts=counts, before=None, after=None, video=None, processing=False)


@app.route("/history")
def history():
    """View detection history"""
    detections = get_all_detections()
    detection_list = [dict(detection) for detection in detections]
    return render_template("history.html", detections=detection_list)


@app.route("/export_csv")
def export_csv():
    """Export detection history as CSV"""
    detections = get_all_detections()
    
    # Create CSV in memory
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(['Timestamp', 'Filename', 'Type', 'Cars', 'Buses', 'Trucks', 'Motorcycles', 'Total'])
    
    # Write data
    for detection in detections:
        writer.writerow([
            detection['timestamp'],
            detection['filename'],
            detection['file_type'],
            detection['cars'],
            detection['buses'],
            detection['trucks'],
            detection['motorcycles'],
            detection['total_vehicles']
        ])
    
    # Convert to bytes
    output.seek(0)
    bytes_data = io.BytesIO(output.getvalue().encode('utf-8'))
    bytes_data.seek(0)
    
    return send_file(
        bytes_data,
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'traffic_detections_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    )


@app.route("/stats")
def stats():
    """Get statistics from database"""
    detections = get_all_detections()
    
    total_cars = sum(d['cars'] for d in detections)
    total_buses = sum(d['buses'] for d in detections)
    total_trucks = sum(d['trucks'] for d in detections)
    total_motorcycles = sum(d['motorcycles'] for d in detections)
    total_processed = len(detections)
    
    return jsonify({
        "total_processed": total_processed,
        "total_cars": total_cars,
        "total_buses": total_buses,
        "total_trucks": total_trucks,
        "total_motorcycles": total_motorcycles,
        "total_vehicles": total_cars + total_buses + total_trucks + total_motorcycles
    })


if __name__ == "__main__":
    app.run(debug=True)