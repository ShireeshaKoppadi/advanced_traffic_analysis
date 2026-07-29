# 🚦 AI Traffic Detection System

A real-time vehicle detection and counting system using YOLOv8 for traffic analysis.

## Features

- 📸 **Image Detection** - Upload images to detect and count vehicles
- 🎬 **Video Detection** - Process videos with frame-by-frame vehicle detection
- 📹 **Live Camera Feed** - Real-time vehicle counting from webcam
- 📊 **Detection History** - Database tracking of all detections
- 📥 **Export to CSV** - Download detection records
- 🎯 **Accurate Detection** - Detects Cars, Buses, Trucks, and Motorcycles

## Tech Stack

- **Backend**: Flask (Python)
- **AI/ML**: YOLOv8 (Ultralytics)
- **Computer Vision**: OpenCV
- **Database**: SQLite
- **Frontend**: HTML + JavaScript
- **Video Processing**: imageio

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/GandhamSRILAKSHMI1357/traffic-project-.git
cd traffic-project-
```

2. **Install dependencies**
```bash
pip install flask opencv-python ultralytics imageio
```

3. **Download YOLOv8 model**
The model will be automatically downloaded on first run, or download manually:
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"
```

4. **Run the application**
```bash
python app.py
```

5. **Open in browser**
Visit `http://localhost:5000`

## Usage

### Image Detection
1. Go to "Image" tab
2. Upload an image
3. View detected vehicles with counts

### Video Detection
1. Go to "Video" tab
2. Upload a video file
3. Wait for processing to complete
4. Download the annotated video

### Live Camera
1. Go to "Live Camera" tab
2. Click "Start Live Camera"
3. Watch real-time vehicle counts update
4. Capture frames anytime
5. Click "Stop Camera" to close

### View History
1. Click "📊 History" button
2. See all detection records
3. Export as CSV for analysis

## Project Structure

```
advanced_traffic_analysis/
├── app.py                 # Main Flask application
├── templates/
│   ├── index.html        # Main dashboard
│   └── history.html      # Detection history page
├── static/
│   ├── uploads/          # Uploaded media files
│   └── outputs/          # Processed results
├── traffic_detections.db # SQLite database
└── yolov8m.pt           # YOLOv8 model weights
```

## API Endpoints

- `POST /detect_image` - Process image upload
- `POST /detect_video` - Process video upload
- `GET /camera_feed` - Stream live camera feed
- `GET /camera_counts` - Get live vehicle counts
- `POST /stop_camera` - Stop camera stream
- `GET /capture_frame` - Capture single frame from camera
- `GET /history` - View detection history
- `GET /export_csv` - Export records as CSV
- `GET /stats` - Get overall statistics

## Features Added

✨ **Real-time Camera Counting**
- Live vehicle detection from webcam
- Automatic count updates on dashboard
- Frame capture option
- Stop/resume functionality

🗄️ **Database Tracking**
- Automatic saving of all detections
- Timestamp tracking
- Vehicle count history
- Filterable detection records

📊 **Analytics & Export**
- Detection statistics
- CSV export for analysis
- Historical comparison

## Future Improvements

- [ ] Speed vs accuracy trade-off settings
- [ ] Region-specific counting
- [ ] Custom model training
- [ ] Multi-camera support
- [ ] Alert notifications
- [ ] Performance metrics graph

## License

This project is open source and available under the MIT License.

## Author

**GandhamSRILAKSHMI1357**
- GitHub: [@GandhamSRILAKSHMI1357](https://github.com/GandhamSRILAKSHMI1357)

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss proposed changes.

---

**Note**: The YOLOv8 model file (`yolov8m.pt`) is not included in the repository due to size. It will be downloaded automatically on first run.
