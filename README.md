# Bike, Rider, Helmet Detection for Traffic Law Enforcement

## Overview

This project implements a deep learning-based system for detecting bikes, riders, helmets, and helmet violations in videos and live streams. It is designed to support traffic law enforcement by automatically identifying:

- Bikes
- Riders
- Helmets (worn properly)
- No Helmet (helmet violations)
- Number Plates

The core detection model is based on a custom-trained YOLO architecture capable of real-time inference and annotation.

---

## Features

- **Real-time detection** on live webcam video streams.
- **Batch processing** and annotation of uploaded videos.
- Detection and annotation of five key classes: `bike`, `rider`, `helmet`, `no_helmet`, and `number_plate`.
- Web interface for file uploads with optional custom filenames.
- Video streaming endpoints to display processed video feeds with bounding boxes and confidence scores.
- Automatic saving of live and processed videos with annotations.
- Session management for tracking current video and processing state.
- Graceful cleanup of resources upon server shutdown.

---

## Project Structure

app/
├── static/
│ ├── uploads/
│ │ ├── videos/ # Uploaded raw videos
│ │ └── images/ # Uploaded raw images
├── output/
│ ├── images/ # Extracted frames with annotations
│ └── videos/
│ ├── live-videos/ # Recorded live videos
│ ├── annotated-videos/ # Annotated live videos
│ └── annotated-upload-videos/ # Annotated uploaded videos
├── templates/ # Flask HTML templates
│ ├── file-upload.html # File upload page
│ ├── live-detection.html # Live webcam detection page
│ └── file-upload-detection.html # Uploaded video processing page
├── routes/
│ ├── upload_routes.py # Routes for file upload and management
│ └── video_routes.py # Routes for video capture, processing, streaming
├── model/
│ └── project_model_kaggle.pt # Custom YOLO model weights
└── app.py # Flask app initialization and setup


---

## Installation & Setup

### Prerequisites

- Python 3.7 or higher
- OpenCV (`cv2`)
- Flask
- PyTorch (for loading and running the YOLO model)
- Other dependencies listed in `requirements.txt`

### Setup

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/traffic-bike-helmet-detection.git
   cd traffic-bike-helmet-detection

### Create and activate a python virtual enviroment
python -m venv venv
source venv/bin/activate    # On Windows use: venv\Scripts\activate

### install required packages
pip install -r requirements.txt

### run the flask app
flask run


Usage
Upload Video/Image

    Navigate to /upload_file to upload a video or image.

    You can specify a custom filename or keep the original.

    Supported video formats: .mp4, .avi, .mov.

    Supported image formats: .png, .jpg, .jpeg.

Live Detection

    Visit /live_detection to open the webcam feed.

    Start and stop live video recording with bounding box annotations for the five classes.

    The processed video will be saved in the output directory.

Uploaded Video Processing

    After uploading a video, navigate to /upload_video_detection/<filename>.

    Start processing the video with the YOLO model.

    View the live processed video feed with annotated detections.

## Detection Details
### the system detects and anotates the following classes

| Class          | Description                      |
| -------------- | -------------------------------- |
| `bike`         | Two-wheeled vehicles             |
| `rider`        | Person riding the bike           |
| `helmet`       | Helmet worn correctly            |
| `no_helmet`    | Rider without helmet (violation) |
| `number_plate` | Vehicle registration plate       |


## API endpoint
| Route                                | Method | Description                               |
| ------------------------------------ | ------ | ----------------------------------------- |
| `/upload_file`                       | GET    | Render upload page                        |
| `/upload_file`                       | POST   | Upload video/image with optional filename |
| `/live_detection`                    | GET    | Launch live webcam detection              |
| `/start_video`                       | GET    | Start live video recording                |
| `/stop_video`                        | GET    | Stop live video recording                 |
| `/video_feed`                        | GET    | Stream live annotated video feed          |
| `/upload_video_detection/<filename>` | GET    | Page to process uploaded video            |
| `/start_upload_video_processing`     | GET    | Begin processing the uploaded video       |
| `/uploaded_video_processed_feed`     | GET    | Stream annotated processed uploaded video |

Model

    The YOLO model was trained separately on a custom dataset labeled with the five classes.

    Model inference is done frame-by-frame using the my_model() function in the Flask app.

    Confidence thresholds can be adjusted in the code for filtering detections.


Feel free to contribute or raise issues to improve this project!