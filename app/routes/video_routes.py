from flask import Blueprint, render_template, request, jsonify, Response, session, url_for, redirect

import os
import cv2
import uuid
import atexit
import pandas as pd
from app import my_model


video_routes = Blueprint('video_routes', __name__)


# Create directories for frames and videos if they don’t exist
live_video_path = os.path.join('app', 'output', 'videos', 'live-videos')
uploaded_video_path = os.path.join('app', 'static', 'uploads', 'videos')

annotated_video_path = os.path.join('app', 'output', 'videos', 'annotated-videos')
annotated_upload_video_path = os.path.join('app', 'output', 'videos', 'annotated-upload-videos')
frames_path = os.path.join('app', 'output', 'images')

if not os.path.exists(live_video_path):
    os.makedirs(live_video_path)
if not os.path.exists(annotated_video_path):
    os.makedirs(annotated_video_path)
if not os.path.exists(frames_path):
    os.makedirs(frames_path)



# initialize path in case of uploaded video
video_path = None
# Initialize variables for video writer
cap = None
out = None
out2 = None
is_recording = False
is_processing = False
video_filename = ""
video_uid = None
uploaded_video_name = None

path = os.path.join('app', 'output', 'videos')
# Generating unique identifier for each video
def generate_uuid():
    return str(uuid.uuid4())

# Ensuring resources are cleaned upon exit
def cleanup():
    global out, is_recording, cap
    # Stop the recording if it’s active
    print("Server interrupted: stopping recording and releasing resources.")
    if is_recording:
        print("Server interrupted: stopping recording and releasing resources.")
        is_recording = False
        out.release()  # Release VideoWriter to save video
        out = None
        
    if is_processing:
        print("Server interrupted: stopping processing and releasing resources.")
        is_processing = False
    if 'uploaded_video_name' in session:
        session.pop('uploaded_video_name')

    if out2 is not None:
        out2.release()
    

    if cap is not None:
        cap.release()  # Release the camera
    print("Resources have been released.")

# Register the cleanup function to be called on exit
atexit.register(cleanup)

# Start recording for live video
@video_routes.route('/start_video')
def start_video():
    global out, is_recording, video_filename, cap

    if cap is None or not cap.isOpened():
        return jsonify({"Status": "Camera is not initialized or cannot be opened "})


    video_uid = generate_uuid()
    session['video_uid'] = video_uid
   

    if not is_recording:
        is_recording = True
        video_filename = os.path.join('app','output','videos','live-videos', f"video_{video_uid}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' codec for .mp4
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # frame_rate = 20
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        out = cv2.VideoWriter(video_filename, fourcc, frame_rate, (frame_width, frame_height))
        if not out.isOpened():
            return jsonify({"status": " Failed to open video writer"})
        return jsonify({"status": "Recording started"})
    else:
        return jsonify({"status": "Already recording"})

# Stop recording For live video
@video_routes.route('/stop_video')
def stop_video():
    global out, out2, is_recording
    print('stop video called')
    if out2 is not None:
        out2.release()
        out2 = None
    if out is not None:
        out.release()
        out = None

    if is_recording:
        is_recording = False
        
        # return jsonify({"status": f"Recording stopped. Video saved as {video_filename}"})
        return jsonify({"status": f"Recording stopped. Video saved "})
    else:
        if cap is not None:
            cap.release()
        
        if 'uploaded_video_name' in session:
            session.pop('uploaded_video_name')
        if 'is_processing' in session:
            session.pop('is_processing')
    

        return jsonify({"status": "Recording not active"})

# Video stream generator
def generate_frames(video_uid):
    
    i = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        img = frame
        # img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
        # img = cv2.resize(img, (640, 640))
        

        # loading the model
        # model = torch.hub.load("ultralytics/yolov5", "custom", path="app/model/project_model_kaggle.pt")
        model = my_model()
        results = model(img)
        detections = results.pandas().xyxy[0]
        # print(detections)
        # assign a unique ID to each object 
        detections['id'] = range(1, len(detections) + 1)

        # Save individual frames only if recording is active
        if is_recording:
            file_name = os.path.join( frames_path, f'{video_uid }_frame_{i}.jpg')
            # print(f"{file_name} saved,video file id {video_uid}")
            # cv2.imwrite(file_name, frame)
            i += 1

            color_map = {
                'bike': (0, 255, 0),
                'helmet': (0, 0, 255),
                'no_helmet': (255, 0, 0),
                'number_plate': (255, 255, 0),
                'rider': (0, 255, 255)
            }
            # drawing boundin box
            for index, row in detections.iterrows():

                confidence = row['confidence']
                if confidence < 0.5:
                    continue
                x1 = int(row['xmin'])
                y1 = int(row['ymin'])
                x2 = int(row['xmax'])
                y2 = int(row['ymax'])

                class_name = row['name']
                confidence = row['confidence']

                # get color for the current class
                color = color_map.get(class_name, (0, 0, 0)) # default to black if class not found

                # draw rectangle
                cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
                # put text
                cv2.putText(img, f'{class_name}: {confidence:.2f}', (x1, y1-10) , cv2.FONT_HERSHEY_SIMPLEX, 1, color , 2)

           
            # Write the frame to the video file
            if out:
                out.write(img)

            # # Encode the frame to JPEG for live streaming
            # top_left = (50, 50)   # (x, y)
            # bottom_right = (200, 200)  # (x, y)

            # # Draw the rectangle on the image
            # # Parameters: image, top-left corner, bottom-right corner, color (BGR), thickness
            # cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)  # Green color, thickness 2

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield frame to the video feed
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@video_routes.route('/video_feed')
def video_feed():
    id = session.get('video_uid')
    print(f"video id : {id}")
    return Response(generate_frames(id), mimetype='multipart/x-mixed-replace; boundary=frame')

# live video read and display on page  step 1
@video_routes.route('/live_detection')
def live_detection():

    global cap 
    # cap = cv2.VideoCapture(1) # for external camera
    cap = cv2.VideoCapture(0) # for web camp

    # unique_id = generate_uuid()
    # print(f"unique id for the video : {unique_id}")
    
    return render_template('live-detection.html')



# section 2
# Dealing with existing videos
# uploaded video read and display on page step 1
@video_routes.route('/upload_video_detection/<filename>')
def upload_video_detection(filename):
    global uploaded_video_name
    uploaded_video_name = filename

    # uploaded_video_path = os.path.join('app', 'static', 'uploads', 'videos', filename)

    print('uploaded video name : ', uploaded_video_name)
    session['uploaded_video_name'] = filename

  
    return render_template('file-upload-detection.html', is_video=True, is_image=False, filename=filename)


# def process_video():

# Start processing for uploaded video
@video_routes.route('/start_upload_video_processing')
def start_upload_video_processing():
    global cap, is_processing, uploaded_video_name
    print(f"uploaded video name : {uploaded_video_name}")
    cap = cv2.VideoCapture(os.path.join(uploaded_video_path, uploaded_video_name))
   
    session['is_processing'] = True

    print(uploaded_video_name)
    if not uploaded_video_name:
        return jsonify({'message': 'could not find uploaded video!'})
    
    # frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

    # print(f'Frame height: {frame_height}, Frame width: {frame_width}, Frame rate: {frame_rate}')
    is_processing = session.get('is_processing')
    if is_processing:

        return jsonify({'message': 'Video processing started!', 'status': 'success' }) 
    
    return jsonify({'message': 'Video processing not active', 'status': 'failed'})


@video_routes.route('/processed_result')
def processed_result():
    pass


# peform inference on video
def frame_inference(frame):
    # img = cv2.imread(os.path.join('app/static/uploads/images', filename))
    # preparing image for processing
    img = frame
    # img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img, (640, 640))
    

    # loading the model
    # model = torch.hub.load("ultralytics/yolov5", "custom", path="app/model/project_model_kaggle.pt")
    model = my_model()
    results = model(img)
    detections = results.pandas().xyxy[0]
    # print(detections)
    # assign a unique ID to each object 
    detections['id'] = range(1, len(detections) + 1)

    color_map = {
        'bike': (0, 255, 0),
        'helmet': (0, 0, 255),
        'no_helmet': (255, 0, 0),
        'number_plate': (255, 255, 0),
        'rider': (0, 255, 255)
    }

    # draw bounding boxes on the image
    for index, row in detections.iterrows():
        confidence = row['confidence']
        if confidence < 0.4:
            continue
        x1 = int(row['xmin'])
        y1 = int(row['ymin'])
        x2 = int(row['xmax'])
        y2 = int(row['ymax'])

        class_name = row['name']
        confidence = row['confidence']

        color = color_map.get(class_name, (0, 0, 0))

        # draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        # put text
        cv2.putText(img, f'{class_name} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def generate_uploaded_video_frames():
    global out2, cap

    while True:
        success, frame = cap.read()  # Read the frame from the camera
        print(success)
        if not success:
            cap.release()
            print("video stoped")
            break  # If frame isn't read successfully, stop the stream
            
        # print('Frame received')
        frame = frame_inference(frame)

        if out2:
            out2.write(frame)
            
        # Encode the frame to JPEG for live streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue  # Skip this frame if encoding failed
            
        frame = buffer.tobytes()  # Convert the buffer to bytes
            
        # Yield the frame as a multipart response
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
    # After the loop is done, release the camera and VideoWriter
    cap.release()
    if out2:
        out2.release()

         

@video_routes.route('/uploaded_video_processed_feed')
def uploaded_video_processed_feed():
    global video_filename , cap, out2
    video_filename = session.get('uploaded_video_name')
    # print(" called fro processing : ", video_filename)
    # print(f"video file name : {video_filename}")
    # print(uploaded_video_name)
    cap = cv2.VideoCapture(os.path.join(uploaded_video_path, video_filename))
    

    video_storage_path = os.path.join('app','output','videos','annotated-upload-videos', video_filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' codec for .mp4
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    out2 = cv2.VideoWriter(video_storage_path, fourcc, frame_rate, (frame_width, frame_height))
        
     # Check if the video capture object was successfully opened
    if not cap.isOpened():
        # return f"Error: Could not open video file {video_filename}", 500
        return jsonify({'message': f"Error: Could not open video file {video_filename}", 'status': 'failed'}), 500
    
    
    return Response(generate_uploaded_video_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# reloader test
# @video_routes.route('/reloader')
# def reloader_frames():
#     global cap
#     cap = cv2.VideoCapture(os.path.join('app','static','uploads','videos','test_vid3.mp4'))  # Open the camera (0 is usually the default webcam)
#     cap.releas()
    
    # def generate_frames():
    #     while True:
    #         success, frame = cap.read()  # Read the frame from the camera
    #         if not success:
    #             cap.release()
    #             print("video stoped")
    #             break  # If frame isn't read successfully, stop the stream
            
    #         print('Frame received')
            
    #         # Encode the frame to JPEG for live streaming
    #         ret, buffer = cv2.imencode('.jpg', frame)
    #         if not ret:
    #             continue  # Skip this frame if encoding failed
            
    #         frame = buffer.tobytes()  # Convert the buffer to bytes
            
    #         # Yield the frame as a multipart response
    #         yield (b'--frame\r\n'
    #                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    # Return the generator wrapped in a Response with the proper mimetype for streaming
    
