from flask import Flask,  request,  jsonify , Blueprint, render_template
import os

upload_route = Blueprint('upload_route', __name__)

# Create uploads directory if it doesn't exist
os.makedirs('app/static/uploads/videos', exist_ok=True)
os.makedirs('app/static/uploads/images', exist_ok=True)

# showing the uploads page
@upload_route.route('/upload_file')
def upload_page():
    return render_template('file-upload.html')
# Route to handle the file upload, preview, and custom filename
@upload_route.route('/upload_file', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Get the custom filename provided by the user
    custom_filename = request.form.get('filename')

    # If no custom filename is provided, use the original filename
    if not custom_filename:
        custom_filename = file.filename
        # Get the file extension and append it to the custom filename
        file_extension = os.path.splitext(file.filename)[1].lower()
        new_filename = custom_filename + file_extension
        
    new_filename = custom_filename
    filename = new_filename
    
    # Check if the file is a video or an image and save accordingly
    if file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        # Save video to the videos folder
        video_filename = os.path.join('app', 'static','uploads', 'videos', new_filename)
        file.save(video_filename)
        video_name = filename
        
        return jsonify({'message': f'Video {video_name} saved', 'output_name': video_name, 'is_video': True}), 200
    
    elif file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Save image to the images folder
        image_filename = os.path.join('app', 'static', 'uploads', 'images', new_filename)
        print(image_filename)
        file.save(image_filename)
        image_name = filename
        
        return jsonify({'message': f'image {image_name} saved', 'output_name': image_name, 'is_video': False}), 200

    else:
        # Handle unsupported file types
        return jsonify({'error': 'Unsupported file type'}), 400
