from flask import Flask, render_template, request, jsonify

from app.routes.image_routes import image_routes
from app.routes.video_routes import video_routes
from app.routes.upload_routes import upload_route

import sqlite3

app = Flask(__name__, template_folder='app/templates', static_folder='app/static')
app.secret_key = 'secret_key' # for video id session
# Registering the blueprints
app.register_blueprint(image_routes, url_prefix='/image')
app.register_blueprint(video_routes, url_prefix='/video')
app.register_blueprint(upload_route, url_prefix='/upload')





@app.route('/')
def home():
    return render_template('index.html')


@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        # Get form data
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')

        # Insert data into database
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute('INSERT INTO messages (name, email, message) VALUES (?, ?, ?)', (name, email, message))

        # Commit changes and close connection
        conn.commit()
        conn.close()

        # Return response (this will be handled via JavaScript)
        return jsonify({'status': 'success', 'message': 'Message sent successfully!'})

    # Render contact page for GET request
    return render_template('contact.html')

@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=True, port=3000)