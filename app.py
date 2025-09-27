import os
import time
import tempfile
import logging
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, redirect, url_for, session, flash, make_response
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from datetime import datetime
import pdfkit
from database import get_db_connection

# Setup logging for debugging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__,
            static_folder="static",
            template_folder="templates")
app.secret_key = "deepfake"

# PDFKIT config for wkhtmltopdf (ensure wkhtmltopdf.exe exists at this path)
path_to_wkhtmltopdf = os.path.join(app.static_folder, 'wkhtmltopdf.exe')
config = pdfkit.configuration(wkhtmltopdf=path_to_wkhtmltopdf)

# Load ML model once when app starts
MODEL_PATH = 'saved_model/model'
try:
    model = load_model(MODEL_PATH)
    app.logger.info("Model loaded successfully.")
except Exception as e:
    app.logger.error(f"Error loading model: {e}")
    model = None

class_names = ['Real', 'Fake']

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form['username']
        password = request.form['password']
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()
        if user and check_password_hash(user['password'], password):
            session['username'] = username
            return redirect(url_for("home"))
        flash('Invalid username or password', 'danger')
    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        full_name = request.form['full_name']
        email = request.form['email']
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return render_template("registration.html")

        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE username = ? OR email = ?', (username, email)).fetchone()
        if user:
            flash('Username or email already exists', 'danger')
        else:
            conn.execute('INSERT INTO users (full_name, email, username, password) VALUES (?, ?, ?, ?)',
                         (full_name, email, username, generate_password_hash(password)))
            conn.commit()
            conn.close()
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for("login"))
        conn.close()
    return render_template("registration.html")


@app.route("/logout")
@login_required
def logout():
    session.pop('username', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for("login"))


def create_spectrogram(file):
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        file.save(temp_file.name)
        audio_file = temp_file.name

    try:
        fig = plt.figure(figsize=(3, 3), dpi=72)
        ax = fig.add_subplot(1, 1, 1)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        y, sr = librosa.load(audio_file, sr=None)
        ms = librosa.feature.melspectrogram(y=y, sr=sr)
        log_ms = librosa.power_to_db(ms, ref=np.max)
        librosa.display.specshow(log_ms, sr=sr, ax=ax)
        plt.axis('off')
        plt.savefig('static/melspectrogram.png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        image_data = load_img('static/melspectrogram.png', target_size=(224, 224))
    finally:
        # Clean up temp audio file
        try:
            os.remove(audio_file)
        except Exception as e:
            app.logger.warning(f"Could not delete temp file: {e}")

    return image_data


def predictions(image_data, model):
    img_array = np.array(image_data)
    img_array1 = img_array / 255.0
    img_batch = np.expand_dims(img_array1, axis=0)
    prediction = model.predict(img_batch)
    class_label = np.argmax(prediction)
    return class_label, prediction


@app.route("/")
@app.route("/home", methods=["GET", "POST"])
def home():
    return render_template("home.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/features")
def features():
    return render_template("features.html")


@app.route("/download_report", methods=["POST"])
@login_required
def download_report():
    detection_message = request.form['detection_message']
    percentage = request.form['percentage']
    execution_time = request.form['execution_time']
    audio_path = request.form['audio_path']
    today_date = datetime.today().strftime('%Y-%m-%d')

    rendered = render_template("download_report.html",
                               detection_message=detection_message,
                               percentage=percentage,
                               execution_time=execution_time,
                               audio_path=audio_path,
                               today_date=today_date)

    pdf = pdfkit.from_string(rendered, False, configuration=config)
    response = make_response(pdf)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'attachment; filename=report.pdf'
    return response


@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    try:
        if model is None:
            flash('Model is not loaded. Please try again later.', 'danger')
            return render_template("audio_upload.html")

        if request.method == "POST":
            file = request.files.get("audio_file")
            if file and file.filename.endswith(".wav"):
                start_time = time.time()
                spec = create_spectrogram(file)
                class_label, prediction = predictions(spec, model)
                detection_message = f"Detection Result: {class_names[class_label]}"
                end_time = time.time()
                execution_time = round(end_time - start_time, 2)
                percentage = "%.2f" % (prediction[0][class_label] * 100)
                return render_template("audio_Result.html",
                                       detection_message=detection_message,
                                       percentage=percentage,
                                       execution_time=execution_time,
                                       audio_path=file.filename)
            flash('Upload a valid audio file (.wav)', 'danger')
        return render_template("audio_upload.html")
    except Exception as e:
        app.logger.error(f"Error in upload: {e}")
        flash('An error occurred during file upload. Please try again.', 'danger')
        return render_template("audio_upload.html")


if __name__ == "__main__":
    app.run(debug=True)
