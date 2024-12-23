from flask import Flask, request, render_template, redirect, url_for
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load BLIP model and processor
processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')

def generate_caption(image_path):
    """Generate caption for an image."""
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=20)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image_file' in request.files and request.files['image_file'].filename:
            file = request.files['image_file']
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            caption = generate_caption(file_path)
            return render_template('index.html', caption=caption)
        elif 'image_url' in request.form and request.form['image_url']:
            image_url = request.form['image_url']
            response = requests.get(image_url, stream=True)
            if response.status_code == 200:
                with open("uploads/temp.jpg", "wb") as f:
                    f.write(response.content)
                caption = generate_caption("uploads/temp.jpg")
                return render_template('index.html', caption=caption)
            else:
                return render_template('index.html', error="Invalid URL or unable to fetch image.")
    return render_template('index.html')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
