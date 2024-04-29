from flask import Flask, request, render_template_string
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow_datasets as tfds
from sklearn.neighbors import NearestNeighbors
from PIL import Image
import io
import base64

app = Flask(__name__)

MODEL_PATH = '../model.h5'
FEATURES_PATH = '../features_.npy'

# Load TensorFlow model
model = load_model(MODEL_PATH)
feature_extractor = tf.keras.Model(inputs=model.input, outputs=model.get_layer('global_average_pooling2d').output)

# Load Nearest Neighbors model and features
nn = NearestNeighbors(n_neighbors=10, algorithm='ball_tree')
features = np.load(FEATURES_PATH)
nn.fit(features)


# Helper function to preprocess image from upload
def preprocess_image(upload):
    image = Image.open(upload)
    image = np.array(image)
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32)
    return np.expand_dims(image, axis=0)


def resize_and_maintain_aspect(image, target_height=100):
    initial_height = tf.shape(image)[0]
    initial_width = tf.shape(image)[1]
    ratio = tf.cast(target_height, tf.float32) / tf.cast(initial_height, tf.float32)
    new_width = tf.cast(tf.round(tf.cast(initial_width, tf.float32) * ratio), tf.int32)
    image = tf.image.resize(image, [target_height, new_width])
    return tf.cast(image, tf.uint8)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        original_image_data = file.read()
        encoded_image_data = base64.b64encode(original_image_data).decode('utf-8')
        image_for_processing = preprocess_image(file)
        features = feature_extractor.predict(image_for_processing)
        distances, indices = nn.kneighbors(features)

        images = []
        for index in indices[0]:
            img_tensor, _ = next(iter(tfds.load('caltech101', split='all', as_supervised=True).skip(index).take(1)))
            # Use the resize_and_maintain_aspect function to resize the images
            img_tensor = resize_and_maintain_aspect(img_tensor, target_height=100)  # Adjust target_height as needed
            img_buffer = io.BytesIO()
            pil_img = Image.fromarray(img_tensor.numpy())  # Convert tensor to numpy array
            pil_img.save(img_buffer, format='JPEG')
            img_buffer.seek(0)
            img_data = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            images.append(img_data)

        return render_template_string('''
        <!DOCTYPE html>
        <html>
            <head>
                <title>Image Results</title>
            </head>
            <body>
                <h1>Uploaded Image</h1>
                <img src="data:image/jpeg;base64,{{upload_data}}"/>
                <h1>Similar Images</h1>
                {% for img in images %}
                <img src="data:image/jpeg;base64,{{img}}"/>
                {% endfor %}
            </body>
        </html>
        ''', upload_data=encoded_image_data, images=images)
    else:
        return '''
        <html>
            <body>
                <h1>Upload new File</h1>
                <form action="" method="post" enctype="multipart/form-data">
                    <input type="file" name="file" />
                    <input type="submit" value="Upload" />
                </form>
            </body>
        </html>
        '''

if __name__ == '__main__':
    app.run(debug=True)
