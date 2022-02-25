from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow import expand_dims
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads/'
model = load_model('cnn_leaf_diseases_ef0_basic3_224-10_model.h5')

class_dict = ['Cassava Bacterial Blight', 'Cassava Brown Streak Disease', 'Cassava Green Mottle', 'Cassava Mosaic Disease', 'Healthy']

def predict_label(img_path, model):
    loaded_img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(loaded_img)
    img_array = expand_dims(img_array, axis=0)
    classes = model.predict(img_array)
    predicted_bit = class_dict[np.argmax(classes)]
    predicted_value = round(np.max(classes)*100, 2)
    return predict_bit, predict_value
    
    
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if request.files:
            image = request.files['image']
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(img_path)
            prediction, prediction_value = predict_label(img_path, model)
            return render_template('index.html', uploaded_image=image.filename, prediction=predict_bit, prediction_value=predict_value)
    return render_template('index.html')

@app.route('/display/<filename>')
def send_uploaded_image(filename=''):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
