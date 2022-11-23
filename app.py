from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array, load_img
from tensorflow import expand_dims
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads/'
model = load_model('cnn_leaf_diseases_ef0_adv-02_224-15-10_model.h5')

class_dict = ['Cassava Bacterial Blight', 'Cassava Brown Streak Disease', 'Cassava Green Mottle', 'Cassava Mosaic Disease', 'Healthy']

def predict_label(img_path, model):
    loaded_img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(loaded_img)
    img_array = expand_dims(img_array, axis=0)
    classes = model.predict(img_array)
    predict_class = class_dict[np.argmax(classes)]
    predict_value = round(np.max(classes)*100, 2)
    value_cbb = round(classes[0][0]*100, 2)
    value_cbsd = round(classes[0][1]*100, 2)
    value_cgm = round(classes[0][2]*100, 2)
    value_cmd = round(classes[0][3]*100, 2)
    value_heal = round(classes[0][4]*100, 2)
    return predict_class, predict_value, value_cbb, value_cbsd, value_cgm, value_cmd, value_heal

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if request.files:
            image = request.files['image']
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(img_path)
            predict_class, predict_value, value_cbb, value_cbsd, value_cgm, value_cmd, value_heal = predict_label(img_path, model)
            return render_template('index.html', uploaded_image=image.filename, prediction=predict_class, prediction_value=predict_value, cbb_value=value_cbb, cbsd_value=value_cbsd, cgm_value=value_cgm, cmd_value=value_cmd, heal_value=value_heal)

    return render_template('index.html')

@app.route('/display/<filename>')
def send_uploaded_image(filename=''):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)