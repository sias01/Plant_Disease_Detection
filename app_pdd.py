import os
from flask import Flask, render_template, request
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np
import pandas as pd
#import disease_detector
from disease_detector import *
#from posixpath import basename
from flask import Flask, render_template,redirect, request
import joblib
import os
from os.path import join, dirname, realpath
from flask import Flask, request, redirect, render_template
#from werkzeug.utils import secure_filename
import os


disease_info = pd.read_csv((r"C:\Users\Shreyas Desai\Documents\Plant-Disease-Detection\disease.csv" ), encoding='cp1252')
supplement_info = pd.read_csv((r"C:\Users\Shreyas Desai\Documents\Plant-Disease-Detection\supplement_info.csv"),encoding='cp1252')


UPLOAD_FOLDER = join(dirname(realpath('save.jpg')), "./file_input")

ALLOWED_EXTENSIONS = {'png', 'PNG', 'jpg', 'JPG', 'jpeg', 'JPEG', 'gif', 'GIF'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
app.secret_key = 'key'

os.environ["FLASK_ENV"] = "development"
port = 5000

location = './templates'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/')
def home_page():
    return render_template('./home.html')

@app.route('/contact')
def contact():
    return render_template('./contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('./index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('./mobile-device.html')

def pipe(img_path):
    img_256 = prepare_img_256(img_path)
    x = pipe_loc(img_256, model2)
       
    result = x

    return result

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename
        file_path = os.path.join('static/uploads/', filename)
        image.save(file_path)
        #print(file_path)
        pred = pipe(file_path)
        title = disease_info['disease_name'][pred]
        description =disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]
        return render_template('submit.html' , title = title , desc = description , prevent = prevent , 
                               image_url = image_url , pred = pred ,sname = supplement_name , simage = supplement_image_url , buy_link = supplement_buy_link)

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', supplement_image = list(supplement_info['supplement image']),
                           supplement_name = list(supplement_info['supplement name']), disease = list(disease_info['disease_name']), buy = list(supplement_info['buy link']))

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = port, debug = True)