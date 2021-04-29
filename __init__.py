
from flask import Flask,render_template,request
from flask import json
from flask import jsonify
from main import compute

from werkzeug.utils import secure_filename

import numpy, cv2
from cv2 import *


import sys
app = Flask(__name__)

PROCESSED_FOLDER = os.path.dirname(os.path.abspath(__file__))

app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
@app.route('/')
def home():
    return render_template("index.html")


@app.route('/team')
def team():
    return render_template("team.html")


@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        if(f):
            print(f)
            #read image file string data
            # data = f
            filename = secure_filename(f.filename) # save file 
            filepath = os.path.join(app.config['PROCESSED_FOLDER'], filename);
            f.save(filepath)
            # img = cv2.imread(filepath)

            arr = compute(filepath)
            # arr = []
            return json.jsonify({ 'ans':arr})
        else:
            print("Aaaaa") 	


if __name__ == '__main__':
    app.run(debug=True)