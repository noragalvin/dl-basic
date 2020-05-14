import os

from flask import (
    Flask, render_template, request, json, jsonify
)

from PIL import Image
import io
import numpy as np
import cv2


def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev'
    )

    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    @app.route('/', methods=['GET'])
    def index():
        return render_template('index.html')

    @app.route('/recognize', methods=['POST'])
    def recognize():
        f = request.files['file']
        in_memory_file = io.BytesIO()
        f.save(in_memory_file)
        data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
        color_image_flag = 1
        img = cv2.imdecode(data, color_image_flag)
        print(img)
        return 'Detected!'


    return app
