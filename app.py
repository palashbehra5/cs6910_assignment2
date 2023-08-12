from flask import Flask, request, jsonify
from viz import occlusion
import cv2

app = Flask(__name__)

@app.route('/occlusion_res', methods = ['POST'])
def occlusion_res():

    img_path = "sample_images\\dog_1.jpg"
    model_path = "models\\models_petimages\\resnet.pth"
    occlusion_results = occlusion(img_path,
                                  model_path,
                                  2,
                                  window_size=4)

    return jsonify({"results" : occlusion_results})