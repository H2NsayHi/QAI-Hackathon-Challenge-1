import json
from flask import Flask, request, json
import base64
import numpy as np
import cv2
import torch
torch.set_num_threads(4)
app = Flask(__name__)

# TODO
# Import model
from run import run_output
# from extract_pairs import Extractor
# extractor = Extractor()

# Health-checking method
@app.route('/healthCheck', methods=['GET'])
def health_check():
    """
    Health check the server
    Return:
    Status of the server
        "OK"
    """
    return "OK"

# Inference method
@app.route('/infer', methods=['POST'])
def infer():
    """
    Do inference on input image
    Return:
    Dictionary Object following this schema
        {
            "image_name": <Image Name>
            "infers":
            [
                {
                    "food_name_en": <Food Name in Englist>
                    "food_name_vi": <Food Name in Vietnamese>
                    "food_price": <Price of food>
                }
            ]
        }
    """

    # Read data from request
    image_name = request.form.get('image_name')
    encoded_img = request.form.get('image')

    # Convert base64 back to bytes
    img = base64.b64decode(encoded_img)
    img = np.frombuffer(img, dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    # TODO
    # Call model for inference
    try:
        pairs = run_output(img)
        response = {
            "image_name": image_name,
            "infers": []
        }
        for pair in pairs:
            print(pair[0],pair[1],pair[2])
            dct = {
                'food_name_en': pair[2],
                'food_name_vi': pair[0],
                'food_price': pair[1]
            }
            response['infers'].append(dct)
        return json.dumps(response)
        
    except:
        return None
    

if __name__ == "__main__":
    app.run(debug=True, port=5000, host='0.0.0.0')
