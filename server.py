from PIL import Image
import numpy as np
from flask import Flask, request
import flask
import json
import io
import utils

# Khởi tạo model.
global model
model = None
# Khởi tạo flask app
app = Flask(__name__)

# Khai báo các route 1 cho API
@app.route("/", methods=["GET"])
# Khai báo hàm xử lý dữ liệu.
def _hello_world():
	return "Hello world"

# Khai báo các route 2 cho API
@app.route("/predict", methods=["POST"])
# Khai báo hàm xử lý dữ liệu.
def _predict():
	data = {"success": False}
	if request.files.get("file"):
		# Lấy file ảnh người dùng upload lên
		image = request.files["file"].read()
		# Convert sang dạng array image
		image = Image.open(io.BytesIO(image))
		# resize ảnh
		img_width,img_height = 150,150
		image_rz = utils._preprocess_image(image,
			(img_width, img_height))

		#du doan image
		prediction = model.predict_classes(image_rz)
		if prediction[0][0] == 0:
			data["Animal"] = 'Cat'
		elif prediction[0][0] == 1:
			data["Animal"] = 'Dog'
		data["success"] = True
	return json.dumps(data, ensure_ascii=False, cls=utils.NumpyEncoder)

if __name__ == "__main__":
	print("App run!")
	# Load model
	model = utils._load_model()
	app.run(debug=False, host='127.0.0.1', threaded=False)