from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image  # Import the Image module from Pillow
import numpy as np

app = Flask(__name__)

dic = {0: 'Cat', 1: 'Dog'}

model_path = 'C:\\Users\\PNW_checkout\\Desktop\\ML Project\\model.h5'
model = load_model(model_path)

model.make_predict_function()

def predict_label(img_path):
    i = Image.open(img_path)  # Use Image.open from Pillow instead of image.load_img
    i = i.resize((100, 100))  # Resize the image using Pillow
    i = image.img_to_array(i) / 255.0
    i = i.reshape(1, 100, 100, 3)
    probabilities = model.predict(i)
    predicted_class = np.argmax(probabilities)
    return dic[predicted_class]


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")


@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = predict_label(img_path)

	return render_template("index.html", prediction = p, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)




