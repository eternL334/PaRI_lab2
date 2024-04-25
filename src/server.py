from flask import Flask
from flask import render_template, redirect
from flask import request, url_for
from flask import send_file
from waitress import serve
from werkzeug.utils import secure_filename
import os   
import cv2 as cv
from model import Model

app = Flask(__name__)
UPLOAD_FOLDER = 'src/static'
text = ''
model = Model("src/LinkNet_model.pt")

def make_prediction():
    image = cv.imread(os.path.join(UPLOAD_FOLDER, "image.png"))
    segmentate_mask = model.segmentate(image)
    dots_mask = model.dots(image)
    result = model(segmentate_mask, dots_mask)

    cv.imwrite(os.path.join(UPLOAD_FOLDER, "segmentate.png"), segmentate_mask * 255)

    text = [f"{len(result)}"]
    for centroid, counts in result:
        text.append(f"{centroid[0]}, {centroid[1]}; {counts[0]}, {counts[1]}, {counts[2]}")

    return text

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/", methods=['POST'])
def index_post():
    global n
    try:
        file = request.files['image']
        filename = secure_filename(file.filename)
        file.save(os.path.join(UPLOAD_FOLDER, filename))
        image = cv.imread(os.path.join(UPLOAD_FOLDER, filename))
        os.remove(os.path.join(UPLOAD_FOLDER, filename))
        cv.imwrite(os.path.join(UPLOAD_FOLDER, "image.png"), image)

        return redirect(url_for('predict'))
    except Exception as e:
        app.logger.warning(f"{e}")
        return redirect(url_for('fail'))

@app.route("/predict")
def predict():
    try:
        text = make_prediction()
        with open(os.path.join(UPLOAD_FOLDER, "text.txt"), "w") as f:
            f.write("\n".join(text))
    except Exception as e:
        app.logger.warning(f"{e}")
        return redirect(url_for('fail'))

    return render_template("predict.html", text=text)

@app.route("/predict", methods=['POST'])
def download():
    try:
        return send_file(
                "static/text.txt",
                download_name='text.txt',
                as_attachment=True
            )
    except Exception as e:
        app.logger.warning(f"{e}")
        return redirect(url_for('fail'))

@app.route("/fail")
def fail():
    return render_template("fail.html")

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port='5000')
