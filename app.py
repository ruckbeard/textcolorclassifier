from flask import Flask
from flask import render_template
from predict import predict_text_color
import random

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/predict')
def predict_random():
    red = random.choice(range(0,256))
    green = random.choice(range(0, 256))
    blue = random.choice(range(0, 256))
    p_text_color = predict_text_color(red, green, blue)
    return render_template('predict.html',
                           color='#%02x%02x%02x' % (red, green, blue),
                           p_text_color="#ffffff" if p_text_color[0] == 1 else "#000000",
                           a_text_color="#000000" if red*0.299 + green*0.587 + blue*0.114 > 186 else "#ffffff")

@app.route('/predict/<int:red>/<int:green>/<int:blue>')
def predict(red, green, blue):
    p_text_color = predict_text_color(red, green, blue)
    return render_template('predict.html',
                           color='#%02x%02x%02x' % (red, green, blue),
                           p_text_color="#ffffff" if p_text_color[0] == 1 else "#000000",
                           a_text_color="#000000" if red*0.299 + green*0.587 + blue*0.114 > 186 else "#ffffff")

if __name__ == '__main__':
    app.run()
