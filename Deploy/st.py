from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

from flask import Flask,render_template,url_for,request
import pandas as pd 
import joblib

# load the model from disk
SEQUENCE_LENGTH = 1042
EMBEDDING_SIZE = 100

lstm = load_model("LSTM.h5")
int2label = {0:"Buyable",1:"Not Buyable"}

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home1.html')

@app.route('/predict',methods=['POST'])
def predict():
	

	if request.method == 'POST':
		message = request.form['message']
		print(message)
		tokenizer = Tokenizer()
		
		tokenizer.fit_on_texts(message)
		x = tokenizer.texts_to_sequences(message)
		x = np.array(x)
		x = pad_sequences(x, maxlen=SEQUENCE_LENGTH)
		prediction = lstm.predict(x)[0]
		prediction = int2label[np.argmax(prediction)]
		print(prediction)
		return render_template('result.html',prediction = prediction)



if __name__ == '__main__':
	app.run(debug=False)