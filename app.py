from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as model_file:
	model = pickle.load(model_file)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
	feature = float(list(request.form.values())[0])
	final_feature = np.array([feature]).reshape(-1, 1)
	prediction = model.predict(final_feature)
	return render_template('index.html', prediction_text='Predicted Salary: ${:.2f}'.format(prediction[0]))
	# features = [float(x) for x in request.form.values()]
	# final_features = np.array(features).reshape(1, -1)
	# prediction = model.predict(final_features)
	# return render_template('index.html', prediction_text='Predicted salary:'.format(prediction[0]))

if __name__ == '__main__':
	app.run(debug=False, host='0.0.0.0')
