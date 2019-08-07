from flask import Flask, render_template, request, Response
import json
from os.path import join
import interface

#Main function to serve Astronomaly

webapp_dir = join('..','..','webapp')


app = Flask(__name__,
 	static_folder = join(webapp_dir,'public'),
 	template_folder=join(webapp_dir,'public'))


@app.route('/')
def index():
	return render_template('index.html')

@app.route('/getindex', methods=["POST"])
def get_index():
	if request.method == "POST":
		ind = request.get_json()
		ind = interface.get_original_id_from_index(int(ind))
		return json.dumps(ind)
	else:
		return "Hello"

@app.route('/getmetadata', methods=["POST"])
def get_metadata():
	if request.method == "POST":
		id = str(request.get_json())
		output = interface.get_metadata(id)
		return json.dumps(output)
	else:
		return "Hello"

@app.route('/getlightcurve', methods=["POST"])
def get_light_curve():
	if request.method == "POST":
		id = str(request.get_json())
		# print(id)
		output = interface.get_light_curve(id)
		output = json.dumps(output)
		# print(output)
		return output
	else:
		return "Hello"

@app.route('/getfeatures', methods=["POST"])
def get_features():
	if request.method == "POST":
		id = str(request.get_json())
		# print(id)
		output = interface.get_features(id)
		output = json.dumps(output)
		# print(output)
		return output
	else:
		return "Hello"



@app.route('/image', methods=["GET", "POST"])
def get_image():
	if request.method == "POST":
		id = str(request.get_json())
		output = interface.get_image_cutout(id)
		return Response(output.getvalue(), mimetype='image/png')
	else:
		return "Hello"

@app.route('/cluster', methods=["GET","POST"])
def get_clusters():
	if request.method == "POST":
		technique = request.get_json()
		if technique == 'tsne':
			output = interface.get_tsne_data(input_key='auto', 
				color_by_column='anomaly_score')
			js = json.dumps(output)
			# print(js)
			return js


@app.route('/sort', methods=["GET","POST"])
def sort_data():
	if request.method == "POST":
		column = (str)(request.get_json())
		print(column)
		if column == "random":
			interface.randomise_ml_scores()
		else:
			interface.sort_ml_scores(column)
		return json.dumps("success")


if __name__ == "__main__":
	app.run()