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

@app.route('/image', methods=["GET", "POST"])
def get_image():
	if request.method == "POST":
		id = request.get_json()
		id = interface.get_original_id_from_index(int(id))
		output = interface.get_image_cutout(id)
		return Response(output.getvalue(), mimetype='image/png')
	else:
		return "Hello"

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