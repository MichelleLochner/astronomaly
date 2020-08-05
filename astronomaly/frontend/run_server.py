from flask import Flask, render_template, request, Response
import json
from os.path import join
from astronomaly.frontend.interface import Controller
import logging
import argparse

# Main function to serve Astronomaly

parser = argparse.ArgumentParser(description='Run the Astronomaly server')
help_str = 'Location of the script Astronomaly should run. \
    See the scripts folder for examples.'
parser.add_argument('script', help=help_str)
args = parser.parse_args()
script = args.script

webapp_dir = join('..', '..', 'webapp')


app = Flask(__name__,
            static_folder=join(webapp_dir, 'public'),
            template_folder=join(webapp_dir, 'public'))

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

controller = Controller(script)


@app.route('/')
def index():
    """
    Serves the main page
    """
    return render_template('index.html')


@app.route('/getindex', methods=["POST"])
def get_index():
    """
    Returns the actual index (e.g. "obj287") of an instance given its position
    in the array.
    """
    if request.method == "POST":
        ind = request.get_json()
        ind = controller.get_original_id_from_index(int(ind))
        return json.dumps(ind)
    else:
        return ""


@app.route('/getdatatype', methods=["POST"])
def get_data_type():
    """
    Serves the data type we're working with (e.g. "image", "light_curve",
    "raw_features")
    """
    if request.method == "POST":
        return json.dumps(controller.get_data_type())
    else:
        return ""


@app.route('/getmetadata', methods=["POST"])
def get_metadata():
    """
    Serves the metadata for a particular instance
    """
    if request.method == "POST":
        idx = str(request.get_json())
        output = controller.get_metadata(idx)
        return json.dumps(output)
    else:
        return ""


@app.route('/getlightcurve', methods=["POST"])
def get_light_curve():
    """
    Serves the display data for a light curve
    """
    if request.method == "POST":
        idx = str(request.get_json())
        output = controller.get_display_data(idx)
        output = json.dumps(output)
        return output
    else:
        return ""


@app.route('/getfeatures', methods=["POST"])
def get_features():
    """
    Serves the features ready to be displayed in a table.
    """
    if request.method == "POST":
        idx = str(request.get_json())
        output = controller.get_features(idx)
        output = json.dumps(output)
        return output
    else:
        return ""


@app.route('/getrawfeatures', methods=["POST"])
def get_raw_features():
    """
    Serves raw features ready for basic plotting
    """
    if request.method == "POST":
        idx = str(request.get_json())
        output = controller.get_display_data(idx)
        output = json.dumps(output)
        return output
    else:
        return ""


@app.route('/getimage', methods=["GET", "POST"])
def get_image():
    """
    Serves the current instance as an image to be displayed
    """
    if request.method == "POST":
        idx = str(request.get_json())
        output = controller.get_display_data(idx)
        if output is None:
            return ""
        return Response(output.getvalue(), mimetype='image/png')
    else:
        return ""


@app.route('/cluster', methods=["GET", "POST"])
def get_clusters():
    """
    Serves the data to be displayed on the clustering tab
    """
    if request.method == "POST":
        technique = request.get_json()
        if technique == 'tsne':
            output = controller.get_cluster_data(color_by_column='score')
            js = json.dumps(output)
            return js


@app.route('/sort', methods=["GET", "POST"])
def sort_data():
    """
    Sorts the data by a requested column
    """
    if request.method == "POST":
        column = (str)(request.get_json())
        if column == "trained_score":
            controller.run_active_learning()

        if column == "random":
            controller.randomise_ml_scores()
        else:
            controller.sort_ml_scores(column)
        return json.dumps("success")


@app.route('/label', methods=["GET", "POST"])
def get_label():
    """
    Records the label given to an instance by a human
    """
    if request.method == "POST":
        out_dict = request.get_json()
        idx = out_dict['id']
        label = (float)(out_dict['label'])
        controller.set_human_label(idx, label)
        return json.dumps("success")


if __name__ == "__main__":
    controller.run_pipeline()
    host = 'http://127.0.0.1:5000/'
    print('##### Astronomaly server now running #####')
    print('Point your browser to', host)
    print()
    app.run()
