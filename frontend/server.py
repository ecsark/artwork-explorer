import cPickle as pk
import errno
import os
import random

import caffe
import scipy.stats
from flask import Flask, g, request, render_template, redirect
from flask_bootstrap import Bootstrap
from sqlalchemy import *
from werkzeug import secure_filename
from multiprocessing.dummy import Pool as ThreadPool

from mona_lisa import VGGExtractor, HogExtractor, LBPExtractor, PretrainedSVC, ZeroScoreRecommender, Recommender

UPLOAD_FOLDER = '/Users/ecsark/Documents/visualdb/project/artwork-explorer/frontend/static/upload/'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
img_dir = "/Users/ecsark/Documents/visualdb/project/wikiart/images/"

app = Flask(__name__, template_folder=template_dir)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = os.urandom(24)
Bootstrap(app)

proj_root = '/Users/ecsark/Documents/visualdb/project/artwork-explorer/'
caffe_root = '/Users/ecsark/Documents/visualdb/caffe/'
model_weights = caffe_root + 'models/vgg/model.caffemodel'
model_def = caffe_root + 'models/vgg/deploy.prototxt'
vgg_ft = VGGExtractor(model_weights, model_def)
hog_ft = HogExtractor()
lbp_ft = LBPExtractor()
style_svc = PretrainedSVC(proj_root + 'model/svc_vgg_fc7.pk')
artist_svc = PretrainedSVC(proj_root + 'model/vgg_artists.pk')
year_svr = pk.load(open(proj_root + 'model/vgg_year.pk', 'rb'))
style_recommender = ZeroScoreRecommender(pk.load(open(proj_root + 'model/style_decision_votes_train.pk', 'rb')))
hog_recommender = Recommender(pk.load(open(proj_root + 'model/hog_train.pk', 'rb')))
lbp_recommender = Recommender(pk.load(open(proj_root + 'model/lbp_train.pk', 'rb')))
artist_recommender = ZeroScoreRecommender(pk.load(open(proj_root + 'model/artist_decision_votes_train.pk', 'rb')))
year_recommender = ZeroScoreRecommender(pk.load(open(proj_root + 'model/year_decision_scores_train.pk', 'rb')))

engine = create_engine("postgresql://localhost/artdb")

style_name = ["Art Nouveau", "Baroque",
              "Expressionism", "Impressionism",
              "Neoclassicism", "Post-Impressionism",
              "Realism", "Romanticism",
              "Surrealism", "Symbolism"]


def get_style_name(style_id):
	return style_name[style_id]


@app.before_request
def before_request():
	"""
	This function is run at the beginning of every web request 
	(every time you enter an address in the web browser).
	We use it to setup a database connection that can be used throughout the request

	The variable g is globally accessible
	"""
	try:
		g.conn = engine.connect()
	except:
		print "uh oh, problem connecting to database"
		import traceback
		traceback.print_exc()
		g.conn = None


@app.teardown_request
def teardown_request(exception):
	"""
	At the end of the web request, this makes sure to close the database connection.
	If you don't the database could run out of memory!
	"""
	try:
		g.conn.close()
	except Exception as e:
		print e


@app.route('/')
def index():
	return render_template("index.html")


def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


class ImageContainer:
	def __init__(self, img_id, style, artist, year, name, url):
		self.img_id = img_id
		self.style = get_style_name(style)
		self.artist = artist
		self.name = name
		self.year = int(year) if year is not None else None
		self.url = url


def extract_feature(image_file_name):
	img = caffe.io.load_image(image_file_name)
	im = img[:, :, (2, 1, 0)] * 255

	def extract_(args):
		return args[0].extract(args[1])

	pool = ThreadPool(3)
	ft_vgg, ft_hog, ft_lbp = tuple(pool.map(extract_,
	                                        [(vgg_ft, img),
	                                         (hog_ft, im),
	                                         (lbp_ft, im)]))
	# ft_vgg = vgg_ft.extract(img)
	# ft_hog = hog_ft.extract(im)
	# ft_lbp = lbp_ft.extract(im)
	return {'vgg': ft_vgg, 'hog': ft_hog, 'lbp': ft_lbp}


def analyze_style(features):
	pred = style_svc.predict(features['vgg'])[0]
	votes = style_svc.get_votes(features['vgg'])
	rec_1 = style_recommender.recommend(votes, k=32)
	rec_2 = lbp_recommender.recommend(features['lbp'], k=16, from_list=rec_1)
	rec = hog_recommender.recommend(features['hog'], k=8, from_list=rec_2)
	return pred, rec, votes


def analyze_artist(features):
	pred = artist_svc.predict(features['vgg'])[0]
	votes = artist_svc.get_votes(features['vgg'])
	rec_1 = artist_recommender.recommend(votes, k=32)
	rec_2 = lbp_recommender.recommend(features['lbp'], k=16, from_list=rec_1)
	rec = hog_recommender.recommend(features['hog'], k=8, from_list=rec_2)
	return pred, rec, votes


def analyze_year(features):
	pred = year_svr.predict(features['vgg'])[0]
	votes = [pred]
	rec_1 = year_recommender.recommend(votes, k=32)
	rec_2 = lbp_recommender.recommend(features['lbp'], k=16, from_list=rec_1)
	rec = hog_recommender.recommend(features['hog'], k=8, from_list=rec_2)
	return pred, rec, votes


def get_result_id(src):
	return random.sample(range(0, 8000), 8)


def extract_artist_info(file_name):
	a = file_name.split('.')[0].split('/')[-1].split('_')
	artist = a[0].replace("-", " ").title()
	b = a[-1].split('-')
	year = None
	name = None
	if len(b) > 1 and b[-1].isdigit():
		for idx, n in enumerate(reversed(b)):
			if n.isdigit():
				if len(n) == 4:
					year = int(n)
					name = ' '.join(b[:len(b) - idx])
					break
			else:
				break
	name = ' '.join(b).title() if name is None else name
	return artist, name, year


def build_img_info(img_id, table_name):
	if img_id is not None:
		if table_name == "tests":
			cursor = g.conn.execute("SELECT * FROM tests WHERE id = %s", img_id)
		elif table_name == "imgs":
			cursor = g.conn.execute("SELECT * FROM imgs WHERE id = %s", img_id)
		else:
			return None
		entry = cursor.fetchone()
		if entry is not None:
			url = "/static/img/" + entry['name']
			artist, name, year = extract_artist_info(entry['name'])
			return ImageContainer(img_id, entry['style'], artist, year, name, url)
		cursor.close()


def parse_name(st):
	return ' '.join(st.split('-')).title()


def analyze_and_render(absolute_file_url, get_url, source=None):
	features = extract_feature(absolute_file_url)

	def build_(analyze_fn):
		return analyze_fn(features)

	pool = ThreadPool(3)
	styles, artists, years = tuple(pool.map(build_, [analyze_style, analyze_artist, analyze_year]))

	style_pred, style_rec, style_votes = styles
	artist_pred, artist_rec, artist_votes = artists
	year_pred, year_rec, year_votes = years

	style_results = [build_img_info(str(i), 'imgs') for i in style_rec]
	artist_results = [build_img_info(str(i), 'imgs') for i in artist_rec]
	year_results = [build_img_info(str(i), 'imgs') for i in year_rec]

	style_show = sorted([(style_name[i], v) for i, v in enumerate(style_votes)], key=lambda x: x[1])
	artist_show = artist_svc.get_top_k_classes(artist_votes, 10)
	artist_min = min([a[1] for a in artist_show]) - 10
	artist_show = sorted([(parse_name(a[0]), a[1] - artist_min) for a in artist_show], key=lambda x: x[1])

	norm = scipy.stats.norm(year_pred, 40)
	year_start = int(year_pred / 10) * 10 - 40
	year_show = [(str(y), norm.pdf(y) * 10) for y in range(year_start, year_start + 100, 10)]

	internal_source = source is not None
	if source is None:
		source = ImageContainer(0, style_pred, artist_pred.title(), year_pred, None, get_url)

	return render_template("show_image.html", source=source, internal_source=internal_source,
	                       styles=style_show, style_pics=style_results,
	                       artists=artist_show, artist_pics=artist_results,
	                       years=year_show, year_pics=year_results)


@app.route('/query_url')
def query_url():
	img_url = request.args.get("img_url")
	if img_url is None:
		return "invalid image url"
	try:
		return analyze_and_render(img_url, img_url)
	except Exception as e:
		print e


@app.route('/upload_file', methods=['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		f = request.files['file']
		if f and allowed_file(f.filename):
			filename = secure_filename(f.filename)
			absolute_file_url = os.path.join(app.config['UPLOAD_FOLDER'], filename)
			f.save(absolute_file_url)
			return analyze_and_render(absolute_file_url, '/static/upload/' + filename)


@app.route('/query')
def query():
	img_id = request.args.get("img_id")
	if img_id is not None:
		source = build_img_info(img_id, "tests")
		if source is None:
			return "img does not exist"
		filename = source.url[12:]
		absolute_file_url = img_dir + filename
		return analyze_and_render(absolute_file_url, '/static/img/' + filename, source)


@app.route('/user_grouping', methods=['GET', 'POST'])
def user_grouping():
	uid = request.args.get('uid')
	if uid is None:
		uid = 'Please type your name'
	img_id = random.randint(0, 1999)
	source = build_img_info(img_id, "tests")
	results_id = get_result_id(int(img_id))
	results_list = list()
	for i in results_id:
		img = build_img_info(str(i), "imgs")
		results_list.append(img)

	return render_template("user_grouping.html", source=source, results_list=results_list, uid=uid, all_id=results_id)


@app.route('/group_result', methods=['GET', 'POST'])
def group_result():
	selected = request.form.getlist('check')
	all_id = request.form.get('all_id')
	img_id = request.form.get('img_id')
	uid = request.form.get('uid')
	if img_id != None:
		g.conn.execute("INSERT INTO user_group (id, source_id, pics, uid, all_id) VALUES (DEFAULT, %s, %s, %s, %s);",
		               img_id, selected, uid, all_id)

	if uid != None:
		return redirect('/user_grouping?uid=' + uid)
	else:
		return redirect('/user_grouping')


@app.route('/user_test')
def get_test_img():
	img_id = random.randint(0, 1999)
	source = build_img_info(img_id, "tests")
	return render_template("user_test.html", source=source, img_id=img_id)


@app.route('/new_pair')
def new_pair():
	select = request.args.get('style')
	img_id = request.args.get('img_id')
	cursor = g.conn.execute("INSERT INTO user_conv (id, src, style) VALUES (DEFAULT, %s, %s);", img_id, select)
	return redirect('user_test')


if __name__ == "__main__":
	import click


	@click.command()
	@click.option('--debug', is_flag=True)
	@click.option('--threaded', is_flag=True)
	@click.argument('HOST', default='0.0.0.0')
	@click.argument('PORT', default=7788, type=int)
	def run(debug, threaded, host, port):
		"""
		This function handles command line parameters.
		Run the server using

				python server.py

		Show the help text using

				python server.py --help

		"""
		try:
			os.symlink(img_dir, static_dir + "/img")
		except OSError, e:
			if e.errno == errno.EEXIST:
				os.remove(static_dir + "/img")
				os.symlink(img_dir, static_dir + "/img")

		HOST, PORT = host, port
		print "running on %s:%d" % (HOST, PORT)
		app.run(host=HOST, port=PORT, debug=debug, threaded=threaded)


	run()
