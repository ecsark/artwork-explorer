#!/usr/bin/env python2.7

import os
import errno
from flask import Flask, g, request, render_template
from sqlalchemy import *
import random

tmpl_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
img_dir = "/Volumes/Alex/Columiba_course/Visual_DB/Project/database/wikiart/images/"

app = Flask(__name__, template_folder=tmpl_dir)

app.secret_key = os.urandom(24)


DATABASEURI = "postgresql://localhost/artdb"
engine = create_engine(DATABASEURI)

class img_info:
  def __init__(self, img_id, style, name, url):
    self.img_id = img_id
    self.style = style
    self.name = name
    self.url = url

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
    import traceback; traceback.print_exc()
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
    pass

@app.route('/')
def index():
  return render_template("index.html")

def get_result_id(src):
  print src
  return random.sample(range(0, 8000), 6)

  
def build_img_info(img_id, table):
  if img_id != None:
    if table == "tests":
      cursor = g.conn.execute("SELECT * FROM tests WHERE id = %s", img_id)
    elif table == "imgs":
      cursor = g.conn.execute("SELECT * FROM imgs WHERE id = %s", img_id)
    else:
      return None   
    entry = cursor.fetchone() 
    if entry != None:
      url = "/static/img/" + entry['name']
      return img_info(img_id, entry['style'], entry['name'], url)
  cursor.close()
  return None   

@app.route('/query')
def query():
  img_id = request.args.get("img_id")
  if img_id != None:
    source = build_img_info(img_id, "tests")
    if source != None:
      results_id = get_result_id(int(img_id))
      results_list = list()
      for i in results_id:
        img = build_img_info(str(i), "imgs")
        results_list.append(img)
      return render_template("show_image.html", source = source, results_list= results_list)
    else:
      return "img does not exist" 


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
