import flask
import optparse
import tornado.wsgi
import tornado.httpserver

# Obtain the flask app object
app = flask.Flask(__name__)

@app.route('/')
def index():
    return flask.render_template('index.html')

@app.route('/<path:path>')
def serve_page(path):
    return flask.send_from_directory('results', path)

def start_tornado(app, port=5000):
    http_server = tornado.httpserver.HTTPServer(
        tornado.wsgi.WSGIContainer(app))
    http_server.listen(port)
    print("Tornado server starting on port {}".format(port))
    tornado.ioloop.IOLoop.instance().start()

def start_from_terminal(app):
    """
    Parse command line options and start the server.
    """
    parser = optparse.OptionParser()
    parser.add_option(
        '-p', '--port',
        help="which port to serve content on",
        type='int', default=5000)

    opts, args = parser.parse_args()

    start_tornado(app, opts.port)

if __name__ == '__main__':
    start_from_terminal(app)
