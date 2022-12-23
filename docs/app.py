import logging

from flask import Flask
from flask_cors import *

app = Flask(__name__)
CORS(app, resources=r'/*', supports_credentials=True)

app = Flask(__name__, static_url_path='/', static_folder='build/html/')


@app.route('/')
@app.route('/<path:path>')
def serve_sphinx_docs(path='index.html'):
    return app.send_static_file(path)


if __name__ != "__main__":
    gunicorn_logger = logging.getLogger("gunicorn.error")
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)
