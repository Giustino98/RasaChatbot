# This is a sample Python script.

# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import cv2

from actions import actions
from flask import Response, Flask, render_template


app = Flask(__name__)


# app.run(host="127.0.0.1", port=8000, debug=True,
#           threaded=True, use_reloader=False)

def gen():
    while True:
        # encodedImage = actions.ActionStartWebcam.encodedImage
        frame = actions.ActionStartWebcam.encodedImage
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(frame) + b'\r\n')


@app.route("/video_feed")
def video_feed():
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/")
def index():
    # return the rendered template
    return render_template('index.html')


if __name__ == '__main__':


    endpoint.run(
        "actions",  # action package
        constants.DEFAULT_SERVER_PORT,  # port of the web server
        "*",  # cors origins
    )

    app.run(debug=True)
