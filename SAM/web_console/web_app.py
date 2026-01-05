import webbrowser
from threading import Timer
from flask import Flask, request, jsonify

app = Flask(__name__, static_folder="web_console", static_url_path="")


@app.route("/")
def index():
    # Flask will automatically serve static/index.html if requested at root
    return app.send_static_file("index.html")


received_texts = []


@app.route("/submit", methods=["POST"])
def submit():
    data = request.json  # expects JSON
    text = data.get("text", "")
    received_texts.append(text)
    print(text)
    return jsonify(status="ok", received=text)


@app.route("/received")
def get_received():
    return jsonify(received_texts)


counter = 0


@app.route("/data")
def data():
    global counter
    counter += 1
    return jsonify(counter=counter)


def open_browser():
    return
    # webbrowser.open("http://127.0.0.1:5000")


if __name__ == "__main__":
    # open browser a short moment after server starts
    Timer(1, open_browser).start()
    app.run(debug=True)

