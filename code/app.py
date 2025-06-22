from flask import Flask, request, jsonify
from analyze import analyze_video
import tempfile

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze():
    video = request.files.get('video')
    if not video:
        return jsonify({'error': 'No video uploaded'}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp:
        video.save(temp.name)
        result = analyze_video(temp.name)

    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
