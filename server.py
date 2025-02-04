from flask import Flask, request, jsonify, Response
import requests
from flask_cors import CORS
import torch

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Ollama API Endpoint
OLLAMA_URL = "http://localhost:11434/api/generate"

# Check CUDA availability
use_cuda = torch.cuda.is_available()
device = "cuda" if use_cuda else "cpu"
if use_cuda:
    print(f"CUDA is available: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available, using CPU.")

@app.route("/generate", methods=["POST"])
def generate():
    try:
        # Retrieve the prompt and model from the request
        data = request.json
        prompt = data.get("prompt", "")
        model = data.get("model", "llama3.2")
        stream = data.get("stream", False)

        # Dynamically check CUDA availability for each request
        use_cuda = torch.cuda.is_available()
        device = "cuda" if use_cuda else "cpu"

        # Prepare payload for Ollama
        payload = {
            "model": model,
            "prompt": prompt,
            "keep_alive": 0,
            "stream": stream,
            "device": device
        }

        # Send request to Ollama
        response = requests.post(OLLAMA_URL, json=payload, stream=stream)
        response.raise_for_status()

        if stream:
            # Stream response as it comes
            def generate_stream():
                for chunk in response.iter_content(chunk_size=8192):
                    yield chunk

            return Response(generate_stream(), content_type="application/json")

        return jsonify(response.json())  # Return full response

    except requests.exceptions.Timeout:
        return jsonify({"error": "Request timed out"}), 504
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Request error: {str(e)}"}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)  # Enable debug for better logs
