# I will use this Flask API Server Later to deploy  

from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_ollama import ChatOllama
import torch 

# Check CUDA availability
use_cuda = torch.cuda.is_available()
device = "cuda" if use_cuda else "cpu"
if use_cuda:
    print(f"CUDA is available: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available, using CPU.")

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Initialize the LLM with direct access to Ollama
local_llm = "llama3.2"
llm = ChatOllama(model=local_llm, temperature=0)

@app.route("/generate", methods=["POST"])
def generate():
    try:
        # Retrieve prompt from the request
        data = request.json
        prompt = data.get("prompt", "")

        # Generate response
        response = llm.invoke(prompt).content
        return jsonify({"response": response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
