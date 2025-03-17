import runpod
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import io
import base64

# Load the MiniCPM-o-2_6 model at startup
print("Loading MiniCPM-o-2_6 model...")
model = AutoModel.from_pretrained(
    'openbmb/MiniCPM-o-2_6',
    trust_remote_code=True,
    attn_implementation='sdpa',  # You can also use 'flash_attention_2' if available
    torch_dtype=torch.bfloat16,
    init_vision=True,   # enable vision support
    init_audio=False,   # disable audio (if not needed)
    init_tts=False      # disable TTS (if not needed)
)
model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-o-2_6', trust_remote_code=True)
print("Model loaded successfully.")

def handler(event):
    print("Worker Start")
    input_data = event.get('input', {})
    prompt = input_data.get('prompt', '')
    image_data = input_data.get('image', None)  # Expecting a base64‑encoded image string

    # If an image is provided, decode it and build the message list
    if image_data:
        try:
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            print("Image received and decoded.")
            msgs = [{'role': 'user', 'content': [image, prompt]}]
        except Exception as e:
            print("Error decoding image:", e)
            return {"error": "Invalid image data."}
    else:
        print("No image provided; using prompt only.")
        msgs = [{'role': 'user', 'content': [prompt]}]

    # Generate a response from the model using chat inference
    response = model.chat(
        msgs=msgs,
        tokenizer=tokenizer,
        max_new_tokens=256  # Adjust token limit as needed
    )
    print("Generated response:", response)
    return {"response": response}

if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})
