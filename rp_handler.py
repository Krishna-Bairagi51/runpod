import runpod
import torch
import base64
import io
import json
import re
from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

# Load the Gemma model at startup
print("Loading Gemma-3-12b-it model...")
model_id = "google/gemma-3-12b-it"

model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="eager"
).eval()
processor = AutoProcessor.from_pretrained(model_id)

device = next(model.parameters()).device
print(f"Model loaded successfully on: {device}")

def handler(event):
    print("Worker Start")
    input_data = event.get('input', {})
    prompt = input_data.get('prompt', '')
    image_data = input_data.get('image', None)  # Expecting a base64‑encoded image string

    # Process Image
    images = []
    if image_data:
        try:
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            images.append(image)
            print("Image received and decoded.")
        except Exception as e:
            print("Error decoding image:", e)
            return {"error": "Invalid image data."}

    # Optimized Structured Prompt
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are an advanced AI trained for retail product recognition. Given images of a product, identify and extract structured details such as product name, category, parent category, brand, pack size, weight, MRP, and size."
                }
            ]
        },
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": image} for image in images],
                {
                    "type": "text",
                    "text" : (
                        "Analyze the given images and return a structured JSON response containing the following details:\n"
                        "- **product_name**: The exact name of the product.\n"
                        "- **categ_name**: The specific category this product belongs to.\n"
                        "- **parent_categ_name**: The broader category under which the product falls.\n"
                        "- **brand_name**: The brand of the product.\n"
                        "- **pack_size**: The package size (return 0.0 if not available).\n"
                        "- **weight**: Provide the weight of the product if visible, else return 0.0.\n"
                        "- **mrp**: The maximum retail price of the product.\n"
                        "- **size**: Provide the size of the product if visible, else 0.0.\n"
                        "- **description**: A concise and informative description of the product, including its key features, usage, and any unique aspects.\n\n"
                        "Return only the JSON object without any additional text."
                    )
                }
            ]
        }
    ]

    # Process Input
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(device)

    # Generate Response
    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=256, do_sample=False)

    # Decode Response
    decoded = processor.decode(generation[0], skip_special_tokens=True).strip()

    # Extract JSON using regex
    json_match = re.search(r'\{.*\}', decoded, re.DOTALL)

    if json_match:
        json_output = json_match.group(0)
        try:
            json_data = json.loads(json_output)  # Validate JSON
            print("Generated JSON response:", json.dumps(json_data, indent=4))
            return json_data  # Return structured JSON response
        except json.JSONDecodeError:
            print("❌ Invalid JSON output. Raw output:", decoded)
            return {"error": "Invalid JSON format", "raw_output": decoded}
    else:
        print("❌ No valid JSON detected. Raw output:", decoded)
        return {"error": "No JSON detected", "raw_output": decoded}

if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})
