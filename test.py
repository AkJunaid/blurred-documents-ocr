# ============================================================
# DeepSeek-OCR PDF Scanner for Local Machine
# ============================================================

# ============================================================
# Imports
# ============================================================

import os
import torch
from PIL import Image
from pdf2image import convert_from_path
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# PDF PATH
# ============================================================

PDF_PATH = "/home/junaid/Document-Understanding/scan_000.pdf"

# ============================================================
# Convert PDF to Images
# ============================================================

print("Converting PDF pages to images...")

pages = convert_from_path(
    PDF_PATH,
    dpi=300
)

print(f"Total Pages: {len(pages)}")

# ============================================================
# Load DeepSeek OCR Model
# ============================================================

MODEL_NAME = "deepseek-ai/DeepSeek-VL2-small"

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
).eval()

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Model loaded on:", device)

# ============================================================
# OCR Function
# ============================================================

def run_ocr(image: Image.Image):

    prompt = """
Extract all readable text from this document image.
Preserve formatting and line breaks.
Return only the extracted text.
"""

    conversation = [
        {
            "role": "User",
            "content": "<image>\n" + prompt,
            "images": [image]
        },
        {
            "role": "Assistant",
            "content": ""
        }
    ]

    inputs = tokenizer(
        conversations=conversation,
        images=[image],
        return_tensors="pt",
        force_batchify=True
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False
        )

    result = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return result

# ============================================================
# Process All Pages
# ============================================================

all_text = []

for idx, page in enumerate(pages):

    print(f"\nProcessing Page {idx+1}/{len(pages)}")

    text = run_ocr(page)

    cleaned = text.strip()

    all_text.append(
        f"\n\n================ PAGE {idx+1} ================\n\n{cleaned}"
    )

# ============================================================
# Save Output
# ============================================================

final_text = "\n".join(all_text)

OUTPUT_FILE = "/home/junaid/Document-Understanding/deepseek_ocr_output.txt"

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(final_text)

print("\nOCR COMPLETE")
print("Saved to:", OUTPUT_FILE)

# ============================================================
# Preview
# ============================================================

print("\n================ OCR PREVIEW ================\n")
print(final_text[:5000])