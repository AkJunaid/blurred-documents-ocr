from pdf2image import convert_from_path
import cv2
import numpy as np
import pytesseract
import os
import json
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def deskew(image):
    coords = np.column_stack(np.where(image == 0))
    if len(coords) == 0:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def process_pdf(pdf_path):
    print(f'Converting PDF {pdf_path} to images...')
    images = convert_from_path(pdf_path, dpi=300)
    results = []
    os.makedirs('temp_pages', exist_ok=True)
    for i, img in enumerate(images):
        temp_path = f'temp_pages/page_{i}.jpg'
        img.save(temp_path, 'JPEG')
        processed = preprocess_image(temp_path)
        raw_text = perform_ocr(processed)
        corrected = context_aware_correction(raw_text)
        data = extract_structured_data(corrected)
        results.append({'page': i+1, 'raw': raw_text, 'corrected': corrected, 'data': data})
    return results

def process_file(file_path):
    if file_path.lower().endswith('.pdf'):
        return process_pdf(file_path)
    else:
        processed = preprocess_image(file_path)
        raw_text = perform_ocr(processed)
        corrected = context_aware_correction(raw_text)
        return [{'page': 1, 'raw': raw_text, 'corrected': corrected, 'data': extract_structured_data(corrected)}]

def preprocess_image(image_path):
    print("Stage 1: Preprocessing Image...")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")

    # 1. Proper grayscale conversion (not just one channel)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Upscale for better OCR (2x)
    scale = 2
    gray = cv2.resize(gray, None, fx=scale, fy=scale, 
                      interpolation=cv2.INTER_CUBIC)

    # 3. Light denoise only
    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    # 4. Adaptive threshold (better than Otsu for uneven lighting)
    binary = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=15
    )

    # 5. Deskew
    deskewed = deskew(binary)

    return deskewed

def perform_ocr(processed_image):
    print("Stage 2: OCR...")
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(
        processed_image, 
        config=custom_config,
        lang='eng'
    )
    return text

def is_garbage_text(text):
    if not text.strip():
        return True
    
    alphanumeric_count = sum(1 for char in text if char.isalnum())
    total_count = len(text.replace(" ", "").replace("\n", ""))
    
    if total_count == 0:
        return True
        
    ratio = alphanumeric_count / total_count
    return ratio < 0.6  # Less than 60% alphanumeric means likely garbage

def context_aware_correction(raw_text):
    print("Stage 3: Context-aware correction...")
    
    if is_garbage_text(raw_text):
        print("Warning: Raw OCR is largely garbage/noise. Skipping LLM correction to prevent hallucinations.")
        return "[IMAGE QUALITY TOO LOW TO PRODUCE MEANINGFUL TEXT]"

    prompt = f"""This text was extracted from a legal document via OCR and may contain misread characters. 
    Correct likely OCR errors while preserving meaning and legal terminology.
    If a word is completely illegible and cannot be guessed from context, mark it with [?].
    Preserve the original formatting and line breaks as much as possible. Do not rewrite or summarize.
    Raw OCR Text:
    {raw_text}
    """
    
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    return completion.choices[0].message.content

def extract_structured_data(corrected_text):
    print("Stage 4: Structured output...")
    
    if corrected_text == "[IMAGE QUALITY TOO LOW TO PRODUCE MEANINGFUL TEXT]":
        return {"error": "Image quality too low"}
    return {"status": "success", "text": corrected_text[:50]}

if __name__ == '__main__':
    pdf_path = '/home/junaid/Document-Understanding/scan_000.pdf'
    print(f'Processing {pdf_path}...')
    try:
        results = process_file(pdf_path)
        with open('document_data.json', 'w') as f:
            json.dump(results, f, indent=4)
        print('Processing complete. Results saved to document_data.json')
    except Exception as e:
        print(f'Error processing file: {e}')
