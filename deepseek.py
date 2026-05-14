# pip install torch==2.6.0 transformers==4.46.3 tokenizers==0.20.3 einops addict easydict --quiet
# pip install flash-attn==2.7.3 --no-build-isolation --quiet
# pip install pdf2image pillow --quiet
# apt-get install -y poppler-utils > /dev/null
import os

# Avoid importing TensorFlow; this script uses PyTorch only.
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import tempfile
from io import StringIO
from pathlib import Path

from transformers import AutoModel, AutoTokenizer
import torch
from pdf2image import convert_from_path
from PIL import Image

MODEL_NAME = "deepseek-ai/DeepSeek-OCR"
SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
SUPPORTED_PDF_EXTS = {".pdf"}

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModel.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    use_safetensors=True,
)
model = model.eval().cuda().to(torch.bfloat16)


def extract_markdown_from_output(output_text):
    """Extract clean markdown from the model's output text"""
    lines = output_text.split('\n')
    markdown_lines = []
    
    for line in lines:
        # Skip lines with special tokens and detection coordinates
        if '<|ref|>' in line or '<|det|>' in line or '<|/ref|>' in line or '<|/det|>' in line:
            # Extract the actual text between tags
            # Pattern: <|ref|>type<|/ref|><|det|>coords<|/det|>\nActual text
            continue
        elif line.startswith('==') or 'image size:' in line or 'tokens:' in line or 'compression ratio:' in line:
            continue
        elif line.strip():
            markdown_lines.append(line)
    
    return '\n'.join(markdown_lines)


def capture_model_output(func, *args, **kwargs):
    """Capture stdout from model.infer() call"""
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()
    
    try:
        result = func(*args, **kwargs)
        output = captured_output.getvalue()
    finally:
        sys.stdout = old_stdout
    
    return result, output


def _resolve_output_md_path(input_path: Path, output_path) -> Path:
    if output_path is None:
        raise ValueError("output_path is required")
    output_path = Path(output_path)
    if output_path.suffix.lower() == ".md":
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path / f"{input_path.stem}.md"


def _images_to_pdf(image_paths: list[Path], pdf_path: Path) -> None:
    if not image_paths:
        raise ValueError("No images provided for PDF conversion")

    images = []
    for path in image_paths:
        with Image.open(path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            images.append(img.copy())

    first, rest = images[0], images[1:]
    first.save(pdf_path, "PDF", save_all=True, append_images=rest)


def _infer_image_markdown(
    image_path: str,
    temp_dir: str,
    base_size: int,
    image_size: int,
) -> str:
    prompt = "<image>\n<|grounding|>Convert the document to markdown."
    _, output = capture_model_output(
        model.infer,
        tokenizer,
        prompt=prompt,
        image_file=image_path,
        output_path=temp_dir,
        base_size=base_size,
        image_size=image_size,
        crop_mode=True,
        save_results=False,
        test_compress=True,
    )
    return extract_markdown_from_output(output)


def ocr_pdf_to_markdown(
    pdf_path,
    output_md_path,
    dpi=200,
    base_size=1024,
    image_size=640,
):
    """
    Convert a scanned PDF to markdown using DeepSeek-OCR
    
    Args:
        pdf_path: Path to the input PDF file
        output_md_path: Markdown file path or output directory
        dpi: DPI for PDF to image conversion (higher = better quality, slower)
        base_size: Base size parameter for OCR model
        image_size: Image size parameter for OCR model
    """
    pdf_path = Path(pdf_path)
    output_md_path = _resolve_output_md_path(pdf_path, output_md_path)
    print(f"Processing PDF: {pdf_path}")
    
    # Convert PDF to images
    print("Converting PDF pages to images...")
    images = convert_from_path(str(pdf_path), dpi=dpi)
    print(f"Found {len(images)} pages")
    
    # Create temporary directory for intermediate images
    with tempfile.TemporaryDirectory() as temp_dir:
        all_markdown = []
        
        for page_num, image in enumerate(images, 1):
            print(f"\nProcessing page {page_num}/{len(images)}...")
            
            # Save temporary image
            temp_image_path = Path(temp_dir) / f"page_{page_num}.png"
            image.convert("RGB").save(temp_image_path, "PNG")
            
            try:
                markdown_text = _infer_image_markdown(
                    str(temp_image_path),
                    temp_dir,
                    base_size,
                    image_size,
                )
                
                if markdown_text.strip():
                    all_markdown.append(f"# Page {page_num}\n\n{markdown_text}\n\n---\n")
                    print(
                        f"[OK] Extracted {len(markdown_text)} characters from page {page_num}"
                    )
                else:
                    all_markdown.append(f"# Page {page_num}\n\n[No text extracted from this page]\n\n---\n")
                    print(f"[WARN] No text extracted from page {page_num}")
                
            except Exception as e:
                print(f"[ERROR] Error processing page {page_num}: {e}")
                all_markdown.append(f"# Page {page_num}\n\n[Error processing this page: {e}]\n\n---\n")
    
    # Save combined markdown
    print(f"\nSaving markdown to: {output_md_path}")
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_markdown))
    
    print("[OK] Done")
    return str(output_md_path)


def ocr_image_to_markdown(
    image_path,
    output_md_path,
    dpi=200,
    base_size=1024,
    image_size=640,
):
    """
    Convert an image to markdown using DeepSeek-OCR
    Image inputs are converted into a single-page PDF before OCR.

    Args:
        image_path: Path to the input image file
        output_md_path: Markdown file path or output directory
        dpi: DPI for PDF to image conversion (higher = better quality, slower)
        base_size: Base size parameter for OCR model
        image_size: Image size parameter for OCR model
    """
    image_path = Path(image_path)
    output_md_path = _resolve_output_md_path(image_path, output_md_path)
    print(f"Processing image: {image_path}")

    with tempfile.TemporaryDirectory() as temp_dir:
        pdf_path = Path(temp_dir) / f"{image_path.stem}.pdf"
        _images_to_pdf([image_path], pdf_path)
        return ocr_pdf_to_markdown(
            pdf_path,
            output_md_path,
            dpi=dpi,
            base_size=base_size,
            image_size=image_size,
        )


def batch_process_inputs(
    input_dir,
    output_dir,
    dpi=200,
    base_size=1024,
    image_size=640,
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(
        [
            path
            for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_PDF_EXTS
        ]
    )
    image_files = sorted(
        [
            path
            for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTS
        ]
    )
    print(f"Found {len(pdf_files)} PDFs and {len(image_files)} images")

    outputs = []
    for pdf_file in pdf_files:
        outputs.append(
            ocr_pdf_to_markdown(
                pdf_file,
                output_dir,
                dpi=dpi,
                base_size=base_size,
                image_size=image_size,
            )
        )

    if image_files:
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle_pdf = Path(temp_dir) / "images_bundle.pdf"
            _images_to_pdf(image_files, bundle_pdf)
            outputs.append(
                ocr_pdf_to_markdown(
                    bundle_pdf,
                    output_dir / "images_bundle.md",
                    dpi=dpi,
                    base_size=base_size,
                    image_size=image_size,
                )
            )

    return outputs


def process_path(
    input_path,
    output_path,
    dpi=200,
    base_size=1024,
    image_size=640,
):
    input_path = Path(input_path)

    if input_path.is_dir():
        return batch_process_inputs(
            input_path,
            output_path,
            dpi=dpi,
            base_size=base_size,
            image_size=image_size,
        )

    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    suffix = input_path.suffix.lower()
    if suffix in SUPPORTED_PDF_EXTS:
        return [
            ocr_pdf_to_markdown(
                input_path,
                output_path,
                dpi=dpi,
                base_size=base_size,
                image_size=image_size,
            )
        ]
    if suffix in SUPPORTED_IMAGE_EXTS:
        return [
            ocr_image_to_markdown(
                input_path,
                output_path,
                dpi=dpi,
                base_size=base_size,
                image_size=image_size,
            )
        ]

    raise ValueError(f"Unsupported file type: {input_path.suffix}")


if __name__ == "__main__":
    # Example usage: set input path to a PDF, image, or directory.
    input_path = "/home/junaid/Document-Understanding/data/input"
    output_path = "/home/junaid/Document-Understanding/data/output"
    process_path(input_path, output_path, dpi=200)