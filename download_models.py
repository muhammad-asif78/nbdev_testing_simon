from paddleocr import PaddleOCR

# Initialize PaddleOCR to download models
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=True,
    lang='en'
)
print("PaddleOCR models downloaded successfully")