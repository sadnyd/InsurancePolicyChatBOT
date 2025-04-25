from services.pdf_service import PDFExtractionService

# Change 'your_test.pdf' to the name of the PDF you want to test
pdf_path = "test.pdf"
service = PDFExtractionService(data_dir="./data")
service.extract_and_save(pdf_path)
