from docling.document_converter import DocumentConverter

if __name__ == "__main__":
    converter = DocumentConverter()
    result = converter.convert("/app/sample.pdf")
    print(result.document.export_to_markdown())
