from pdf2image import convert_from_path
import os

# Define the path to the PDF file
pdf_path = r"Script\\Data.pdf"

# Output folder for images
output_folder = "pdf_images"
os.makedirs(output_folder, exist_ok=True)

# Convert PDF to images (each page as an image)
images = convert_from_path(pdf_path, dpi=300)  # Adjust DPI for quality

# Save each page as an image
for i, image in enumerate(images):
    image_path = os.path.join(output_folder, f"page_{i+1}.png")
    image.save(image_path, "PNG")

print(f"Converted {len(images)} pages to images in '{output_folder}'")
