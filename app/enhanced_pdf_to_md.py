import pymupdf as fitz
import os
import ollama

# Helper function to extract text from PDF blocks
def _get_text_from_block(block: dict) -> str:
    text = ""
    if "lines" in block:
        for line in block["lines"]:
            for span in line["spans"]:
                text += span["text"]
            text += "\n"
    return text

# Main function
def pdf_to_markdown(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    markdown_text = ""

    output_dir = "files"
    os.makedirs(output_dir, exist_ok=True)

    img_count = 0
    
    # Process each page in the PDF document
    for page_number in range(len(doc)):
        page = doc.load_page(page_number)
        markdown_text += f"## Page {page_number + 1}\n\n"

        page_dict = page.get_text("dict", sort=True)
        blocks = page_dict.get("blocks", [])
        
        # Process each block (text or image) on the page
        for i, block in enumerate(blocks):
            # Handle text blocks
            if block["type"] == 0:
                markdown_text += _get_text_from_block(block)
                markdown_text += "\n"
            
            # Handle image blocks
            elif block["type"] == 1:
                img_count += 1
                print(f"\n--- Processing Image {img_count} on Page {page_number + 1} ---")

                # Gather context from surrounding text blocks
                before_text = ""
                for j in range(i - 1, -1, -1):
                    if blocks[j]["type"] == 0:
                        before_text = _get_text_from_block(blocks[j]) + before_text
                
                after_text = ""
                for j in range(i + 1, len(blocks)):
                    if blocks[j]["type"] == 0:
                        after_text += _get_text_from_block(blocks[j])

                before_context = before_text.strip()[-200:]
                after_context = after_text.strip()[:200]
                context = (before_context + " " + after_context).strip()
                print(f"  [Context] Gathered {len(context)} characters: \"{context[:100]}...\"")
                
                # Process and save the image
                try:
                    image_bytes = block["image"]
                    image_ext = block["ext"]
                    
                    img_count += 1
                    image_filename = f"{os.path.basename(pdf_path).replace('.pdf', '')}{img_count}.{image_ext}"
                    image_path = os.path.join(output_dir, image_filename)
                    
                    print(f"  [Image] Saving image to: {image_path}")
                    with open(image_path, "wb") as img_file:
                        img_file.write(image_bytes)

                    # Generate AI description for the image using context
                    try:
                        vision_prompt = f'Provide a concise, one-sentence description for the following image or icon. If an image, the caption should describe what it shows and why it is important. If an icon, the caption should include what it is used for, not what it looks like. Choose either Before Context or After Context, but not both. Before Context: "{before_context} After Context: {after_context}"'
                        if not context.strip():
                            vision_prompt = 'Provide a concise, one-sentence description for the following image or icon. If an image, the caption should describe what it shows and why it is important. If an icon, the caption should include what it is used for, not what it looks like.'

                        print("  [AI] Calling vision model for description...")
                        vision_response = ollama.chat(
                            model='gemma3:4b',
                            messages=[{
                                'role': 'user',
                                'content': vision_prompt,
                                'images': [image_path]
                            }]
                        )
                        image_desc = vision_response['message']['content'].strip()
                        print(f"  [AI] Generated description: \"{image_desc}\"")

                        normalized_image_path = image_path.replace('\\', '/')
                        markdown_text += f"![{image_desc}]({normalized_image_path})\n\n"
                        print(f"--- Finished Processing Image {img_count} ---")

                    except Exception as e:
                        print(f"Warning: Could not process image with Ollama on page {page_number + 1}. Error: {e}")
                        normalized_image_path = image_path.replace('\\', '/')
                        markdown_text += f"![{os.path.basename(normalized_image_path)}]({normalized_image_path})\n\n"

                except Exception as e:
                    print(f"Warning: Could not extract an image on page {page_number + 1}. Error: {e}")

    # Save the generated markdown to a file
    output_md_path = os.path.join(output_dir, os.path.basename(pdf_path).replace(".pdf", ".md"))
    
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(markdown_text)
    
    return output_md_path

# # Test code for the PDF to Markdown converter
# if __name__ == "__main__":
#     test_pdf_path = "files/Xgig-Serialytics.pdf" 

#     if not os.path.exists(test_pdf_path):
#         print(f"Test PDF file '{test_pdf_path}' does not exist. Please provide a valid PDF file for testing.")
#     else:
#         print("Starting PDF to Markdown conversion with AI image descriptions...")
#         output_md_path = pdf_to_markdown(test_pdf_path)
        
#         if os.path.exists(output_md_path):
#             print(f"\nMarkdown file created successfully: {output_md_path}")
#         else:
#             print("\nFailed to create Markdown file.")