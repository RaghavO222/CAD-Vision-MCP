from anthropic import AsyncAnthropic
from typing import List, Tuple
import base64
import io
from PIL import Image
from dotenv import load_dotenv
import os
# Load environment variables from the .env file in the same directory
load_dotenv()

# Fetch the API key from the environment
api_key = os.getenv("ANTHROPIC_API_KEY")

if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in .env file.")

anthropic_client = AsyncAnthropic(
    api_key=api_key,
    timeout=300.0
)

# Define the exact beta version string required by the SDK
FILES_API_BETA = ["files-api-2025-04-14"]

def scale_image_for_claude(image_bytes: bytes, max_dim: int = 7500) -> bytes:
    """
    Checks if an image exceeds Claude's 8000px maximum limit on either dimension.
    If it does, downscales it proportionally while maintaining aspect ratio.
    """
    try:
        # img = Image.open(io.BytesIO(image_bytes))
        # width, height = img.size
        
        # if width > max_dim or height > max_dim:
        #     if width > height:
        #         new_width = max_dim
        #         new_height = int(height * (max_dim / width))
        #     else:
        #         new_height = max_dim
        #         new_width = int(width * (max_dim / height))
                
        #     print(f"Image dimensions ({width}x{height}) exceed limit. Resizing to {new_width}x{new_height}...")
            
        #     img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
        #     output_bytes = io.BytesIO()
        #     img_format = img.format if img.format else "PNG"
        #     img.save(output_bytes, format=img_format)
        #     return output_bytes.getvalue()
        img = Image.open(io.BytesIO(image_bytes))
        img.load()  # forces decode now, so corrupt files raise here, not later

        # Convert CMYK / P / RGBA / etc. to RGB
        if img.mode != "RGB":
            img = img.convert("RGB")

        width, height = img.size
        if width > max_dim or height > max_dim:
            if width > height:
                new_size = (max_dim, int(height * (max_dim / width)))
            else:
                new_size = (int(width * (max_dim / height)), max_dim)
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        output = io.BytesIO()
        img.save(output, format="JPEG", quality=90)
        return output.getvalue(), "image/jpeg"
    except Exception as e:
        print(f"Failed to check/resize image: {e}")
        
    return image_bytes

async def call_claude(
    meta_prompt: str, 
    files_data: List[Tuple[str, bytes, str]] = None
) -> str:
    """
    Sends a request to Claude with an optional list of files (PDFs or Images).
    
    :param meta_prompt: The core text prompt instruction.
    :param files_data: A list of tuples containing (filename, file_bytes, media_type).
    """
    if files_data is None:
        files_data = []
    
    message_content = []
    has_pdf = False

    # 1. Process files dynamically
    for filename, file_bytes, media_type in files_data:
        
        # Handle PDFs via the Files API
        if media_type == "application/pdf":
            has_pdf = True
            uploaded_file = await anthropic_client.beta.files.upload(
                file=(filename, file_bytes, "application/pdf"),
                betas=FILES_API_BETA
            )
            message_content.append({
                "type": "document",
                "source": {
                    "type": "file",
                    "file_id": uploaded_file.id
                },
                "title": filename
            })
            
        # Handle Images (PNG, JPG, JPEG, WebP) inline
        elif media_type in ["image/png", "image/jpeg", "image/webp"]:
            # file_bytes = scale_image_for_claude(file_bytes, max_dim=7500)
            file_bytes, media_type = scale_image_for_claude(file_bytes, max_dim=7500)

            base64_image = base64.b64encode(file_bytes).decode("utf-8")
            message_content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64_image
                }
            })

    # 2. Append the main prompt text
    message_content.append({
        "type": "text",
        "text": meta_prompt
    })
    
    # 3. Prepare call parameters (This fixes the 'NoneType' error)
    params = {
        "model": "claude-opus-4-8",
        "max_tokens": 60000,
        "messages": [
            {
                "role": "user",
                "content": message_content
            }
        ],
        # Swapping to adaptive mode removes the deprecation warning entirely
        "thinking": {
            "type": "adaptive" 
        },
        "output_config": {
            "effort": "high"  # Uses the native reasoning engine efficiently
        }
    }

    # Only include betas when using PDFs (important!)
    if has_pdf:
        params["betas"] = FILES_API_BETA

    # 4. Call Claude

    response = await anthropic_client.beta.messages.create(**params)
    
    # 5. FIXED HERE: Dynamically find and extract the text block block

    for block in response.content:
        if block.type == "text":
            return block.text


    
    raise ValueError("The API response did not contain a valid text block.")