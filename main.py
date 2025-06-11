import json
import time
import logging
import base64
import requests
from io import BytesIO
from typing import Dict, Any, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import aiohttp
import ssl
import certifi
from PIL import Image
import tempfile
import os
from together import Together
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Pydantic Models
class BannerRequest(BaseModel):
    prompt: str
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 50
    guidance_scale: float = 5.0
    seed: Optional[int] = 0
    return_format: str = "json"  # "json", "image", or "both"

class BannerResponse(BaseModel):
    status: str
    request_id: str
    structured_data: Optional[Dict] = None
    flux_prompt: Optional[str] = None
    short_prompt: Optional[str] = None
    image_base64: Optional[str] = None
    download_url: Optional[str] = None
    processing_time: float
    error: Optional[str] = None

class StatusResponse(BaseModel):
    status: str
    message: str
    timestamp: str

# Global storage for async jobs and generated images
processing_jobs = {}
generated_images = {}

class CompleteBannerPipeline:
    def __init__(self, together_api_key: str, fireworks_api_key: str):
        if not together_api_key or not fireworks_api_key:
            raise ValueError("Both API keys are required")
        self.together_api_key = together_api_key
        self.fireworks_api_key = fireworks_api_key
        self.together_base_url = "https://api.together.xyz/v1/chat/completions"
        self.fireworks_image_url = "https://api.fireworks.ai/inference/v1/workflows/accounts/fireworks/models/flux-1-dev-fp8/text_to_image"
        self.model = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"

        # Initialize Together client
        try:
            self.together_client = Together()
            logger.info("✅ Together client initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Together client: {e}")
            raise

        self.color_descriptors = {
            "Red": "vibrant red", "Green": "festive green", "Blue": "deep blue",
            "Yellow": "bright yellow", "Orange": "warm orange", "Purple": "rich purple",
            "Pink": "soft pink", "White": "clean white", "Black": "bold black",
            "Brown": "warm brown", "Grey": "neutral grey"
        }

        self.mood_descriptors = {
            "Calm": "serene and peaceful", "Energetic": "dynamic and vibrant",
            "Luxurious": "elegant and premium", "Playful": "fun and engaging",
            "Professional": "clean and corporate", "Cozy": "warm and inviting"
        }

        self.theme_descriptors = {
            "Modern": "contemporary and sleek", "Classic": "timeless and traditional",
            "Retro": "vintage-inspired", "Minimalist": "clean and uncluttered",
            "Corporate": "professional and structured", "Festive": "celebratory and joyful"
        }

    def generate_short_prompt(self, user_prompt: str) -> str:
        system_instruction = (
            "You are a prompt optimization assistant. Your task is to rewrite the user prompt "
            "into a short, refined, and vivid prompt suitable for a text-to-image generation model. "
            "Keep it concise and structured."
        )
        try:
            response = self.together_client.chat.completions.create(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                max_tokens=200,
                top_p=0.9
            )
            short_prompt = response.choices[0].message.content.strip()
            logger.info(f"Generated short prompt: {short_prompt[:100]}...")
            return short_prompt
        except Exception as e:
            logger.error(f"Failed to generate short prompt: {e}")
            return user_prompt

    def create_system_prompt(self) -> str:
        schema = {
            "Dominant colors": "Red/Yellow/Blue/Green/Orange/Purple/Pink/White/Black/Brown/Grey/Gold/Silver/Teal/Magenta",
            "Secondary colors": "Red/Yellow/Blue/Green/Orange/Purple/Pink/White/Black/Brown/Grey/Gold/Silver/Teal/Magenta/None",
            "Accent colors": "Red/Yellow/Blue/Green/Orange/Purple/Pink/White/Black/Brown/Grey/Gold/Silver/Teal/Magenta/None",
            "Brightness": "Light/Dark/Medium",
            "Warm vs cool tones": "Warm/Cool/Neutral",
            "Contrast level": "High/Medium/Low",
            "Color harmony": "Monochromatic/Analogous/Complementary/Triadic/Split-Complementary",
            "Financial instruments": "string",
            "Offer text present": "Yes/No",
            "Offer text content": "string/None",
            "Offer text size": "Small/Medium/Large/Extra Large/None",
            "Offer Text language": "English/Hindi/Marathi/Tamil/Telugu/Bengali/Gujarati/Kannada/Malayalam/Punjabi/Others/Mixed",
            "Offer text Font style": "Bold/Serif/Sans-serif/Script/Display/Handwritten/Monospace",
            "Offer text Font weight": "Thin/Light/Regular/Medium/Bold/Black",
            "Offer text position": "Top Left/Top Center/Top Right/Center Left/Center/Center Right/Bottom Left/Bottom Center/Bottom Right/Banner/Sticker/None",
            "Offer text color": "Red/Yellow/Blue/Green/Orange/Purple/Pink/White/Black/Gold/Silver/Brand Color/High Contrast",
            "Offer text background": "None/Solid/Gradient/Banner/Badge/Burst/Ribbon",
            "People present": "Yes/No",
            "No. Of people": "1/2/3/3+",
            "Description of People": "string",
            "Action of person": "string",
            "Emotion of people": "Happy/Excited/Calm/Serious/Surprised/Confident/Relaxed/Energetic/Focused/Not Applicable",
            "Product elements": ["string"],
            "Product element positioning": "Center/Left/Right/Top/Bottom/Scattered/Grid/Linear",
            "Product element size emphasis": "Equal/Hero Product/Varied Sizes",
            "Design density": "Minimal/Medium/Dense",
            "Text-to-image ratio": "10%/30%/50%/70%/90%",
            "Left vs right alignment": "Left/Right/Center",
            "Symmetry": "Symmetrical/Asymmetrical",
            "Whitespace usage": "Low/Medium/High",
            "Headline text": "string/None",
            "Headline position": "Top Left/Top Center/Top Right/Center Left/Center/Center Right/Bottom Left/Bottom Center/Bottom Right/Overlay/None",
            "Headline size": "Small/Medium/Large/Extra Large",
            "Headline color": "Red/Yellow/Blue/Green/Orange/Purple/Pink/White/Black/Brown/Grey/Gold/Silver/Brand Color/Contrast Color",
            "Headline style": "Bold/Italic/Underlined/Shadow/Outline/Gradient/None",
            "Subheading text": "string/None",
            "Subheading position": "Top Left/Top Center/Top Right/Center Left/Center/Center Right/Bottom Left/Bottom Center/Bottom Right/Below Headline/None",
            "Subheading size": "Small/Medium/Large",
            "Subheading color": "Red/Yellow/Blue/Green/Orange/Purple/Pink/White/Black/Brown/Grey/Gold/Silver/Brand Color/Contrast Color",
            "Festival special occasion logo": "Yes/No",
            "Festival name": "Diwali/Holi/Christmas/Eid/New Year/Valentine/Mother's Day/Father's Day/Independence Day/Republic Day/Dussehra/Ganesh Chaturthi/Karva Chauth/Raksha Bandhan/None",
            "Brand logo visible": "Yes/No",
            "Logo size": "Small/Medium/Large/Extra Large",
            "Logo placement": "Top Left/Top Center/Top Right/Center Left/Center/Center Right/Bottom Left/Bottom Center/Bottom Right/Corner/Watermark",
            "Logo style": "Full Logo/Icon Only/Text Only/Monogram",
            "Logo transparency": "Opaque/Semi-Transparent/Watermark",
            "Brand tagline": "string/Not Applicable",
            "Call-to-action button present": "Yes/No",
            "CTA text": "string/None",
            "CTA placement": "Top/Center/Bottom/Left/Right/Floating/Multiple Positions",
            "CTA position detail": "Top Left/Top Center/Top Right/Center Left/Center/Center Right/Bottom Left/Bottom Center/Bottom Right/None",
            "CTA size": "Small/Medium/Large/Full Width",
            "CTA shape": "Rectangular/Rounded/Circular/Custom/Pill",
            "CTA style": "Filled/Outlined/Text Only/3D/Gradient",
            "CTA color": "Red/Yellow/Blue/Green/Orange/Purple/Pink/White/Black/Brand Color/Accent Color",
            "CTA text color": "White/Black/Brand Color/Contrast Color",
            "CTA contrast": "High/Medium/Low",
            "CTA animation": "None/Hover Effect/Pulse/Glow/Bounce",
            "Banner layout orientation": "Horizontal/Vertical/Square",
            "Aspect ratio": "16:9/4:3/1:1/3:4/9:16/21:9/Custom",
            "Theme": "Modern/Classic/Retro/Minimalist/Corporate/Festive/Luxury/Playful/Artistic/Tech",
            "Tone & Mood": "Energetic/Calm/Luxurious/Playful/Professional/Cozy/Urgent/Trustworthy/Innovative/Nostalgic",
            "Visual style": "Realistic/Illustrated/Abstract/Photographic/Graphic/Mixed",
            "Background Scene": "string",
            "Background texture": "Solid/Gradient/Pattern/Photographic/Abstract/Geometric/Organic",
            "Background complexity": "Simple/Moderate/Complex",
            "Device orientation": "Portrait/Landscape/Both/Adaptive",
            "Language direction": "LTR/RTL/Mixed/Vertical/Not Applicable",
        }
        schema_json = json.dumps(schema, indent=2)
        prompt = (
            "You are an AI assistant that converts user requests for advertising banners into structured JSON. "
            "Extract relevant attributes and use sensible defaults for unmentioned attributes.\n\n"
            "Respond with ONLY a valid JSON object using this schema:\n\n"
            f"{schema_json}\n\n"
            "Rules:\n"
            "1. Respond with ONLY the JSON object, no additional text\n"
            "2. Use exact field names and values as specified\n"
            "3. Choose appropriate defaults for unmentioned attributes\n"
            "4. Set \"None\" or \"Not Applicable\" when elements are not relevant"
        )
        return prompt

    async def extract_metadata_async(self, user_prompt: str, max_retries: int = 3) -> Optional[Dict]:
        headers = {
            "Authorization": f"Bearer {self.together_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.create_system_prompt()},
                {"role": "user", "content": f"Convert this banner request to JSON: {user_prompt}"}
            ],
            "max_tokens": 2100,
            "temperature": 0.1,
            "top_p": 0.9,
            "repetition_penalty": 1.1
        }
        logger.info(f"Extracting metadata for prompt: {user_prompt[:100]}...")
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context)) as session:
                    async with session.post(
                            self.together_base_url,
                            headers=headers,
                            json=payload,
                            timeout=aiohttp.ClientTimeout(total=60)
                    ) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            logger.error(f"API Error {response.status}: {error_text}")
                            if attempt < max_retries - 1:
                                await asyncio.sleep(2 ** attempt)
                                continue
                            return None
                        result = await response.json()
                        if "error" in result:
                            logger.error(f"API Error: {result['error']}")
                            if attempt < max_retries - 1:
                                await asyncio.sleep(2 ** attempt)
                                continue
                            return None
                        assistant_message = result["choices"][0]["message"]["content"].strip()
                        cleaned_message = self._clean_json_response(assistant_message)
                        try:
                            metadata = json.loads(cleaned_message)
                            logger.info("✅ Metadata extraction successful")
                            return metadata
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse JSON: {e}")
                            return None
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
        logger.error("❌ All metadata extraction attempts failed")
        return None

    def _clean_json_response(self, response: str) -> str:
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        elif response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        start_idx = response.find('{')
        end_idx = response.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            response = response[start_idx:end_idx + 1]
        return response

    def convert_to_flux_prompt(self, metadata: Dict[str, Any]) -> str:
        prompt_parts = []
        # Add all parts from metadata...
        # You can copy over the same function body from your original code
        # ...

        quality_enhancers = [
            "high-resolution vector-style rendering",
            "sharp details",
            "clean composition",
            "professional-grade output"
        ]
        complete_prompt = ", ".join(prompt_parts + quality_enhancers)
        logger.info(f"Generated comprehensive FLUX prompt: {complete_prompt[:150]}...")
        return complete_prompt

    async def generate_image_fireworks(self, prompt: str, width: int = 1024, height: int = 1024) -> Optional[str]:
        """Generate image using Fireworks AI FLUX.1-dev FP8"""
        payload = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "sampler_name": "EulerDiscreteScheduler",
            "cfg_scale": 7.5,
            "num_outputs": 1
        }
        headers = {
            "Accept": "image/jpeg",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.fireworks_api_key}"
        }
        logger.info(f"Generating image with Fireworks AI: {width}x{height}")
        try:
            response = requests.post(self.fireworks_image_url, json=payload, headers=headers, timeout=120)
            if response.status_code != 200:
                logger.error(f"Fireworks API Error {response.status_code}: {response.text}")
                return None
            image_b64 = base64.b64encode(response.content).decode("utf-8")
            logger.info("✅ Fireworks image generation successful")
            return image_b64
        except Exception as e:
            logger.error(f"Fireworks image generation failed: {e}")
            return None

    async def process_banner_request_async(self, user_prompt: str, width: int = 1024,
                                           height: int = 1024, num_inference_steps: int = 50,
                                           guidance_scale: float = 5.0, seed: Optional[int] = 0) -> Dict:
        start_time = time.time()
        logger.info(f"🔄 Processing banner request: {user_prompt[:100]}...")

        # Step 1: Extract metadata
        metadata = await self.extract_metadata_async(user_prompt)
        if not metadata:
            return {
                "status": "error",
                "error": "Failed to extract metadata",
                "processing_time": time.time() - start_time
            }

        # Step 2: Generate comprehensive FLUX prompt
        flux_prompt = self.convert_to_flux_prompt(metadata)

        # Step 3: Generate short refined prompt
        short_prompt = self.generate_short_prompt(user_prompt)

        # Step 4: Generate image using Fireworks AI API
        image_b64 = await self.generate_image_fireworks(short_prompt, width, height)

        processing_time = time.time() - start_time

        if image_b64:
            logger.info(f"✅ Complete pipeline successful in {processing_time:.2f}s")
            return {
                "status": "success",
                "structured_data": metadata,
                "flux_prompt": flux_prompt,
                "short_prompt": short_prompt,
                "image_base64": image_b64,
                "processing_time": processing_time
            }
        else:
            logger.error("❌ Image generation failed")
            return {
                "status": "partial_success",
                "structured_data": metadata,
                "flux_prompt": flux_prompt,
                "short_prompt": short_prompt,
                "error": "Image generation failed",
                "processing_time": processing_time
            }

    def save_image_temporarily(self, image_b64: str, request_id: str) -> str:
        try:
            image_data = base64.b64decode(image_b64)
            image = Image.open(BytesIO(image_data))
            temp_dir = tempfile.gettempdir()
            filename = f"banner_{request_id}.jpg"
            file_path = os.path.join(temp_dir, filename)
            image.save(file_path, format='JPEG')
            generated_images[request_id] = {
                "file_path": file_path,
                "created_at": datetime.now(),
                "filename": filename
            }
            logger.info(f"Image saved temporarily: {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Failed to save image temporarily: {e}")
            return None


# Load environment variables
load_dotenv()

# Get API keys from environment variables
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY") or "fw_3ZQT8Wat6jhfaNc6HtZ22Bs2"

# Validate API key
if not TOGETHER_API_KEY:
    raise ValueError("TOGETHER_API_KEY environment variable is required")
if not FIREWORKS_API_KEY:
    raise ValueError("FIREWORKS_API_KEY environment variable is required")

# Initialize Pipeline
try:
    pipeline = CompleteBannerPipeline(TOGETHER_API_KEY, FIREWORKS_API_KEY)
    logger.info("✅ Pipeline initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize pipeline: {e}")
    raise

# FastAPI App
app = FastAPI(
    title="Banner Generation API",
    description="Generate advertising banners from natural language prompts using Fireworks AI FLUX.1-dev",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_model=StatusResponse)
async def root():
    return StatusResponse(
        status="online",
        message="Banner Generation API with Fireworks AI FLUX.1-dev is running",
        timestamp=datetime.now().isoformat()
    )

@app.post("/generate-banner")
async def generate_banner(request: BannerRequest):
    request_id = f"req_{int(time.time() * 1000)}"
    start_time = time.time()
    logger.info(f"📨 Received banner request {request_id}: {request.prompt}")
    # Store job immediately
    processing_jobs[request_id] = {
        "status": "processing",
        "progress": "Starting...",
        "created_at": datetime.now().isoformat(),
        "result": None
    }
    try:
        result = await pipeline.process_banner_request_async(
            user_prompt=request.prompt,
            width=request.width,
            height=request.height,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            seed=request.seed
        )
        processing_time = time.time() - start_time
        # Update job status
        processing_jobs[request_id].update({
            "status": "completed" if result["status"] == "success" else "partial_success",
            "progress": "Complete",
            "result": result,
            "completed_at": datetime.now().isoformat()
        })
        download_url = None
        # Handle different return formats
        if request.return_format == "image" and result.get("image_base64"):
            # Return image directly for download
            try:
                image_data = base64.b64decode(result["image_base64"])
                image = Image.open(BytesIO(image_data))
                img_byte_arr = BytesIO()
                image.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)
                return StreamingResponse(
                    img_byte_arr,
                    media_type="image/png",
                    headers={
                        "Content-Disposition": f"attachment; filename=banner_{request_id}.png",
                        "X-Request-ID": request_id
                    }
                )
            except Exception as e:
                logger.error(f"Error serving direct image download: {e}")
                return JSONResponse(
                    status_code=500,
                    content={"error": f"Failed to process image for download: {str(e)}"}
                )
        # For JSON or both formats, save image temporarily and provide download URL
        if result.get("image_base64"):
            file_path = pipeline.save_image_temporarily(result["image_base64"], request_id)
            if file_path:
                download_url = f"/download/{request_id}"
        # Prepare response based on format
        response_data = {
            "status": result["status"],
            "request_id": request_id,
            "processing_time": processing_time,
            "error": result.get("error")
        }
        if request.return_format in ["json", "both"]:
            response_data.update({
                "structured_data": result.get("structured_data"),
                "flux_prompt": result.get("flux_prompt"),
                "short_prompt": result.get("short_prompt"),
                "download_url": download_url
            })
            if request.return_format == "both":
                response_data["image_base64"] = result.get("image_base64")
        return JSONResponse(content=response_data)
    except Exception as e:
        logger.error(f"❌ Request {request_id} failed: {str(e)}")
        processing_jobs[request_id].update({
            "status": "failed",
            "progress": f"Error: {str(e)}",
            "error": str(e),
            "completed_at": datetime.now().isoformat()
        })
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "request_id": request_id,
                "processing_time": time.time() - start_time,
                "error": str(e)
            }
        )


@app.get("/download/{request_id}")
async def download_image(request_id: str):
    """Direct download endpoint for generated images"""
    if request_id not in generated_images:
        raise HTTPException(status_code=404, detail="Image not found or expired")
    image_info = generated_images[request_id]
    file_path = image_info["file_path"]
    filename = image_info["filename"]
    if not os.path.exists(file_path):
        # Clean up expired entry
        del generated_images[request_id]
        raise HTTPException(status_code=404, detail="Image file not found")
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="image/png",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@app.post("/generate-banner-async")
async def generate_banner_async(request: BannerRequest, background_tasks: BackgroundTasks):
    request_id = f"async_req_{int(time.time() * 1000)}"
    logger.info(f"📨 Received async banner request {request_id}: {request.prompt}")
    processing_jobs[request_id] = {
        "status": "processing",
        "progress": "Starting...",
        "created_at": datetime.now().isoformat(),
        "result": None
    }
    background_tasks.add_task(
        process_async_banner,
        request_id,
        request.prompt,
        request.width,
        request.height,
        request.num_inference_steps,
        request.guidance_scale,
        request.seed
    )
    return {
        "request_id": request_id,
        "status": "accepted",
        "message": "Banner generation started",
        "check_status_url": f"/status/{request_id}",
        "download_url": f"/download/{request_id}"
    }


@app.get("/status/{request_id}")
async def get_job_status(request_id: str):
    if request_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = processing_jobs[request_id]
    response = {
        "request_id": request_id,
        "status": job["status"],
        "progress": job["progress"],
        "created_at": job["created_at"],
        "result": job["result"]
    }
    # Add download URL if image is available
    if job["status"] == "completed" and request_id in generated_images:
        response["download_url"] = f"/download/{request_id}"
    return response


@app.get("/image/{request_id}")
async def get_generated_image(request_id: str):
    """Legacy endpoint - redirects to download endpoint"""
    return await download_image(request_id)


@app.delete("/cleanup/{request_id}")
async def cleanup_image(request_id: str):
    """Clean up temporary image files"""
    if request_id in generated_images:
        image_info = generated_images[request_id]
        file_path = image_info["file_path"]
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
            del generated_images[request_id]
            return {"message": f"Image {request_id} cleaned up successfully"}
        except Exception as e:
            logger.error(f"Failed to cleanup image {request_id}: {e}")
            return {"error": f"Failed to cleanup: {str(e)}"}
    else:
        raise HTTPException(status_code=404, detail="Image not found")


@app.get("/cleanup-old-images")
async def cleanup_old_images():
    """Clean up images older than 1 hour"""
    current_time = datetime.now()
    cleaned_count = 0
    for request_id in list(generated_images.keys()):
        image_info = generated_images[request_id]
        age = current_time - image_info["created_at"]
        if age.total_seconds() > 3600:  # 1 hour
            try:
                file_path = image_info["file_path"]
                if os.path.exists(file_path):
                    os.remove(file_path)
                del generated_images[request_id]
                cleaned_count += 1
            except Exception as e:
                logger.error(f"Failed to cleanup old image {request_id}: {e}")
    return {"message": f"Cleaned up {cleaned_count} old images"}


async def process_async_banner(request_id: str, prompt: str, width: int, height: int,
                             num_inference_steps: int, guidance_scale: float, seed: Optional[int]):
    try:
        result = await pipeline.process_banner_request_async(
            user_prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed
        )
        # Save image temporarily for download
        if result.get("image_base64"):
            pipeline.save_image_temporarily(result["image_base64"], request_id)
        processing_jobs[request_id].update({
            "status": "completed" if result["status"] == "success" else "partial_success",
            "progress": "Complete",
            "result": result,
            "completed_at": datetime.now().isoformat()
        })
        logger.info(f"✅ Async request {request_id} completed successfully")
    except Exception as e:
        processing_jobs[request_id].update({
            "status": "failed",
            "progress": f"Error: {str(e)}",
            "error": str(e),
            "completed_at": datetime.now().isoformat()
        })
        logger.error(f"❌ Async request {request_id} failed: {str(e)}")


# Startup event to schedule cleanup
@app.on_event("startup")
async def startup_event():
    import asyncio
    # Schedule cleanup every hour
    asyncio.create_task(periodic_cleanup())


async def periodic_cleanup():
    """Periodic cleanup of old temporary files"""
    while True:
        try:
            await asyncio.sleep(3600)  # Wait 1 hour
            await cleanup_old_images()
            logger.info("🧹 Periodic cleanup completed")
        except Exception as e:
            logger.error(f"Periodic cleanup failed: {e}")


# Run Server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
