import gradio as gr
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import torch

# Load the pre-trained model (you can try other fine-tuned versions too)
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    revision="fp16" if torch.cuda.is_available() else None,
)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# Cartoonify function
def cartoonify(image, strength=0.8, prompt="cartoon style, animated, colorful"):
    image = image.convert("RGB").resize((512, 512))
    result = pipe(prompt=prompt, image=image, strength=strength, guidance_scale=7.5)
    cartoon_img = result.images[0]
    return cartoon_img

# Gradio interface
interface = gr.Interface(
    fn=cartoonify,
    inputs=[
        gr.Image(type="pil"),
        gr.Slider(0.3, 1.0, value=0.8, label="Cartoon Strength")
    ],
    outputs=gr.Image(type="pil"),
    title="🖼️ Cartoon Image Generator",
    description="Upload a photo and convert it to a cartoon-style image using Stable Diffusion.",
)

# Launch the app
interface.launch()
