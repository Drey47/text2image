import streamlit as st
from diffusers import DiffusionPipeline
import torch

def load_model():
    st.write("Loading...")
    pass

st.title('Géneration d\'images à partir d\'un prompt')
st.button('chager le model', on_click = load_model)

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")


# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()

user_prompt = st.text_area('Entrer votre prompt: ', height=100)

images = pipe(prompt=user_prompt).images[0]
st.image(images)

