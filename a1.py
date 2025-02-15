import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
import speech_recognition as sr
from PIL import Image

def recognize_speech_from_mic():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    try:
        with mic as source:
            st.write("Adjusting for ambient noise. Please wait.")
            recognizer.adjust_for_ambient_noise(source)
            st.write("Listening for your command...")
            audio = recognizer.listen(source)
    except Exception as e:
        st.error(f"Error accessing microphone: {e}")
        return None

    try:
        st.write("Recognizing speech...")
        transcription = recognizer.recognize_google(audio)
        st.write(f"You said: {transcription}")
        return transcription
    except sr.RequestError:
        st.error("API was unreachable or unresponsive")
    except sr.UnknownValueError:
        st.error("Unable to recognize speech")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

    return None

def generate_image_from_text(prompt):
    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    pipe = pipe.to("cpu")

    try:
        image = pipe(prompt).images[0]
        return image
    except Exception as e:
        st.error(f"Failed to generate image: {e}")
        return None

def main():
    st.title("Speech to Image Generator")
    
    if st.button("Start Speech Recognition"):
        prompt = recognize_speech_from_mic()
    else:
        prompt = st.text_input("Or enter your prompt manually:")

    if prompt:
        st.write(f"Generating image for: {prompt}")
        image = generate_image_from_text(prompt)
        if image:
            st.image(image)
        else:
            st.error("Failed to generate image.")
    else:
        st.warning("No valid speech input recognized. Please try again.")

if __name__ == "__main__":
    main()