import streamlit as st
import os
import json
import google.generativeai as genai
from PIL import Image
import io
import time
import base64
from streamlit_extras.app_logo import add_logo
from streamlit_extras.colored_header import colored_header


# --- Configuration (Move to a separate config file in a real app) ---

MODEL_NAME = "gemini-2.0-flash"  # Use a model name supported by your API key.
CONTEXT_FOLDER = "context_crops"  # Make sure this folder exists.


# --- Helper Functions ---

def load_api_key(filepath=".env"):
    """Loads the API key."""
    try:
        with open(filepath, "r") as f:
            for line in f:
                if line.startswith("GOOGLE_API_KEY"):
                    return line.strip().split("=")[1].strip()
    except FileNotFoundError:
        pass
    api_key = os.environ.get("GOOGLE_API_KEY")
    if api_key:
        return api_key
    raise ValueError("Google API key not found.")


def configure_genai():
    """Configures the Google Generative AI library."""
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
    except Exception as e:
        raise RuntimeError(f"Failed to configure Google AI: {e}") from e


generation_config = {
    "temperature": 0.5,
    "top_p": 0.5,
    "top_k": 32,
    "max_output_tokens": 4096,
    "response_mime_type": "text/plain",
}


def load_context_images(context_folder):
    """Loads context images."""
    context_images = []
    for crop_name in os.listdir(context_folder):
        crop_path = os.path.join(context_folder, crop_name)
        if os.path.isdir(crop_path):
            for stage_name in os.listdir(crop_path):
                input_folder = os.path.join(crop_path, stage_name)
                if os.path.isdir(input_folder):
                    image_files = [
                        f
                        for f in os.listdir(input_folder)
                        if f.lower().endswith((".png", ".jpg", ".jpeg"))
                    ]
                    for image_file in image_files[:2]:
                        image_path = os.path.join(input_folder, image_file)
                        try:
                            with open(image_path, "rb") as f:
                                image_data = f.read()
                                description = f"{crop_name} - {stage_name}:"
                                context_images.append((description, image_data))
                        except Exception as e:
                            print(f"Error loading context image {image_path}: {e}")
    return context_images


def call_llm(image_data, prompt_text):
    """Calls the Gemini LLM."""
    try:
        image_part = {"mime_type": "image/jpeg", "data": image_data}

        context_images = load_context_images(CONTEXT_FOLDER)
        prompt_parts = [
            prompt_text,
            "\n\nINPUT IMAGE:",
            image_part,
            "\n\nUse the following examples to match the crop in the input image:\n",
        ]

        for description, context_image_data in context_images:
            prompt_parts.append(description)
            prompt_parts.append(
                {"mime_type": "image/jpeg", "data": context_image_data}
            )

        model = genai.GenerativeModel(MODEL_NAME, generation_config=generation_config)
        response = model.generate_content(prompt_parts)

        if response.text:
            try:
                response_text = response.text.replace("```json", "").replace("```", "")
                return json.loads(response_text)
            except json.JSONDecodeError:
                st.error(f"Error: Invalid JSON response: {response.text}")
                return None
        else:
            st.error("Error: Empty or non-text response from LLM.")
            return None

    except Exception as e:
        st.error(f"Error calling LLM: {type(e).__name__} - {e}")
        return None


def process_image(image_data, prompt_text):
    """Processes image, returns structured data."""
    llm_result = call_llm(image_data, prompt_text)

    if llm_result:
        try:
            crops = llm_result.get("crop_name", [])
            scores = llm_result.get("confidence_score", [])
            stages = llm_result.get("stage_of_plant_growth", [])
            description = llm_result.get("description", "")

            crops = crops if isinstance(crops, list) else [crops]
            scores = scores if isinstance(scores, list) else [scores]
            stages = stages if isinstance(stages, list) else [stages]
            description = description if description else "No description."

            return crops, scores, stages, description

        except (KeyError, IndexError) as e:
            st.error(f"Error processing LLM result: {e}. Result: {llm_result}")
            return [], [], [], "Error parsing results"
    else:
        return [], [], [], "No results from LLM"


# --- Streamlit App ---

def main():
    """Main Streamlit app."""
    global GOOGLE_API_KEY
    try:
        GOOGLE_API_KEY = load_api_key()
    except ValueError as e:
        st.error(str(e))
        st.stop()

    configure_genai()

    # --- Custom CSS ---
    st.markdown(
        """
        <style>
        /* General page styling */
        .reportview-container {
            background: linear-gradient(to bottom right, #f0f2f6, #c8d9e8); /* Soft gradient */
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; /* Modern font */
        }

        /* Title styling */
        .title {
            color: #2e7d32; /* Dark green */
            text-align: center;
            padding-bottom: 0.5rem;
            font-size: 3.5rem; /* Larger font size */
            font-weight: bold;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2); /* Text shadow for depth */
        }

        /* Header description styling */
        .header-description {
            text-align: center;
            color: #555; /* Subdued color for description */
            margin-bottom: 2rem; /* Increased spacing */
            font-size: 1.2rem;
        }
         /* File uploader styling */
        .stFileUploader {
            padding: 1.5em;  /* More padding */
            border: 2px dashed #4CAF50;  /* Green dashed border */
            border-radius: 10px; /* Rounded corners */
            background-color: #f8f8f8; /* Light background */
            transition: border-color 0.3s ease, box-shadow 0.3s ease; /* Smooth transition */

        }

        .stFileUploader:hover {
            border-color: #388e3c; /* Darker green on hover */
            box-shadow: 0 0 10px rgba(76, 175, 80, 0.5); /* Green glow effect */

        }


        /* Button styling */
        .stButton>button {
            display: block;
            margin: 0.5em auto; /* Centered buttons */
            color: white !important; /* Keep text white, even after click */
            background-color: #4CAF50;
            padding: 0.8em 1.5em; /* More padding */
            border: none;
            border-radius: 25px; /* Rounded buttons */
            text-align: center;
            transition: all 0.3s ease;
            width: auto;  /* Let the button size to its content */
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2); /* Subtle shadow */
            font-size: 1.1rem; /* Slightly larger font */
            cursor: pointer; /* Hand cursor on hover */
        }

        .stButton>button:hover {
            background-color: #388e3c;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3); /* More pronounced shadow */
            transform: translateY(-2px); /* Slight upward movement */
        }

        /* Image styling */
        .stImage {
            border-radius: 12px; /* More rounded corners */
            box-shadow: 4px 6px 10px rgba(0, 0, 0, 0.2); /* Stronger shadow */
            transition: transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out; /* Smooth transition */
            overflow: hidden;  /* Ensures the image stays within the rounded borders */

        }

        .stImage:hover {
            transform: scale(1.03); /* Slightly larger on hover */
            box-shadow: 6px 8px 15px rgba(0, 0, 0, 0.3); /* Enhanced shadow */
        }


        /* Progress bar styling */
        .stProgress .st-bo {
            background-color: #4CAF50;
        }

        /* Result card styling */
        .result-card {
            background-color: rgba(255, 255, 255, 0.95); /* Slight transparency */
            border-radius: 15px; /* More rounded corners */
            box-shadow: 0 6px 15px rgba(0, 0, 0, 0.15); /* More pronounced shadow */
            padding: 1.5em;
            margin-bottom: 1.5em;
            transition: transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out;
            border-left: 8px solid #4CAF50; /* Thicker, green border */
            backdrop-filter: blur(5px); /* Slight blur effect */
        }

        .result-card:hover {
            transform: translateY(-8px); /* More pronounced lift */
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);
        }

        /* Result title styling */
        .result-title {
            color: #2e7d32;
            font-size: 1.5em; /* Larger font size */
            font-weight: bold;
            margin-bottom: 0.7em;
            border-bottom: 2px solid #4CAF50; /* Underline */
            padding-bottom: 0.3em;
            display: inline-block; /* Makes the border only span the text width */
        }

        /* Reset button styling */
        .reset-button {
            background-color: #f44336; /* Red color */
        }

        .reset-button:hover {
            background-color: #d32f2f;
        }

        /* Button placement */
        .button-container {
            display: flex;
            justify-content: center; /* Center the buttons */
            gap: 2em; /* More space between buttons */
            margin-bottom: 2em; /* More space before the image */
        }
        .st-emotion-cache-1y4p8pa {
            padding: 0;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )
    colored_header(
    label="Crop Field Analyzer",
    description="Upload an image to analyze and identify the crops, their growth stages, and confidence scores.",
    color_name="green-70",
    )


    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None

    uploaded_file = st.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png"], key="file_uploader"
    )
    st.session_state.uploaded_file = uploaded_file  # Update session state

    try:
        with open("st_prompt.txt", "r") as f:
            prompt_text = f.read()
    except FileNotFoundError:
        st.error("Error: Could not find 'st_prompt.txt'.")
        st.stop()

    if st.session_state.uploaded_file is not None:  # Conditionally display image
        image = Image.open(st.session_state.uploaded_file)
        image.thumbnail((300, 900))  # Resize image to max 400x400
        # No resizing here, display the image in a responsive manner
        # Use columns for responsive layout.  This is better than fixed width.
        col1, col2, col3, col4 = st.columns([1, 2, 4, 3])  # Adjust ratios as needed
        with col3:
            st.image(image, caption="Uploaded Image", use_container_width=False)


    # --- Button Container ---
    with st.container():
        st.markdown("<div class='button-container'>", unsafe_allow_html=True)
        analyze_button = st.button(
            "Analyze Image"
        )  # Analyze button (no key needed here)
        if st.button("Reset", type="primary"):
            st.session_state.clear()
            st.session_state.uploaded_file = None  # Clear uploaded_file
            st.experimental_rerun()  # Use st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


    if st.session_state.uploaded_file is not None:  # Conditionally process and display results
        image_data = st.session_state.uploaded_file.getvalue()

        if analyze_button:  # Use the button variable
            with st.spinner("Analyzing..."):
                crops, scores, stages, description = process_image(image_data, prompt_text)
                time.sleep(1)  # Keep for visual feedback.  Consider removing in production.

            st.subheader("Results:")
            if crops:
                for i, crop in enumerate(crops):
                    with st.container():
                        st.markdown(f"<div class='result-card'>", unsafe_allow_html=True)
                        st.markdown(
                            f"<h3 class='result-title'>Crop {i+1}</h3>",
                            unsafe_allow_html=True,
                        )
                        st.write(f"**Crop:** {crop}")
                        st.write(
                            f"**Score:** {scores[i] if i < len(scores) else 'N/A'}"
                        )
                        st.write(
                            f"**Stage:** {stages[i] if i < len(stages) else 'N/A'}"
                        )
                        st.markdown("</div>", unsafe_allow_html=True)
                with st.container():
                    st.markdown(f"<div class='result-card'>", unsafe_allow_html=True)
                    st.markdown(
                        f"<h3 class='result-title'>Description</h3>", unsafe_allow_html=True
                    )
                    st.write(f"**Description:** {description}")
                    st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.error("Could not identify the crop or an error occurred.")


if __name__ == "__main__":
    main()