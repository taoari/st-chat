import json
import logging
import streamlit as st  # 1.34.0
import time
import tiktoken
import base64
from audio_recorder_streamlit import audio_recorder

from datetime import datetime
from streamlit_float import *

from openai import OpenAI  # 1.30.1
# from openai import AzureOpenAI  # 1.30.1

# Code adopted from: 
# 1. https://github.com/pierrelouisbescond/streamlit-chat-ui-improvement
# 2. https://github.com/enricd/the_omnichat

##### Global variables

AVAILABLE_MODELS = [
    "gpt-4o", 
    "gpt-4-turbo", 
    "gpt-3.5-turbo", 
]

##### Initialize logger and client

logger = logging.getLogger()
logging.basicConfig(encoding="UTF-8", level=logging.INFO)

# Secrets to be stored in /.streamlit/secrets.toml
# OPENAI_API_ENDPOINT = "https://xxx.openai.azure.com/"
# OPENAI_API_KEY = "efpgishhn2kwlnk9928avd6vrh28wkdj" (this is a fake key 😉)

# To be used with standard OpenAI API
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# # To be used with standard Azure OpenAI API
# client = AzureOpenAI(
#     azure_endpoint=st.secrets["OPENAI_API_ENDPOINT"],
#     api_key=st.secrets["OPENAI_API_KEY"],
#     api_version="2024-02-15-preview",
# )

##### Initialize session_state

# Adapted from https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps

default_state = {
    'messages': [], 
    'settings': {},
    'prev_speech_hash': None,
    }

for k, v in default_state.items():
    if k not in st.session_state:
        st.session_state[k] = v

##### Utils

# This function logs the last question and answer in the chat messages
def log_feedback(icon):
    # We display a nice toast
    st.toast("Thanks for your feedback!", icon="👌")

    # We retrieve the last question and answer
    last_messages = json.dumps(st.session_state["messages"][-2:])

    # We record the timestamp
    activity = datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ": "

    # And include the messages
    activity += "positive" if icon == "👍" else "negative"
    activity += ": " + last_messages

    # And log everything
    logger.info(activity)

def add_image_to_messages(key='uploaded_image'):
    picture = st.session_state[key]

    if picture:

        img_type = picture.type
        img = base64.b64encode(picture.read()).decode('utf-8')

        st.session_state["messages"].append(
            {
                "role": "user", 
                "content": [{
                    "type": "image_url",
                    "image_url": {"url": f"data:{img_type};base64,{img}"}
                }]
            })

def speech_recognition(speech_input, model="whisper-1"):
    transcript = client.audio.transcriptions.create(
        model=model, 
        file=("audio.wav", speech_input),
    )
    return transcript.text

def speech_sythesis(text, model, voice):
    response =  client.audio.speech.create(
        model=model,
        voice=voice,
        input=text,
    )
    return response.content

def llm_stream(messages, model='gpt-3.5-turbo', temperature=0.0):
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        stream=True,
        max_tokens=512,  # Limited to 512 tokens for demo purposes
    )
    return stream

##### Streamlit UI
        
def setup_header():
    st.set_page_config(page_title="Streamlit Chat", page_icon="🤩", initial_sidebar_state='collapsed')

    st.title("🤩 Streamlit Chat")

    # hack to hide deploy button: 
    # https://discuss.streamlit.io/t/how-to-hide-or-remove-the-deploy-button-that-appears-at-the-top-right-corner-of-the-streamlit-app/55325
    st.markdown(
        r"""
        <style>
        .stDeployButton {
                visibility: hidden;
            }
        </style>
        """, unsafe_allow_html=True
    )

def setup_siderbar():
    # Sidebar
    with st.sidebar:
        model = st.selectbox("Select a model:", AVAILABLE_MODELS, index=0)
        with st.popover("⚙️ Model parameters"):
            model_temp = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.3, step=0.1)

        audio_prompt = None
        speech_input = audio_recorder("Press to talk:", icon_size="2x", neutral_color="#6ca395", )
        if speech_input:
            st.audio(speech_input, format="audio/wav")
            speech_hash = hash(speech_input)
            if speech_hash != st.session_state['prev_speech_hash']:
                st.session_state['prev_speech_hash'] = speech_hash
                try:
                    audio_prompt = speech_recognition(speech_input)
                except:
                    pass

        tts_voice, tts_model = None, None
        audio_response = st.toggle("Audio response", value=False)
        if audio_response:
            cols = st.columns(2)
            with cols[0]:
                tts_voice = st.selectbox("Select a voice:", ["alloy", "echo", "fable", "onyx", "nova", "shimmer"])
            with cols[1]:
                tts_model = st.selectbox("Select a model:", ["tts-1", "tts-1-hd"], index=0)

        settings = {
            'model': model,
            'temperature': model_temp,
            'audio_response': audio_response,
            'tts_voice': tts_voice,
            'tts_model': tts_model,
            'audio_prompt': audio_prompt,
        }
        st.session_state['settings'] = settings


def setup_sidebar_export():

    with st.sidebar:

        st.divider()

        # Converts the list of messages into a JSON Bytes format
        json_messages = json.dumps(st.session_state["messages"]).encode("utf-8")

        # And the corresponding Download button
        st.download_button(
            label="📥 Export chat history",
            data=json_messages,
            file_name="chat_conversation.json",
            mime="application/json",
        )

        debug = st.toggle("Debug", value=False)
        if debug:
            st.json(st.session_state)

def setup_action_buttons():

    # If there is at least one message in the chat, we display the options
    if len(st.session_state["messages"]) >= 0:

        action_buttons_container = st.container()
        action_buttons_container.float(
            "bottom: 6.9rem;background-color: var(--default-backgroundColor); padding-top: 1rem;"
        )

        # We set the space between the icons thanks to a share of 100
        cols_dimensions = [2, 2, 14.3, 14.3, 10, 10, 8, 8]
        cols_dimensions.append(100 - sum(cols_dimensions))

        col0, col1, col2, col3, col4, col5, col6, col7, col8 = action_buttons_container.columns(
            cols_dimensions
        )

        with col2:

            # We set the message back to 0 and rerun the app
            # (this part could probably be improved with the cache option)
            if st.button("🧹 Clear"):
                st.session_state["messages"] = []
                st.rerun()

        with col3:
            if st.button("🔁 Redo"):
                if len(st.session_state["messages"]) > 0:
                    st.session_state["messages"].pop(-1)
                st.rerun()

        with col4:

            with st.popover("📁"): # "🎨"
                # https://stackoverflow.com/questions/71789004/how-does-one-hide-the-the-file-uploader-in-streamlit-after-the-image-is-loaded
                picture = st.file_uploader(
                    "Choose a file", type=["jpg", "png", "bmp"], 
                    accept_multiple_files=False,
                    key='uploaded_image',
                    on_change=lambda : add_image_to_messages('uploaded_image'),
                )

        with col5:

            with st.popover("📸"):
                activate_camera = st.checkbox("Activate camera")
                if activate_camera:
                    holder = st.empty()
                    st.camera_input(
                        "Take a picture", 
                        key="camera_image",
                        on_change=lambda : add_image_to_messages('camera_image'),
                    )

        with col6:
            icon = "👍"

            # The button will trigger the logging function
            if st.button(icon):
                log_feedback(icon)

        with col7:
            icon = "👎"

            # The button will trigger the logging function
            if st.button(icon):
                log_feedback(icon)

        with col8:

            # We initiate a tokenizer
            enc = tiktoken.get_encoding("cl100k_base")

            # We encode the messages
            tokenized_full_text = enc.encode(
                " ".join([item["content"] for item in st.session_state["messages"] if isinstance(item["content"], str)])
            )

            # And display the corresponding number of tokens
            label = f"💬 {len(tokenized_full_text)} tokens"
            st.link_button(label, "https://platform.openai.com/tokenizer")
            
def setup_chat():
    user_avatar = "👩‍💻"
    assistant_avatar = "🤖"
    settings = st.session_state['settings']

    # We rebuild the previous conversation stored in st.session_state["messages"] with the corresponding emojis
    for message in st.session_state["messages"]:
        with st.chat_message(
            message["role"],
            avatar=assistant_avatar if message["role"] == "assistant" else user_avatar,
        ):
            if isinstance(message["content"], str):
                st.markdown(message["content"])
            else:
                for content in message["content"]:
                    if content["type"] == 'image_url':
                        st.image(content['image_url']['url'])

    # A chat input will add the corresponding prompt to the st.session_state["messages"]
    prompt = st.chat_input("How can I help you?")
    if not prompt and settings['audio_prompt']:
        prompt = settings['audio_prompt']
        settings['audio_prompt'] = None

    if prompt:
        # and display it in the chat history
        with st.chat_message("user", avatar=user_avatar):
            st.markdown(prompt)
        st.session_state["messages"].append({"role": "user", "content": prompt})

    if prompt or (len(st.session_state['messages']) > 0 and st.session_state['messages'][-1]['role'] == 'user'):

        # Display assistant response in chat message container
        with st.chat_message("assistant", avatar=assistant_avatar):
            messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state["messages"]
                ]
            stream = llm_stream(messages, model=settings["model"], temperature=settings['temperature'])

            response = st.write_stream(stream)
        st.session_state["messages"].append({"role": "assistant", "content": response})
        
        # TTS if enabled
        if settings['audio_response']:
            audio_bytes = speech_sythesis(response, model=settings['tts_model'], voice=settings['tts_voice'])
            st.audio(audio_bytes, format='audio/mp3', autoplay=True)

    st.write("")

def main():
    setup_header()
    setup_action_buttons()
    setup_siderbar()

    # Chat should be at the end
    setup_chat()

    setup_sidebar_export()

if __name__ == '__main__':
    main()