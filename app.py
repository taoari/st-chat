import json
import logging
import streamlit as st  # 1.34.0
import time
import tiktoken
from io import BytesIO
from PIL import Image
import base64

from datetime import datetime
from streamlit_float import *

from openai import OpenAI  # 1.30.1
# from openai import AzureOpenAI  # 1.30.1


# Code adopted from: https://github.com/pierrelouisbescond/streamlit-chat-ui-improvement

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

# Model Choice - Name to be adapter to your deployment
if "model" not in st.session_state:
    st.session_state["model"] = "gpt-4o"

# Adapted from https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps
if "messages" not in st.session_state:
    st.session_state["messages"] = []


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


@st.experimental_dialog("🎨 Upload a picture")
def upload_document():
    # st.warning(
    #     "This is a demo dialog window. You need to process the file afterwards.",
    #     icon="💡",
    # )
    picture = st.file_uploader(
        "Choose a file", type=["jpg", "png", "bmp"], label_visibility="hidden"
    )
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
        st.rerun()
        

# Function to convert file to base64
def get_image_base64(pil_img):
    buffered = BytesIO()
    pil_img.save(buffered, format=pil_img.format)
    img_byte = buffered.getvalue()

    return base64.b64encode(img_byte).decode('utf-8')

def file_to_base64(file):
    with open(file, "rb") as f:

        return base64.b64encode(f.read())

def base64_to_image(base64_string):
    base64_string = base64_string.split(",")[1]
    return Image.open(BytesIO(base64.b64decode(base64_string)))

##### Streamlit UI


def main():

    st.set_page_config(page_title="Streamlit Chat", page_icon="🤩")

    st.title("🤩 Streamlit Chat")

    user_avatar = "👩‍💻"
    assistant_avatar = "🤖"

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
    if prompt := st.chat_input("How can I help you?"):

        # and display it in the chat history
        with st.chat_message("user", avatar=user_avatar):
            st.markdown(prompt)
        st.session_state["messages"].append({"role": "user", "content": prompt})

    if prompt or (len(st.session_state['messages']) > 0 and st.session_state['messages'][-1]['role'] == 'user'):

        # Display assistant response in chat message container
        with st.chat_message("assistant", avatar=assistant_avatar):
            stream = client.chat.completions.create(
                model=st.session_state["model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state["messages"]
                ],
                stream=True,
                max_tokens=512,  # Limited to 300 tokens for demo purposes
            )
            response = st.write_stream(stream)
        st.session_state["messages"].append({"role": "assistant", "content": response})

    st.write("")

    # If there is at least one message in the chat, we display the options
    if len(st.session_state["messages"]) >= 0:

        action_buttons_container = st.container()
        action_buttons_container.float(
            "bottom: 6.9rem;background-color: var(--default-backgroundColor); padding-top: 1rem;"
        )

        # We set the space between the icons thanks to a share of 100
        cols_dimensions = [7, 14.9, 14.5, 14.5, 8.7, 8.6, 8.7]
        cols_dimensions.append(100 - sum(cols_dimensions))
        # print(cols_dimensions)

        col0, col1, col2, col3, col4, col5, col6, col7 = action_buttons_container.columns(
            cols_dimensions
        )

        with col1:

            # Converts the list of messages into a JSON Bytes format
            json_messages = json.dumps(st.session_state["messages"]).encode("utf-8")

            # And the corresponding Download button
            st.download_button(
                label="📥 Save!",
                data=json_messages,
                file_name="chat_conversation.json",
                mime="application/json",
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

            if st.button("🎨"):
                upload_document()

        with col5:
            icon = "👍"

            # The button will trigger the logging function
            if st.button(icon):
                log_feedback(icon)

        with col6:
            icon = "👎"

            # The button will trigger the logging function
            if st.button(icon):
                log_feedback(icon)

        with col7:

            # We initiate a tokenizer
            enc = tiktoken.get_encoding("cl100k_base")

            # We encode the messages
            tokenized_full_text = enc.encode(
                " ".join([item["content"] for item in st.session_state["messages"] if isinstance(item["content"], str)])
            )

            # And display the corresponding number of tokens
            label = f"💬 {len(tokenized_full_text)} tokens"
            st.link_button(label, "https://platform.openai.com/tokenizer")

    # else:

    #     # At the first run of a session, we temporarly display a message
    #     if "disclaimer" not in st.session_state:
    #         with st.empty():
    #             for seconds in range(3):
    #                 st.warning(
    #                     "‎ You can click on 👍 or 👎 to provide feedback regarding the quality of responses.",
    #                     icon="💡",
    #                 )
    #                 time.sleep(1)
    #             st.write("")
    #             st.session_state["disclaimer"] = True


if __name__ == '__main__':
    main()