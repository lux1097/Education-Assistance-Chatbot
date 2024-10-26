import time
import streamlit as st
import json
from collections import defaultdict

from openai import OpenAI

import cv2
from PIL import Image
from vit_inferencer import get_emotion
from ultralytics import YOLO

face_model = YOLO('yolov8n-face.pt')

client = OpenAI(api_key='*****')

st.set_page_config(
    page_title="Streamlit quizz app",
    page_icon="❓",
)

# Custom CSS for the buttons
st.markdown("""
<style>
div.stButton > button:first-child {
    display: block;
    margin: 0 auto;
</style>
""", unsafe_allow_html=True)

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o" # "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "prompt" not in st.session_state:
    st.session_state.prompt = None

if 'emotion_counts' not in st.session_state:
    st.session_state.emotion_counts = defaultdict(int)

# Initialize session variables if they do not exist
default_values = {
    'current_index': 0,
    'current_question': 0,
    'score': 0,
    'selected_option': None, 
    'answer_submitted': False,
    'start_quiz': False,
    'start_time': None,
    'is_dialog': False,
    'first_time': True,
    'stop_detection': False,
    'taken_hint': False
    }


for key, value in default_values.items():
    st.session_state.setdefault(key, value)

# Load quiz data
with open('quiz_data.json', 'r', encoding='utf-8') as f:
    quiz_data = json.load(f)

def restart_quiz():
    st.session_state.current_index = 0
    st.session_state.score = 0
    st.session_state.selected_option = None
    st.session_state.answer_submitted = False


def submit_answer():
    # Check if an option has been selected
    if st.session_state.selected_option is not None:
        # Mark the answer as submitted
        st.session_state.answer_submitted = True
        # Check if the selected option is correct
        if st.session_state.selected_option == quiz_data[st.session_state.current_index]['answer']:
            st.session_state.score += 10
    else:
        # If no option selected, show a message and do not mark as submitted
        st.warning("Please select an option before submitting.")


def next_question():
    st.session_state.current_index += 1
    st.session_state.selected_option = None
    st.session_state.answer_submitted = False
    st.session_state['start_time'] = time.time()
    st.session_state.taken_hint = False
    st.session_state.emotion_counts.clear()
    st.session_state.messages = []
    st.session_state.prompt = None


# Function to get a response to a question using OpenAI's API
def get_reply(ques):
    st.session_state.messages.append({"role": "user", "content": ques})
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=False,
        )
        # response = st.write_stream(stream)
        response = stream.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()


# Dialog function to display a message if necessary
@st.dialog("You still thinking?")
def display_dialog(emotions_captured):
    st.write(f"Looks like you are {emotions_captured}!!")
    st.write(f"Check get support tab")


# Title and description 
st.title("Education Assistance Chatbot")
tab1, tab2, tab3 = st.tabs(["Get Support", "Homework", "Emotion"])

# Tab for the Chatbot interface
with tab1:

    # Display existing messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Now prompt for user input
    st.session_state.prompt = st.chat_input("Type something...")

    if st.session_state.prompt:
        st.session_state.messages.append({"role": "user", "content": st.session_state.prompt})
        with st.chat_message("user"):
            st.markdown(st.session_state.prompt)

        with st.chat_message("assistant"):
            stream = client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            )
            response = st.write_stream(stream)
            # response = stream.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": response})

# Tab for the quiz interface
with tab2:
    col1, col2 = st.columns([9, 1])

    if st.button('Start Quiz') and st.session_state.start_time is None:
        st.session_state.start_quiz = True

        st.session_state.start_time = time.time()

    if st.session_state.start_time is not None:

        st.session_state['start_time'] = time.time()
        st.session_state[st.session_state['start_time']] = []

        # Progress bar
        progress_bar_value = (st.session_state.current_index + 1) / len(quiz_data)
        st.metric(label="Score", value=f"{st.session_state.score} / {len(quiz_data) * 10}")
        st.progress(progress_bar_value)

        # Display the question and answer options
        question_item = quiz_data[st.session_state.current_index]
        st.subheader(f"Question {st.session_state.current_index + 1}")
        st.title(f"{question_item['question']}")
        st.write(question_item['information'])

        st.markdown(""" ___""")

        # Answer selection
        options = question_item['options']
        correct_answer = question_item['answer']

        st.session_state['current_question'] = question_item['question']
        st.session_state['current_options'] = question_item['options']
        st.session_state['current_answer'] = question_item['answer']

        if st.session_state.answer_submitted:
            for i, option in enumerate(options):
                label = option
                if option == correct_answer:
                    # get_reply("how can you help")
                    st.success(f"{label} (Correct answer)")
                elif option == st.session_state.selected_option:
                    st.error(f"{label} (Incorrect answer)")
                else:
                    st.write(label)
        else:
            for i, option in enumerate(options):
                if st.button(option, key=i, use_container_width=True):
                    st.session_state.selected_option = option

        st.markdown(""" ___""")

        # Submission button and response logic
        if st.session_state.answer_submitted:
            if st.session_state.current_index < len(quiz_data) - 1:
                st.button('Next', on_click=next_question)
            else:
                st.write(f"Quiz completed! Your score is: {st.session_state.score} / {len(quiz_data) * 10}")
                if st.button('Restart', on_click=restart_quiz):
                    pass
        else:
            if st.session_state.current_index < len(quiz_data):
                st.button('Submit', on_click=submit_answer)

# Tab for emotion detection
with tab3:
    st.header("Emotion Detection")
    # Emotion detection while loop to capture video feed
    if st.session_state.start_quiz and not st.session_state.stop_detection:
        st.write("Emotion detected started")
        stframe = st.empty()
        capture = cv2.VideoCapture(0)
        frame_placeholder = st.empty()
        while capture.isOpened():
            ret, frame = capture.read()
            if not ret:
                break

            results = face_model(frame)
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box[:4])
                    # Extract face region
                    face_img = frame[y1:y2, x1:x2]
                    rgb_frame = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                    pil_frame = Image.fromarray(rgb_frame)
                    # Frame is sent to emotion recognition model
                    emotion_label, emotion_confidence = get_emotion(pil_frame)
                    # update_emotion_counts(emotion_label)
                    st.session_state.emotion_counts[emotion_label] += 1
                    if st.session_state.emotion_counts["Confused"] > 5 and not st.session_state.taken_hint:
                        # Prompt the student 
                        display_dialog("Confused")
                        st.session_state.taken_hint = True
                        # Prompt template that is passed to chatgpt
                        guide_prompt = (
                            "A high school student is doing a homework in online platform. " +
                            "Looks like the student is struggling to answer the current quiz question. " +
                            "Please help the student to make him answer on his own by providing him brief hints. " +
                            "DO NOT give the answer!"
                            "The following is the question and options presented to the student."
                        ) + f"\nQuestion: {st.session_state['current_question']}" + f"\nOptions: {st.session_state['current_options']}"
                        get_reply(guide_prompt)
                        

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Put emotion label on the bounding box
                    label = f"{emotion_label}: {emotion_confidence:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            frame_resized = cv2.resize(frame, (320, 240))
            cv2.imshow("Emotion Detection", frame_resized)

            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),channels="RGB")
            if cv2.waitKey(1) & 0xFF == ord("q"):
                st.session_state.stop_detection = True
                st.write("Emotion detection stopped")
                break

        capture.release()
        cv2.destroyAllWindows()
