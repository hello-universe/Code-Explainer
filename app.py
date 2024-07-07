"""
Install the Google AI Python SDK

$ pip install google-generativeai

See the getting started guide for more information:
https://ai.google.dev/gemini-api/docs/get-started/python
"""

import os
import gradio as gr

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Create the model
# See https://ai.google.dev/api/python/google/generativeai/GenerativeModel

def codeExplaination(input_code):
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
        # safety_settings = Adjust safety settings
        # See https://ai.google.dev/gemini-api/docs/safety-settings
        system_instruction="Your task is to act as a Code Explainer.\nI'll give you a Code Snippet.\n"
                           "First, compute the final output of the code.\nThen, your job is to explain the Code Snippet"
                           " step-by-step.\nBreak down the code into as many steps as possible.\nShare intermediate "
                           "checkpoints along with results.\nState your Steps and Checkpoints in your output.\nFew good "
                           "examples of Python code output between #### separator:\n"
                           "####\nExample 1: Code Snippet\nx = 10\ndef foo():\n    global x\n    x = 5\nfoo()\nprint(x)"
                           "\n\nCorrect answer: 5\nExplanation: Inside the foo function, the global keyword is used to "
                           "modify the global variable x to be 5.\nSo, print(x) outside the function prints the modified "
                           "value, which is 5.\n---------------------\n"
                           "Example 2: Code Snippet\ndef modify_list(input_list):\n    input_list.append(4)\n    "
                           "input_list = [1, 2, 3]\nmy_list = [0]\nmodify_list(my_list)\nprint(my_list)\n\n"
                           "Correct answer: [0, 4]\nExplanation: Inside the modify_list function, an element 4 is appended "
                           "to input_list.\nThen, input_list is reassigned to a new list [1, 2, 3], but this change doesn't "
                           "affect the original list.\nSo, print(my_list) outputs [0, 4].\n####"
                           ,
    )

    chat_session = model.start_chat(
        history=[
        ]
    )
    response = chat_session.send_message(input_code)

    return response.text


#define app UI

theme = gr.themes.Base(
    primary_hue="blue",
    text_size="lg",
    spacing_size="lg",
    font=[gr.themes.GoogleFont('poppins'), 'ui-sans-serif', 'system-ui', 'sans-serif'],
)

demo = gr.Interface(
    fn=codeExplaination,
    inputs=gr.TextArea(lines=15, placeholder="Enter code...", label="Input Code", show_copy_button = True),
    outputs=gr.Markdown(label="Explanation"),
    title = "Code Explainer",
    allow_flagging="never",
    theme = theme,
    css="footer {visibility: hidden}"
)

demo.launch()