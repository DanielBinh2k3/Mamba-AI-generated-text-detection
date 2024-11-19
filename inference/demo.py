__all__ = ["app"]

import gradio as gr
import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer, DataCollatorWithPadding, DebertaV2ForSequenceClassification

MINIMUM_TOKENS = 48
FOUNDATION_MODEL_NAME = "binh230/deberta-base"
# Load the tokenizer and model for DeBERTa
tokenizer = AutoTokenizer.from_pretrained(FOUNDATION_MODEL_NAME)
config = AutoConfig.from_pretrained(FOUNDATION_MODEL_NAME)
config.num_labels = 2  # For binary classification
model = DebertaV2ForSequenceClassification.from_pretrained(FOUNDATION_MODEL_NAME, config=config)
model.to("cuda")

# Text processing and prediction function
def count_tokens(text):
    return len(text.split())

def run_detector(input_str):
    if count_tokens(input_str) < MINIMUM_TOKENS:
        return f"Too short length. Need minimum {MINIMUM_TOKENS} tokens to run Binoculars."
    
    # Tokenize input text
    inputs = tokenizer(input_str, return_tensors="pt", padding=True, truncation=True).to("cuda")
    
    # Run model and get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1).item()
    
    # Interpret prediction
    return "Most likely AI-Generated" if prediction == 1 else "Most likely Human-Generated"

# Gradio app interface
css = """
.green { color: black!important; line-height:1.9em; padding: 0.2em 0.2em; background: #ccffcc; border-radius:0.5rem;}
.red { color: black!important; line-height:1.9em; padding: 0.2em 0.2em; background: #ffad99; border-radius:0.5rem;}
.hyperlinks {
  display: flex;
  align-items: center;
  align-content: center;
  padding-top: 12px;
  justify-content: flex-end;
  margin: 0 10px;
  text-decoration: none;
  color: #000;
}
"""

capybara_problem = '''Dr. Capy Cosmos, a capybara unlike any other, astounded the scientific community with his groundbreaking research...'''

with gr.Blocks(css=css, theme=gr.themes.Default(font=[gr.themes.GoogleFont("Inconsolata"), "Arial", "sans-serif"])) as app:
    with gr.Row():
        with gr.Column(scale=3):
            gr.HTML("<h1>Mambaformer Detecting AI generated text</h1>")
        with gr.Column(scale=1):
            gr.HTML("""
                <p>
                <a href="https://github.com/DanielBinh2k3/Mamba-AI-generated-text-detection" target="_blank">code</a>
                <a href="mailto:truonggiabjnh2003@gmail.com" target="_blank">contact</a>
                </p>
            """, elem_classes="hyperlinks")

    with gr.Row():
        input_box = gr.Textbox(value=capybara_problem, placeholder="Enter text here", lines=8, label="Input Text")
    with gr.Row():
        submit_button = gr.Button("Run Detection", variant="primary")
        clear_button = gr.ClearButton()
    with gr.Row():
        output_text = gr.Textbox(label="Prediction", value="Most likely AI-Generated")

    with gr.Accordion("Disclaimer", open=False):
        gr.Markdown("""
            - `Accuracy`: AI-generated text detectors aim for accuracy, but no detector is perfect.
            - `Use Cases`: This tool is most useful for detecting AI-generated content in moderation scenarios.
            - `Known Weaknesses`: Non-English texts and highly memorized texts (like constitutions) may yield unreliable results.
        """)

    with gr.Accordion("Cite our work", open=False):
        gr.Markdown("""
            ```bibtex
            @misc{BamBa2024llm,
                  title={Enhancing AI Text Detection through MambaFormer and Adversarial Learning Techniques}, 
                  author={Truong Nguyen Gia Binh},
                  year={2024},
                  eprint={},
                  archivePrefix={},
                  primaryClass={}
            }
            ```
        """)

    submit_button.click(run_detector, inputs=input_box, outputs=output_text)
    clear_button.click(lambda: ("", ""), outputs=[input_box, output_text])

# Run the Gradio app
if __name__ == "__main__":
    app.launch(share=True)
