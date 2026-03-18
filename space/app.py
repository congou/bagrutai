"""
BagrutAI — HuggingFace Space app
Serves the fine-tuned Gemma 2 2B LoRA model as a Gradio chat interface.

Deploy instructions:
1. Create a new Space at https://huggingface.co/new-space
2. Choose "Gradio" SDK, "CPU basic (free)" hardware
3. Upload this file and requirements.txt to the Space
4. Change ADAPTER_REPO below to your HuggingFace username
"""

import os
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ⚠️ Change "congou" to your actual HuggingFace username!
ADAPTER_REPO = "congou/bagrutai-lora"
BASE_MODEL = "google/gemma-2-2b-it"

SYSTEM_PROMPT = "אתה מורה עזר לאזרחות שמכין תלמידים לבגרות בישראל."
MAX_NEW_TOKENS = 300

# HF token from Space Secrets (needed for gated Gemma model)
HF_TOKEN = os.environ.get("HF_TOKEN")

# Load model at startup
print("Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_REPO, token=HF_TOKEN)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    torch_dtype=torch.float32,
    token=HF_TOKEN,
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_REPO, token=HF_TOKEN)
model.eval()
print("Model ready!")


def chat_fn(message, history):
    """Generate a response to a civics question."""
    user_input = f"{SYSTEM_PROMPT}\n\n{message}"
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_input}],
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    answer_tokens = out[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()


# Build the Gradio interface
demo = gr.ChatInterface(
    fn=chat_fn,
    multimodal=False,
    title="BagrutAI — מורה עזר לאזרחות",
    description="שאל שאלות מחומר האזרחות וקבל תשובות בסגנון הבגרות",
    examples=[
        "הציגו את עיקרי חוק השבות.",
        "הציגו את המושג ביקורת שיפוטית.",
        "הציגו את המושג דמוקרטיה מתגוננת.",
        "הציגו את עיקרי הסדר הסטטוס־קוו.",
        "מהם התנאים ההכרחיים לקיום בחירות דמוקרטיות?",
    ],
    cache_examples=False,
)

demo.launch()
