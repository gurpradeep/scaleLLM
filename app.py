import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from optimum.onnxruntime import ORTModelForSeq2SeqLM

text = """The Hugging Face Pro plan provides enhanced compute access, 
persistent Spaces, private repositories, and priority inference endpoints for developers building AI applications."""

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

inputs = tokenizer(text, return_tensors="pt")

# Base model
base_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

start = time.time()
_ = base_model.generate(**inputs, max_new_tokens=60)
print(f"Base model latency: {time.time() - start:.2f}s")

# Optimized model
opt_model = ORTModelForSeq2SeqLM.from_pretrained("optimized-bart-int8")

start = time.time()
_ = opt_model.generate(**inputs, max_new_tokens=60)
print(f"Optimized model latency: {time.time() - start:.2f}s")
