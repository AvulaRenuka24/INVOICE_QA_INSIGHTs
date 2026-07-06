import sys
import time
import logging

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Logging Configuration

logging.basicConfig(
    filename="logs/llm_calls.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Custom Exception
class LLMUnavailable(Exception):
    """Raised when the LLM cannot be loaded or used."""
    pass

# Model Configuration
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto"
    )

except Exception as e:
    raise LLMUnavailable(f"Unable to load model: {e}")

# Generate Function
def generate(messages, max_tokens=256):
    """
    Generate a response from Qwen.

    Parameters:
        messages (list): Chat messages
        max_tokens (int): Maximum new tokens

    Returns:
        str: Generated text
    """

    try:
        start_time = time.time()

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(
            text,
            return_tensors="pt"
        ).to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0,
            do_sample=False
        )

        generated_tokens = outputs[0][inputs.input_ids.shape[-1]:]

        response = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        )

        end_time = time.time()

        latency = end_time - start_time

        token_count = len(generated_tokens)

        logging.info(
            f"Latency={latency:.2f}s | Tokens={token_count}"
        )

        return response.strip()

    except Exception as e:
        raise LLMUnavailable(f"Generation failed: {e}")

# Command Line Test

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python llm.py \"Your Prompt\"")
        sys.exit(1)

    prompt = sys.argv[1]

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    print(generate(messages))