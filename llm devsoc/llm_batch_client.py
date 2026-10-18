"""Send prompts from a text file to the Groq API and save the responses as JSON."""

import argparse
import json
import os
import sys
import time

import requests

API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-20b"  # check console.groq.com/docs/models for current names
REQUEST_TIMEOUT_SECONDS = 30
DELAY_BETWEEN_REQUESTS_SECONDS = 1       # stay under rate limits


def read_prompts(file_path):
    """Return a list of non-empty lines from the input text file."""
    with open(file_path, "r", encoding="utf-8") as input_file:
        return [line.strip() for line in input_file if line.strip()]


def get_llm_response(prompt, api_key, model):
    """Send one prompt to the API and return the model's reply text."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    response = requests.post(
        API_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT_SECONDS
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def save_results(results, output_path):
    """Write the list of results to a JSON file."""
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(results, output_file, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Batch-query an LLM API from a text file.")
    parser.add_argument("--input", default="prompts.txt", help="text file, one prompt per line")
    parser.add_argument("--output", default="responses.json", help="where to save the JSON")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        sys.exit("Set the GROQ_API_KEY environment variable first.")

    prompts = read_prompts(args.input)
    results = []

    for index, prompt in enumerate(prompts, start=1):
        print(f"[{index}/{len(prompts)}] {prompt[:50]}")
        try:
            response_text = get_llm_response(prompt, api_key, args.model)
            error_message = None
        except requests.RequestException as error:
            response_text = None
            error_message = str(error)

        results.append({
            "id": index,
            "prompt": prompt,
            "response": response_text,
            "error": error_message,
        })
        time.sleep(DELAY_BETWEEN_REQUESTS_SECONDS)

    save_results(results, args.output)
    print(f"Saved {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()