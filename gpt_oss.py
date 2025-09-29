import os
from groq import Groq
client = Groq(

    api_key="your_api_key",

)
import pandas as pd
import json
from tqdm import tqdm
import time
from groq import Groq

def generate_code_for_prompts(csv_path, output_json_path):
    """
    Reads prompts from a CSV, generates code using Groq, and saves to JSON.

    Args:
        csv_path (str): Path to the input CSV file ('dev_v2.csv').
        output_json_path (str): Path for the output JSON file ('submission.json').
    """
    print(f"\nLoading prompts from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file {csv_path} was not found.")
        return

    # Initialize the Groq client
    # client = Groq()

    # This is the system instruction that guides the model's behavior.
    # It's designed to produce clean, correct, and properly formatted code.
    system_content = (
        "You are an expert Python programmer. Your task is to write a clean, "
        "efficient, and robust Python function based on the user's instruction. "
        "The instruction will be in Bengali. "
        "If you call any function make sure you have defined it"
        "If you have made any import make sure to import it"
        "Make sure you consider all the edge cases and both positive and negative corner cases"
        "Make sure the code is extremely robust and can work on any test cases provided"
        "Check if the syntax are correct and parenthesis opening and closing are done properly"
        "Your code must handle all possible edge cases. "
        "The final output must ONLY be the Python code, enclosed in a single "
        "```python ... ``` markdown block. Do not add any explanation or text "
        "Do not add any comments"
        "The example test_list are {test_list}\n\n"
       "here's the instruction:"
    )

    responses = []
    print(f"Generating code for {len(df)} prompts...")

    # Using tqdm for a nice progress bar
    #for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Generating Code"):
    for index, row in tqdm(df.iloc[404:].iterrows(), total=df.shape[0] - 404, desc="Generating Code"):

        try:
            # if row['id']==2:
            #   break
            print("current row id", row['id'])
            system = system_content.format(test_list=row['test_list'])
            print(system)
            messages = [
                {
                    "role": "system",
                    "content": system
                },
                {
                    "role": "user",
                    "content": row['instruction']
                }
            ]

            completion = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=messages,
                temperature=1,
                max_tokens=8192,
                top_p=1,
                stream=False,
                stop=None
            )

            # Clean the generated code to remove markdown fences and extra whitespace
            generated_code = completion.choices[0].message.content.strip()
            if generated_code.startswith("```python"):
                generated_code = generated_code[9:].strip()
            if generated_code.endswith("```"):
                generated_code = generated_code[:-3].strip()

            responses.append({
                "id": row['id'],
                "response": generated_code
            })

            # Handle rate limiting
            total_tokens = completion.usage.total_tokens if completion.usage else 0
            req_sleep = 60.0 / 30  # 2 seconds for 30 req/min
            tok_sleep = total_tokens / (8000 / 60.0) if total_tokens > 0 else 0
            sleep_time = max(req_sleep, tok_sleep)
            time.sleep(sleep_time)

        except Exception as e:
            print(f"Error generating code for ID {row['id']}: {e}")
            # Add a blank response on error to maintain submission file structure
            responses.append({
                "id": row['id'],
                "response": ""
            })
            # Still sleep the minimum for requests
            time.sleep(60.0 / 30)

    # Save the results to the submission file
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(responses, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Successfully generated code and saved to {output_json_path}")

if __name__ == "__main__":
    # Define file paths
    dev_csv_path = '/content/test_v1.csv'
    submission_file_path = 'submission.json'

    # Step 1: Generate the code and create submission.json
    generate_code_for_prompts(dev_csv_path, submission_file_path)
