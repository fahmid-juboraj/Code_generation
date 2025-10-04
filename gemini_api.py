import os
import json
import time
import pandas as pd
import google.generativeai as genai
import ast
import sys
import signal
from tqdm import tqdm


try:
    # A more secure way to handle the API key, especially in Colab
    from google.colab import userdata
    GOOGLE_API_KEY = "your_api_key"  # Use userdata.get for Colab secrets
except (ImportError, KeyError):
    # Fallback for local environments
    GOOGLE_API_KEY = "your_api_key"  # Use env var or paste here

genai.configure(api_key=GOOGLE_API_KEY)

print("Libraries imported and API key configured.")


def generate_code_for_prompts(csv_path, input_json_path, output_json_path, target_ids):
    """
    Reads prompts from a CSV for specific IDs, loads existing code from JSON,
    generates/fixes code using Gemini, and saves to JSON.

    Args:
        csv_path (str): Path to the input CSV file ('test_v1.csv').
        input_json_path (str): Path to the input JSON file with existing code.
        output_json_path (str): Path for the output JSON file ('submission.json').
        target_ids (list): List of specific IDs to process.
    """
    print(f"\nLoading prompts from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file {csv_path} was not found.")
        return

    print(f"\nLoading existing code from {input_json_path}...")
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            input_json = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file {input_json_path} was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: The file {input_json_path} is not a valid JSON.")
        return

    # Initialize the Gemini model
    # Using gemini-1.5-flash as it's fast and capable for code generation/fixing.
    model = genai.GenerativeModel('gemini-2.5-flash')

    # This is the system instruction that guides the model's behavior.
    # It's designed to fix code to be clean, correct, robust, and error-free.
    system_prompt = (
        "You are an expert Python competitive programmer of ICPC level with extensive experience in debugging and fixing code. "
        "Your task is to scrutinize, validate, and fix the provided code to ensure it passes all hidden test cases, "
        "including edge cases, corner cases, positive, negative, and zero cases, with no compilation, syntax, runtime, or assertion errors. "
        "The instruction will be in Bengali—translate it accurately to English before proceeding. "
        "First, deeply analyze the instruction: Understand exact requirements, input types, output expectations, and constraints. "
        "Evaluate the existing code: Simulate its execution on the given test_list examples and identify failures. "
        "Generate additional extremely hard test cases where the code might fail, such as: "
        "- Extreme values (e.g., float('inf'), float('nan'), very large integers). "
        "- Empty inputs (lists, strings, dicts, etc.). "
        "- Single-element or zero-element collections. "
        "- Negative, zero, very large/small numbers. "
        "- Invalid types (add type checks or graceful handling). "
        "- Boundary conditions, overflows, underflows. "
        "- Duplicates, unsorted/sorted data, case sensitivity. "
        "- Special characters, Unicode, Bengali-specific encodings. "
        "- Error-prone scenarios (division by zero, index errors, key errors)—prefer preventive checks over try-except. "
        "- Performance for large inputs—optimize to O(n) or better. "
        "For each hard test case, mentally simulate failures, identify issues, and fix the code to handle them perfectly. "
        "If the function requires imports, include them at the top. Define helper functions fully if needed. "
        "Double-check syntax: Proper indentation, matching parentheses/brackets, no typos. "
        "Ensure the code is Pythonic, readable, self-contained—no external dependencies beyond standard libraries. "
        "Do not use time.sleep or import sys unless absolutely necessary. "
        "The final output MUST be ONLY the complete fixed Python code, enclosed in a single "
        "```python ... ``` markdown block. Do not add any explanations, comments, or extra text. "
        "The code itself should have NO comments. "
        "The example test_list are {test_list}\n\n"
        "here's the instruction:"
    )

    # Create a dictionary from input_json for quick lookup by id
    json_code_dict = {item['id']: item['response'] for item in input_json}

    # Filter df to target_ids only
    filtered_df = df[df['id'].isin(target_ids)]

    responses = []
    print(f"Fixing code for {len(filtered_df)} specific IDs...")

    # Using tqdm for a nice progress bar
    for index, row in tqdm(filtered_df.iterrows(), total=filtered_df.shape[0], desc="Fixing Code"):
        try:
            print(f"Processing row id {row['id']}")
            existing_code = json_code_dict.get(row['id'], "")

            # Format system prompt with test_list
            formatted_system = system_prompt.format(test_list=row['test_list'])

            if existing_code:
                # If code exists, prompt to fix it
                user_prompt = (
                    f"Scrutinize and fix the following code to ensure no errors and passes all test cases, including hidden ones. "
                    f"Generate hard test cases to identify issues and fix accordingly. "
                    f"Existing code: ```python\n{existing_code}\n```\n\n"
                    f"Instruction: {row['instruction']}"
                )
            else:
                # If no code, generate new but warn (though per requirement, assume exists)
                user_prompt = (
                    f"Generate robust code for the instruction, considering all hard test cases to avoid any errors. "
                    f"Instruction: {row['instruction']}"
                )

            # Generate content with system instruction and user prompt
            response = model.generate_content(
                [formatted_system, user_prompt],
                generation_config=genai.types.GenerationConfig(
                    temperature=0.9,
                    max_output_tokens=8192,
                    top_p=1
                )
            )

            # Clean the generated code to remove markdown fences and extra whitespace
            generated_code = response.text.strip()
            if generated_code.startswith("```python"):
                generated_code = generated_code[9:]
            if generated_code.endswith("```"):
                generated_code = generated_code[:-3]
            generated_code = generated_code.strip()

            responses.append({
                "id": row['id'],
                "response": generated_code
            })
        except Exception as e:
            print(f"Error fixing code for ID {row['id']}: {e}")
            # Add existing or blank response on error
            responses.append({
                "id": row['id'],
                "response": existing_code if existing_code else ""
            })

        # Rate limiting: Gemini free tier has limits, sleep 2 seconds
        time.sleep(2)

    # Save the results to the submission file
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(responses, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Successfully fixed code and saved to {output_json_path}")


# ==============================================================================
# Step 3: Evaluate the Generated Code
# ==============================================================================
# Timeout handler to prevent infinite loops
def handler(signum, frame):
    raise TimeoutError("Execution timed out after 30 seconds")


def evaluate_combined_data(res_data, ref_data):
    # Convert to DataFrames for easy merging
    res_df = pd.DataFrame(res_data)[['id', 'response']]
    ref_df = pd.DataFrame(ref_data)
    # Drop the response column from ref_df if it exists
    if 'response' in ref_df.columns:
        ref_df = ref_df.drop(columns=['response'])

    # Merge the data on 'id'
    combined_df = ref_df.merge(res_df, on='id', how='left')

    # Convert back to list of dictionaries
    combined_data = combined_df.to_dict('records')

    global_correct = 0
    global_total = len(combined_data)

    for entry in combined_data:
        entry_id = entry['id']
        response_code = entry.get('response', '')  # Use empty string if response missing
        test_list_raw = '"'+entry['test_list']+'"'
        if response_code is not None:
            response_code = response_code.strip('` \n').replace('python\n', '').strip()


        print(f"Executing Sample ID: {entry_id}")

        # 🚫 Skip code if it contains time.sleep (case-insensitive)
        if "time.sleep" in response_code.lower():
            print(f"⏭️ Skipping Code Execution: contains time.sleep()")
            continue

        correct = 0


        # Parse the test cases safely
        try:
            inner_str = ast.literal_eval(test_list_raw)
            test_cases = ast.literal_eval(inner_str)
        except Exception as e:
            print(f"❌❌❌❌ Failed to parse test_list: {e} ❌❌❌")
            continue

        # Create a shared namespace for exec
        namespace = {}

        try:
            # Set timeout for function definition
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(30)
            exec(response_code, namespace)
            signal.alarm(0)  # cancel timer if finished early
        except TimeoutError:
            print(f"⏱️ Timeout in function definition. Skipping test case execution for this ID.\n")
            continue
        except Exception as e:
            print(f"❌ Error in function definition: {e}. Skipping test case execution for this ID.\n")
            continue

        passed = True
        # Run each assert statement
        for i, assert_stmt in enumerate(test_cases):
            try:
                signal.alarm(30)  # 30 seconds per test case
                exec(assert_stmt, namespace)
                signal.alarm(0)
                correct += 1
            except TimeoutError:
                print(f"⏱️ Test case {i + 1} timed out. Skipping all remaining test cases for this ID.")
                passed = False
                break  # Exit loop on timeout
            except AssertionError:
                print(f"❌ Test case {i + 1} failed: assertion error. Skipping all remaining test cases for this ID.")
                passed = False
                break  # Exit loop on timeout
            except Exception as e:
                print(f"⚠️ Test case {i + 1} exception: {e}. Skipping all remaining test cases for this ID.")
                passed = False
                break  # Exit loop on timeout
            finally:
                signal.alarm(0)
        if passed:
            print(f"✅ ID {entry_id} Passed all test cases.\n")
        else:
            print(f"❌ ID {entry_id} Failed some test cases.\n")

        total = len(test_cases)
        if correct == total:
            global_correct += 1

    return global_correct, global_total


# ==============================================================================
# Main Execution Block
# ==============================================================================
if __name__ == "__main__":
    # Define file paths and target_ids
    dev_csv_path = '/content/test_v1_subset.csv'
    input_json_path = '/content/new_tc_submission.json'  # Path to JSON with existing code
    submission_file_path = 'submission_new.json'
    target_ids = [28,51,91,148,303,357,426] #figure out ids with failed cases

   
    generate_code_for_prompts(dev_csv_path, input_json_path, submission_file_path, target_ids)

   
    if os.name == 'nt':
        print("\nWarning: Timeout functionality is not supported on Windows. "
              "Evaluation will run without time limits.")
    else:
        # Read submission.json
        with open(submission_file_path, 'r', encoding='utf-8') as f:
            res_data = json.load(f)

        # Read reference CSV
        ref_df = pd.read_csv(
            dev_csv_path,
            dtype=str,                # keep everything as string to avoid NaN
            keep_default_na=False     # empty cells stay '', not NaN
        )
        # Ensure 'id' is numeric to merge cleanly (adjust to int if your JSON ids are ints)
        ref_df['id'] = ref_df['id'].astype(int)
        ref_data = ref_df.to_dict('records')

        # Evaluate the combined data
        correct, all_total = evaluate_combined_data(res_data, ref_data)

        # Calculate and print accuracy
        scores = {
            "accuracy": correct / all_total if all_total > 0 else 0.0
        }
        print(f"\nPass@1: {correct}/{all_total} = {scores['accuracy']:.2f}")
