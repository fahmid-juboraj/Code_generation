#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fix Failed Codes using GPT-OSS:20B via Ollama
"""

import subprocess
import sys
import os
import time
import json
import re
import ast
from datetime import datetime
from pathlib import Path

import pandas as pd

# ---------- INSTALL DEPENDENCIES ----------
print("📦 Installing required packages...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "openai", "tqdm"])
print("✅ Packages installed successfully!")

from openai import OpenAI
from tqdm import tqdm

# ---------- DISPLAY HELPERS ----------
def print_status(message, level="INFO"):
    levels = {
        "INFO": "\033[94m[INFO]\033[0m",
        "SUCCESS": "\033[92m[SUCCESS]\033[0m",
        "WARNING": "\033[93m[WARNING]\033[0m",
        "ERROR": "\033[91m[ERROR]\033[0m",
        "PROCESS": "\033[95m[PROCESS]\033[0m"
    }
    print(f"{levels.get(level, '[INFO]')} {message}")

# ---------- INSTALL OLLAMA ----------
print_status("Setting up Ollama...", "PROCESS")
result = os.system("curl -fsSL https://ollama.com/install.sh | sh 2>/dev/null")
if result == 0:
    print_status("Ollama installed successfully!", "SUCCESS")
else:
    print_status("Ollama installation had warnings but may still work", "WARNING")

print("Starting Ollama server...")
os.system("nohup ollama serve > /tmp/ollama_serve_stdout.log 2>/tmp/ollama_serve_stderr.log &")
time.sleep(5)
running = os.system("ps aux | grep -E 'ollama serve' | grep -v grep > /dev/null 2>&1")
if running == 0:
    print_status("Ollama server is running!", "SUCCESS")
else:
    print_status("Ollama server failed to start.", "ERROR")

print_status("Downloading GPT-OSS:20B Model (~13GB)", "PROCESS")
start_time = time.time()
result = os.system("ollama pull gpt-oss:20b")
end_time = time.time()
if result == 0:
    print_status(f"Model downloaded successfully in {(end_time - start_time)/60:.1f} minutes!", "SUCCESS")
else:
    print_status("Model download failed.", "ERROR")

print("\n📋 Available models:")
os.system("ollama list")

# ---------- CHAT WRAPPER ----------
class GPTOSSChat:
    def __init__(self, system_message="You are a helpful AI assistant."):
        self.client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        self.conversation_history = []
        self.system_message = system_message
        
    def set_system_message(self, message):
        self.system_message = message
        print_status(f"System message updated: {message[:100]}...", "INFO")
    
    def clear_history(self):
        self.conversation_history = []
        print_status("Conversation history cleared", "INFO")
    
    def chat(self, user_input, include_history=True):
        try:
            messages = [{"role": "system", "content": self.system_message}]
            if include_history:
                messages.extend(self.conversation_history)
            messages.append({"role": "user", "content": user_input})
            print_status("Thinking...", "PROCESS")
            response = self.client.chat.completions.create(
                model="gpt-oss:20b",
                messages=messages
            )
            assistant_response = response.choices[0].message.content
            if include_history:
                self.conversation_history.append({"role": "user", "content": user_input})
                self.conversation_history.append({"role": "assistant", "content": assistant_response})
            return assistant_response
        except Exception as e:
            print_status(f"Error: {str(e)}", "ERROR")
            return None

chat = GPTOSSChat()
print_status("Chat interface ready!", "SUCCESS")

# ---------- CONFIG ----------
CANDIDATE_CSVS = [
    Path("./failed_with_details_test-3_en.csv"),
]
CSV_PATH = next((p for p in CANDIDATE_CSVS if p.exists()), None)
if CSV_PATH is None:
    raise FileNotFoundError("Could not find a CSV. Checked: " + ", ".join(map(str, CANDIDATE_CSVS)))

OUT_JSON = Path("error_submission_fixed.json")
GEN_FAILS_CSV = Path("gen_failures.csv")
MAX_PROMPT_CHARS = 3500
RUN_ASSERT_TESTS = True

# ---------- HELPERS ----------
NBSP = "\u00A0"
ZWSP = "\u200B"
DEF_RE = re.compile(r"^\s*def\s+([A-Za-z_]\w*)\s*\(", re.M)

def normalize_ws(s: str) -> str:
    return ("" if s is None else str(s)).replace(NBSP, " ").replace(ZWSP, "").strip()

def extract_text(obj):
    if isinstance(obj, str):
        return obj
    for attr in ("content", "text", "message", "output"):
        if hasattr(obj, attr):
            return getattr(obj, attr)
        if isinstance(obj, dict) and attr in obj:
            return obj[attr]
    return str(obj)

def clean_code_fences(text: str) -> str:
    if not isinstance(text, str):
        return text
    t = text.strip()
    t = re.sub(r"^```[a-zA-Z0-9_+-]*\s*\n?", "", t)
    t = re.sub(r"\n?```$", "", t)
    return t.strip()

def ensure_str(x):
    return "" if x is None else str(x)

def shorten(s: str, max_chars=MAX_PROMPT_CHARS) -> str:
    s = ensure_str(s)
    if len(s) <= max_chars:
        return s
    half = max_chars // 2
    return s[:half] + "\n...\n" + s[-half:]

def extract_expected_fn_names(test_list_text: str):
    pat = re.compile(r"\b([A-Za-z_]\w*)\s*\(", re.MULTILINE)
    seen, out = set(), []
    for n in pat.findall(test_list_text or ""):
        if n not in seen:
            seen.add(n); out.append(n)
    return out

def strip_comments_markdown(s: str) -> str:
    lines = s.splitlines()
    cleaned = [ln for ln in lines if not ln.strip().startswith("```")]
    while cleaned and cleaned[0].lstrip().startswith("#"):
        cleaned.pop(0)
    while cleaned and cleaned[-1].lstrip().startswith("#"):
        cleaned.pop()
    return "\n".join(cleaned).strip()

def has_any_expected_def(code: str, expected_names) -> bool:
    if not expected_names:
        return bool(DEF_RE.search(code))
    defs = DEF_RE.findall(code)
    return any(d in expected_names for d in defs)

def parse_test_list_str(test_list_str: str):
    test_list_str = (test_list_str or "").strip()
    if not test_list_str:
        return [], None
    try:
        tests = ast.literal_eval(test_list_str)
        if not isinstance(tests, list) or not all(isinstance(s, str) for s in tests):
            return [], "test_list is not a list[str]"
        return tests, None
    except Exception as e:
        return [], f"literal_eval failed: {e}"

def build_exec_namespace():
    import math, cmath, re as _re, heapq, itertools, collections, functools, statistics, fractions, decimal, random
    ns = {
        "__name__": "__main__",
        "math": math, "cmath": cmath, "re": _re, "heapq": heapq,
        "itertools": itertools, "collections": collections, "functools": functools,
        "statistics": statistics, "fractions": fractions, "decimal": decimal, "random": random,
    }
    return ns

def exec_user_code_into_ns(code_str: str, ns: dict):
    compiled = compile(code_str, "<model_code>", "exec")
    exec(compiled, ns, ns)
    return ns

def run_assert_tests(assert_strings, ns):
    for s in assert_strings:
        code = compile(s, "<test>", "exec")
        exec(code, ns, ns)

def format_system_message():
    return (
        "You are a Competitive Programmer. Your task is to REPAIR or REWRITE a function so that "
        "it passes the provided assertion tests.\n"
        "OUTPUT RULES:\n"
        "• Output ONLY valid Python code (no prose, comments, markdown).\n"
        "• Use Python 3 standard library only.\n"
        "• Define the EXACT function name/signature expected by the tests.\n"
        "• No printing, no input, no side effects.\n"
    )

def build_user_message(rid, instr_s, clean_s, tests_s, reason_s, fn_hint):
    return f"""
Task ID: {rid}

INSTRUCTION:
{instr_s}

FAILING CODE:
{clean_s}

TEST CASES:
{tests_s}

FAILURE REASON:
{reason_s}

Fix the code to pass all tests.
Expected function(s): {fn_hint}

Return ONLY corrected Python code:
""".strip()

# ---------- LOAD DATA ----------
data = pd.read_csv(CSV_PATH, dtype=str, keep_default_na=False)

# ---------- MAIN LOOP ----------
results = []
fail_rows = []

for _, row in tqdm(data.iterrows(), total=len(data), desc="Fixing failed codes"):
    rid = ensure_str(row.get("id", "")).strip()
    instr = ensure_str(row.get("instruction_en", ""))
    tests = ensure_str(row.get("test_list", ""))
    fail_reason = ensure_str(row.get("failures", ""))
    code_clean = ensure_str(row.get("generated_code_clean", ""))

    instr_s = shorten(instr, 3000)
    tests_s = shorten(tests, 3400)
    reason_s = shorten(fail_reason, 1500)
    clean_s = shorten(code_clean, 3000)

    expected_fns = extract_expected_fn_names(tests_s)
    fn_hint = ", ".join(expected_fns[:5]) if expected_fns else "N/A"

    system_msg = format_system_message()
    user_msg = build_user_message(rid, instr_s, clean_s, tests_s, reason_s, fn_hint)

    chat.set_system_message(system_msg)
    code_response = chat.chat(user_msg, include_history=False)

    text = normalize_ws(clean_code_fences(extract_text(code_response)))
    text = strip_comments_markdown(text)

    def_ok = has_any_expected_def(text, expected_fns)
    test_result = {"ok": True, "error": None}

    if RUN_ASSERT_TESTS and tests_s.strip():
        asserts, parse_err = parse_test_list_str(tests)
        if parse_err:
            test_result = {"ok": False, "error": f"TEST_LIST_PARSE_ERROR: {parse_err}"}
        else:
            try:
                ns = build_exec_namespace()
                exec_user_code_into_ns(text, ns)
                run_assert_tests(asserts, ns)
            except AssertionError as ae:
                test_result = {"ok": False, "error": f"ASSERTION_FAILED: {ae}"}
            except Exception as ex:
                test_result = {"ok": False, "error": f"TEST_RUNTIME_ERROR: {type(ex).__name__}: {ex}"}

    if not def_ok or not test_result["ok"]:
        reason = []
        if not def_ok:
            reason.append("NO_MATCHING_DEF")
        if not test_result["ok"]:
            reason.append(test_result["error"] or "TESTS_FAILED")
        fail_rows.append({
            "id": rid,
            "reason": " | ".join(reason),
            "expected_functions": ", ".join(expected_fns),
            "preview": text[:2000],
        })

    results.append({"id": int(rid) if rid.isdigit() else rid, "response": text})

# ---------- SAVE OUTPUT ----------
OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print_status(f"Saved {len(results)} repaired responses → {OUT_JSON.resolve()}", "SUCCESS")

if fail_rows:
    pd.DataFrame(fail_rows).to_csv(GEN_FAILS_CSV, index=False)
    print_status(f"Guardrail/test failures saved → {GEN_FAILS_CSV.resolve()}", "WARNING")
else:
    print_status("All generations passed guardrails and tests ✅", "SUCCESS")
