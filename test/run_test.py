#!/usr/bin/env python3

import subprocess
import sys
import os
import re

# Config
SOURCE_FILE = "simple.qke"
CONVERTED_FILE = "simple_legal.mlir"
VMFB_FILE = "simple.vmfb"
FUNC_NAME = "__nvqpp__mlirgen__kernel"
INPUT_ARG = "1xi64=1"

def run(cmd, check=True, capture_output=False, text=True):
    print(f"> {' '.join(cmd)}")
    return subprocess.run(cmd, check=check, capture_output=capture_output, text=text)

def check_tools():
    for tool in ["quake2iree-opt", "iree-compile", "iree-run-module"]:
        if not shutil.which(tool):
            print(f"Error: Required tool '{tool}' not found in PATH.")
            sys.exit(1)

def convert_quake_to_iree():
    run([
        "quake2iree-opt",
        SOURCE_FILE,
        "-convert-quake-to-iree",
        "-o", CONVERTED_FILE
    ])

def compile_to_vmfb():
    run([
        "iree-compile",
        CONVERTED_FILE,
        "--target-backends=cpu",
        "--output-format=vm-bytecode",
        "-o", VMFB_FILE
    ])

def run_module():
    result = run([
        "iree-run-module",
        "--module=" + VMFB_FILE,
        "--function=" + FUNC_NAME,
        "--input=" + INPUT_ARG,
        "--print_output"
    ], capture_output=True)
    return result.stdout.strip()

def parse_result(output):
    # Expect output like: [tensor<i1>: false] or a raw 0/1 or list
    match = re.search(r"\bfalse\b|\btrue\b|tensor<\w+>:\s*(\w+)", output)
    if match:
        val = match.group(0).strip()
        if "false" in val:
            return "0"
        elif "true" in val:
            return "1"
    elif output.strip() in ["0", "1"]:
        return output.strip()
    else:
        print(f"Warning: Unrecognized output format: '{output}'")
        return None

def test_result(measured):
    if measured != "0":
        print(f"❌ Test failed: expected '0', got '{measured}'")
        sys.exit(1)
    print("✅ Test passed: most probable result = 0 (as expected)")

def main():
    import shutil
    check_tools()
    convert_quake_to_iree()
    compile_to_vmfb()
    output = run_module()
    print(f"[IREE Output] {output}")
    measured = parse_result(output)
    test_result(measured)

if __name__ == "__main__":
    main()
