# test/test_mlir_conversion.py

import os
import subprocess
import sys

def main():
    input_file = "test.mlir"
    output_file = "output.mlir"
    script = ".run_q2i_opt.sh"

    # Clean up previous output
    if os.path.exists(output_file):
        os.remove(output_file)

    # Run conversion
    result = subprocess.run(
        [script, input_file, output_file],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        print("❌ Conversion script failed:")
        print(result.stderr)
        sys.exit(1)

    if not os.path.exists(output_file):
        print("❌ Output MLIR file was not created.")
        sys.exit(1)

    with open(output_file) as f:
        contents = f.read()
        if "module" not in contents:
            print("❌ Output does not appear to be valid MLIR (missing 'module').")
            sys.exit(1)

    print("✅ MLIR conversion test passed.")

if __name__ == "__main__":
    main()
