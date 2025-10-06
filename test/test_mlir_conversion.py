import os
import subprocess
import sys
from pathlib import Path

def run_test(input_file: Path, q2i_opt_path: Path):
    prefix = input_file.stem.split('_')[0]  # get "01"
    output_file = input_file.with_name(f"{prefix}_output.mlir")

    # Clean previous output
    if output_file.exists():
        output_file.unlink()

    print(f"🔧 Running test for: {input_file.name}")

    # Check the binary exists and is executable
    if not q2i_opt_path.exists() or not os.access(q2i_opt_path, os.X_OK):
        print(f"❌ Error: {q2i_opt_path} not found or not executable.")
        return False

    # Run the q2i-opt tool
    result = subprocess.run(
        [
            str(q2i_opt_path),
            str(input_file),
            "--quake-to-standard",
            "-o", str(output_file)
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Check result
    if result.returncode != 0:
        print(f"❌ {input_file.name}: Conversion failed.")
        print(result.stderr)
        return False

    if not output_file.exists():
        print(f"❌ {input_file.name}: Output file was not created.")
        return False

    with open(output_file) as f:
        contents = f.read()
        if "module" not in contents:
            print(f"❌ {input_file.name}: Output missing 'module'.")
            return False

    print(f"✅ {input_file.name}: Test passed.")
    return True


def main():
    test_dir = Path(__file__).parent
    q2i_opt_path = test_dir.parent / "build" / "tool" / "q2i-opt"  # adjust if path differs

    passed = 0
    total = 0  

    # test 1
    test_file = test_dir / "01_quake_alloca.mlir"
    total += 1
    if run_test(test_file, q2i_opt_path): passed += 1

    # test 2 
    test_file = test_dir / "02_quake_veq_size.mlir"
    total += 1
    if run_test(test_file, q2i_opt_path): passed += 1

    
    # test 3 
    #test_file = test_dir / "03_quake_init_state.mlir"
    #if run_test(test_file, q2i_opt_path):
    #    passed += 1


    print(f"\n ✅ {passed}/{total} tests passed.")
    if passed != total:
        sys.exit(1)


if __name__ == "__main__":
    main()
