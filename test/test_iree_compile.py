import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR / "data"
Q2I_OPT = TEST_DIR.parent / "build" / "tool" / "q2i-opt"
IREE_BACKEND = "llvm-cpu"

LOWERED_MLIR_INPUTS = [
    "01_output.mlir",
    "02_output.mlir",
    "03_output.mlir",
    "04_output.mlir",
    "05_output.mlir",
    "06_output.mlir",
]

QUAKE_INPUTS = [
    "01_quake_alloca.mlir",
    "02_quake_veq_size.mlir",
    "03_quake_dealloc.mlir",
    "04_quake_concat.mlir",
    "05_quake_extractref.mlir",
    "06_quake_initializestate.mlir",
]


def require_iree_compile(test_case: unittest.TestCase) -> str:
    iree_compile = shutil.which("iree-compile")
    if iree_compile is None:
        test_case.skipTest("iree-compile not found in PATH")
    return iree_compile


def run_iree_compile(iree_compile: str, input_file: Path, output_file: Path):
    return subprocess.run(
        [
            iree_compile,
            str(input_file),
            f"--iree-hal-target-backends={IREE_BACKEND}",
            "-o",
            str(output_file),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def run_q2i_conversion(input_file: Path, output_file: Path):
    return subprocess.run(
        [
            str(Q2I_OPT),
            str(input_file),
            "--quake-to-standard",
            "-o",
            str(output_file),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


class TestIreeCompileAvailable(unittest.TestCase):
    def test_iree_compile_available(self):
        iree_compile = require_iree_compile(self)
        self.assertTrue(os.access(iree_compile, os.X_OK), f"{iree_compile} is not executable")


class TestIreeCompileLoweredMlirInputs(unittest.TestCase):
    def test_lowered_mlir_inputs_compile_with_iree(self):
        iree_compile = require_iree_compile(self)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            for lowered_mlir_input in LOWERED_MLIR_INPUTS:
                with self.subTest(input=lowered_mlir_input):
                    input_file = DATA_DIR / lowered_mlir_input
                    output_file = temp_dir_path / f"{input_file.stem}.vmfb"

                    result = run_iree_compile(iree_compile, input_file, output_file)

                    self.assertEqual(
                        result.returncode,
                        0,
                        f"iree-compile failed for {lowered_mlir_input}:\n{result.stderr}",
                    )
                    self.assertTrue(output_file.exists(), f"IREE output not created: {output_file}")
                    self.assertGreater(output_file.stat().st_size, 0, f"IREE output is empty: {output_file}")


class TestQuakeToIreeEndToEnd(unittest.TestCase):
    def test_quake_inputs_lower_and_compile_with_iree(self):
        iree_compile = require_iree_compile(self)
        self.assertTrue(Q2I_OPT.exists(), f"q2i-opt not found at {Q2I_OPT}")
        self.assertTrue(os.access(Q2I_OPT, os.X_OK), f"q2i-opt is not executable: {Q2I_OPT}")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            for quake_input in QUAKE_INPUTS:
                with self.subTest(input=quake_input):
                    input_file = DATA_DIR / quake_input
                    lowered_file = temp_dir_path / f"{input_file.stem}_lowered.mlir"
                    iree_output_file = temp_dir_path / f"{input_file.stem}.vmfb"

                    q2i_result = run_q2i_conversion(input_file, lowered_file)
                    self.assertEqual(
                        q2i_result.returncode,
                        0,
                        f"q2i-opt failed for {quake_input}:\n{q2i_result.stderr}",
                    )
                    self.assertTrue(lowered_file.exists(), f"Lowered MLIR not created: {lowered_file}")

                    iree_result = run_iree_compile(iree_compile, lowered_file, iree_output_file)
                    self.assertEqual(
                        iree_result.returncode,
                        0,
                        f"iree-compile failed for lowered {quake_input}:\n{iree_result.stderr}",
                    )
                    self.assertTrue(iree_output_file.exists(), f"IREE output not created: {iree_output_file}")
                    self.assertGreater(
                        iree_output_file.stat().st_size,
                        0,
                        f"IREE output is empty: {iree_output_file}",
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
