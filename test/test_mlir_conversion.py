import os
import subprocess
import tempfile
import unittest
from pathlib import Path

TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR / "data"
Q2I_OPT = TEST_DIR.parent / "build" / "tool" / "q2i-opt"


def run_conversion(input_file: Path, output_file: Path) -> tuple[bool, str, Path]:
    """Run q2i-opt on input_file and return (success, stderr, output_path)."""
    result = subprocess.run(
        [str(Q2I_OPT), str(input_file), "--quake-to-standard", "-o", str(output_file)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    return result.returncode == 0, result.stderr, output_file


class TestQ2IOptAvailable(unittest.TestCase):
    def test_binary_exists_and_executable(self):
        self.assertTrue(Q2I_OPT.exists(), f"q2i-opt not found at {Q2I_OPT}")
        self.assertTrue(os.access(Q2I_OPT, os.X_OK), f"q2i-opt is not executable")


class TestMlirConversions(unittest.TestCase):

    def _assert_conversion(self, input_name: str):
        input_file = DATA_DIR / input_name
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / f"{input_file.stem}_lowered.mlir"
            success, stderr, output_file = run_conversion(input_file, output_file)
            self.assertTrue(success, f"q2i-opt failed for {input_name}:\n{stderr}")
            self.assertTrue(output_file.exists(), f"Output file not created: {output_file}")
            contents = output_file.read_text()
            self.assertIn("module", contents, f"Output missing 'module' keyword in {output_file.name}")

    def test_01_quake_alloca(self):
        self._assert_conversion("01_quake_alloca.mlir")

    def test_02_quake_veq_size(self):
        self._assert_conversion("02_quake_veq_size.mlir")

    def test_03_quake_dealloc(self):
        self._assert_conversion("03_quake_dealloc.mlir")

    def test_04_quake_concat(self):
        self._assert_conversion("04_quake_concat.mlir")

    def test_05_quake_extractref(self):
        self._assert_conversion("05_quake_extractref.mlir")

    def test_06_quake_initializestate(self):
        self._assert_conversion("06_quake_initializestate.mlir")


if __name__ == "__main__":
    unittest.main(verbosity=2)
