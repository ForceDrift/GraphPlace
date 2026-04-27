import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.append(str(Path(__file__).resolve().parents[1]))

from graphplace.parser.iccad2015.iccad2015_parser import ICCAD2015Parser
from graphplace.parser.ispd.ispd_parser import ISPDParser

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _is_lfs_pointer(path: Path) -> bool:
    if not path.exists():
        return False
    with path.open(mode="r", encoding="utf-8") as handle:
        first_line = handle.readline().strip()
    return first_line.startswith("version https://git-lfs.github.com/spec/v1")


def _def_has_nets_section(path: Path) -> bool:
    with path.open(mode="r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.lstrip()
            if stripped.startswith("NETS "):
                return True
            if stripped.startswith("SPECIALNETS "):
                return False
    return False


def _collect_aux_files(aux_path: Path) -> list[Path]:
    with aux_path.open(mode="r", encoding="utf-8") as handle:
        tokens = handle.read().strip().split()

    files: list[Path] = []
    for token in tokens:
        if token.endswith(":"):
            continue
        if token.startswith("RowBasedPlacement"):
            continue
        if token.startswith("#"):
            continue
        files.append(aux_path.parent / token)
    return files


class ParserTestCase(unittest.TestCase):
    def test_ispd2005_parser(self) -> None:
        root = DATA_DIR / "ispd2005_sample"
        aux_path = root / "adaptec1.inf.aux"
        if not aux_path.exists():
            self.skipTest("ISPD2005 dataset not found under data/")

        required = [aux_path] + _collect_aux_files(aux_path)
        missing = [path for path in required if not path.exists()]
        if missing:
            self.skipTest("ISPD2005 aux references missing files under data/")
        if any(_is_lfs_pointer(path) for path in required):
            self.skipTest("ISPD2005 dataset is a Git LFS pointer; run git lfs pull")

        design = ISPDParser(str(aux_path)).parse()

        self.assertEqual(design.name, "adaptec1.inf")
        self.assertGreater(len(design.nodes), 0)
        self.assertGreater(len(design.nets), 0)
        self.assertGreater(design.die_hx, design.die_lx)
        self.assertGreater(design.die_hy, design.die_ly)

    def test_ispd2026_parser(self) -> None:
        root = DATA_DIR / "ispd2026_sample" / "aes_cipher_top" / "TCP_250_UTIL_0.40"
        node_csv = root / "node.csv"
        nets_csv = root / "nets.csv"
        def_file = root / "contest.def"
        required = [node_csv, nets_csv, def_file]
        missing = [path for path in required if not path.exists()]
        if missing:
            self.skipTest("ISPD2026 dataset not found under data/")
        if any(_is_lfs_pointer(path) for path in required):
            self.skipTest("ISPD2026 dataset is a Git LFS pointer; run git lfs pull")

        design = ISPDParser(str(root)).parse()

        self.assertEqual(design.name, "aes_cipher_top")
        self.assertGreater(len(design.nodes), 0)
        self.assertGreater(len(design.nets), 0)
        self.assertGreater(design.die_hx, design.die_lx)
        self.assertGreater(design.die_hy, design.die_ly)

    def test_iccad2015_parser(self) -> None:
        root = DATA_DIR / "iccad2015_sample" / "superblue1"
        def_file = root / "superblue1.def"
        if not def_file.exists():
            self.skipTest("ICCAD2015 dataset not found under data/")
        if _is_lfs_pointer(def_file):
            self.skipTest("ICCAD2015 dataset is a Git LFS pointer; run git lfs pull")

        design = ICCAD2015Parser(str(def_file)).parse()
        has_nets = _def_has_nets_section(def_file)

        self.assertEqual(design.name, "superblue1")
        self.assertGreater(len(design.nodes), 0)
        if has_nets:
            self.assertGreater(len(design.nets), 0)
        self.assertGreater(design.die_hx, design.die_lx)
        self.assertGreater(design.die_hy, design.die_ly)


if __name__ == "__main__":
    unittest.main()
