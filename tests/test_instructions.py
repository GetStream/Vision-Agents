import pytest
from vision_agents.core.instructions import Instructions, InstructionsReadError


class TestInstructions:
    @pytest.mark.parametrize(
        "input_text, file_data, full_reference",
        [
            ("", "", ""),
            ("some prompt", "", "some prompt"),
            (
                "read @file1.md and @file2.md",
                "file1 data",
                "read @file1.md and @file2.md\n\n\n## Referenced Documentation:\n\n### file1.md\nfile1 data\n\n### file2.md\nfile1 data",
            ),
        ],
    )
    def test_parse_success(self, tmp_path, input_text, file_data, full_reference):
        for name in ("file1.md", "file2.md"):
            file_path = tmp_path / name
            file_path.write_text(file_data, encoding="utf-8")

        instructions = Instructions(input_text=input_text, base_dir=tmp_path)
        assert instructions.input_text == input_text
        assert instructions.full_reference == full_reference

    def test_parse_not_a_file(self, tmp_path):
        file_path = tmp_path / "file1.md"
        file_path.mkdir()
        input_text = "read @file1.md"
        with pytest.raises(InstructionsReadError, match="reason - path is not a file"):
            Instructions(input_text=input_text, base_dir=tmp_path)

    def test_parse_file_doesnt_exist(self, tmp_path):
        input_text = "read @file1.md"
        with pytest.raises(InstructionsReadError, match="reason - file not found"):
            Instructions(input_text=input_text, base_dir=tmp_path)

    def test_parse_file_not_md(self, tmp_path):
        input_text = "read @file1.txt"
        file_path = tmp_path / "file1.txt"
        file_path.write_text("abcdef", encoding="utf-8")
        with pytest.raises(InstructionsReadError, match="reason - file is not .md"):
            Instructions(input_text=input_text, base_dir=tmp_path)

    def test_parse_file_hidden(self, tmp_path):
        input_text = "read @.file1.md"
        file_path = tmp_path / ".file1.md"
        file_path.write_text("abcdef", encoding="utf-8")
        with pytest.raises(
            InstructionsReadError, match='reason - filename cannot start with "."'
        ):
            Instructions(input_text=input_text, base_dir=tmp_path)

    def test_parse_file_outside_base_dir(self, tmp_path):
        file_path1 = tmp_path / "file1.md"
        base_dir = tmp_path / "another-dir"
        base_dir.mkdir()
        file_path2 = base_dir / "file1.md"

        input_text = f"read @{file_path1}"
        file_path1.write_text("abcdef", encoding="utf-8")
        file_path2.write_text("abcdef", encoding="utf-8")

        with pytest.raises(
            InstructionsReadError, match="reason - path outside the base directory"
        ):
            Instructions(input_text=input_text, base_dir=base_dir)
