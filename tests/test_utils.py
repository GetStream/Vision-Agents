import pytest
import os
import tempfile
import numpy as np
import av
import asyncio
from stream_agents.core.utils.utils import parse_instructions, Instructions
from getstream.video.rtc.track_util import PcmData


class TestParseInstructions:
    """Test suite for the parse_instructions function."""
    
    def test_parse_instructions_no_mentions(self):
        """Test parsing text with no @ mentions."""
        text = "This is a simple instruction without any mentions."
        result = parse_instructions(text)
        
        assert isinstance(result, Instructions)
        assert result.input_text == text
        assert result.markdown_contents == {}
    
    def test_parse_instructions_single_mention(self):
        """Test parsing text with a single @ mention."""
        text = "Please read @nonexistent.md for more information."
        result = parse_instructions(text)
        
        assert result.input_text == text
        assert result.markdown_contents == {"nonexistent.md": ""}  # File doesn't exist
    
    def test_parse_instructions_multiple_mentions(self):
        """Test parsing text with multiple @ mentions."""
        text = "Check @file1.md and @file2.md for details. Also see @guide.md."
        result = parse_instructions(text)
        
        assert result.input_text == text
        assert result.markdown_contents == {"file1.md": "", "file2.md": "", "guide.md": ""}
    
    def test_parse_instructions_duplicate_mentions(self):
        """Test parsing text with duplicate @ mentions."""
        text = "Read @nonexistent.md and then @nonexistent.md again."
        result = parse_instructions(text)
        
        assert result.input_text == text
        # Should only include unique filenames
        assert result.markdown_contents == {"nonexistent.md": ""}
    
    def test_parse_instructions_non_markdown_mentions(self):
        """Test parsing text with @ mentions that are not markdown files."""
        text = "Check @user123 and @file.txt for information."
        result = parse_instructions(text)
        
        assert result.input_text == text
        # Should only capture .md files
        assert result.markdown_contents == {}
    
    def test_parse_instructions_mixed_mentions(self):
        """Test parsing text with both markdown and non-markdown @ mentions."""
        text = "Check @user123, @nonexistent.md, and @config.txt for details."
        result = parse_instructions(text)
        
        assert result.input_text == text
        # Should only capture .md files
        assert result.markdown_contents == {"nonexistent.md": ""}
    
    def test_parse_instructions_complex_filenames(self):
        """Test parsing text with complex markdown filenames."""
        text = "See @my-file.md, @file_with_underscores.md, and @file-with-dashes.md."
        result = parse_instructions(text)
        
        assert result.input_text == text
        assert result.markdown_contents == {"my-file.md": "", "file_with_underscores.md": "", "file-with-dashes.md": ""}
    
    def test_parse_instructions_edge_cases(self):
        """Test parsing text with edge cases."""
        # Empty string
        result = parse_instructions("")
        assert result.input_text == ""
        assert result.markdown_contents == {}
        
        # Only @ symbol
        result = parse_instructions("@")
        assert result.input_text == "@"
        assert result.markdown_contents == {}
        
        # @ without filename
        result = parse_instructions("Check @ for details")
        assert result.input_text == "Check @ for details"
        assert result.markdown_contents == {}
        
        # @ with spaces in filename (should not match)
        result = parse_instructions("Check @my file.md for details")
        assert result.input_text == "Check @my file.md for details"
        assert result.markdown_contents == {}
    
    def test_parse_instructions_case_sensitivity(self):
        """Test that parsing is case sensitive."""
        text = "Check @README.md and @readme.md for information."
        result = parse_instructions(text)
        
        assert result.input_text == text
        # Should treat as different files due to case sensitivity
        # Both files exist in the current directory, so they should have content
        assert "README.md" in result.markdown_contents
        assert "readme.md" in result.markdown_contents
        # Content should not be empty since these files exist
        assert len(result.markdown_contents["README.md"]) > 0
        assert len(result.markdown_contents["readme.md"]) > 0
    
    def test_parse_instructions_special_characters(self):
        """Test parsing with special characters in filenames."""
        text = "Check @file-1.md, @file_2.md, and @file.3.md for details."
        result = parse_instructions(text)
        
        assert result.input_text == text
        assert result.markdown_contents == {"file-1.md": "", "file_2.md": "", "file.3.md": ""}
    
    def test_parse_instructions_multiline_text(self):
        """Test parsing multiline text with @ mentions."""
        text = """Please review the following files:
        - @setup.md for installation instructions
        - @api.md for API documentation
        - @troubleshooting.md for common issues
        """
        result = parse_instructions(text)
        
        assert result.input_text == text
        assert result.markdown_contents == {"setup.md": "", "api.md": "", "troubleshooting.md": ""}


class TestInstructions:
    """Test suite for the Instructions dataclass."""
    
    def test_instructions_initialization(self):
        """Test Instructions dataclass initialization."""
        input_text = "Test instruction"
        markdown_contents = {"file1.md": "# File 1 content"}
        
        instructions = Instructions(input_text, markdown_contents)
        
        assert instructions.input_text == input_text
        assert instructions.markdown_contents == markdown_contents
    
    def test_instructions_empty_markdown_files(self):
        """Test Instructions with empty markdown files dict."""
        input_text = "Simple instruction"
        markdown_contents = {}
        
        instructions = Instructions(input_text, markdown_contents)
        
        assert instructions.input_text == input_text
        assert instructions.markdown_contents == {}
    
    def test_instructions_equality(self):
        """Test Instructions equality comparison."""
        instructions1 = Instructions("test", {"file.md": "content"})
        instructions2 = Instructions("test", {"file.md": "content"})
        instructions3 = Instructions("different", {"file.md": "content"})
        
        assert instructions1 == instructions2
        assert instructions1 != instructions3


class TestParseInstructionsFileReading:
    """Test suite for file reading functionality in parse_instructions."""
    
    def test_parse_instructions_with_existing_files(self):
        """Test parsing with actual markdown files that exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test markdown files
            file1_path = os.path.join(temp_dir, "readme.md")
            file2_path = os.path.join(temp_dir, "guide.md")
            
            with open(file1_path, 'w', encoding='utf-8') as f:
                f.write("# README\n\nThis is a test readme file.")
            
            with open(file2_path, 'w', encoding='utf-8') as f:
                f.write("# Guide\n\nThis is a test guide file.")
            
            text = "Please read @readme.md and @guide.md for information."
            result = parse_instructions(text, base_dir=temp_dir)
            
            assert result.input_text == text
            assert result.markdown_contents["readme.md"] == "# README\n\nThis is a test readme file."
            assert result.markdown_contents["guide.md"] == "# Guide\n\nThis is a test guide file."
    
    def test_parse_instructions_with_mixed_existing_nonexisting_files(self):
        """Test parsing with mix of existing and non-existing files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create only one test file
            file1_path = os.path.join(temp_dir, "readme.md")
            with open(file1_path, 'w', encoding='utf-8') as f:
                f.write("# README\n\nThis file exists.")
            
            text = "Check @readme.md and @nonexistent.md for details."
            result = parse_instructions(text, base_dir=temp_dir)
            
            assert result.input_text == text
            assert result.markdown_contents["readme.md"] == "# README\n\nThis file exists."
            assert result.markdown_contents["nonexistent.md"] == ""  # Empty for non-existing file
    
    def test_parse_instructions_with_custom_base_dir(self):
        """Test parsing with custom base directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test file in subdirectory
            subdir = os.path.join(temp_dir, "docs")
            os.makedirs(subdir)
            file_path = os.path.join(subdir, "api.md")
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("# API Documentation\n\nThis is the API docs.")
            
            text = "See @api.md for API information."
            result = parse_instructions(text, base_dir=subdir)
            
            assert result.input_text == text
            assert result.markdown_contents["api.md"] == "# API Documentation\n\nThis is the API docs."
    
    def test_parse_instructions_file_read_error_handling(self):
        """Test handling of file read errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a file that will cause read errors (permission issues, etc.)
            file_path = os.path.join(temp_dir, "readme.md")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("test content")
            
            # Make file unreadable (this might not work on all systems)
            try:
                os.chmod(file_path, 0o000)  # No permissions
                
                text = "Read @readme.md for information."
                result = parse_instructions(text, base_dir=temp_dir)
                
                assert result.input_text == text
                assert result.markdown_contents["readme.md"] == ""  # Empty due to read error
            finally:
                # Restore permissions for cleanup
                os.chmod(file_path, 0o644)
    
    def test_parse_instructions_unicode_content(self):
        """Test parsing with unicode content in markdown files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "unicode.md")
            
            # Write unicode content
            unicode_content = "# Unicode Test\n\nHello 世界! 🌍\n\nThis has émojis and àccénts."
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(unicode_content)
            
            text = "Check @unicode.md for unicode content."
            result = parse_instructions(text, base_dir=temp_dir)
            
            assert result.input_text == text
            assert result.markdown_contents["unicode.md"] == unicode_content
    
    def test_parse_instructions_default_base_dir(self):
        """Test that default base directory is current working directory."""
        # This test verifies that when no base_dir is provided, it uses os.getcwd()
        text = "Read @readme.md for information."
        result = parse_instructions(text)  # No base_dir provided
        
        assert result.input_text == text
        # Content will not be empty since readme.md exists in current directory
        assert "readme.md" in result.markdown_contents
        assert len(result.markdown_contents["readme.md"]) > 0


# Shared fixtures for integration tests

