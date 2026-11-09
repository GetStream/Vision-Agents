from pathlib import Path

import pytest

pytest_plugins = ["pytester"]

ROOT_CONFTEST = Path(__file__).parent.parent / "conftest.py"


@pytest.fixture
def guarded(pytester: pytest.Pytester) -> pytest.Pytester:
    pytester.makeconftest(
        "import importlib.util\n"
        f"spec = importlib.util.spec_from_file_location('root_conftest', {str(ROOT_CONFTEST)!r})\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(mod)\n"
        "pytest_make_collect_report = mod.pytest_make_collect_report\n"
    )
    return pytester


class TestCollectGuard:
    def test_shadowed_test_file_fails_collection(self, guarded: pytest.Pytester):
        for plugin, test in (("a", "test_one_a"), ("b", "test_one_b")):
            pkg = guarded.mkdir(plugin) / "tests"
            pkg.mkdir()
            (pkg / "__init__.py").write_text("")
            (pkg / "test_x.py").write_text(f"def {test}():\n    pass\n")
        result = guarded.runpytest_subprocess(
            "--import-mode=importlib",
            "--collect-only",
            "-q",
            "a/tests/test_x.py",
            "b/tests/test_x.py",
        )
        assert result.ret == pytest.ExitCode.INTERRUPTED
        result.stdout.fnmatch_lines(
            ["*b/tests/test_x.py collected the tests of *a/tests/test_x.py*"]
        )

    def test_unique_test_files_collect(self, guarded: pytest.Pytester):
        for plugin, test in (("a", "test_one_a"), ("b", "test_one_b")):
            pkg = guarded.mkdir(plugin) / "tests"
            pkg.mkdir()
            (pkg / "test_x.py").write_text(f"def {test}():\n    pass\n")
        result = guarded.runpytest_subprocess(
            "--import-mode=importlib",
            "--collect-only",
            "-q",
            "a/tests/test_x.py",
            "b/tests/test_x.py",
        )
        assert result.ret == pytest.ExitCode.OK
        result.stdout.fnmatch_lines(
            ["*a/tests/test_x.py::test_one_a*", "*b/tests/test_x.py::test_one_b*"]
        )
