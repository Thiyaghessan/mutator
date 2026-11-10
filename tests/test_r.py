import shutil
import subprocess
from pathlib import Path

import pytest

from mutator.cli import run
from mutator.functions import is_private, sites_in_file
from mutator.runner import test_plan as plan_command
from mutator.sites import apply_site


def _write(root: Path, relative: str, source: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def _sites(tmp_path: Path, source: str, relative: str = "R/box.R"):
    path = _write(tmp_path, relative, source)
    return sites_in_file(source, path, tmp_path, relative)


def _described(sites) -> set[tuple[str, str, str]]:
    return {(site.category, site.original, site.mutant) for site in sites}


def test_comparison_operators_are_flipped(tmp_path):
    source = """compare <- function(a, b) {
  c(a == b, a != b, a < b, a <= b, a > b, a >= b)
}
"""
    found = _described(_sites(tmp_path, source))
    assert found == {
        ("equality", "==", "!="),
        ("equality", "!=", "=="),
        ("comparison", "<", "<="),
        ("comparison", "<=", "<"),
        ("comparison", ">", ">="),
        ("comparison", ">=", ">"),
    }


def test_and_becomes_or_and_or_becomes_and(tmp_path):
    source = """both <- function(a, b) {
  c(a && b, a || b, a & b, a | b)
}
"""
    assert _described(_sites(tmp_path, source)) == {
        ("logical", "&&", "||"),
        ("logical", "||", "&&"),
    }


def test_true_and_false_are_swapped(tmp_path):
    source = """flags <- function() {
  list(keep = TRUE, drop = FALSE, text = "TRUE")
}
"""
    assert _described(_sites(tmp_path, source)) == {
        ("boolean", "TRUE", "FALSE"),
        ("boolean", "FALSE", "TRUE"),
    }


def test_numbers_change_by_one(tmp_path):
    source = """numbers <- function() {
  c(0, 7, 12L, 2.5, 1e3, 0x10, "9")
}
"""
    assert _described(_sites(tmp_path, source)) == {
        ("constant", "0", "1"),
        ("constant", "7", "8"),
        ("constant", "12L", "13L"),
        ("constant", "2.5", "3.5"),
    }


def test_if_and_while_conditions_are_negated(tmp_path):
    source = """wait <- function(ready, done) {
  while (ready) {
    ready <- step()
  }
  if (done) {
    stop("finished")
  }
  return(ready)
}
"""
    found = {item for item in _described(_sites(tmp_path, source)) if item[0] == "conditional"}
    assert found == {
        ("conditional", "ready", "!(ready)"),
        ("conditional", "done", "!(done)"),
    }


def test_statements_are_deleted_but_returns_are_not(tmp_path):
    source = """tidy <- function(rows) {
  log_start()
  if (nrow(rows) == 0L) {
    warn_empty()
    return(NULL)
  }
  rows <- sort_rows(rows)
  rows
}
"""
    deleted = sorted(
        (site.line, site.original) for site in _sites(tmp_path, source) if site.category == "statement"
    )
    assert deleted == [
        (2, "log_start()"),
        (3, 'if (nrow(rows) == 0L) {\n    warn_empty()\n    return(NULL)\n  }'),
        (4, "warn_empty()"),
        (7, "rows <- sort_rows(rows)"),
    ]
    # `return(NULL)` and the final `rows`, which is the value the function gives back, stay.


def test_a_deleted_statement_is_described_on_one_line(tmp_path):
    source = """tidy <- function(rows) {
  if (ready) {
    go()
  }
  rows
}
"""
    site = next(site for site in _sites(tmp_path, source) if site.category == "statement" and site.line == 2)
    assert site.mutant == ""
    assert site.description == "delete if (ready) { ..."


def test_sites_belong_to_the_function_crapper_names(tmp_path):
    source = """top_level <- 1 == 1

first <- function(x) {
  x > 1
}

.hidden = function(x) {
  helper <- function(y) y < 2
  helper(x)
}
"""
    sites = _sites(tmp_path, source, "R/transforms/dates.R")
    assert [(site.namespace, site.form_id, site.line, site.original) for site in sites] == [
        ("transforms.dates", "defn/first", 4, ">"),
        ("transforms.dates", "defn/first", 4, "1"),
        ("transforms.dates", "defn-/.hidden", 8, "helper <- function(y) y < 2"),
        ("transforms.dates", "defn-/.hidden", 8, "<"),
        ("transforms.dates", "defn-/.hidden", 8, "2"),
    ]
    assert is_private("r", ".hidden", source, 7, 10) is True
    assert is_private("r", "first", source, 3, 5) is False


def test_coverage_is_judged_on_the_statement_that_holds_the_site(tmp_path):
    source = """choose <- function(x) {
  if (x >
      1) {
    y <- TRUE
  }
  total <- add(
    x,
    2
  )
  total
}
"""
    anchors = {(site.original, site.line): site.anchor_line for site in _sites(tmp_path, source)}
    # A site in the condition is judged on the line the `if` starts on.
    assert anchors[("1", 3)] == 2
    # A site inside the braces is judged on its own statement, not on the `if`.
    assert anchors[("TRUE", 4)] == 4
    # A site on a later line of a long statement is judged on its first line.
    assert anchors[("2", 8)] == 6


def test_a_package_is_tested_with_testthat(tmp_path):
    _write(tmp_path, "DESCRIPTION", "Package: demo\n")
    source = _write(tmp_path, "R/box.R", "f <- function() 1\n")
    command, cwd = plan_command(tmp_path, source, "r", None)
    assert command == ["Rscript", "-e", "testthat::test_local(stop_on_failure = TRUE)"]
    assert cwd == tmp_path.resolve()


def test_loose_scripts_run_each_test_file_and_stop_at_the_first_failure(tmp_path):
    source = _write(tmp_path, "R/box.R", "f <- function() 1\n")
    _write(tmp_path, "tests/test_a.R", "")
    _write(tmp_path, "tests/test_b.R", "")
    _write(tmp_path, "tests/run_all.R", "")
    command, cwd = plan_command(tmp_path, source, "r", None)
    assert command[:2] == ["Rscript", "-e"]
    assert command[3:] == ["tests/test_a.R", "tests/test_b.R"]
    assert cwd == tmp_path.resolve()


def test_the_test_command_option_replaces_the_default(tmp_path):
    source = _write(tmp_path, "R/box.R", "f <- function() 1\n")
    command, cwd = plan_command(tmp_path, source, "r", "Rscript tests/run_all.R")
    assert command == "Rscript tests/run_all.R"
    assert cwd == tmp_path.resolve()


EVERY_KIND = """sign_of <- function(x, strict = TRUE) {
  note("called")
  if (x > 0 && strict) {
    return(1)
  }
  while (x < -5) {
    x <- x + 2L
  }
  if (x == 0) { note("zero"); return(0) }
  return(-1)
}
"""

needs_r = pytest.mark.skipif(shutil.which("Rscript") is None, reason="needs R")


@needs_r
def test_every_mutant_is_still_r_that_parses(tmp_path):
    sites = _sites(tmp_path, EVERY_KIND)
    assert {site.category for site in sites} == {
        "comparison",
        "equality",
        "logical",
        "boolean",
        "constant",
        "conditional",
        "statement",
    }
    mutants = tmp_path / "mutants"
    mutants.mkdir()
    for index, site in enumerate(sites):
        text = apply_site(EVERY_KIND, site.start, site.end, site.mutant)
        (mutants / f"{index}.R").write_text(text, encoding="utf-8")
    check = subprocess.run(
        ["Rscript", "-e", "for (f in list.files(commandArgs(TRUE)[[1]], full.names = TRUE)) parse(f)", str(mutants)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert check.returncode == 0, check.stderr


@needs_r
def test_a_real_run_kills_what_the_tests_check_and_keeps_what_they_do_not(tmp_path, capsys):
    _write(
        tmp_path,
        "R/box.R",
        "sign_of <- function(x) {\n"
        "  note <- 0\n"
        "  if (x > 0) {\n"
        "    return(1)\n"
        "  }\n"
        "  return(-1)\n"
        "}\n",
    )
    _write(
        tmp_path,
        "tests/run_all.R",
        'source("R/box.R")\n'
        "ok <- sign_of(5) == 1 && sign_of(-5) == -1\n"
        "if (!ok) quit(status = 1L)\n",
    )
    code = run(
        ["--root", str(tmp_path), "--no-coverage", "--test-command", "Rscript tests/run_all.R", "R/box.R"]
    )
    out = capsys.readouterr().out
    statuses = {}
    for line in out.splitlines():
        if line.startswith(("KILLED", "SURVIVED")):
            status, place, description = line.split(None, 2)
            statuses[(int(place.rsplit(":", 1)[1]), description)] = status
    # The tests never look at `note`, and never call sign_of(0) or sign_of(1).
    assert statuses == {
        (2, "delete note <- 0"): "SURVIVED",
        (2, "0 -> 1"): "SURVIVED",
        (3, "delete if (x > 0) { ..."): "KILLED",
        (3, "x > 0 -> !(x > 0)"): "KILLED",
        (3, "> -> >="): "SURVIVED",
        (3, "0 -> 1"): "SURVIVED",
        (4, "1 -> 2"): "KILLED",
        (6, "1 -> 2"): "KILLED",
    }
    assert code == 3
    assert (tmp_path / "R" / "box.R").read_text(encoding="utf-8").startswith("sign_of <- function(x) {\n  note <- 0\n")
