import shutil
import subprocess
from pathlib import Path

import pytest

from mutator.cli import run
from mutator.coverage import ExpressionHits, covered_lines
from mutator.crapper_link import ensure_crapper
from mutator.functions import is_private, sites_in_file
from mutator.runner import test_plan as plan_command
from mutator.sites import apply_site


def _r_has(*packages: str) -> bool:
    if shutil.which("Rscript") is None:
        return False
    load = "; ".join(f"library({package})" for package in packages)
    return subprocess.run(["Rscript", "-e", load], capture_output=True, check=False).returncode == 0


needs_r = pytest.mark.skipif(shutil.which("Rscript") is None, reason="needs R")
needs_covr = pytest.mark.skipif(not _r_has("covr"), reason="needs R with covr")
needs_testthat = pytest.mark.skipif(not _r_has("testthat", "pkgload"), reason="needs R with testthat and pkgload")


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


def test_a_site_is_covered_when_the_smallest_expression_round_it_ran():
    source = b"f <- function(x) {\n  if (x > 0) 1 else 2\n}\n"
    # covr rows for line 2: the condition ran, `1` ran, `2` did not.
    hits = ExpressionHits.from_rows(source, [(2, 7, 2, 11, 1), (2, 14, 2, 14, 1), (2, 21, 2, 21, 0)])
    line_two = source.index(b"  if")
    assert hits.covers(line_two + 6, line_two + 11) is True  # x > 0
    assert hits.covers(line_two + 13, line_two + 14) is True  # 1
    assert hits.covers(line_two + 20, line_two + 21) is False  # 2
    # The word `if` is inside no expression, so the first one inside the statement decides.
    assert hits.covers(line_two + 2, line_two + 21) is True
    # Nothing was measured on line 1, so a site there is not covered.
    assert hits.covers(5, 6) is False


BRANCHES = """pick <- function(x) {
  if (x > 0) 1 else 2
  if (x > 5) {
    "big"
  } else if (x < -5) {
    "small"
  } else {
    "mid"
  }
}
"""


@needs_covr
def test_real_coverage_is_right_for_a_one_line_if_and_an_else_if_never_reached(tmp_path):
    path = _write(tmp_path, "R/box.R", BRANCHES)
    _write(tmp_path, "tests/test_box.R", 'stopifnot(pick(9) == "big")\n')
    ensure_crapper().runners.run_coverage(tmp_path, [path], None)
    hits = covered_lines(tmp_path, path, "r")
    assert isinstance(hits, ExpressionHits)
    covered = {
        (site.line, site.description): hits.covers(site.start, site.end)
        for site in sites_in_file(BRANCHES, path, tmp_path, "R/box.R")
    }
    # Line 2 holds a part that ran and a part that did not.
    assert covered[(2, "0 -> 1")] is True
    assert covered[(2, "1 -> 2")] is True
    assert covered[(2, "2 -> 3")] is False
    # The outer condition ran. The `else if` was never reached.
    assert covered[(3, "5 -> 6")] is True
    assert covered[(5, "x < -5 -> !(x < -5)")] is False
    assert covered[(5, "< -> <=")] is False
    assert covered[(5, "5 -> 6")] is False


def test_without_expression_data_the_line_report_is_used(tmp_path):
    path = _write(tmp_path, "R/box.R", "f <- function(x) {\n  x > 1\n}\n")
    report = tmp_path / "target" / "coverage" / "r" / "lcov.info"
    report.parent.mkdir(parents=True)
    report.write_text("SF:R/box.R\nDA:2,1\nend_of_record\n", encoding="utf-8")
    assert covered_lines(tmp_path, path, "r") == {2}


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


def test_loose_scripts_with_a_testthat_folder_are_run_by_testthat(tmp_path):
    source = _write(tmp_path, "R/box.R", "f <- function() 1\n")
    _write(tmp_path, "tests/testthat/helper-values.R", "")
    _write(tmp_path, "tests/testthat/test-box.R", "")
    command, cwd = plan_command(tmp_path, source, "r", None)
    assert command[:2] == ["Rscript", "-e"]
    assert "testthat::test_dir" in command[2]
    assert command[3:] == ["tests/testthat"]
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


def _deleted(tmp_path: Path, source: str) -> list[tuple[int, str]]:
    return sorted(
        (site.line, site.description)
        for site in _sites(tmp_path, source)
        if site.category == "statement"
    )


def test_a_return_written_another_way_is_still_a_return(tmp_path):
    source = """quiet <- function(x) {
  if (x) {
    `return`(1)
  }
  if (!x) {
    invisible(return(2))
  }
  note(x)
  3
}
"""
    assert _deleted(tmp_path, source) == [
        (2, "delete if (x) { ..."),
        (5, "delete if (!x) { ..."),
        (8, "delete note(x)"),
    ]


def test_the_value_of_the_last_if_or_switch_is_what_the_function_gives_back(tmp_path):
    source = """label <- function(x) {
  note(x)
  if (x > 0) {
    note("positive")
    "positive"
  } else if (x < 0) {
    "negative"
  } else {
    switch(as.character(x),
      "0" = {
        note("zero")
        "zero"
      },
      "other"
    )
  }
}
"""
    assert _deleted(tmp_path, source) == [
        (2, "delete note(x)"),
        (4, 'delete note("positive")'),
        (11, 'delete note("zero")'),
    ]


def test_an_if_that_is_not_last_does_not_give_the_function_value(tmp_path):
    source = """tidy <- function(x) {
  if (x) {
    cleanup()
  }
  x
}
"""
    assert _deleted(tmp_path, source) == [(2, "delete if (x) { ..."), (3, "delete cleanup()")]


def test_names_join_crapper_for_a_script_outside_the_r_folder(tmp_path):
    source = "`.hidden` <- function(x) {\n  x > 1\n}\n"
    sites = _sites(tmp_path, source, "scripts/build.R")
    assert {(site.namespace, site.form_id) for site in sites} == {("scripts.build", "defn-/.hidden")}


PACKAGE_SOURCE = "sign_of <- function(x) {\n  if (x > 0) {\n    return(1)\n  }\n  return(-1)\n}\n"
PACKAGE_TEST = 'test_that("sign", {\n  expect_equal(sign_of(5), expected_one)\n  expect_equal(sign_of(-5), -1)\n})\n'


def _killed_and_survived(out: str) -> dict[tuple[int, str], str]:
    statuses = {}
    for line in out.splitlines():
        if line.startswith(("KILLED", "SURVIVED")):
            status, place, description = line.split(None, 2)
            statuses[(int(place.rsplit(":", 1)[1]), description)] = status
    return statuses


EXPECTED_FOR_SIGN_OF = {
    (2, "delete if (x > 0) { ..."): "KILLED",
    (2, "x > 0 -> !(x > 0)"): "KILLED",
    (2, "> -> >="): "SURVIVED",
    (2, "0 -> 1"): "SURVIVED",
    (3, "1 -> 2"): "KILLED",
    (5, "1 -> 2"): "KILLED",
}


@needs_testthat
def test_a_package_is_tested_against_the_mutant_not_the_original(tmp_path, capsys):
    _write(
        tmp_path,
        "DESCRIPTION",
        "Package: demo\nTitle: Demo\nVersion: 0.0.1\nDescription: Demo.\nLicense: MIT\nEncoding: UTF-8\n",
    )
    _write(tmp_path, "NAMESPACE", "export(sign_of)\n")
    _write(tmp_path, "R/box.R", PACKAGE_SOURCE)
    _write(tmp_path, "tests/testthat/helper-values.R", "expected_one <- 1\n")
    _write(tmp_path, "tests/testthat/test-box.R", PACKAGE_TEST)
    code = run(["--root", str(tmp_path), "--no-coverage", "R/box.R"])
    assert _killed_and_survived(capsys.readouterr().out) == EXPECTED_FOR_SIGN_OF
    assert code == 3


@needs_testthat
def test_loose_scripts_with_testthat_are_tested_against_the_mutant(tmp_path, capsys):
    _write(tmp_path, "R/box.R", PACKAGE_SOURCE)
    # The helper loads the source by a path relative to the test folder.
    _write(tmp_path, "tests/testthat/helper-values.R", 'source("../../R/box.R")\nexpected_one <- 1\n')
    _write(tmp_path, "tests/testthat/test-box.R", PACKAGE_TEST)
    code = run(["--root", str(tmp_path), "--no-coverage", "R/box.R"])
    assert _killed_and_survived(capsys.readouterr().out) == EXPECTED_FOR_SIGN_OF
    assert code == 3


@needs_testthat
def test_a_package_in_a_subfolder_is_tested_against_the_mutant(tmp_path, capsys):
    package = "packages/demo"
    _write(
        tmp_path,
        f"{package}/DESCRIPTION",
        "Package: demo\nTitle: Demo\nVersion: 0.0.1\nDescription: Demo.\nLicense: MIT\nEncoding: UTF-8\n",
    )
    _write(tmp_path, f"{package}/NAMESPACE", "export(sign_of)\n")
    _write(tmp_path, f"{package}/R/box.R", PACKAGE_SOURCE)
    _write(tmp_path, f"{package}/tests/testthat/helper-values.R", "expected_one <- 1\n")
    _write(tmp_path, f"{package}/tests/testthat/test-box.R", PACKAGE_TEST)
    code = run(["--root", str(tmp_path), "--no-coverage", f"{package}/R/box.R"])
    assert _killed_and_survived(capsys.readouterr().out) == EXPECTED_FOR_SIGN_OF
    assert code == 3


def test_a_return_in_quotes_or_parentheses_is_still_a_return(tmp_path):
    source = """quiet <- function(x) {
  if (x == 1) {
    "return"(1)
  }
  if (x == 2) {
    (return(2))
  }
  if (x == 3) {
    invisible((return(3)))
  }
  note(x)
  4
}
"""
    assert _deleted(tmp_path, source) == [
        (2, "delete if (x == 1) { ..."),
        (5, "delete if (x == 2) { ..."),
        (8, "delete if (x == 3) { ..."),
        (11, "delete note(x)"),
    ]


def test_a_site_in_a_body_covr_does_not_measure_alone_is_not_covered():
    source = b"f <- function(x) {\n  while (FALSE) x <- x + 1\n}\n"
    # covr gives one row for the whole loop on line 2, and it ran.
    hits = ExpressionHits.from_rows(source, [(2, 3, 2, 26, 1)])
    loop = source.index(b"while")
    body = (source.index(b"x <- x + 1"), source.index(b"x <- x + 1") + len(b"x <- x + 1"))
    one = source.index(b"1")
    assert hits.covers(loop, loop + 24) is True
    # The loop ran, which does not show that its body did.
    assert hits.covers(one, one + 1, within=body) is False
    # A row inside the body would show it.
    measured = ExpressionHits.from_rows(source, [(2, 3, 2, 26, 1), (2, 17, 2, 26, 1)])
    assert measured.covers(one, one + 1, within=body) is True


LOOPS = """spin <- function(x) {
  while (FALSE) x <- x + 1
  for (i in integer(0)) x <- x + 2
  while (x > 100) {
    x <- x + 3
  }
  repeat break
  f <- function(y) y + 5
  x
}
tiny <- function(x) x + 6
"""


@needs_covr
def test_real_coverage_does_not_claim_a_body_ran_when_covr_cannot_tell(tmp_path):
    path = _write(tmp_path, "R/box.R", LOOPS)
    _write(tmp_path, "tests/test_box.R", "stopifnot(spin(1) == 1, tiny(1) == 7)\n")
    ensure_crapper().runners.run_coverage(tmp_path, [path], None)
    hits = covered_lines(tmp_path, path, "r")
    covered = {
        (site.line, site.description): hits.covers(site.start, site.end, site.within)
        for site in sites_in_file(LOOPS, path, tmp_path, "R/box.R")
    }
    # Bodies that never ran, with and without braces.
    assert covered[(2, "1 -> 2")] is False
    assert covered[(3, "2 -> 3")] is False
    assert covered[(5, "3 -> 4")] is False
    assert covered[(8, "5 -> 6")] is False
    # Conditions that were evaluated, and whole statements that were reached.
    assert covered[(3, "0 -> 1")] is True
    assert covered[(4, "100 -> 101")] is True
    assert covered[(2, "delete while (FALSE) x <- x + 1")] is True
    assert covered[(8, "delete f <- function(y) y + 5")] is True
    # A function whose whole body is one expression, and which ran.
    assert covered[(11, "6 -> 7")] is True
