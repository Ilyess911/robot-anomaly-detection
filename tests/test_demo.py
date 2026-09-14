"""The browser demo runs end to end, or the suite says so.

A demo advertised in the README and never executed is worse than no demo. This
drives the Streamlit app headless through the framework's own test harness, so a
crash in the page surfaces here rather than in front of whoever clicked the link.

Streamlit is not in ``requirements-lock.txt`` and continuous integration does not
install it: nothing published depends on the demo. The test skips when it is
absent rather than failing, and runs locally after
``pip install -r requirements-demo.txt``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit", reason="demo dependency, see requirements-demo.txt")

from streamlit.testing.v1 import AppTest

#: AppTest resolves a relative path against the calling file, not the working
#: directory, so the path is built from the repository root instead.
APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


@pytest.fixture(scope="module")
def app():
    running = AppTest.from_file(str(APP), default_timeout=300)
    running.run()
    return running


def test_the_demo_renders_without_raising(app):
    assert not app.exception, [str(item.value) for item in app.exception]


def test_the_demo_shows_the_four_headline_numbers(app):
    labels = {metric.label for metric in app.metric}
    assert labels == {"Executions scored", "Flagged", "F1, failure class", "ROC-AUC"}


def test_the_demo_exposes_the_threshold_as_the_thing_to_play_with(app):
    """The slider is the point of the page.

    Everything this project concludes says the threshold is where the value is
    lost, so the demo has to let a visitor move it. A page that only displayed a
    score would be a screenshot.
    """
    sliders = [slider for slider in app.slider if "threshold" in slider.label.lower()]
    assert sliders, "the alarm threshold must be adjustable"
    assert sliders[0].value == 95.0
