from backend.services.ml_service import MLService
from preprocess import preprocess, strip_shortcuts


FAKE_STORY = (
    "BREAKING: Obama secretly signed an order to ban the Pledge of Allegiance in all schools! "
    "Patriots are FURIOUS and the liberal media is hiding it. Share this before it gets deleted!!!"
)


def test_source_markers_are_removed() -> None:
    text = strip_shortcuts(
        "WASHINGTON (Reuters) - Lawmakers met on November 3, 2016. Featured image via Getty Images"
    )
    lowered = text.lower()
    for marker in ("reuters", "november 3", "2016", "featured image", "getty"):
        assert marker not in lowered
    assert "Lawmakers met on" in text


def test_dateline_does_not_change_the_preprocessed_text() -> None:
    assert preprocess("", "WASHINGTON (Reuters) - " + FAKE_STORY) == preprocess("", FAKE_STORY)


def test_dateline_cannot_flip_a_fake_prediction() -> None:
    model = MLService()
    plain = model.predict("Obama bans Pledge of Allegiance", FAKE_STORY)
    with_dateline = model.predict("Obama bans Pledge of Allegiance", "WASHINGTON (Reuters) - " + FAKE_STORY)
    assert plain[0] == "FAKE"
    assert with_dateline == plain
