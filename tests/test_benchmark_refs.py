import pytest

from bartleby.benchmark.refs import ModelRef, check_slug_collisions, parse_refs


def test_parse_splits_on_first_slash_keeping_colons():
    ref = ModelRef.parse("ollama/gemma4:e2b")
    assert (ref.provider, ref.model) == ("ollama", "gemma4:e2b")


def test_parse_requires_slash():
    with pytest.raises(SystemExit):
        ModelRef.parse("ollama:gemma4:e2b")  # the old colon form is gone
    with pytest.raises(SystemExit):
        ModelRef.parse("gemma4")


def test_parse_rejects_non_string():
    with pytest.raises(SystemExit):
        ModelRef.parse({"ollama": "gemma4"})  # a stray YAML mapping entry


def test_unknown_provider_refused():
    with pytest.raises(SystemExit):
        ModelRef.parse("anthropic/claude")


def test_slug_normalizes_colons_and_slashes():
    assert ModelRef("ollama", "gemma4:e2b").slug == "ollama_gemma4-e2b"
    assert ModelRef("ollama", "library/phi4-mini").slug == "ollama_library-phi4-mini"


def test_str_roundtrips():
    assert str(ModelRef.parse("openai/gpt-5-nano")) == "openai/gpt-5-nano"


def test_local_distinguishes_reference_providers():
    assert ModelRef("ollama", "m").local
    assert not ModelRef("openai", "m").local


def test_parse_refs_comma_separated():
    refs = parse_refs("ollama/gemma4:e2b, openai/gpt-5-nano")
    assert [str(r) for r in refs] == ["ollama/gemma4:e2b", "openai/gpt-5-nano"]


def test_parse_refs_none_passthrough():
    assert parse_refs(None) is None


def test_slug_collision_refused():
    with pytest.raises(SystemExit):
        check_slug_collisions([ModelRef("ollama", "a:b"), ModelRef("ollama", "a-b")])


def test_no_collision_passes():
    check_slug_collisions([ModelRef("ollama", "a:b"), ModelRef("openai", "a-b")])


def test_exact_duplicate_refused():
    with pytest.raises(SystemExit):
        check_slug_collisions([ModelRef("ollama", "a:b"), ModelRef("ollama", "a:b")])
