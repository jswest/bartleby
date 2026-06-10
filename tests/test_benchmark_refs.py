import pytest

from bartleby.benchmark.refs import ModelRef, check_slug_collisions, parse_flag_refs


def test_yaml_form_splits_on_first_colon():
    ref = ModelRef.from_yaml("ollama:gemma4:e2b")
    assert (ref.provider, ref.model) == ("ollama", "gemma4:e2b")


def test_flag_form_splits_on_first_slash_keeping_colons():
    ref = ModelRef.from_flag("ollama/gemma4:e2b")
    assert (ref.provider, ref.model) == ("ollama", "gemma4:e2b")


def test_yaml_form_requires_separator():
    with pytest.raises(SystemExit):
        ModelRef.from_yaml("gemma4")


def test_flag_form_requires_slash():
    with pytest.raises(SystemExit):
        ModelRef.from_flag("ollama:gemma4:e2b")


def test_unknown_provider_refused():
    with pytest.raises(SystemExit):
        ModelRef.from_yaml("anthropic:claude")


def test_slug_normalizes_colons_and_slashes():
    assert ModelRef("ollama", "gemma4:e2b").slug == "ollama_gemma4-e2b"
    assert ModelRef("ollama", "library/phi4-mini").slug == "ollama_library-phi4-mini"


def test_str_roundtrips_yaml_form():
    assert str(ModelRef.from_yaml("openai:gpt-5-nano")) == "openai:gpt-5-nano"


def test_local_distinguishes_reference_providers():
    assert ModelRef("ollama", "m").local
    assert not ModelRef("openai", "m").local


def test_parse_flag_refs_comma_separated():
    refs = parse_flag_refs("ollama/gemma4:e2b, openai/gpt-5-nano")
    assert [str(r) for r in refs] == ["ollama:gemma4:e2b", "openai:gpt-5-nano"]


def test_parse_flag_refs_none_passthrough():
    assert parse_flag_refs(None) is None


def test_slug_collision_refused():
    with pytest.raises(SystemExit):
        check_slug_collisions([ModelRef("ollama", "a:b"), ModelRef("ollama", "a-b")])


def test_no_collision_passes():
    check_slug_collisions([ModelRef("ollama", "a:b"), ModelRef("openai", "a-b")])


def test_exact_duplicate_refused():
    with pytest.raises(SystemExit):
        check_slug_collisions([ModelRef("ollama", "a:b"), ModelRef("ollama", "a:b")])
