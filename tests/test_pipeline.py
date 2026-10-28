from src.pipeline import NeuroSearchPipeline

def test_chunking_overlap_and_empty():
    chunks = NeuroSearchPipeline._chunk("one two three four five six", words=4, overlap=1)
    assert chunks == ["one two three four", "four five six"]
    assert NeuroSearchPipeline._chunk("", words=4, overlap=1) == []

def test_overlap_validation():
    p = NeuroSearchPipeline()
    try:
        p.ingest([{"name": "a.txt", "bytes": b"hello"}], chunk_words=10, overlap_words=10)
    except ValueError as e:
        assert "smaller" in str(e)
    else:
        raise AssertionError("Expected invalid overlap to fail")
