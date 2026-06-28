from app.nlp.ner_pipeline import CommerceNERPipeline


def test_rapidfuzz_fallback_parses_sugar_message():
    pipeline = CommerceNERPipeline(model_dir="/nonexistent/path")  # forces fallback
    results = pipeline.parse("Nabagurishije isukari ibiro bitatu")
    assert len(results) >= 1
    assert results[0].product_name == "SUGAR"
    assert results[0].quantity == 3


def test_rapidfuzz_fallback_parses_oil_message():
    pipeline = CommerceNERPipeline(model_dir="/nonexistent/path")
    results = pipeline.parse("namavuta litre imwe")
    assert len(results) >= 1
    assert results[0].product_name == "OIL"


def test_rapidfuzz_fallback_handles_unrecognized_message():
    pipeline = CommerceNERPipeline(model_dir="/nonexistent/path")
    results = pipeline.parse("xyz completely unrelated text")
    assert results == []


def test_rapidfuzz_fallback_parses_digit_quantity():
    pipeline = CommerceNERPipeline(model_dir="/nonexistent/path")
    results = pipeline.parse("isukari 5")
    assert results[0].quantity == 5.0
