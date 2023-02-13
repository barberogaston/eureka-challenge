from housing_inference.api.load import load_model, load_performance

def test_load_model(model, models,usi):
    result = load_model(models, usi)
    assert model.get_params() == result.get_params()


def test_load_performance(model_performance, models, usi):
    assert model_performance == load_performance(models, usi)
