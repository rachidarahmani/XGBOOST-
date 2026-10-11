def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_valid(client, valid_payload):
    r = client.post("/predict", json=valid_payload)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert body["label"] in ("bon", "mauvais")
    assert 0 <= body["probability_good"] <= 1


def test_predict_red_wine(client, valid_payload):
    valid_payload["wine_type"] = "red"
    assert client.post("/predict", json=valid_payload).status_code == 200


def test_label_matches_prediction(client, valid_payload):
    body = client.post("/predict", json=valid_payload).json()
    assert (body["label"] == "bon") == (body["prediction"] == 1)


def test_predict_extreme_value_is_clipped(client, valid_payload):
    valid_payload["chlorides"] = 9999.0  # écrêté, ne doit pas faire planter
    assert client.post("/predict", json=valid_payload).status_code == 200


def test_predict_missing_field(client, valid_payload):
    valid_payload.pop("alcohol")
    assert client.post("/predict", json=valid_payload).status_code == 422


def test_predict_negative_value(client, valid_payload):
    valid_payload["alcohol"] = -5
    assert client.post("/predict", json=valid_payload).status_code == 422


def test_predict_ph_out_of_range(client, valid_payload):
    valid_payload["ph"] = 20
    assert client.post("/predict", json=valid_payload).status_code == 422


def test_predict_invalid_type(client, valid_payload):
    valid_payload["wine_type"] = "rose"
    assert client.post("/predict", json=valid_payload).status_code == 422


def test_model_info(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    assert "metrics" in r.json()
    assert "importances" in r.json()
