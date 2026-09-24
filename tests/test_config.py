from backend.config import load_config


def test_config_loads():
    config = load_config()
    assert "detection" in config
    assert "video" in config
    assert config["video"]["processing_fps"] > 0


def test_optional_stages_default_off():
    config = load_config()
    assert config["segmentation"]["enabled"] is False
    assert config["ocr"]["enabled"] is False
    assert config["vlm"]["enabled"] is False
