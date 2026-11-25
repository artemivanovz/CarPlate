"""Приём RTSP: пароль с & не должен обрывать адрес, пакеты кадра не выкидываем."""

import pytest

from anprgate.video import mask_credentials, quote_rtsp_userinfo, rtsp_ffmpeg_options


def test_password_with_ampersand_is_percent_encoded():
    uri = "rtsp://user:a&b@10.0.0.8:554/streaming/Channels/101"
    encoded = quote_rtsp_userinfo(uri)
    assert encoded == "rtsp://user:a%26b@10.0.0.8:554/streaming/Channels/101"
    assert "a&b@" not in encoded
    masked = mask_credentials(encoded)
    assert "a%26b" not in masked
    assert "10.0.0.8" in masked


def test_quote_leaves_file_path_alone():
    assert quote_rtsp_userinfo(r"data\videos\clip.mp4") == r"data\videos\clip.mp4"


def test_rtsp_options_keep_whole_frames(monkeypatch):
    monkeypatch.delenv("ANPR_RTSP_TRANSPORT", raising=False)
    monkeypatch.delenv("RTSP_TRANSPORT", raising=False)
    options = rtsp_ffmpeg_options()
    assert options["rtsp_transport"] == "tcp"
    assert options["allowed_media_types"] == "video"
    assert int(options["buffer_size"]) >= 1_000_000
    assert "low_delay" not in options.values()
    assert "nobuffer" not in options.values()


@pytest.mark.parametrize("userinfo", ["user:first@private_tail", "user:p%40ss", "user:", "user"])
def test_mask_credentials_removes_entire_userinfo_in_error_and_source(userinfo):
    url = f"rtsp://{userinfo}@camera.example:554/stream"
    masked = mask_credentials(f"Failed opening {url}")
    assert "private_tail" not in masked
    assert "first" not in masked
    assert "p%40ss" not in masked
    assert "camera.example:554/stream" in masked


def test_multiple_urls_and_ipv6_are_redacted_independently():
    text = 'rtsp://one:abc@tail@[::1]:554/x and rtsps://two:xyz@host/y'
    assert mask_credentials(text) == 'rtsp://one:***@[::1]:554/x and rtsps://two:***@host/y'
    assert mask_credentials('rtsp://host/path@label') == 'rtsp://host/path@label'


@pytest.mark.parametrize("password", ["first/secret", "first?secret", "first#secret", "first\"secret", "first'secret", "first secret"])
def test_malformed_or_quoted_password_never_leaks_into_error_text(password):
    masked = mask_credentials(f'Failed opening "rtsp://user:{password}@camera.example/stream"')
    assert "first" not in masked
    assert "secret" not in masked


def test_adjacent_quoted_urls_are_redacted_independently():
    value = '["rtsp://one:first@camera.example","rtsp://two:second@other.example"]'
    assert mask_credentials(value) == '["rtsp://one:***@camera.example","rtsp://two:***@other.example"]'


@pytest.mark.parametrize("password, encoded", [
    ("a@b", "a%40b"), ("p%40ss", "p%40ss"), ("a%26b", "a%26b"),
    ("a%b", "a%25b"), ("a%25b", "a%25b"), ("p%FF", "p%FF"),
    ("пароль", "%D0%BF%D0%B0%D1%80%D0%BE%D0%BB%D1%8C"),
])
def test_userinfo_is_encoded_exactly_once(password, encoded):
    uri = f"rtsp://user:{password}@camera.example:554/stream"
    result = quote_rtsp_userinfo(uri)
    assert result == f"rtsp://user:{encoded}@camera.example:554/stream"
    assert quote_rtsp_userinfo(result) == result
