import configparser
import json

import pytest

from tools import configure_journey_sync as setup


@pytest.mark.parametrize("accessible", [True, False])
def test_setup_keeps_keys_local_and_enables_only_after_bucket_check(tmp_path, monkeypatch, capsys, accessible):
    boto3 = pytest.importorskip("boto3")
    monkeypatch.setattr(setup, "ROOT", tmp_path)
    (tmp_path / "config.json").write_text('{"map_sync_enabled":false,"world_seed":42}')
    entries = iter(["local-test-key", "local-test-secret"])
    monkeypatch.setattr(setup, "getpass", lambda prompt: next(entries))

    class Client:
        def head_bucket(self, **kwargs):
            assert kwargs["Bucket"] == setup.BUCKET
            if not accessible:
                raise OSError("unreachable")

        def close(self):
            pass

    monkeypatch.setattr(boto3, "client", lambda *args, **kwargs: Client())
    if accessible:
        setup.main()
        credentials = configparser.RawConfigParser()
        credentials.read(tmp_path / "cache/journey-credentials.ini")
        assert credentials["journey"]["aws_secret_access_key"] == "local-test-secret"
        assert json.loads((tmp_path / "config.json").read_text()) == {
            "map_sync_enabled": True, "world_seed": 42}
    else:
        with pytest.raises(SystemExit, match="Setup was not saved"):
            setup.main()
        assert not (tmp_path / "cache").exists()
        assert json.loads((tmp_path / "config.json").read_text())["map_sync_enabled"] is False
    output = capsys.readouterr()
    assert "local-test-key" not in output.out + output.err
    assert "local-test-secret" not in output.out + output.err
