import importlib


def test_app_package_exports_main():
    module = importlib.import_module("app")

    assert hasattr(module, "main"), "app package should expose main entrypoint"
