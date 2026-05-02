from importlib import import_module


def run_app(cfg):
    app_name = cfg.get("app")
    if not app_name:
        raise ValueError("Config must include 'app'.")
    module = import_module(f"app.{app_name}.train")
    module.main(cfg)
