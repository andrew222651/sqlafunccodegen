import pathlib

import dukpy


GRAPHILE_VERSION = "4.13.0"
PROJECT_ROOT = pathlib.Path(__file__).parent.parent


def main():
    dukpy.install_jspackage(
        "graphile-build-pg", GRAPHILE_VERSION, str(PROJECT_ROOT / "js_modules")
    )
    introspection_query_path = PROJECT_ROOT / pathlib.Path(
        "js_modules/graphile-build-pg/node8plus/plugins/introspectionQuery.js"
    )
    compiled_path = PROJECT_ROOT / "sqlafunccodegen" / "introspectionQuery.js"
    compiled_path.write_text(
        dukpy.babel_compile(introspection_query_path.read_text())["code"]  # type: ignore
    )


if __name__ == "__main__":
    main()
