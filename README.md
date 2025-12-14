# sqlafunccodegen

Generate type-annotated Python functions that wrap PostgreSQL functions, using
asyncpg and, optionally, SQLAlchemy.
Like [sqlacodegen](https://github.com/agronholm/sqlacodegen)
but for functions instead of tables.

Benefit over PostgREST: You can call functions and execute other SQL within
a transaction.

Usage:
```bash
sqlafunccodegen --help
```

Capabilities:
* "func" mode: functions directly wrap `sqlalchemy.func.<function_name>`
* "python" and "asyncpg_only" modes: functions execute a select statement and
  return results
  * Pydantic models for user-defined composite types
  * set-returning functions return iterables
* all modes:
  * many basic types
  * enums
  * arrays
  * constraints in domains are NOT checked but the underlying type is used
  * comments as docstrings
  * functions with overloads NOT supported
  * polymorphic pseudo-types NOT supported
  * `IN`, `INOUT`, and `VARIADIC` params NOT supported
  * default values are NOT available

Generated code dependencies:
* asyncpg
* Pydantic 2
* SQLAlchemy 2 (except for "asyncpg_only" mode)

Examples
* input: [`tests/schema.ddl`](tests/schema.ddl)
* "python" mode output: [`tests/out_python.py`](tests/out_python.py)
* "func" mode output: [`tests/out_func.py`](tests/out_func.py)
* "asyncpg_only" mode output: [`tests/out_asyncpg_only.py`](tests/out_asyncpg_only.py)

# See also: sqlc

[sqlc](https://github.com/sqlc-dev/sqlc) can be used with its [Python plugin](https://github.com/sqlc-dev/sqlc-gen-python)
to generate Python functions that wrap SQL snippets that live in the application project rather than
the database. This is similar to the [Python codegen for EdgeDB](https://docs.edgedb.com/libraries/python/api/codegen).
As of v1.26.0, a schema file is required, which can be generated [using pg_dump](https://stackoverflow.com/a/14486505/371334).
A complete `sqlc.yaml` file might look like this:
```yaml
version: "2"
plugins:
- name: py
  wasm:
    url: https://downloads.sqlc.dev/plugin/sqlc-gen-python_1.2.0.wasm
    sha256: a6c5d174c407007c3717eea36ff0882744346e6ba991f92f71d6ab2895204c0e
sql:
  - engine: "postgresql"
    database:
      uri: "postgres://user:pass@127.0.0.1:5432/dbname?sslmode=disable"
    queries: "application_queries.sql"
    schema: "pg_dump_output.sql"
    codegen:
    - out: sqlc_out
      plugin: py
      options:
        package: "foo"
        emit_async_querier: true
```
You can use `sqlc` to call Postgres functions by writing queries like `select * from f()`. As of v1.26.0,
sqlc lacks some type features that sqlafunccodegen has such as JSON types, composite-type table columns, and
using immutable types for inputs. sqlc
uses dataclasses instead of Pydantic models.
