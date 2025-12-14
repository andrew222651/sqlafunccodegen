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

