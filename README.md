# sqlafunccodegen

Generate type-annotated Python functions that wrap PostgreSQL functions, using
asyncpg and, optionally, SQLAlchemy.
Like [sqlacodegen](https://github.com/agronholm/sqlacodegen)
but for functions instead of tables.

Usage:
```bash
sqlafunccodegen --help
```

Capabilities:
* "sqlalchemy" mode: functions behave like `sqlalchemy.func.<function_name>`
  * functions can be used inside SQLAlchemy query expressions
* "asyncpg_only" mode: functions execute a select statement and
  return results
  * Python `Enum` types
  * Pydantic models for user-defined composite types
  * set-returning functions return lists
  * constraints in domains are NOT checked but the underlying type is used
* all modes:
  * casts are added for convenience
  * many basic types
  * arrays
  * comments as docstrings
  * functions with overloads NOT supported
  * polymorphic pseudo-types NOT supported
  * `IN`, `INOUT`, and `VARIADIC` params NOT supported
  * default values are NOT available

Generated code dependencies:
* asyncpg
* Pydantic 2 (for "asyncpg_only" mode)
* SQLAlchemy 2 (for "sqlalchemy" mode)

Examples
* input: [`tests/schema.ddl`](tests/schema.ddl)
* "sqlalchemy" mode output: [`tests/out_func.py`](tests/out_func.py)
* "asyncpg_only" mode output: [`tests/out_asyncpg_only.py`](tests/out_asyncpg_only.py)

