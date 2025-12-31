# sqlafunccodegen

Generate type-annotated Python functions that wrap PostgreSQL functions, using
asyncpg and, optionally, SQLAlchemy.
Like [sqlacodegen](https://github.com/agronholm/sqlacodegen)
but for functions instead of tables.

Usage:
```bash
sqlafunccodegen --help
```

* all modes:
  * many basic types
  * arrays
  * comments as docstrings
  * casts on inputs are added for convenience
* "sqlalchemy" mode: functions behave like `sqlalchemy.func.<function_name>`
  * functions can be used inside SQLAlchemy query expressions
  * note: if a function call cannot return null, you can wrap the call with
    `sqlalchemy.NotNullable` and `None` will be removed as a possible Python
    type for the expression
* "asyncpg_only" mode: functions execute a select statement and
  return results
  * Python `Enum` types
  * Pydantic models for user-defined composite types
  * set-returning functions return lists
  * constraints in domains are NOT checked but the underlying type is used
  * note: sqlafunccodegen assumes that JSON/JSONB types are encoded as Python
    JSON types as in
    https://magicstack.github.io/asyncpg/current/usage.html#example-automatic-json-conversion
* not supported:
  * functions with overloads
  * polymorphic pseudo-types
  * `IN`, `INOUT`, and `VARIADIC` params
  * default values

Generated code dependencies:
* asyncpg
* Pydantic 2 (for "asyncpg_only" mode)
* SQLAlchemy 2 (for "sqlalchemy" mode)

Examples
* input: [`tests/schema.ddl`](tests/schema.ddl)
* "sqlalchemy" mode output: [`tests/out_sqlalchemy.py`](tests/out_sqlalchemy.py)
* "asyncpg_only" mode output: [`tests/out_asyncpg_only.py`](tests/out_asyncpg_only.py)

