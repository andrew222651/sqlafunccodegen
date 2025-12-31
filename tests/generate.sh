#!/usr/bin/env bash

cp tests/out_sqlalchemy.py tests/out_sqlalchemy.py.bak
cp tests/out_asyncpg_only.py tests/out_asyncpg_only.py.bak
docker run -d --name testdb --rm -e POSTGRES_PASSWORD=postgres -p 5432:5432 postgres:16-bookworm
sleep 5
psql postgresql://postgres:postgres@localhost:5432/postgres -f tests/schema.ddl
python sqlafunccodegen/main.py --mode sqlalchemy > tests/out_sqlalchemy.py
python sqlafunccodegen/main.py --mode asyncpg_only > tests/out_asyncpg_only.py
docker kill testdb
diff tests/out_sqlalchemy.py tests/out_sqlalchemy.py.bak
diff tests/out_asyncpg_only.py tests/out_asyncpg_only.py.bak
