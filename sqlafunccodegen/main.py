import asyncio
import pathlib
from collections import Counter, defaultdict
from enum import StrEnum
from functools import lru_cache
from typing import Annotated, Sequence

import dukpy
import typer
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.ext.asyncio.engine import create_async_engine


SCHEMA = "public"


async def get_graphile_data(dest: str) -> Sequence[dict]:
    engine = create_async_engine("postgresql+asyncpg://" + dest)

    version_q = select(func.current_setting("server_version_num"))
    async with (
        AsyncSession(engine, expire_on_commit=False) as session,
        session.begin(),
    ):
        server_version_num = (await session.execute(version_q)).scalar_one()

    jsi = dukpy.JSInterpreter()
    pkg_path = pathlib.Path(__file__).parent.resolve()
    jsi.loader.register_path(str(pkg_path))
    jsi.evaljs("m = require('introspectionQuery')")
    introspection_q = jsi.evaljs(
        "m.makeIntrospectionQuery(dukpy['serverVersionNum'])",
        serverVersionNum=server_version_num,
    )
    assert isinstance(introspection_q, str)

    async with (
        AsyncSession(engine, expire_on_commit=False) as session,
        session.begin(),
    ):
        conn = await session.connection()
        return (
            (await conn.exec_driver_sql(introspection_q, ([SCHEMA], False)))
            .scalars()
            .all()
        )


class Mode(StrEnum):
    sqlalchemy = "sqlalchemy"
    asyncpg_only = "asyncpg_only"


def main(
    dest: Annotated[
        str,
        typer.Option(help="user:pass@host:port/dbname"),
    ] = "postgres:postgres@localhost:5432/postgres",
    mode: Annotated[
        Mode,
        typer.Option(
            help="""
'asyncpg_only' mode generates functions that take and return Python
values, while 'sqlalchemy' mode generates functions that act
as sqlalchemy.func functions and can be used within a
SQLAlchemy SQL expression. 'asyncpg_only' mode doesn't
require SQLAlchemy. Note: For 'asyncpg_only' mode, we assume that
JSON/JSONB types are encoded as Python JSON types as in
https://magicstack.github.io/asyncpg/current/usage.html#example-automatic-json-conversion
            """.strip()
        ),
    ] = Mode.sqlalchemy,
):
    dest = dest.removeprefix("postgres://").removeprefix("postgresql://")
    python_generator = PythonGenerator(dest=dest)
    print(python_generator.generate(mode))


class TypeStrings(BaseModel):
    python_type: str
    # python type for function inputs (if different from output type)
    python_type_in: str | None = None
    sqla_type: str | None = None


# types within a schema have unique names
# https://magicstack.github.io/asyncpg/current/usage.html#type-conversion
# https://github.com/sqlalchemy/sqlalchemy/blob/52b03301b3ca6cb285b5290de4dfe6806f4836f1/lib/sqlalchemy/sql/sqltypes.py#L3730
pg_catalog_types = {
    "int2": TypeStrings(python_type="int", sqla_type="postgresql.INTEGER"),
    "int4": TypeStrings(python_type="int", sqla_type="postgresql.INTEGER"),
    "int8": TypeStrings(python_type="int", sqla_type="postgresql.INTEGER"),
    "text": TypeStrings(python_type="str", sqla_type="postgresql.TEXT"),
    "varchar": TypeStrings(python_type="str", sqla_type="postgresql.VARCHAR"),
    "char": TypeStrings(python_type="str", sqla_type="postgresql.CHAR"),
    "uuid": TypeStrings(python_type="UUID", sqla_type="sqlalchemy.UUID"),
    "bool": TypeStrings(python_type="bool", sqla_type="sqlalchemy.Boolean"),
    "void": TypeStrings(python_type="None"),
    # we assume json types correspond to python json types. sqlalchemy does this
    # "SQLAlchemy sets default type decoder for json and jsonb types using the
    # python builtin json.loads function", asyncpg by itself requires a custom
    # encoder https://magicstack.github.io/asyncpg/current/usage.html#example-automatic-json-conversion
    "json": TypeStrings(
        python_type="JsonValue",
        sqla_type="postgresql.JSON",
        python_type_in="JsonFrozen",
    ),
    "jsonb": TypeStrings(
        python_type="JsonValue",
        sqla_type="postgresql.JSONB",
        python_type_in="JsonFrozen",
    ),
    "bit": TypeStrings(
        python_type="asyncpg.BitString", sqla_type="postgresql.BIT"
    ),
    "varbit": TypeStrings(
        python_type="asyncpg.BitString", sqla_type="postgresql.VARBINARY"
    ),
    "box": TypeStrings(python_type="asyncpg.Box"),
    "bytea": TypeStrings(python_type="bytes", sqla_type="postgresql.BYTEA"),
    "cidr": TypeStrings(
        python_type="IPv4Network | IPv6Network", sqla_type="postgresql.CIDR"
    ),
    "inet": TypeStrings(
        python_type="IPv4Interface | IPv6Interface | IPv4Address | IPv6Address",
        sqla_type="postgresql.INET",
    ),
    "macaddr": TypeStrings(python_type="str", sqla_type="postgresql.MACADDR"),
    "circle": TypeStrings(python_type="asyncpg.Circle"),
    "date": TypeStrings(
        python_type="datetime.date", sqla_type="postgresql.DATE"
    ),
    "time": TypeStrings(
        python_type="datetime.time", sqla_type="postgresql.TIME(timezone=False)"
    ),
    "timetz": TypeStrings(
        python_type="datetime.time", sqla_type="postgresql.TIME(timezone=True)"
    ),
    "timestamp": TypeStrings(
        python_type="datetime.datetime",
        sqla_type="postgresql.TIMESTAMP(timezone=False)",
    ),
    "timestamptz": TypeStrings(
        python_type="datetime.datetime",
        sqla_type="postgresql.TIMESTAMP(timezone=True)",
    ),
    "interval": TypeStrings(
        python_type="datetime.timedelta",
        sqla_type="postgresql.INTERVAL",
    ),
    "line": TypeStrings(python_type="asyncpg.Line"),
    "lseg": TypeStrings(python_type="asyncpg.LineSegment"),
    "path": TypeStrings(python_type="asyncpg.Path"),
    "point": TypeStrings(python_type="asyncpg.Point"),
    "polygon": TypeStrings(python_type="asyncpg.Polygon"),
    "float4": TypeStrings(
        python_type="float", sqla_type="postgresql.FLOAT(24)"
    ),
    "float8": TypeStrings(
        python_type="float", sqla_type="postgresql.FLOAT(53)"
    ),
    "bigint": TypeStrings(python_type="int", sqla_type="postgresql.BIGINT"),
    "numeric": TypeStrings(
        python_type="Decimal", sqla_type="postgresql.NUMERIC"
    ),
    "money": TypeStrings(python_type="str", sqla_type="postgresql.MONEY"),
}

IMPORTS = """\
import datetime
from decimal import Decimal
from enum import StrEnum
from ipaddress import (
    IPv4Address, IPv6Address,
    IPv4Interface, IPv6Interface,
    IPv4Network, IPv6Network,
)
from typing import Annotated, Any, Mapping, Sequence, TypeAliasType, TypeVar, Union
from uuid import UUID

import asyncpg
"""

SQLALCHEMY_IMPORTS = """\
import sqlalchemy
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.functions import GenericFunction
"""

PYDANTIC_IMPORTS = """\
import pydantic
"""

DEFINITIONS = """\
JsonValue = TypeAliasType("JsonValue", Union[dict[str, "JsonValue"], list["JsonValue"], str, int, float, bool, None])
JsonFrozen = TypeAliasType("JsonFrozen", Union[Mapping[str, "JsonFrozen"], Sequence["JsonFrozen"], str, int, float, bool, None])
"""

PYDANTIC_DEFINITIONS = """\
def __convert_output(t, v):
    S = pydantic.create_model(
        'S',
        f=(t, ...),
        __config__=pydantic.ConfigDict(arbitrary_types_allowed=True),
    )
    return S.model_validate({'f': v}).f  # type: ignore


def __convert_input(v):
    class S(pydantic.BaseModel):
        model_config=pydantic.ConfigDict(arbitrary_types_allowed=True)
        f: Any

    return S(f=v).model_dump()["f"]  # type: ignore
"""


def docstringize(s: str) -> str:
    if s.startswith("'"):
        s = "''" + s + "''"
    else:
        s = '""' + s + '""'

    return s


class PythonGenerator:
    out_enums: set[str]
    class_ids_to_generate: set[str]

    def __init__(self, dest: str):
        self.dest = dest

        self.out_enums = set()
        self.class_ids_to_generate = set()

        self.graphile_data = asyncio.run(get_graphile_data(dest=self.dest))

        self.graphile_type_by_id = {
            obj["id"]: obj
            for obj in self.graphile_data
            if obj["kind"] == "type"
        }
        self.graphile_class_by_id = {
            obj["id"]: obj
            for obj in self.graphile_data
            if obj["kind"] == "class"
        }
        self.graphile_attrs_by_class_id = defaultdict(list)
        for obj in self.graphile_data:
            if obj["kind"] == "attribute":
                self.graphile_attrs_by_class_id[obj["classId"]].append(obj)
        for key in self.graphile_attrs_by_class_id:
            self.graphile_attrs_by_class_id[key].sort(key=lambda x: x["num"])

    @lru_cache(maxsize=None)
    def graphile_type_to_python(
        self, graphile_type_id: str, in_: bool, mode: Mode
    ) -> str:
        graphile_type = self.graphile_type_by_id[graphile_type_id]

        if graphile_type["isPgArray"]:
            item_graphile_type = self.graphile_type_by_id[
                graphile_type["arrayItemTypeId"]
            ]
            item_python_type = self.graphile_type_to_python(
                item_graphile_type["id"],
                in_=in_,
                mode=mode,
            )
            # in postgres, arrays can have any dimension, but asyncpg doesn't
            # support more than one
            if in_:
                return f"Sequence[{item_python_type}] | None"
            else:
                return f"list[{item_python_type}] | None"
        elif graphile_type["namespaceName"] == "pg_catalog":
            if graphile_type["name"] in pg_catalog_types:
                ts = pg_catalog_types[graphile_type["name"]]
                if in_ and ts.python_type_in:
                    pt = ts.python_type_in
                else:
                    pt = ts.python_type

                # we have to avoid `None | None` because it's an error
                return f"Union[{pt}, None]"
        elif graphile_type["namespaceName"] == SCHEMA:
            if graphile_type["enumVariants"]:
                if mode == Mode.sqlalchemy:
                    return "str | None"
                enum_name = f"Enum__{graphile_type['name']}"
                e = f"class {enum_name}(StrEnum):\n"
                if graphile_type["description"]:
                    e += f"    {docstringize(repr(graphile_type['description']))}\n"
                e += "\n".join(
                    f"    {ev} = '{ev}'" for ev in graphile_type["enumVariants"]
                )
                self.out_enums.add(e)
                return f"{enum_name} | None"
            elif graphile_type["classId"] is not None:
                if mode == Mode.sqlalchemy:
                    if in_:
                        return (
                            "tuple | Mapping[str, Any] | asyncpg.Record | None"
                        )
                    else:
                        return "asyncpg.Record | None"
                self.class_ids_to_generate.add(graphile_type["classId"])
                return f"Model__{graphile_type['name']} | None"
            elif graphile_type["domainBaseTypeId"] is not None:
                return self.graphile_type_to_python(
                    graphile_type["domainBaseTypeId"],
                    in_=in_,
                    mode=mode,
                )
        return "Any"

    @lru_cache(maxsize=None)
    def graphile_type_to_sqla(self, graphile_type_id: str) -> str:
        graphile_type = self.graphile_type_by_id[graphile_type_id]

        if graphile_type["isPgArray"]:
            item_graphile_type = self.graphile_type_by_id[
                graphile_type["arrayItemTypeId"]
            ]
            item_sqla_type = self.graphile_type_to_sqla(
                item_graphile_type["id"]
            )
            if item_sqla_type != "None":
                # ARRAY(None) doesn't seem to work
                return f"postgresql.ARRAY({item_sqla_type})"
        elif graphile_type["namespaceName"] == "pg_catalog":
            if graphile_type["name"] in pg_catalog_types:
                sqla_type = pg_catalog_types[graphile_type["name"]].sqla_type
                if sqla_type is not None:
                    return sqla_type
        elif graphile_type["namespaceName"] == SCHEMA:
            if graphile_type["enumVariants"]:
                out_enum = ", ".join(
                    repr(ev) for ev in graphile_type["enumVariants"]
                )
                return f"postgresql.ENUM({out_enum}, name={repr(graphile_type['name'])})"
            elif graphile_type["domainBaseTypeId"] is not None:
                return self.graphile_type_to_sqla(
                    graphile_type["domainBaseTypeId"],
                )
        # we could do `TypeEngine[$python_type]` as a fallback if there's any
        # benefit. for user-defined types, we could define classes that
        # inherit from UserDefinedType, if there's a need.
        return "None"

    def generate(self, mode: Mode) -> str:
        graphile_procedures = [
            gd for gd in self.graphile_data if gd["kind"] == "procedure"
        ]

        # Graphile doesn't seem to support overloads
        name_count = Counter(
            procedure["name"] for procedure in graphile_procedures
        )
        overloads = {
            procedure["name"]
            for procedure in graphile_procedures
            if name_count[procedure["name"]] > 1
        }
        assert not overloads

        generated_procedures = [
            self.generate_procedure(procedure, mode=mode)
            for procedure in graphile_procedures
        ]
        out_procedures = "\n\n".join(
            sorted(gp for gp in generated_procedures if gp is not None)
        )

        ret = IMPORTS
        if mode == Mode.sqlalchemy:
            ret += SQLALCHEMY_IMPORTS
        if mode == Mode.asyncpg_only:
            ret += PYDANTIC_IMPORTS

        ret += DEFINITIONS
        if mode == Mode.asyncpg_only:
            ret += PYDANTIC_DEFINITIONS

        if mode == Mode.asyncpg_only:
            # can affect `self.out_enums`.
            # note that if we generate in asyncpg_only mode first, then
            # sqlalchemy mode, we will have these extra values in
            # `self.out_enums`.
            out_models = "\n\n".join(sorted(self.generate_models()))
        else:
            out_models = ""

        if mode == Mode.asyncpg_only:
            ret += "\n".join(sorted(self.out_enums)) + "\n\n"
            ret += out_models
        ret += out_procedures

        return ret

    def generate_models(self) -> list[str]:
        completed_class_ids = set()
        ret = []
        # items can get added to the set while we're iterating
        while self.class_ids_to_generate:
            class_id = self.class_ids_to_generate.pop()
            if class_id in completed_class_ids:
                continue
            class_ = self.graphile_class_by_id[class_id]
            out = f"class Model__{class_['name']}(pydantic.BaseModel):\n"
            graphile_type = self.graphile_type_by_id[class_["typeId"]]
            if graphile_type["description"]:
                out += f"    {repr(graphile_type['description'])}\n"
            out += """
    model_config=pydantic.ConfigDict(arbitrary_types_allowed=True)

    @pydantic.model_validator(mode="before")
    @classmethod
    def validate_model(cls, data):
        if isinstance(data, asyncpg.Record):
            return dict(data.items())
        elif isinstance(data, tuple):
            # not sure when this can happen
            return dict(
                (k, v)
                for k, v in zip(cls.model_fields, data)
            )
        else:
            return data
"""

            attrs = self.graphile_attrs_by_class_id[class_id]
            for attr in attrs:
                basic_attr_type = (
                    "'"
                    + self.graphile_type_to_python(
                        # composite types can't have polymorphic types in them
                        attr["typeId"],
                        in_=False,
                        mode=Mode.asyncpg_only,
                    )
                    + "'"
                )
                if attr["description"]:
                    attr_type = (
                        f"Annotated[{basic_attr_type}, pydantic.Field("
                        f"description={repr(attr['description'])}"
                        ")]"
                    )
                else:
                    attr_type = basic_attr_type
                out += f"    {attr['name']}: {attr_type}\n"

            ret.append(out)
            completed_class_ids.add(class_id)
        return ret

    def generate_procedure(self, procedure: dict, mode: Mode) -> str | None:
        # if any arg mode is not IN, we don't handle it
        # variadic params aren't supported in Graphile
        if any(am != "i" for am in procedure["argModes"]):
            return None

        return_type = self.graphile_type_by_id[procedure["returnTypeId"]]
        arg_types = [
            self.graphile_type_by_id[arg_id]
            for arg_id in procedure["argTypeIds"]
        ]

        polymorphic = bool(
            ({at["name"] for at in arg_types} | {return_type["name"]})
            & {
                "any",
                "anyelement",
                "anyarray",
                "anynonarray",
                "anyenum",
                "anyrange",
                "anymultirange",
                "anycompatible",
                "anycompatiblearray",
                "anycompatiblenonarray",
                "anycompatiblerange",
                "anycompatiblemultirange",
            }
        )
        if polymorphic:
            return None

        scalar_return_type = self.graphile_type_to_python(
            return_type["id"], in_=False, mode=mode
        )

        params_list = []
        for arg_type, arg_name in zip(arg_types, procedure["argNames"]):
            s = f"{arg_name}: "
            pt = self.graphile_type_to_python(
                arg_type["id"], in_=True, mode=mode
            )
            if mode == Mode.asyncpg_only:
                s += pt
            else:
                # see https://github.com/sqlalchemy/sqlalchemy/blob/0a1be6dcfab7b17613835248a368e6b822a53319/lib/sqlalchemy/sql/_typing.py#L229
                # and https://github.com/sqlalchemy/sqlalchemy/blob/0a1be6dcfab7b17613835248a368e6b822a53319/lib/sqlalchemy/sql/functions.py#L1852
                s += f"{pt} | sqlalchemy.ColumnExpressionArgument[{pt}]"
            params_list.append(s)
        out_params = ", ".join(params_list)

        if mode == Mode.sqlalchemy:
            # e.g. if a function takes a JSONB, we can't just pass a python
            # string because that becomes `TEXT` in pg which is a type
            # error.
            out_args_list = []
            for arg_type, arg_name in zip(arg_types, procedure["argNames"]):
                sqla_type = self.graphile_type_to_sqla(arg_type["id"])
                if sqla_type == "None":
                    out_args_list.append(arg_name)
                else:
                    out_args_list.append(
                        f"sqlalchemy.cast({arg_name}, type_={sqla_type})"
                    )
            out_args = ", ".join(out_args_list)
        else:
            # somehow raw asyncpg works without casting
            out_args = ", ".join(
                f"__convert_input({arg_name})"
                for arg_name in procedure["argNames"]
            )

        if procedure["description"] is None:
            out_docstring = ""
        else:
            out_docstring = docstringize(repr(procedure["description"]))

        if mode == "asyncpg_only":
            dollar_args = ", ".join(
                f"${i+1}" for i in range(len(procedure["argNames"]))
            )
            func_expr = f"{procedure['name']}({dollar_args})"
            seq = f"'select {func_expr}', {out_args}"
            if procedure["returnsSet"]:
                # we won't get None
                out_return_type = f"list[{scalar_return_type}]"
                ret_expr = f"[__convert_output({scalar_return_type}, r[0]) for r in await conn.fetch({seq})]"
            else:
                out_return_type = scalar_return_type
                ret_expr = f"__convert_output({scalar_return_type}, await conn.fetchval({seq}))"
            return f"""async def {procedure['name']}(
    conn: asyncpg.Connection, {out_params}
) -> {out_return_type}:
    {out_docstring}
    return {ret_expr}"""
        else:
            # can be "None" and parens don't matter, see https://github.com/sqlalchemy/sqlalchemy/blob/a96c607cc53f16b364d5025c5a5de1470661d3c9/lib/sqlalchemy/sql/functions.py#L1623
            sqla_ret = self.graphile_type_to_sqla(return_type["id"])
            out_params_comma = out_params + "," if out_params else ""
            out_args_comma = out_args + "," if out_args else ""

            return f"""class {procedure['name']}(GenericFunction[{scalar_return_type}]):
    inherit_cache = True
    type = {sqla_ret}
    package = "sqlafunccodegen"
    def __init__(self, {out_params_comma} **kwargs):
        {out_docstring}

        super().__init__({out_args_comma} **kwargs)
"""


def cli() -> int:
    typer.run(main)
    return 0


if __name__ == "__main__":
    exit(cli())
