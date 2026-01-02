import datetime
from decimal import Decimal
from enum import StrEnum
from ipaddress import (
    IPv4Address, IPv6Address,
    IPv4Interface, IPv6Interface,
    IPv4Network, IPv6Network,
)
from typing import Annotated, Any, Mapping, Sequence, TypeVar, Union
from uuid import UUID

import asyncpg
import sqlalchemy
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.functions import GenericFunction
type JsonValue = Union[dict[str, "JsonValue"], list["JsonValue"], str, int, float, bool, None]
type JsonFrozen = Union[Mapping[str, "JsonFrozen"], Sequence["JsonFrozen"], str, int, float, bool, None]
class array_id(GenericFunction[list[Union[int, None]] | None]):
    inherit_cache = True
    type = postgresql.ARRAY(postgresql.INTEGER)
    package = "sqlafunccodegen"
    def __init__(self, arr: Sequence[Union[int, None]] | None | sqlalchemy.ColumnExpressionArgument[Sequence[Union[int, None]] | None], **kwargs):
        

        super().__init__(sqlalchemy.cast(arr, type_=postgresql.ARRAY(postgresql.INTEGER)), **kwargs)


class c2vector_id(GenericFunction[asyncpg.Record | None]):
    inherit_cache = True
    type = None
    package = "sqlafunccodegen"
    def __init__(self, c: tuple | Mapping[str, Any] | asyncpg.Record | None | sqlalchemy.ColumnExpressionArgument[tuple | Mapping[str, Any] | asyncpg.Record | None], **kwargs):
        

        super().__init__(c, **kwargs)


class circle_id(GenericFunction[Union[asyncpg.Circle, None]]):
    inherit_cache = True
    type = None
    package = "sqlafunccodegen"
    def __init__(self, c: Union[asyncpg.Circle, None] | sqlalchemy.ColumnExpressionArgument[Union[asyncpg.Circle, None]], **kwargs):
        

        super().__init__(c, **kwargs)


class complex_array_id(GenericFunction[list[asyncpg.Record | None] | None]):
    inherit_cache = True
    type = None
    package = "sqlafunccodegen"
    def __init__(self, ca: Sequence[tuple | Mapping[str, Any] | asyncpg.Record | None] | None | sqlalchemy.ColumnExpressionArgument[Sequence[tuple | Mapping[str, Any] | asyncpg.Record | None] | None], **kwargs):
        

        super().__init__(ca, **kwargs)


class complex_id(GenericFunction[asyncpg.Record | None]):
    inherit_cache = True
    type = None
    package = "sqlafunccodegen"
    def __init__(self, z: tuple | Mapping[str, Any] | asyncpg.Record | None | sqlalchemy.ColumnExpressionArgument[tuple | Mapping[str, Any] | asyncpg.Record | None], **kwargs):
        

        super().__init__(z, **kwargs)


class get_mood(GenericFunction[str | None]):
    inherit_cache = True
    type = postgresql.ENUM('happy', 'sad', 'neutral', name='mood')
    package = "sqlafunccodegen"
    def __init__(self, _mood: str | None | sqlalchemy.ColumnExpressionArgument[str | None], **kwargs):
        

        super().__init__(sqlalchemy.cast(_mood, type_=postgresql.ENUM('happy', 'sad', 'neutral', name='mood')), **kwargs)


class get_range(GenericFunction[Any]):
    inherit_cache = True
    type = None
    package = "sqlafunccodegen"
    def __init__(self,  **kwargs):
        

        super().__init__( **kwargs)


class jsonb_id(GenericFunction[Union[JsonValue, None]]):
    inherit_cache = True
    type = postgresql.JSONB
    package = "sqlafunccodegen"
    def __init__(self, j: Union[JsonFrozen, None] | sqlalchemy.ColumnExpressionArgument[Union[JsonFrozen, None]], **kwargs):
        '''Returns the same jsonb value passed in'''

        super().__init__(sqlalchemy.cast(j, type_=postgresql.JSONB), **kwargs)


class set_of_complex_arrays(GenericFunction[list[asyncpg.Record | None] | None]):
    inherit_cache = True
    type = None
    package = "sqlafunccodegen"
    def __init__(self,  **kwargs):
        

        super().__init__( **kwargs)


class unitthing(GenericFunction[asyncpg.Record | None]):
    inherit_cache = True
    type = None
    package = "sqlafunccodegen"
    def __init__(self, z: tuple | Mapping[str, Any] | asyncpg.Record | None | sqlalchemy.ColumnExpressionArgument[tuple | Mapping[str, Any] | asyncpg.Record | None], **kwargs):
        

        super().__init__(z, **kwargs)

