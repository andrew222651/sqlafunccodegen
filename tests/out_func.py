import datetime
from decimal import Decimal
from enum import Enum
from ipaddress import (
    IPv4Address, IPv6Address,
    IPv4Interface, IPv6Interface,
    IPv4Network, IPv6Network,
)
from typing import Annotated, Any, Iterable, Mapping, Sequence, TypeVar, Union
from typing_extensions import TypeAliasType
from uuid import UUID

import asyncpg
import pydantic
import sqlalchemy
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.functions import GenericFunction
from sqlalchemy.sql.type_api import TypeEngine
from sqlalchemy.sql._typing import _ColumnExpressionArgument
_T = TypeVar('_T')
AnyArray = list[_T] | list['AnyArray']
AnyArrayIn = Sequence[_T] | Sequence['AnyArray']
JsonFrozen = Union[Mapping[str, "JsonFrozen"], Sequence["JsonFrozen"], str, int, float, bool, None]

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
ArrayIn__complex = TypeAliasType('ArrayIn__complex', 'Sequence[Any] | Sequence[ArrayIn__complex] | None')
ArrayIn__int4 = TypeAliasType('ArrayIn__int4', 'Sequence[Union[int, None]] | Sequence[ArrayIn__int4] | None')
Array__complex = TypeAliasType('Array__complex', 'list[asyncpg.Record | None] | list[Array__complex] | None')
Array__int4 = TypeAliasType('Array__int4', 'list[Union[int, None]] | list[Array__int4] | None')

class array_id(GenericFunction[Array__int4]):
    
    inherit_cache = True
    type = postgresql.ARRAY(postgresql.INTEGER)
    def __init__(self, arr: _ColumnExpressionArgument[ArrayIn__int4], **kwargs):
        super().__init__(sqlalchemy.cast(arr, type_=postgresql.ARRAY(postgresql.INTEGER)), **kwargs)


class c2vector_id(GenericFunction[asyncpg.Record | None]):
    
    inherit_cache = True
    type = TypeEngine[asyncpg.Record | None]()
    def __init__(self, c: _ColumnExpressionArgument[Any], **kwargs):
        super().__init__(c, **kwargs)


class circle_id(GenericFunction[Union[asyncpg.Circle, None]]):
    
    inherit_cache = True
    type = TypeEngine[Union[asyncpg.Circle, None]]()
    def __init__(self, c: _ColumnExpressionArgument[Union[asyncpg.Circle, None]], **kwargs):
        super().__init__(c, **kwargs)


class complex_array_id(GenericFunction[Array__complex]):
    
    inherit_cache = True
    type = TypeEngine[Array__complex]()
    def __init__(self, ca: _ColumnExpressionArgument[ArrayIn__complex], **kwargs):
        super().__init__(ca, **kwargs)


class complex_id(GenericFunction[asyncpg.Record | None]):
    
    inherit_cache = True
    type = TypeEngine[asyncpg.Record | None]()
    def __init__(self, z: _ColumnExpressionArgument[Any], **kwargs):
        super().__init__(z, **kwargs)


class get_mood(GenericFunction[str | None]):
    
    inherit_cache = True
    type = TypeEngine[str | None]()
    def __init__(self, _mood: _ColumnExpressionArgument[str | None], **kwargs):
        super().__init__(sqlalchemy.cast(_mood, type_=postgresql.ENUM(name='mood')), **kwargs)


class get_range(GenericFunction[Any]):
    
    inherit_cache = True
    type = TypeEngine[Any]()
    def __init__(self,  **kwargs):
        super().__init__( **kwargs)


class jsonb_id(GenericFunction[Union[pydantic.JsonValue, None]]):
    '''Returns the same jsonb value passed in'''
    inherit_cache = True
    type = postgresql.JSONB()
    def __init__(self, j: _ColumnExpressionArgument[Union[JsonFrozen, None]], **kwargs):
        super().__init__(sqlalchemy.cast(j, type_=postgresql.JSONB), **kwargs)


class set_of_complex_arrays(GenericFunction[Array__complex]):
    
    inherit_cache = True
    type = TypeEngine[Array__complex]()
    def __init__(self,  **kwargs):
        super().__init__( **kwargs)


class unitthing(GenericFunction[asyncpg.Record | None]):
    
    inherit_cache = True
    type = TypeEngine[asyncpg.Record | None]()
    def __init__(self, z: _ColumnExpressionArgument[Any], **kwargs):
        super().__init__(z, **kwargs)

