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
def array_id(
    arr: Any
) -> Any:
    
    return sqlalchemy.cast(getattr(sqlalchemy.func, 'array_id')(sqlalchemy.cast(arr, type_=postgresql.ARRAY(postgresql.INTEGER))), type_=postgresql.ARRAY(postgresql.INTEGER))

def c2vector_id(
    c: Any
) -> Any:
    
    return getattr(sqlalchemy.func, 'c2vector_id')(c)

def circle_id(
    c: Any
) -> Any:
    
    return getattr(sqlalchemy.func, 'circle_id')(c)

def complex_array_id(
    ca: Any
) -> Any:
    
    return getattr(sqlalchemy.func, 'complex_array_id')(ca)

def complex_id(
    z: Any
) -> Any:
    
    return getattr(sqlalchemy.func, 'complex_id')(z)

def get_mood(
    _mood: Any
) -> Any:
    
    return sqlalchemy.cast(getattr(sqlalchemy.func, 'get_mood')(sqlalchemy.cast(_mood, type_=postgresql.ENUM('happy', 'sad', 'neutral', name='mood'))), type_=postgresql.ENUM('happy', 'sad', 'neutral', name='mood'))

def get_range(
    
) -> Any:
    
    return getattr(sqlalchemy.func, 'get_range')()

def jsonb_id(
    j: Any
) -> Any:
    '''Returns the same jsonb value passed in'''
    return sqlalchemy.cast(getattr(sqlalchemy.func, 'jsonb_id')(sqlalchemy.cast(j, type_=postgresql.JSONB)), type_=postgresql.JSONB)

def set_of_complex_arrays(
    
) -> Any:
    
    return getattr(sqlalchemy.func, 'set_of_complex_arrays')()

def unitthing(
    z: Any
) -> Any:
    
    return getattr(sqlalchemy.func, 'unitthing')(z)
