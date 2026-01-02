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
import pydantic
type JsonValue = Union[dict[str, "JsonValue"], list["JsonValue"], str, int, float, bool, None]
type JsonFrozen = Union[Mapping[str, "JsonFrozen"], Sequence["JsonFrozen"], str, int, float, bool, None]
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
class Enum__mood(StrEnum):
    happy = 'happy'
    sad = 'sad'
    neutral = 'neutral'

class Model__c2vector(pydantic.BaseModel):

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
    z1: 'Model__complex | None'
    z2: 'Model__complex | None'
    moods: 'list[Enum__mood | None] | None'


class Model__complex(pydantic.BaseModel):
    'A complex number'

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
    r: Annotated['Union[float, None]', pydantic.Field(description='The real part')]
    i: 'Union[float, None]'
async def array_id(
    conn: asyncpg.Connection, arr: Sequence[Union[int, None]] | None
) -> list[Union[int, None]] | None:
    
    return __convert_output(list[Union[int, None]] | None, await conn.fetchval('select array_id($1)', __convert_input(arr)))

async def c2vector_id(
    conn: asyncpg.Connection, c: Model__c2vector | None
) -> Model__c2vector | None:
    
    return __convert_output(Model__c2vector | None, await conn.fetchval('select c2vector_id($1)', __convert_input(c)))

async def circle_id(
    conn: asyncpg.Connection, c: Union[asyncpg.Circle, None]
) -> Union[asyncpg.Circle, None]:
    
    return __convert_output(Union[asyncpg.Circle, None], await conn.fetchval('select circle_id($1)', __convert_input(c)))

async def complex_array_id(
    conn: asyncpg.Connection, ca: Sequence[Model__complex | None] | None
) -> list[Model__complex | None] | None:
    
    return __convert_output(list[Model__complex | None] | None, await conn.fetchval('select complex_array_id($1)', __convert_input(ca)))

async def complex_id(
    conn: asyncpg.Connection, z: Model__complex | None
) -> Model__complex | None:
    
    return __convert_output(Model__complex | None, await conn.fetchval('select complex_id($1)', __convert_input(z)))

async def get_mood(
    conn: asyncpg.Connection, _mood: Enum__mood | None
) -> Enum__mood | None:
    
    return __convert_output(Enum__mood | None, await conn.fetchval('select get_mood($1)', __convert_input(_mood)))

async def get_range(
    conn: asyncpg.Connection, 
) -> Any:
    
    return __convert_output(Any, await conn.fetchval('select get_range()', ))

async def jsonb_id(
    conn: asyncpg.Connection, j: Union[JsonFrozen, None]
) -> Union[JsonValue, None]:
    '''Returns the same jsonb value passed in'''
    return __convert_output(Union[JsonValue, None], await conn.fetchval('select jsonb_id($1)', __convert_input(j)))

async def set_of_complex_arrays(
    conn: asyncpg.Connection, 
) -> list[list[Model__complex | None] | None]:
    
    return [__convert_output(list[Model__complex | None] | None, r[0]) for r in await conn.fetch('select set_of_complex_arrays()', )]

async def unitthing(
    conn: asyncpg.Connection, z: Model__complex | None
) -> Model__complex | None:
    
    return __convert_output(Model__complex | None, await conn.fetchval('select unitthing($1)', __convert_input(z)))
