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
ArrayIn__complex = TypeAliasType('ArrayIn__complex', 'Sequence[Model__complex | None] | Sequence[ArrayIn__complex] | None')
ArrayIn__int4 = TypeAliasType('ArrayIn__int4', 'Sequence[Union[int, None]] | Sequence[ArrayIn__int4] | None')
Array__complex = TypeAliasType('Array__complex', 'list[Model__complex | None] | list[Array__complex] | None')
Array__int4 = TypeAliasType('Array__int4', 'list[Union[int, None]] | list[Array__int4] | None')
Array__mood = TypeAliasType('Array__mood', 'list[Enum__mood | None] | list[Array__mood] | None')

class Enum__mood(str, Enum):
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
    moods: 'Array__mood'


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
    db_sesh: AsyncSession, arr: ArrayIn__int4
) -> Array__int4:
    
    r = (await db_sesh.execute(
        sqlalchemy.select(
            getattr(sqlalchemy.func, 'array_id')(sqlalchemy.literal(__convert_input(arr), type_=postgresql.ARRAY(postgresql.INTEGER)))
        )
    )).scalar_one_or_none()
    return __convert_output(Array__int4, r)

async def c2vector_id(
    db_sesh: AsyncSession, c: Model__c2vector | None
) -> Model__c2vector | None:
    
    r = (await db_sesh.execute(
        sqlalchemy.select(
            getattr(sqlalchemy.func, 'c2vector_id')(sqlalchemy.literal(__convert_input(c), type_=None))
        )
    )).scalar_one_or_none()
    return __convert_output(Model__c2vector | None, r)

async def circle_id(
    db_sesh: AsyncSession, c: Union[asyncpg.Circle, None]
) -> Union[asyncpg.Circle, None]:
    
    r = (await db_sesh.execute(
        sqlalchemy.select(
            getattr(sqlalchemy.func, 'circle_id')(sqlalchemy.literal(__convert_input(c), type_=None))
        )
    )).scalar_one_or_none()
    return __convert_output(Union[asyncpg.Circle, None], r)

async def complex_array_id(
    db_sesh: AsyncSession, ca: ArrayIn__complex
) -> Array__complex:
    
    r = (await db_sesh.execute(
        sqlalchemy.select(
            getattr(sqlalchemy.func, 'complex_array_id')(sqlalchemy.literal(__convert_input(ca), type_=None))
        )
    )).scalar_one_or_none()
    return __convert_output(Array__complex, r)

async def complex_id(
    db_sesh: AsyncSession, z: Model__complex | None
) -> Model__complex | None:
    
    r = (await db_sesh.execute(
        sqlalchemy.select(
            getattr(sqlalchemy.func, 'complex_id')(sqlalchemy.literal(__convert_input(z), type_=None))
        )
    )).scalar_one_or_none()
    return __convert_output(Model__complex | None, r)

async def get_mood(
    db_sesh: AsyncSession, _mood: Enum__mood | None
) -> Enum__mood | None:
    
    r = (await db_sesh.execute(
        sqlalchemy.select(
            getattr(sqlalchemy.func, 'get_mood')(sqlalchemy.literal(__convert_input(_mood), type_=postgresql.ENUM('happy', 'sad', 'neutral', name='mood')))
        )
    )).scalar_one_or_none()
    return __convert_output(Enum__mood | None, r)

async def get_range(
    db_sesh: AsyncSession, 
) -> Any:
    
    r = (await db_sesh.execute(
        sqlalchemy.select(
            getattr(sqlalchemy.func, 'get_range')()
        )
    )).scalar_one_or_none()
    return __convert_output(Any, r)

async def jsonb_id(
    db_sesh: AsyncSession, j: Union[JsonFrozen, None]
) -> Union[pydantic.JsonValue, None]:
    '''Returns the same jsonb value passed in'''
    r = (await db_sesh.execute(
        sqlalchemy.select(
            getattr(sqlalchemy.func, 'jsonb_id')(sqlalchemy.literal(__convert_input(j), type_=postgresql.JSONB))
        )
    )).scalar_one_or_none()
    return __convert_output(Union[pydantic.JsonValue, None], r)

async def set_of_complex_arrays(
    db_sesh: AsyncSession, 
) -> Iterable[Array__complex]:
    
    r = (await db_sesh.execute(
        sqlalchemy.select(
            getattr(sqlalchemy.func, 'set_of_complex_arrays')()
        )
    )).scalars()
    return (__convert_output(Array__complex, i) for i in r)

async def unitthing(
    db_sesh: AsyncSession, z: Model__complex | None
) -> Model__complex | None:
    
    r = (await db_sesh.execute(
        sqlalchemy.select(
            getattr(sqlalchemy.func, 'unitthing')(sqlalchemy.literal(__convert_input(z), type_=None))
        )
    )).scalar_one_or_none()
    return __convert_output(Model__complex | None, r)
