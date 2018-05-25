import json
from dataclasses import asdict, dataclass
from datetime import datetime as dt
from typing import (Any, Dict, Generic, List, Optional, Type, TypeVar, Union)

from redis import StrictRedis
from typing_extensions import Protocol
import itertools
from utilities.time_utils import date_from_ts_ms, timestamp_ms

JSON = Any
UID = int


T = TypeVar('T', bound='Item')
U = TypeVar('U', bound='Stamped')
R = TypeVar('R', bound='Reference')


class Item(Protocol):

    @classmethod
    def from_json(cls: Type[T], data: JSON, **kwargs: Any) -> T:
        ...

    @property
    def json(self) -> JSON:
        ...


# class Raw(NamedTuple):
#
#     data: JSON
#
#     @classmethod
#     def from_json(cls, data: JSON, **kwargs: Any) -> 'Raw':
#         return cls(data=data)
#
#     @property
#     def json(self) -> JSON:
#         return self.data


@dataclass
class Stamped:

    stamp: dt

    @classmethod
    def from_json(cls: Type[U], data: JSON, **kwargs: Any) -> U:  # type: ignore
        data = dict(data)
        data['stamp'] = date_from_ts_ms(data['stamp'])
        return cls(**data)  # type: ignore

    @property
    def json(self) -> JSON:
        data = asdict(self)
        data['stamp'] = timestamp_ms(self.stamp)
        return data

    @classmethod
    def from_data(cls: Type[U], *args: Any, **kwargs: Any) -> U:   # type: ignore
        stamp = dt.now()
        if args:
            return cls(stamp, *args)  # type: ignore
        else:
            return cls(stamp=stamp, **kwargs)  # type: ignore


@dataclass
class StampedRaw(Stamped):
    data: JSON


class Collection(Generic[T]):
    collection: str = ''
    uri: str
    kwargs: Dict
    item: Type[T]
    redis: StrictRedis

    def __init__(self, redis: StrictRedis, base_uri: str, name: Optional[str] = None,
                 **kwargs: Any) -> None:
        self.redis = redis
        if name is not None:
            self.collection = name
        self.kwargs = kwargs
        self.uri = f"{base_uri}/{self.collection}"

    def delete(self) -> None:
        self.redis.delete(self.uri)

    def load(self, value: bytes) -> T:
        return self.item.from_json(data=json.loads(value), **self.kwargs)

    def dump(self, value: T) -> str:
        return json.dumps(value.json)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} {self.uri}"


class TimeStampedCollection(Collection[Stamped]):

    max_length: int = 1000
    item: Type[Stamped] = StampedRaw

    def push(self, value: Stamped) -> Stamped:
        self.redis.zadd(self.uri, timestamp_ms(value.stamp), self.dump(value))
        if self.max_length > 0:
            self.redis.zremrangebyrank(self.uri, 0, -self.max_length)
        return value

    # Changed!!! (return a list not a union of list and T)
    def get_last(self, length: int = 1, rev: bool = True) -> List[Stamped]:
        length = max(1, min(self.max_length, length))
        if rev:
            items = self.redis.zrevrange(self.uri, 0, length - 1)
        else:
            items = self.redis.zrange(self.uri, 0, length - 1)
        return [self.load(value) for value in items]

    def get_first(self, length: int = 1) -> List[Stamped]:
        return self.get_last(length, rev=False)

    def get_all_in_timeslice(self, start: Optional[dt] = None,
                             end: Optional[dt] = None) -> List[Stamped]:
        start_r: Union[str, float]
        end_r: Union[str, float]
        if start:
            start_r = timestamp_ms(start)
        else:
            start_r = "-inf"
        if end:
            end_r = timestamp_ms(end)
        else:
            end_r = "+inf"
        items = self.redis.zrevrangebyscore(self.uri, end_r, start_r)
        return [self.load(value) for value in items]


@dataclass
class Reference:
    index: bytes
    uri: str

    @classmethod
    def from_json(cls: Type[R], data: JSON, **kwargs: Any) -> R:
        return cls(index=data, uri=f"{kwargs['base_uri']}/{data}")  # type: ignore

    @property
    def json(self) -> JSON:
        return str(self.index)


class ReferenceCollection(Collection[Reference]):

    stamped = False
    item: Type[Reference] = Reference
    max_length: int = -1

    def __init__(self, redis: StrictRedis, base_uri: str, name: Optional[str] = None,
                 **kwargs: Any) -> None:
        super(ReferenceCollection, self).__init__(
            redis=redis, base_uri=base_uri, name=name, **kwargs)
        self.kwargs['base_uri'] = self.uri

    def push(self, index: Optional[Union[str, int]] = None) -> Reference:
        if index is None:
            indices: List[int] = [int(x) for x in self.redis.zrange(self.uri, 0, -1)]
            s_index = next(str(i) for i in itertools.count() if i not in indices)
            index = int(s_index)
        else:
            s_index = str(index)
        if self.stamped:
            score = timestamp_ms(dt.now())
        else:
            score = index
        self.redis.zadd(self.uri, score, s_index)

        if self.max_length > 0:
            self.redis.zremrangebyrank(self.uri, 0, -self.max_length)

        return self.item.from_json(data=s_index, **self.kwargs)

    def has(self, index: Union[str, int]) -> bool:
        return not self.redis.zscore(self.uri, str(index)) is None

    def get(self, index: Union[str, int]) -> Optional[Reference]:
        if index is None or not self.has(index):
            return None
        return self.load(bytes(str(index), 'ascii'))

    def get_indices(self, start: int = 0, end: int = -1) -> List[bytes]:
        return self.redis.zrange(self.uri, start, end, withscores=False)

    def get_all(self, start: int = 0, end: int = -1) -> List[Reference]:
        return [self.load(i) for i in self.get_indices(start=start, end=end)]

    def delete(self) -> None:
        for item in self.get_all():
            self.redis.delete(item.uri)
        self.redis.delete(self.uri)

    def remove(self, index: Union[str, int]) -> None:
        item = self.get(index)
        if not item:
            return
        if hasattr(item, 'delete'):
            item.delete()  # type: ignore
        self.redis.zrem(self.uri, index)
        self.redis.delete(item.uri)

    def get_last_indices(self, length: int = 1) -> List[bytes]:
        if self.max_length > 0:
            length = max(1, min(self.max_length, length))
        return self.redis.zrevrange(self.uri, 0, length - 1, withscores=False)

    def get_last(self, length: int = 1) -> List[Reference]:
        return [self.load(i) for i in self.get_last_indices(length=length)]

    def get_indices_in_timeslice(self, start: Optional[dt] = None,
                                 end: Optional[dt] = None) -> List[bytes]:
        if not self.stamped:
            return []
        start_score = timestamp_ms(start) if start else "-inf"
        end_score = timestamp_ms(end) if end else "+inf"
        return self.redis.zrangebyscore(self.uri, start_score, end_score, withscores=False)

    def get_all_in_timeslice(self, start: Optional[dt] = None,
                             end: Optional[dt] = None) -> List[Reference]:
        return [self.load(i) for i in self.get_indices_in_timeslice(start=start, end=end)]
