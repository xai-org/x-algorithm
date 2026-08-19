# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import base64
import dataclasses
import datetime
import enum
import importlib
import inspect
import json
import logging
import pkgutil
import re
import traceback
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, NoneType, UnionType
from typing import (
    Any,
    Literal,
    NamedTuple,
    Type,
    TypeAliasType,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)

from typing_extensions import dataclass_transform

ConfigType = TypeVar("ConfigType", bound="Config")
CONFIG_REGISTRY: dict[str, Any] = {}
logger = logging.getLogger(__name__)


def _config_class_key(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def make_unregistered_config_error_message(cls_name: str) -> str:
    return (
        f"{cls_name} is not registered to config registry. "
        f"Make sure you import the modules that include {cls_name}."
    )


def unregistered_config_error_message_if_unregistered(cls: type) -> str | None:
    key = _config_class_key(cls)
    if key not in CONFIG_REGISTRY:
        return make_unregistered_config_error_message(key)
    return None


@dataclasses.dataclass
class Config:
    def to_dict(self):
        error_msg = unregistered_config_error_message_if_unregistered(type(self))
        if error_msg is not None:
            logger.warning(
                "%s Subclasses of Config must use @configclass "
                "(otherwise from_dict cannot resolve __class=%r).",
                error_msg,
                _config_class_key(type(self)),
            )
        result = {
            f.name: to_dict(getattr(self, f.name), container=f"<field: {f.name}>")
            for f in dataclasses.fields(self)
            if f.init
        }
        assert "__class" not in result
        result["__class"] = _config_class_key(self.__class__)
        return result

    @classmethod
    def from_dict(
        cls: type[ConfigType],
        d: dict[str, Any],
        ensure_class: str | type = "",
        max_back_compat: bool = False,
        ignore_unregistered_cls: bool = False,
    ) -> ConfigType:
        return from_dict(
            d,
            ensure_class=ensure_class,
            max_back_compat=max_back_compat,
            ignore_unregistered_cls=ignore_unregistered_cls,
        )

    def to_json(self, sort_keys: bool = False):
        d = to_dict(self)
        result = json.dumps(d, indent=4, sort_keys=sort_keys)
        return result

    @classmethod
    def from_json(
        cls: type[ConfigType],
        json_string: str,
        ensure_class: str | type = "",
        max_back_compat: bool = False,
        ignore_unregistered_cls: bool = False,
    ) -> ConfigType:
        return from_dict(
            json.loads(json_string),
            ensure_class=ensure_class,
            max_back_compat=max_back_compat,
            ignore_unregistered_cls=ignore_unregistered_cls,
        )

    def init(self, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def get_type_hints(cls) -> dict[str, type]:
        return get_type_hints(cls)


@overload
def configclass(cls: type[ConfigType]) -> type[ConfigType]: ...


@overload
def configclass(
    *, kw_only: bool = ..., unsafe_hash: bool = ...
) -> Callable[[type[ConfigType]], type[ConfigType]]: ...


@dataclass_transform()
def configclass(
    cls: type[ConfigType] | None = None, *, kw_only: bool = False, unsafe_hash=True
) -> Callable[[type[ConfigType]], type[ConfigType]] | type[ConfigType]:
    @dataclass_transform()
    def _configclass(cls: type[ConfigType]) -> type[ConfigType]:
        assert issubclass(cls, Config), (
            f"Configured class {cls.__name__} must be a subclass of Config."
        )
        cls = dataclasses.dataclass(kw_only=kw_only, unsafe_hash=unsafe_hash)(cls)
        cls_key = _config_class_key(cls)
        existing = CONFIG_REGISTRY.get(cls_key)
        if existing is not None and existing is not cls:
            raise AssertionError(
                f"Duplicate Config Class {cls_key}, trying to register module "
                f"{cls.__module__} but found already {existing.__module__}"
            )
        CONFIG_REGISTRY[cls_key] = cls
        CONFIG_REGISTRY.setdefault(cls.__name__, cls)
        return cls

    if cls is not None:
        return _configclass(cls)
    else:
        return _configclass


def from_dict(
    value: Any,
    ensure_class: str | type = "",
    max_back_compat: bool = False,
    ignore_unregistered_cls: bool = False,
):
    if isinstance(ensure_class, type):
        ensure_class = ensure_class.__name__
    if isinstance(value, tuple):
        return tuple(
            from_dict(
                x, max_back_compat=max_back_compat, ignore_unregistered_cls=ignore_unregistered_cls
            )
            for x in value
        )
    if isinstance(value, list):
        return [
            from_dict(
                x, max_back_compat=max_back_compat, ignore_unregistered_cls=ignore_unregistered_cls
            )
            for x in value
        ]
    if isinstance(value, dict):
        if "__class" in value or ensure_class:
            cls_name = value.get("__class", ensure_class)
            if ensure_class:
                assert cls_name == ensure_class, (
                    f'Inconsistent config class requirement: "{cls_name}" != "{ensure_class}"'
                )

            cls = CONFIG_REGISTRY.get(cls_name)
            if cls is None:
                if ignore_unregistered_cls:
                    return None
                else:
                    error_msg = make_unregistered_config_error_message(cls_name)
                    raise ValueError(error_msg)

            if max_back_compat:
                signature = inspect.signature(cls.__init__)
                compat_value = {}
                for param_name, param_info in signature.parameters.items():
                    if param_name in value:
                        compat_value[param_name] = value[param_name]
                    else:
                        assert param_name == "self" or param_info.default is not param_info.empty, (
                            f"Required arg `{param_name}` not found in the provided dict"
                        )

                for key in value:
                    if key not in compat_value and key != "__class":
                        logger.warning(
                            f"[Compatibility Warning] arg `{key}` in the provided dict does not appear in Config __init__. Ignore for max back compatibility."
                        )
                init_values = {
                    k: from_dict(
                        v,
                        ensure_class=ensure_class,
                        max_back_compat=max_back_compat,
                        ignore_unregistered_cls=ignore_unregistered_cls,
                    )
                    for k, v in compat_value.items()
                    if k != "__class"
                }
            else:
                init_values = {
                    k: from_dict(v, ignore_unregistered_cls=ignore_unregistered_cls)
                    for k, v in value.items()
                    if k != "__class"
                }

            for k, vtype in cls.get_type_hints().items():
                v = init_values.get(k)
                if v is None:
                    continue
                if is_tuple(vtype):
                    init_values[k] = tuple(v)
                elif is_path(vtype):
                    init_values[k] = Path(v)
                elif is_bytes(vtype):
                    init_values[k] = base64.b64decode(v.encode("utf-8"))
                elif is_datetime(vtype):
                    init_values[k] = datetime.datetime.fromisoformat(v)
                else:
                    coerced = _coerce_enums_from_hint(vtype, v)
                    if coerced is not v:
                        init_values[k] = coerced
            obj = cls(**init_values)
            return obj
        else:
            return {k: from_dict(v, max_back_compat=max_back_compat) for k, v in value.items()}
    else:
        assert not ensure_class, f'Can\'t ensure class for non-dictionary for config "{value}"'
    return value


def to_dict(obj: Any, container: str = "???") -> Any:
    if isinstance(obj, Config):
        return obj.to_dict()
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {
            field.name: to_dict(getattr(obj, field.name), container=f"<field: {field.name}>")
            for field in dataclasses.fields(obj)
            if field.init
        }
    elif isinstance(obj, list):
        return [to_dict(x, "<some list>") for x in obj]
    elif isinstance(obj, tuple):
        return tuple(to_dict(x, "<some tuple>") for x in obj)
    elif isinstance(obj, dict):
        return {k: to_dict(v, f"<key: {k}>") for k, v in obj.items()}
    elif isinstance(obj, (str, float, int, bool, NoneType)):
        return obj
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, enum.Enum):
        return obj.name
    elif isinstance(obj, re.Pattern):
        return obj.pattern
    elif isinstance(obj, bytes):
        return base64.b64encode(obj).decode()
    elif isinstance(obj, datetime.datetime):
        return obj.isoformat()
    elif isinstance(obj, type):
        module = obj.__module__
        name = obj.__qualname__ if hasattr(obj, "__qualname__") else obj.__name__
        return f"{module}.{name}"
    elif hasattr(obj, "model_dump"):
        return to_dict(obj.model_dump(exclude_unset=True), container)
    else:
        print(f"Unknown object type {type(obj)} of {obj} at {container}")
    return obj


def resolve_type_hints(dcls: Any) -> dict[str, type[Any]]:
    if isinstance(dcls, Config):
        resolved_hints = type(dcls).get_type_hints()
    elif isinstance(dcls, type):
        if dataclasses.is_dataclass(dcls):
            resolved_hints = get_type_hints(dcls)
            field_names = [field.name for field in dataclasses.fields(dcls)]
            return {name: resolved_hints[name] for name in field_names}
        else:
            return {}
    else:
        resolved_hints = get_type_hints(type(dcls))
    field_names = [field.name for field in dataclasses.fields(dcls)]
    return {name: resolved_hints[name] for name in field_names}


def replace_cli_subs(
    config: Any, kvs: list[str], presets: dict[str, list[str]] | None = None
) -> tuple[Any, dict[str, list[str]]]:
    kvs = replace_presets(kvs, presets or {})
    new_config = config
    kv_used = {}

    for kv in kvs:
        k, v = parse_kv(kv)
        key_path = tuple(k.split("."))
        used = []
        new_config = replace_recursive(new_config, (), key_path, v, used)
        kv_used[k] = [".".join(x) for x in used]
    return new_config, kv_used


def replace_presets(args: list[str], presets: dict[str, list[str]]) -> list[str]:
    kvs = []
    for arg in args:
        if "=" in arg:
            kvs.append(arg)
        else:
            preset = presets.get(arg)
            if not preset:
                raise ValueError(
                    f"Invalid argument {arg!r}, not a key=value replacement and no matching preset found"
                )
            kvs += preset
    return kvs


def replace_recursive(
    dcls: Any,
    current_path: tuple[str, ...],
    key_path: tuple[str, ...],
    value: str,
    used: list[tuple[str, ...]],
) -> Any:
    new_dcls = dcls
    ty_hints = resolve_type_hints(dcls)
    for dataclass_field in dataclasses.fields(dcls):
        name = dataclass_field.name
        if not hasattr(dcls, name):
            continue
        attr = getattr(dcls, name)
        path = current_path + (name,)
        if isinstance(attr, type):
            continue
        if dataclasses.is_dataclass(attr):
            old_val = getattr(dcls, name)
            new_val = replace_recursive(old_val, path, key_path, value, used)
            if old_val != new_val:
                new_dcls = dataclasses.replace(new_dcls, **{name: new_val})
        elif isinstance(attr, list):
            new_list = []
            for i, item in enumerate(attr):
                item_path = path + (str(i),)
                if item_path[-len(key_path) :] == key_path:
                    new_list.append(value)
                    used.append(path)
                elif dataclasses.is_dataclass(item):
                    new_item = replace_recursive(item, item_path, key_path, value, used)
                    new_list.append(new_item)
                else:
                    new_list.append(item)
            if attr != new_list:
                new_dcls = dataclasses.replace(new_dcls, **{name: new_list})
        elif isinstance(attr, dict):
            new_dict: dict[str, Any] = {}
            for k, item in attr.items():
                item_path = path + (k,)
                if item_path[-len(key_path) :] == key_path:
                    new_dict[k] = value
                    used.append(path)
                elif dataclasses.is_dataclass(item):
                    new_item = replace_recursive(item, item_path, key_path, value, used)
                    new_dict[k] = new_item
                else:
                    new_dict[k] = item
            if attr != new_dict:
                new_dcls = dataclasses.replace(new_dcls, **{name: new_dict})

        if path[-len(key_path) :] == key_path:
            if value == "None" and is_optional(ty_hints[name]):
                new_dcls = dataclasses.replace(dcls, **{name: None})
            else:
                ty = ty_hints[name]
                new_dcls = dataclasses.replace(new_dcls, **{name: cast_type(ty, value)})
            used.append(path)
    return new_dcls


def cast_type(ty: type[Any], val: str):
    if ty is bool:
        if val in ("False", "false"):
            return False
        elif val in ("True", "true"):
            return True
        raise ValueError(f"Expected bool [True, true, False, false], got {val!r}")
    elif is_optional(ty) and val == "None":
        return None
    elif is_union(ty):
        subtys = [subty for subty in get_args(ty) if subty is not type(None)]
        for subty in subtys:
            try:
                return cast_type(subty, val)
            except ValueError:
                if subty is subtys[-1]:
                    raise
        raise TypeError(f"Got {val} for union type {ty}")
    elif is_tuple(ty):
        if not val:
            return ()
        vals = val.split(",")
        tys = get_args(ty)
        if len(vals) != len(tys):
            raise TypeError(f"Tuple {ty} has different number of arguments from given value {vals}")
        return tuple(cast_type(a_ty, v) for a_ty, v in zip(tys, vals))
    elif is_list(ty):
        if not val:
            return []
        vals = val.split(",")
        tys = get_args(ty)
        return [cast_type(tys[0], x) for x in vals]
    elif is_dict(ty):
        kty, vty = get_args(ty)
        result = {}
        for item in val.split(","):
            k, v = item.split(":", 1)
            result[cast_type(kty, k.strip())] = cast_type(vty, v.strip())
        return result
    elif is_path(ty):
        return Path(val)
    elif is_bytes(ty):
        return base64.b64decode(val.encode("utf-8"))
    elif is_datetime(ty):
        return datetime.datetime.fromisoformat(val)
    elif get_origin(ty) is Literal:
        options = get_args(ty)
        subtys = tuple([type(o) for o in options])
        for subty in subtys:
            try:
                cast = cast_type(subty, val)
            except ValueError:
                continue
            if cast in options:
                return cast
        raise TypeError(f"Literal {ty} does not allow for {val!r}")
    elif isinstance(ty, enum.EnumMeta):
        try:
            return ty(val)
        except ValueError:
            return ty[val]
    elif isinstance(ty, TypeAliasType):
        return cast_type(ty.__value__, val)
    elif ty is re.Pattern:
        return re.compile(val)
    else:
        return ty(val)


def is_union(ty) -> bool:
    return get_origin(ty) in (Union, UnionType)


def is_optional(ty) -> bool:
    return is_union(ty) and type(None) in get_args(ty)


def is_list(ty) -> bool:
    return get_origin(ty) is list


def is_tuple(ty) -> bool:
    return get_origin(ty) is tuple


def is_dict(ty) -> bool:
    return get_origin(ty) is dict


def is_path(ty) -> bool:
    if ty is Path:
        return True
    o = get_origin(ty)
    return o is Path or (isinstance(ty, UnionType) and set(ty.__args__) == {Path, type(None)})


def is_bytes(ty) -> bool:
    if ty is bytes:
        return True
    o = get_origin(ty)
    return o is bytes or (isinstance(ty, UnionType) and set(ty.__args__) == {bytes, type(None)})


def is_datetime(ty) -> bool:
    if ty is datetime.datetime:
        return True
    o = get_origin(ty)
    return o is datetime.datetime or (
        isinstance(ty, UnionType) and set(ty.__args__) == {datetime.datetime, type(None)}
    )


def _unwrap_optional(ty: Any) -> Any:
    if is_optional(ty):
        non_none = [a for a in get_args(ty) if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
    return ty


def _coerce_enum_value(enum_cls: enum.EnumMeta, value: Any) -> Any:
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(value)
    except (ValueError, TypeError):
        return enum_cls[value]


def _coerce_enums_from_hint(vtype: Any, value: Any) -> Any:
    inner = _unwrap_optional(vtype)
    if isinstance(inner, enum.EnumMeta):
        return _coerce_enum_value(inner, value)
    if is_list(inner) and isinstance(value, list):
        elem_args = get_args(inner)
        if len(elem_args) == 1 and isinstance(elem_args[0], enum.EnumMeta):
            enum_cls = elem_args[0]
            if value and all(isinstance(x, enum_cls) for x in value):
                return value
            return [_coerce_enum_value(enum_cls, x) for x in value]
    return value


def parse_kv(kv: str) -> tuple[str, str]:
    parts = kv.split("=", 1)
    if len(parts) < 2:
        raise ValueError(f"Invalid config option {kv!r}, expected format 'key=value'")
    return tuple(parts)


def get_module(mod_path: str) -> ModuleType:
    try:
        return importlib.import_module(mod_path)
    except ModuleNotFoundError as e:
        raise type(e)(
            f"No module found {mod_path!r}, "
            f"or {mod_path!r} has another import issue: {traceback.format_exc()}"
        ) from e


def get_configs_from_mod(
    mod: ModuleType,
) -> dict[str, Config] | None:
    if "CONFIGS" not in dir(mod):
        return None
    configs: dict[str, Any | Config] = mod.CONFIGS
    assert all(isinstance(config, Config) for config in configs.values())
    return cast(dict[str, Config], configs)


def get_named_config_from_module(
    name: str,
    base_module_name: str,
    on_failure_log_additional_info: bool = False,
) -> Config:
    parts = name.split(".", 1)
    if len(parts) < 2:
        raise ValueError(f"Invalid config {name!r}, expected format 'module.key'")

    mod_name, ver = parts
    try:
        mod = get_module(f"{base_module_name}.{mod_name}")
    except ModuleNotFoundError as e:
        if on_failure_log_additional_info:
            discovered_configs = get_all_configs_from_module(
                base_module_name, ignore_loading_errors=True
            )
            available_configs = sorted(discovered_configs.configs.keys())
            print(f"Available configs: {available_configs}")
        raise e
    configs = get_configs_from_mod(mod)
    if configs is None:
        raise ValueError(f"Module {mod_name!r} does not have CONFIGS")

    if ver not in configs:
        avail = ", ".join(configs.keys())
        raise ValueError(f"Config module {mod_name!r} has no key {ver!r}. Available keys: {avail}")
    return configs[ver]


class DiscoveredConfigsInModule(NamedTuple):
    configs: dict[str, Config]
    failing_modules: list[str]


def get_all_configs_from_module(
    base_module_name: str,
    ignore_loading_errors: bool = False,
) -> DiscoveredConfigsInModule:
    mod = get_module(base_module_name)
    if not hasattr(mod, "__path__"):
        raise ValueError(f"Module '{base_module_name}' is not a package containing configs.")

    configs: dict[str, Config] = {}
    failing_modules: list[str] = []
    for _, name, _is_pkg in pkgutil.iter_modules(mod.__path__):
        try:
            child_mod = get_module(f"{base_module_name}.{name}")
        except Exception:
            failing_modules.append(f"{base_module_name}.{name}")
            logger.warning(
                f"Failed to load eval config module {base_module_name}.{name}",
                exc_info=True,
            )
            if ignore_loading_errors:
                continue
            raise
        child_mod_configs = get_configs_from_mod(child_mod)
        if child_mod_configs is None:
            continue
        child_mod_configs = {
            f"{name}.{config_name}": value for config_name, value in child_mod_configs.items()
        }
        configs.update(child_mod_configs)
    return DiscoveredConfigsInModule(configs=configs, failing_modules=failing_modules)
