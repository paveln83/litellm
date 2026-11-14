from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.types.router import LiteLLM_Params
from litellm.utils import with_named_credential

_ROUTER_OWNED_LITELLM_PARAMS: Final = frozenset({"max_retries"})
_HEADER_LITELLM_PARAMS: Final = frozenset({"extra_headers", "headers"})


def deployment_sets_litellm_param(deployment_params: Mapping[str, object], key: str) -> bool:
    value: Final = deployment_params.get(key)
    if value is None:
        return False
    field: Final = LiteLLM_Params.model_fields.get(key)
    return field is None or value != field.default  # pyright: ignore[reportAny]  # pydantic types FieldInfo.default as Any


def _headers_under_deployment(
    router_headers: Mapping[str, object], deployment_headers: Mapping[str, object]
) -> dict[str, object]:  # mutable-ok: providers merge more headers into the dict they are sent
    deployment_names: Final = frozenset(name.lower() for name in deployment_headers)
    kept: Final = {name: value for name, value in router_headers.items() if name.lower() not in deployment_names}
    return {**kept, **deployment_headers}


def _router_default_under_deployment(key: str, value: object, configured: Mapping[str, object]) -> object:
    if key in _ROUTER_OWNED_LITELLM_PARAMS or not deployment_sets_litellm_param(configured, key):
        return value
    router_headers: Final = as_str_mapping(value) if key in _HEADER_LITELLM_PARAMS else None
    deployment_headers: Final = as_str_mapping(configured.get(key))
    if router_headers is None or deployment_headers is None:
        return None
    return _headers_under_deployment(router_headers, deployment_headers)


def _router_defaults_left_unset(
    router_defaults: Mapping[str, object], configured: Mapping[str, object]
) -> Mapping[str, object]:
    applied: Final = (
        (key, _router_default_under_deployment(key, value, configured)) for key, value in router_defaults.items()
    )
    return MappingProxyType({key: value for key, value in applied if value is not None})


def router_defaults_for_deployment(
    router_defaults: Mapping[str, object], deployment_params: Mapping[str, object]
) -> Mapping[str, object]:
    return _router_defaults_left_unset(router_defaults, with_named_credential(deployment_params))


def params_sent_to_deployment(
    deployment_params: Mapping[str, object], router_defaults: Mapping[str, object], request: Mapping[str, object]
) -> Mapping[str, object]:
    configured: Final = with_named_credential(deployment_params)
    return MappingProxyType({**configured, **_router_defaults_left_unset(router_defaults, configured), **request})
