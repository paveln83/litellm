"""
One router attempt, as the host runs it: the request for the deployment the router core picked.

The Rust router core picks a deployment and hands the attempt to its host; ``PythonHost`` builds
the keyword arguments for that one deployment here and calls ``litellm.<op>`` with them. Nothing
here selects, retries, falls back or records cooldowns: the core does all of that.
"""

import re
import uuid
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from litellm.litellm_core_utils.batch_utils import get_router_metadata_variable_name
from litellm.litellm_core_utils.get_llm_provider_logic import inferred_provider
from litellm.litellm_core_utils.pattern_match_deployments import PatternMatchRouter

# The positional parameters of each endpoint, in litellm's order: a call that passes them
# positionally is read back into keywords before it is routed.
POSITIONAL: Final = MappingProxyType(
    {
        "acompletion": ("model", "messages"),
        "completion": ("model", "messages"),
        "atext_completion": ("model", "prompt"),
        "text_completion": ("model", "prompt"),
        "aembedding": ("model", "input"),
        "embedding": ("model", "input"),
        "aimage_generation": ("prompt", "model"),
        "image_generation": ("prompt", "model"),
        "atranscription": ("model", "file"),
        "aspeech": ("model", "input", "voice"),
        "arerank": ("model", "query", "documents"),
        "amoderation": ("input", "model"),
        "aresponses": ("input", "model"),
        "aanthropic_messages": (),
    }
)


# The arguments an endpoint takes without a default, which a call that leaves one out raises
# ``TypeError`` for before it routes.
REQUIRED: Final = MappingProxyType(
    {
        "atext_completion": ("model", "prompt"),
        "text_completion": ("model", "prompt"),
    }
)


def request_from_call(op: str, args: tuple[object, ...], kwargs: Mapping[str, object]) -> Mapping[str, object]:
    """The call's arguments as one keyword request. Raises ``TypeError`` like the endpoint would."""
    names: Final = POSITIONAL.get(op, ())
    if len(args) > len(names):
        raise TypeError(f"{op}() takes {len(names)} positional arguments but {len(args)} were given")
    request: Final = dict(kwargs)
    for name, value in zip(names, args, strict=False):
        if name in request:
            raise TypeError(f"{op}() got multiple values for argument {name!r}")
        request[name] = value
    missing: Final = [name for name in REQUIRED.get(op, ()) if name not in request]
    if missing:
        listed: Final = " and ".join(repr(name) for name in missing)
        raise TypeError(
            f"{op}() missing {len(missing)} required positional argument{'s' if len(missing) > 1 else ''}: {listed}"
        )
    return request


# Endpoints the Python router serves through ``_ageneric_api_call_with_fallbacks``, which writes
# the routing metadata under ``litellm_metadata``.
_GENERIC_API_OPS: Final = frozenset({"aanthropic_messages", "aresponses"})


def metadata_field(op: str) -> str:
    """``metadata`` or ``litellm_metadata``: where the router writes what it routed the call as."""
    return "litellm_metadata" if op in _GENERIC_API_OPS else get_router_metadata_variable_name(function_name=op)


def prepare_call(op: str, request: Mapping[str, object], model_group_alias: str | None) -> Mapping[str, object]:
    """What the router adds to a request once, before its first attempt
    (``_update_kwargs_before_fallbacks``): a trace id and the model group it was asked for."""
    field: Final = metadata_field(op)
    metadata: Final = dict(_as_mapping(request.get(field)))
    metadata.update({"model_group": request.get("model"), "model_group_alias": model_group_alias})
    return {**request, "litellm_trace_id": request.get("litellm_trace_id") or str(uuid.uuid4()), field: metadata}


def _wildcard_match(pattern: object, requested: str) -> "re.Match[str] | None":
    if not isinstance(pattern, str) or "*" not in pattern:
        return None
    regex: Final = PatternMatchRouter().pattern_to_regex(pattern)
    return re.match(regex, requested) or re.match(regex, f"{inferred_provider(requested)}/{requested}")


def for_requested_model(deployment: Mapping[str, object], requested: str | None) -> Mapping[str, object]:
    """A wildcard deployment as the router serves ``requested`` through it: its ``model`` with
    the wildcards filled in from the requested name (``PatternMatchRouter.route``), by the team's
    public pattern (``_core_team_pattern_routers``) or its own. Any other deployment comes back
    unchanged."""
    params: Final = _as_mapping(deployment.get("litellm_params"))
    model: Final = params.get("model")
    if requested is None or not isinstance(model, str):
        return deployment
    public_name: Final = _as_mapping(deployment.get("model_info")).get("team_public_model_name")
    match: Final = _wildcard_match(public_name, requested) or _wildcard_match(deployment.get("model_name"), requested)
    if match is None:
        return deployment
    return {
        **deployment,
        "litellm_params": {**params, "model": PatternMatchRouter.set_deployment_model_name(match, model)},
    }


def stamp_attempt_error(error: BaseException, deployment: Mapping[str, object]) -> None:
    """What the router stamps on a failed attempt's error: the deployment's own ``num_retries``
    (``_set_deployment_num_retries_on_exception``) and its id (``failed_deployment_id``)."""
    params: Final = _as_mapping(deployment.get("litellm_params"))
    if getattr(error, "num_retries", None) is None and params.get("num_retries") is not None:
        try:
            error.num_retries = int(str(params["num_retries"]))  # pyright: ignore[reportAttributeAccessIssue]  # the router stamps the same attribute  # rebind-ok: the router stamps the error in place
        except ValueError:
            pass
    deployment_id: Final = _as_mapping(deployment.get("model_info")).get("id")
    if deployment_id and not getattr(error, "failed_deployment_id", None):
        try:
            error.failed_deployment_id = deployment_id  # pyright: ignore[reportAttributeAccessIssue]  # the router stamps the same attribute  # rebind-ok: see above
        except AttributeError:
            pass


def _as_mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        entries: Final[Mapping[object, object]] = value  # pyright: ignore[reportUnknownVariableType]  # a request's untyped mapping
        return {str(key): item for key, item in entries.items()}
    return MappingProxyType({})
