from collections.abc import Mapping
from typing import Final

import pytest

from litellm.router_utils.default_litellm_params import params_sent_to_deployment

_ROUTER_DEFAULTS: Final = {"vertex_location": "us-east5", "use_in_pass_through": True, "timeout": None}


@pytest.mark.parametrize(
    ("deployment", "request_params", "sent"),
    [
        ({"vertex_location": "eu"}, {}, {"vertex_location": "eu", "use_in_pass_through": True}),
        ({}, {}, {"vertex_location": "us-east5", "use_in_pass_through": True}),
        ({"vertex_location": None}, {}, {"vertex_location": "us-east5", "use_in_pass_through": True}),
        ({"vertex_location": "eu"}, {"vertex_location": "asia-northeast1"}, {"vertex_location": "asia-northeast1"}),
        ({"use_in_pass_through": False}, {}, {"vertex_location": "us-east5", "use_in_pass_through": True}),
    ],
    ids=[
        "deployment-over-router-default",
        "router-default-fills-unset",
        "router-default-fills-none",
        "request-over-both",
        "deployment-field-default-counts-as-unset",
    ],
)
def test_params_sent_to_deployment_prefer_the_request_then_the_deployment_then_the_router_default(
    deployment: Mapping[str, object], request_params: Mapping[str, object], sent: Mapping[str, object]
) -> None:
    merged: Final = params_sent_to_deployment(deployment, _ROUTER_DEFAULTS, request_params)

    assert {key: merged[key] for key in sent} == sent
    assert "timeout" not in merged


@pytest.mark.parametrize("key", ["headers", "extra_headers"])
def test_router_default_headers_merge_under_the_deployment_headers_by_case_insensitive_name(key: str) -> None:
    router_defaults: Final = {key: {"Helicone-Auth": "Bearer router", "X-Team": "search"}}
    deployment: Final = {key: {"helicone-auth": "Bearer deployment", "X-Region": "eu"}}

    sent: Final = params_sent_to_deployment(deployment, router_defaults, {})

    assert sent[key] == {"X-Team": "search", "helicone-auth": "Bearer deployment", "X-Region": "eu"}


@pytest.mark.parametrize(("router_key", "deployment_key"), [("extra_headers", "headers"), ("headers", "extra_headers")])
def test_a_deployment_header_beats_a_router_default_header_of_the_same_name_under_the_other_key(
    router_key: str, deployment_key: str
) -> None:
    router_defaults: Final = {router_key: {"X-Owner": "router", "X-Team": "search"}}
    deployment: Final = {deployment_key: {"x-owner": "deployment"}}

    sent: Final = params_sent_to_deployment(deployment, router_defaults, {})

    assert sent[router_key] == {"X-Team": "search"}
    assert sent[deployment_key] == {"x-owner": "deployment"}


def test_request_headers_replace_the_merged_router_and_deployment_headers() -> None:
    sent: Final = params_sent_to_deployment(
        {"headers": {"X-Region": "eu"}}, {"headers": {"X-Team": "search"}}, {"headers": {"X-Trace": "1"}}
    )

    assert sent["headers"] == {"X-Trace": "1"}
