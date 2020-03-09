"""Provide strategies for given endpoint(s) definition."""
import asyncio
import re
from base64 import b64encode
from functools import partial
from typing import Any, Callable, Dict, Mapping, Optional, Union, Sequence
from urllib.parse import quote_plus

import hypothesis
import hypothesis.strategies as st
from hypothesis.errors import Unsatisfiable
from hypothesis_jsonschema import from_schema
from hypothesis_jsonschema_unfit import not_from_schema
from pytest import skip

from . import utils
from ._compat import handle_warnings
from .constants import InputType
from .exceptions import InvalidSchema
from .hooks import get_hook
from .models import Case, Endpoint
from .types import Hook

PARAMETERS = frozenset(("path_parameters", "headers", "cookies", "query", "body", "form_data"))


def create_test(
    endpoint: Endpoint,
    test: Callable,
    settings: Optional[hypothesis.settings] = None,
    seed: Optional[int] = None,
    input_type: InputType = InputType.valid,
) -> Callable:
    """Create a Hypothesis test."""
    strategy = endpoint.as_strategy(input_type)
    wrapped_test = hypothesis.given(case=strategy)(test)
    if seed is not None:
        wrapped_test = hypothesis.seed(seed)(wrapped_test)
    original_test = get_original_test(test)
    #
    # def skip_unsatisfiable_invalid_test(*args, **kwargs):
    #     try:
    #         original_test(*args, **kwargs)
    #     except Unsatisfiable:
    #         if input_type is InputType.invalid:
    #             skip("No invalid could be generated for this case.")
    #         raise

    if asyncio.iscoroutinefunction(original_test):
        wrapped_test.hypothesis.inner_test = make_async_test(original_test)  # type: ignore
    if settings is not None:
        wrapped_test = settings(wrapped_test)
    return add_examples(wrapped_test, endpoint)


def skip_unsatisfiable_invalid_test():
    try:
        pass
    except Unsatisfiable:
        skip("No invalid could be generated for this case.")


def make_test_or_exception(
    endpoint: Endpoint,
    func: Callable,
    settings: Optional[hypothesis.settings] = None,
    seed: Optional[int] = None,
    input_type: InputType = InputType.valid,
) -> Union[Callable, InvalidSchema]:
    try:
        return create_test(endpoint, func, settings, seed=seed, input_type=input_type)
    except InvalidSchema as exc:
        return exc


def get_original_test(test: Callable) -> Callable:
    """Get the original test function even if it is wrapped by `hypothesis.settings` decorator.

    Applies only to Hypothesis pre 4.42.4 versions.
    """
    # `settings` decorator is applied
    if getattr(test, "_hypothesis_internal_settings_applied", False) and hypothesis.__version_info__ < (4, 42, 4):
        # This behavior was changed due to a bug - https://github.com/HypothesisWorks/hypothesis/issues/2160
        # And since Hypothesis 4.42.4 is no longer required
        return test._hypothesis_internal_test_function_without_warning  # type: ignore
    return test


def make_async_test(test: Callable) -> Callable:
    def async_run(*args: Any, **kwargs: Any) -> None:
        loop = asyncio.get_event_loop()
        coro = test(*args, **kwargs)
        future = asyncio.ensure_future(coro, loop=loop)
        loop.run_until_complete(future)

    return async_run


def get_example(endpoint: Endpoint) -> Optional[Case]:
    static_parameters = {}
    for name in PARAMETERS:
        parameter = getattr(endpoint, name)
        if parameter is not None and "example" in parameter:
            static_parameters[name] = parameter["example"]
    if static_parameters:
        with handle_warnings():
            strategies = {
                other: from_schema(getattr(endpoint, other))
                for other in PARAMETERS - set(static_parameters)
                if getattr(endpoint, other) is not None
            }
            return _get_case_strategy(endpoint, static_parameters, strategies).example()
    return None


def add_examples(test: Callable, endpoint: Endpoint) -> Callable:
    """Add examples to the Hypothesis test, if they are specified in the schema."""
    example = get_example(endpoint)
    if example:
        test = hypothesis.example(case=example)(test)
    return test


def is_valid_header(headers: Any) -> bool:
    """Verify if the generated headers are valid."""
    # Data could be of any type
    if not isinstance(headers, dict):
        return False
    for name, value in headers.items():
        # only string values can be sent by clients to the app
        if not isinstance(name, str) or not isinstance(value, str):
            return False
        if not name or not value:
            return False
        if not utils.is_ascii_encodable(name) or not utils.is_ascii_encodable(value):
            return False
        if utils.has_invalid_characters(name, value):
            return False
    return True


def is_surrogate(item: Any) -> bool:
    return isinstance(item, str) and bool(re.search(r"[\ud800-\udfff]", item))


def is_valid_query(query: Any) -> bool:
    """Surrogates are not allowed in a query string.

    `requests` and `werkzeug` will fail to send it to the application.
    """
    if not isinstance(query, dict):
        return False
    for name, value in query.items():
        if is_surrogate(name) or is_surrogate(value):
            return False
    return True


def is_valid_cookie(cookies: Any) -> bool:
    if not isinstance(cookies, dict):
        return False
    for name, value in cookies.items():
        if not isinstance(name, str) or not isinstance(value, str):
            return False
        if not utils.is_ascii_encodable(name) or not utils.is_ascii_encodable(value):
            return False
    return True


def is_valid_form_data(form_data: Any) -> bool:
    if isinstance(form_data, Mapping):
        return all(isinstance(key, str) and isinstance(value, (bytes, str, int)) for key, value in form_data.items())

    def is_valid_item(item: Any) -> bool:
        return isinstance(item, Sequence) and len(item) == 2 and isinstance(item[0], str) and isinstance(item[1], (bytes, str, int))

    return isinstance(form_data, (tuple, list, set)) and all(is_valid_item(item) for item in form_data)


def get_schema_strategy(value: Any, input_type: InputType) -> st.SearchStrategy:
    strategy_func = {InputType.valid: from_schema, InputType.invalid: not_from_schema}[input_type]
    return strategy_func(value)


def get_case_strategy(endpoint: Endpoint, input_type: InputType = InputType.valid) -> st.SearchStrategy:
    """Create a strategy for a complete test case.

    Path & endpoint are static, the others are JSON schemas.
    """
    strategies = {}
    static_kwargs: Dict[str, Any] = {"endpoint": endpoint, "input_type": input_type}
    try:
        for parameter in PARAMETERS:
            value = getattr(endpoint, parameter)
            if value is not None:
                strategy = get_schema_strategy(value, input_type)
                if parameter == "path_parameters":
                    strategies[parameter] = strategy.filter(filter_path_parameters).map(quote_all)  # type: ignore
                elif parameter in ("headers", "cookies"):
                    strategies[parameter] = strategy.filter(is_valid_header)  # type: ignore
                elif parameter == "query":
                    strategies[parameter] = strategy.filter(is_valid_query)  # type: ignore
                elif parameter == "cookies":
                    strategies[parameter] = strategy.filter(is_valid_cookie)  # type: ignore
                elif parameter == "form_data":
                    strategies[parameter] = strategy.filter(is_valid_form_data)  # type: ignore
                else:
                    strategies[parameter] = strategy  # type: ignore
            else:
                static_kwargs[parameter] = None
        return _get_case_strategy(endpoint, static_kwargs, strategies)
    except AssertionError:
        raise InvalidSchema("Invalid schema for this endpoint")


def filter_path_parameters(parameters: Dict[str, Any]) -> bool:
    """Single "." chars are excluded from path by urllib3.

    In this case one variable in the path template will be empty, which will lead to 404 in most of the cases.
    Because of it this case doesn't bring much value and might lead to false positives results of Schemathesis runs.
    """
    for value in parameters.values():
        # Disallow composed values
        if isinstance(value, (list, dict)):
            return False
    return not any(value == "." for value in parameters.values())


def quote_all(parameters: Dict[str, Any]) -> Dict[str, Any]:
    return {key: quote_plus(value) if isinstance(value, str) else value for key, value in parameters.items()}


def _get_case_strategy(
    endpoint: Endpoint, extra_static_parameters: Dict[str, Any], strategies: Dict[str, st.SearchStrategy]
) -> st.SearchStrategy:
    static_parameters: Dict[str, Any] = {"endpoint": endpoint, **extra_static_parameters}
    if endpoint.schema.validate_schema and endpoint.method == "GET":
        if endpoint.body is not None:
            raise InvalidSchema("Body parameters are defined for GET request.")
        static_parameters["body"] = None
        strategies.pop("body", None)
    _apply_hooks(strategies, get_hook)
    _apply_hooks(strategies, endpoint.schema.get_hook)
    return st.builds(partial(Case, **static_parameters), **strategies)


def _apply_hooks(strategies: Dict[str, st.SearchStrategy], getter: Callable[[str], Optional[Hook]]) -> None:
    for key, strategy in strategies.items():
        hook = getter(key)
        if hook is not None:
            strategies[key] = hook(strategy)


def register_string_format(name: str, strategy: st.SearchStrategy) -> None:
    if not isinstance(name, str):
        raise TypeError(f"name must be of type {str}, not {type(name)}")
    if not isinstance(strategy, st.SearchStrategy):
        raise TypeError(f"strategy must be of type {st.SearchStrategy}, not {type(strategy)}")
    from hypothesis_jsonschema._from_schema import STRING_FORMATS  # pylint: disable=import-outside-toplevel

    STRING_FORMATS[name] = strategy


def init_default_strategies() -> None:
    register_string_format("binary", st.binary())
    register_string_format("byte", st.binary().map(lambda x: b64encode(x).decode()))
