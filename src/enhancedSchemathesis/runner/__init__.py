from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Tuple, Union
from urllib.parse import urlparse

import hypothesis.errors

from .. import loaders
from ..checks import DEFAULT_CHECKS
from ..models import CheckFunction
from ..schemas import BaseSchema
from ..types import Filter, NotSet, RawAuth
from ..utils import dict_not_none_values, dict_true_values, file_exists, get_base_url, get_requests_auth, import_app
from . import events
from .impl import BaseRunner, SingleThreadRunner, SingleThreadWSGIRunner, ThreadPoolRunner, ThreadPoolWSGIRunner
from collections import OrderedDict

def prepare(  # pylint: disable=too-many-arguments
    schema_uri: Union[str, Dict[str, Any], OrderedDict],
    *,
    # Runtime behavior
    checks: Iterable[CheckFunction] = DEFAULT_CHECKS,
    workers_num: int = 1,
    seed: Optional[int] = None,
    exit_first: bool = False,
    # Schema loading
    loader: Callable = loaders.from_uri,
    base_url: Optional[str] = None,
    auth: Optional[Tuple[str, str]] = None,
    auth_type: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
    request_timeout: Optional[int] = None,
    endpoint: Optional[Filter] = None,
    method: Optional[Filter] = None,
    tag: Optional[Filter] = None,
    app: Optional[str] = None,
    validate_schema: bool = True,
    execute_in_order: Optional[dict] = None,
    # Hypothesis-specific configuration
    hypothesis_deadline: Optional[Union[int, NotSet]] = None,
    hypothesis_derandomize: Optional[bool] = None,
    hypothesis_max_examples: Optional[int] = None,
    hypothesis_phases: Optional[List[hypothesis.Phase]] = None,
    hypothesis_report_multiple_bugs: Optional[bool] = None,
    hypothesis_suppress_health_check: Optional[List[hypothesis.HealthCheck]] = None,
    hypothesis_verbosity: Optional[hypothesis.Verbosity] = None,
) -> Generator[events.ExecutionEvent, None, None]:
    """Prepare a generator that will run test cases against the given API definition."""
    # pylint: disable=too-many-locals
    validate_loader(loader, schema_uri)
    
    if auth is None:
        # Auth type doesn't matter if auth is not passed
        auth_type = None  # type: ignore
    hypothesis_options = prepare_hypothesis_options(
        deadline=hypothesis_deadline,
        derandomize=hypothesis_derandomize,
        max_examples=hypothesis_max_examples,
        phases=hypothesis_phases,
        report_multiple_bugs=hypothesis_report_multiple_bugs,
        suppress_health_check=hypothesis_suppress_health_check,
        verbosity=hypothesis_verbosity,
    )
    return execute_from_schema(
        schema_uri=schema_uri,
        loader=loader,
        base_url=base_url,
        endpoint=endpoint,
        method=method,
        tag=tag,
        app=app,
        validate_schema=validate_schema,
        checks=checks,
        hypothesis_options=hypothesis_options,
        seed=seed,
        workers_num=workers_num,
        exit_first=exit_first,
        auth=auth,
        auth_type=auth_type,
        headers=headers,
        request_timeout=request_timeout,
        execute_in_order=execute_in_order,
    )


def validate_loader(loader: Callable, schema_uri: Union[str, Dict[str, Any]]) -> None:
    """Sanity checking for input schema & loader."""
    if loader not in (
        loaders.from_uri,
        loaders.from_aiohttp,
        loaders.from_dict,
        loaders.from_file,
        loaders.from_path,
        loaders.from_wsgi,
    ):
        # Custom loaders are not checked
        return
    if isinstance(schema_uri, dict):
        if loader is not loaders.from_dict:
            raise ValueError("Dictionary as a schema is allowed only with `from_dict` loader")
    elif loader is loaders.from_dict:
        raise ValueError("Schema should be a dictionary for `from_dict` loader")


def execute_from_schema(
    *,
    schema_uri: Union[str, Dict[str, Any],OrderedDict],
    loader: Callable = loaders.from_uri,
    base_url: Optional[str] = None,
    endpoint: Optional[Filter] = None,
    method: Optional[Filter] = None,
    tag: Optional[Filter] = None,
    app: Optional[str] = None,
    validate_schema: bool = True,
    checks: Iterable[CheckFunction],
    workers_num: int = 1,
    hypothesis_options: Dict[str, Any],
    auth: Optional[RawAuth] = None,
    auth_type: Optional[str] = None,
    headers: Optional[Dict[str, Any]] = None,
    request_timeout: Optional[int] = None,
    execute_in_order: Optional[dict] = None,
    seed: Optional[int] = None,
    exit_first: bool = False,
) -> Generator[events.ExecutionEvent, None, None]:
    """Execute tests for the given schema.

    Provides the main testing loop and preparation step.
    """
    # pylint: disable=too-many-locals
    try:
        if app is not None:
            app = import_app(app)
        schema = load_schema(
            schema_uri,
            base_url=base_url,
            loader=loader,
            app=app,
            validate_schema=validate_schema,
            auth=auth,
            auth_type=auth_type,
            headers=headers,
            endpoint=endpoint,
            method=method,
            tag=tag,
            execute_in_order=execute_in_order,
        )
        runner: BaseRunner
        if workers_num > 1:
            if schema.app:
                runner = ThreadPoolWSGIRunner(
                    schema=schema,
                    checks=checks,
                    hypothesis_settings=hypothesis_options,
                    auth=auth,
                    auth_type=auth_type,
                    headers=headers,
                    seed=seed,
                    workers_num=workers_num,
                    exit_first=exit_first,
                )
            else:
                runner = ThreadPoolRunner(
                    schema=schema,
                    checks=checks,
                    hypothesis_settings=hypothesis_options,
                    auth=auth,
                    auth_type=auth_type,
                    headers=headers,
                    seed=seed,
                    request_timeout=request_timeout,
                    exit_first=exit_first,
                )
        else:
            if schema.app:
                runner = SingleThreadWSGIRunner(
                    schema=schema,
                    checks=checks,
                    hypothesis_settings=hypothesis_options,
                    auth=auth,
                    auth_type=auth_type,
                    headers=headers,
                    seed=seed,
                    exit_first=exit_first,
                    execute_in_order=execute_in_order,
                )
            else:  
                runner = SingleThreadRunner(
                    schema=schema,
                    checks=checks,
                    hypothesis_settings=hypothesis_options,
                    auth=auth,
                    auth_type=auth_type,
                    headers=headers,
                    seed=seed,
                    request_timeout=request_timeout,
                    exit_first=exit_first,
                    execute_in_order=execute_in_order,
                )               
        yield from runner.execute()
    except Exception as exc:
        yield events.InternalError.from_exc(exc)


def load_schema(
    schema_uri: Union[str, Dict[str, Any],OrderedDict],
    *,
    base_url: Optional[str] = None,
    loader: Callable = loaders.from_uri,
    app: Any = None,
    validate_schema: bool = True,
    # Network request parameters
    auth: Optional[Tuple[str, str]] = None,
    auth_type: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
    # Schema filters
    endpoint: Optional[Filter] = None,
    method: Optional[Filter] = None,
    tag: Optional[Filter] = None,
    execute_in_order: Optional[dict] = None,
) -> BaseSchema:
    """Load schema via specified loader and parameters."""
    loader_options = dict_true_values(base_url=base_url, endpoint=endpoint, method=method, tag=tag, app=app, execute_in_order=execute_in_order)

    if not isinstance(schema_uri, dict):
        if file_exists(schema_uri):
            loader = loaders.from_path
        elif app is not None and not urlparse(schema_uri).netloc:
            # If `schema` is not an existing filesystem path or an URL then it is considered as an endpoint with
            # the given app
            loader = loaders.get_loader_for_app(app)
        else:
            loader_options.update(dict_true_values(headers=headers, auth=auth, auth_type=auth_type))

    if "base_url" not in loader_options and not isinstance(schema_uri, dict):
        loader_options["base_url"] = get_base_url(schema_uri)
    if loader is loaders.from_uri and loader_options.get("auth"):
        loader_options["auth"] = get_requests_auth(loader_options["auth"], loader_options.pop("auth_type", None))
    return loader(schema_uri, validate_schema=validate_schema, **loader_options)


def prepare_hypothesis_options(  # pylint: disable=too-many-arguments
    deadline: Optional[Union[int, NotSet]] = None,
    derandomize: Optional[bool] = None,
    max_examples: Optional[int] = None,
    phases: Optional[List[hypothesis.Phase]] = None,
    report_multiple_bugs: Optional[bool] = None,
    suppress_health_check: Optional[List[hypothesis.HealthCheck]] = None,
    verbosity: Optional[hypothesis.Verbosity] = None,
) -> Dict[str, Any]:
    options = dict_not_none_values(
        derandomize=derandomize,
        max_examples=max_examples,
        phases=phases,
        report_multiple_bugs=report_multiple_bugs,
        suppress_health_check=suppress_health_check,
        verbosity=verbosity,
    )
    # `deadline` is special, since Hypothesis allows to pass `None`
    if deadline is not None:
        if isinstance(deadline, NotSet):
            options["deadline"] = None
        else:
            options["deadline"] = deadline
    return options
