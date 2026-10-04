"""Thin wrapper around an AppWorld task so the agents don't depend on AppWorld directly.

The tests swap this for a fake environment with the same methods.

Windows note: AppWorld stops long-running code with signal.SIGALRM, which only exists on
Linux/macOS. On Windows we switch that off (timeout_seconds=None) and use our own timer
instead: a background thread interrupts the main thread if one execution runs too long
(e.g. a model wrote an infinite loop), and the step is reported as timed out.
"""

from __future__ import annotations

import _thread
import ast
import builtins
import json
import os
import re
import threading
import types
from typing import Any

EXECUTION_TIMEOUT_S = 100  # same as AppWorld's own default
IS_WINDOWS = os.name == "nt"


def appworld_kwargs() -> dict[str, Any]:
    """Extra AppWorld(...) arguments needed on this OS."""
    return {"timeout_seconds": None} if IS_WINDOWS else {}


class _Watchdog:
    """Interrupts the main thread after `seconds` unless cancelled first."""

    def __init__(self, seconds: float) -> None:
        self.fired = False
        self._lock = threading.Lock()
        self._done = False
        self._timer = threading.Timer(seconds, self._fire)
        self._timer.daemon = True

    def _fire(self) -> None:
        with self._lock:
            if self._done:
                return
            self.fired = True
        _thread.interrupt_main()

    def __enter__(self) -> "_Watchdog":
        self._timer.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        with self._lock:
            self._done = True
        self._timer.cancel()


def run_with_timeout(fn, *args: Any, seconds: float = EXECUTION_TIMEOUT_S) -> tuple[bool, Any]:
    """Run fn(*args); returns (timed_out, result). Only needed where SIGALRM is missing."""
    try:
        with _Watchdog(seconds) as dog:
            result = fn(*args)
    except KeyboardInterrupt:
        if dog.fired:
            return True, None
        raise
    if dog.fired:
        # The interrupt may have been caught inside the executed cell (IPython reports it as
        # an error); either way the step ran past the limit.
        return True, result
    return False, result


TIMEOUT_MESSAGE = f"Execution failed. Traceback:\nExecution timed out after {EXECUTION_TIMEOUT_S} seconds."

PROTECTED_NAMES = ("apis", "requester")
RESTORED_NOTE = (
    "\n[Note: this step reassigned `{name}`, which must always refer to the app APIs. It has been "
    "restored. Store results in a different variable name.]"
)


def restore_protected(namespace: dict[str, Any], originals: dict[str, Any]) -> str:
    """Put back objects the model must not overwrite (e.g. `apis = apis.api_docs...`).

    Returns a note to append to the step's output, or "" if nothing was changed."""
    notes = []
    for name, obj in originals.items():
        if obj is not None and namespace.get(name) is not obj:
            namespace[name] = obj
            notes.append(RESTORED_NOTE.format(name=name))
    return "".join(notes)


# --------------------------------------------------------------------------- AppWorld context hooks
# These implement the two optional hooks in score/context.py for AppWorld.

_APP_ARG = re.compile(r"""app_name\s*=\s*['"](\w+)['"]""")


def _parse_data(output: str) -> Any:
    """Parse a printed JSON / Python literal, optionally after a 'label:' prefix."""
    text = output.strip()
    candidates = [text]
    if ":" in text.split("\n", 1)[0]:
        candidates.append(text.split(":", 1)[1].strip())
    for c in candidates:
        for parse in (json.loads, ast.literal_eval):
            try:
                return parse(c)
            except Exception:
                continue
    return None


def _schema_shape(x: Any) -> str:
    if isinstance(x, dict):
        return "{" + ", ".join(list(x)[:12]) + ("…" if len(x) > 12 else "") + "}"
    if isinstance(x, list) and x:
        return "[" + _schema_shape(x[0]) + ", …]"
    return type(x).__name__ if x is not None else "None"


def compact_api_doc(doc: dict[str, Any]) -> tuple[str, str]:
    """One line per API: app.api(params) — description. Returns {shape}"""
    name = f"{doc.get('app_name')}.{doc.get('api_name')}"
    params = []
    for p in doc.get("parameters") or []:
        piece = f"{p.get('name')}: {p.get('type', '')}"
        if not p.get("required"):
            piece += f" = {p.get('default')!r}"
        if p.get("constraints"):
            piece += " [" + "; ".join(map(str, p["constraints"])) + "]"
        params.append(piece)
    success = (doc.get("response_schemas") or {}).get("success")
    desc = (doc.get("description") or "").strip()
    return f"doc:{name}", f"{name}({', '.join(params)}) — {desc} Returns {_schema_shape(success)}"


def appworld_reference_from_step(code: str, output: str) -> list[tuple[str, str]]:
    if output.lstrip().startswith("Execution failed") or "api_docs" not in code:
        return []
    data = _parse_data(output)
    if data is None:
        return []
    out: list[tuple[str, str]] = []
    docs = data if isinstance(data, list) else [data]
    if "show_api_doc" in code:
        for d in docs:
            if isinstance(d, dict) and d.get("api_name") and d.get("app_name"):
                out.append(compact_api_doc(d))
    elif "show_api_descriptions" in code and isinstance(data, list):
        m = _APP_ARG.search(code)
        names = [d.get("name") if isinstance(d, dict) else d for d in data]
        names = [n for n in names if isinstance(n, str)]
        if m and names:
            out.append((f"apis:{m.group(1)}", f"{m.group(1)} APIs: {', '.join(names)}"))
    elif "show_app_descriptions" in code and isinstance(data, list):
        names = [d.get("name") if isinstance(d, dict) else d for d in data]
        names = [n for n in names if isinstance(n, str)]
        if names:
            out.append(("apps", f"Apps: {', '.join(names)}"))
    return out


class ApiCatalog:
    """The task's own API documentation (AppWorld ships it with every task): app -> api -> doc."""

    def __init__(self, docs: Any) -> None:
        self.docs: dict[str, dict[str, dict[str, Any]]] = {
            str(app): {str(api): dict(doc) for api, doc in apis.items()} for app, apis in dict(docs).items()
        }

    @property
    def apps(self) -> list[str]:
        return list(self.docs)

    def apis(self, app: str) -> list[str]:
        return list(self.docs.get(app, {}))

    def doc(self, app: str, api: str) -> dict[str, Any] | None:
        d = self.docs.get(app, {}).get(api)
        if d is None:
            return None
        d = dict(d)
        d.setdefault("app_name", app)
        d.setdefault("api_name", api)
        return d

    def signature(self, app: str, api: str) -> str:
        d = self.doc(app, api)
        return compact_api_doc(d)[1] if d else f"{app}.{api}"


def _attr_chain(node: ast.AST) -> list[str]:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return parts[::-1]
    return []


def _str_literal(node: ast.AST | None) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _api_calls(tree: ast.AST) -> list[tuple[str, str, ast.Call]]:
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            chain = _attr_chain(node.func)
            if len(chain) == 3 and chain[0] == "apis":
                calls.append((chain[1], chain[2], node))
    return calls


def _defined_names(tree: ast.AST) -> set[str]:
    """Every name the code itself binds anywhere (generous on purpose: no false alarms)."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.alias):
            names.add((node.asname or node.name).split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            names.update(node.names)
        elif hasattr(ast, "MatchAs") and isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            names.add(node.name)
    return names


def validate_appworld_code(code: str, catalog: ApiCatalog | None, existing: set[str], max_problems: int = 3) -> str | None:
    """Problems that would make this step fail or silently do the wrong thing, or None if it looks fine.

    Checks (all against this task's own API catalog and the session's real variables):
      - syntax errors;
      - calls to apps or APIs that don't exist;
      - parameters an API doesn't have (AppWorld ignores them silently, which gives wrong results);
      - required parameters that are missing;
      - documentation lookups for apps/APIs that don't exist;
      - variables that don't exist yet."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        line = (e.text or "").strip()
        return f"SyntaxError on line {e.lineno}: {e.msg}" + (f"\n    {line}" if line else "")
    problems: list[str] = []
    if catalog is not None:
        for app, api, node in _api_calls(tree):
            if app not in catalog.docs:
                problems.append(f"There is no app `{app}`. Apps: {', '.join(catalog.apps)}.")
                continue
            doc = catalog.doc(app, api)
            if doc is None:
                problems.append(f"The {app} app has no API `{api}`. Its APIs: {', '.join(catalog.apis(app))}.")
                continue
            params = {p.get("name"): p for p in doc.get("parameters") or []}
            given = {k.arg for k in node.keywords if k.arg}
            unknown = sorted(given - set(params))
            if unknown:
                problems.append(f"`{app}.{api}` has no parameter(s) {', '.join(unknown)}. Signature: {catalog.signature(app, api)}")
            spread = any(k.arg is None for k in node.keywords) or bool(node.args)
            missing = [n for n, p in params.items() if p.get("required") and n not in given]
            if missing and not spread:
                problems.append(f"`{app}.{api}` is missing required parameter(s) {', '.join(missing)}. Signature: {catalog.signature(app, api)}")
            if app == "api_docs" and api in ("show_api_doc", "show_api_descriptions"):
                kw = {k.arg: k.value for k in node.keywords if k.arg}
                app_lit, api_lit = _str_literal(kw.get("app_name")), _str_literal(kw.get("api_name"))
                if app_lit and app_lit not in catalog.docs:
                    problems.append(f"There is no app `{app_lit}`. Apps: {', '.join(catalog.apps)}.")
                elif app_lit and api_lit and catalog.doc(app_lit, api_lit) is None:
                    problems.append(f"The {app_lit} app has no API `{api_lit}`. Its APIs: {', '.join(catalog.apis(app_lit))}.")
    defined = existing | _defined_names(tree) | set(dir(builtins))
    loads = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    undefined = sorted(loads - defined)
    if undefined:
        visible = sorted(n for n in existing if not n.startswith("_"))
        problems.append(f"Not defined: {', '.join(undefined)}. Variables that exist: {', '.join(visible[-40:]) or '(none)'}.")
    if not problems:
        return None
    return "\n".join(f"- {p}" for p in problems[:max_problems])


def catalog_reference_for_code(code: str, catalog: ApiCatalog | None) -> list[tuple[str, str]]:
    """Reference entries for every API this code calls or looks up, straight from the catalog."""
    if catalog is None:
        return []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    out: list[tuple[str, str]] = []
    for app, api, node in _api_calls(tree):
        if app == "api_docs":
            kw = {k.arg: k.value for k in node.keywords if k.arg}
            app_lit, api_lit = _str_literal(kw.get("app_name")), _str_literal(kw.get("api_name"))
            if api == "show_api_doc" and app_lit and api_lit and catalog.doc(app_lit, api_lit):
                out.append(compact_api_doc(catalog.doc(app_lit, api_lit)))
            elif api == "show_api_descriptions" and app_lit in catalog.docs:
                out.append((f"apis:{app_lit}", f"{app_lit} APIs: {', '.join(catalog.apis(app_lit))}"))
            elif api == "show_app_descriptions":
                out.append(("apps", f"Apps: {', '.join(catalog.apps)}"))
        elif catalog.doc(app, api):
            out.append(compact_api_doc(catalog.doc(app, api)))
    return out


_APP_API_MENTION = re.compile(r"\b([a-z][a-z_]*)\.([a-z][a-z_0-9]*)\b")


def catalog_reference_for_text(text: str, catalog: ApiCatalog | None) -> list[tuple[str, str]]:
    """Reference entries for every `app.api` named in free text (e.g. the planner's instructions)."""
    if catalog is None or not text:
        return []
    out, seen = [], set()
    for app, api in _APP_API_MENTION.findall(text):
        if (app, api) not in seen and catalog.doc(app, api):
            seen.add((app, api))
            out.append(compact_api_doc(catalog.doc(app, api)))
    return out


def summarize_namespace(ns: dict[str, Any], baseline: set[str], max_vars: int = 40, max_repr: int = 100) -> str:
    """Variables the agents created, with type and a short value preview."""
    lines = []
    for name, val in ns.items():
        if name.startswith("_") or name in baseline or name in PROTECTED_NAMES:
            continue
        if isinstance(val, (types.ModuleType, types.FunctionType, types.BuiltinFunctionType, type)):
            continue
        try:
            r = repr(val)
        except Exception:
            r = "<unprintable>"
        r = r.replace("\n", " ")
        if len(r) > max_repr:
            r = r[: max_repr - 1] + "…"
        size = f" (len {len(val)})" if isinstance(val, (list, dict, tuple, set)) else ""
        lines.append(f"{name}: {type(val).__name__}{size} = {r}")
    return "\n".join(lines[-max_vars:]) if lines else "(no variables yet)"


class AppWorldEnv:
    def __init__(self, task_id: str, experiment_name: str, max_steps: int) -> None:
        from appworld import AppWorld  # imported lazily so tests run without AppWorld data

        self.task_id = task_id
        self.max_steps = max_steps
        self.steps = 0
        # max_interactions is a hard backstop inside AppWorld; our own max_steps is the real budget.
        self.world = AppWorld(
            task_id=task_id,
            experiment_name=experiment_name,
            max_interactions=max_steps + 5,
            **appworld_kwargs(),
        )
        task = self.world.task
        self.instruction: str = task.instruction
        self.supervisor: dict[str, Any] = dict(task.supervisor)
        self.app_descriptions: str = json.dumps(
            [{"name": k, "description": v} for k, v in task.app_descriptions.items()], indent=1
        )
        shell = getattr(self.world, "shell", None)
        self._baseline: set[str] = set(shell.user_ns) if shell is not None else set()
        try:
            self.catalog: ApiCatalog | None = ApiCatalog(task.api_docs)
        except Exception:
            self.catalog = None

    # ---- context hooks (see score/context.py)
    def state_summary(self) -> str:
        shell = getattr(self.world, "shell", None)
        return summarize_namespace(shell.user_ns, self._baseline) if shell is not None else ""

    def reference_from_step(self, code: str, output: str) -> list[tuple[str, str]]:
        # From the catalog for everything the code calls or looks up (works even if the step failed),
        # plus anything parsed from the printed output (e.g. lookups with computed arguments).
        merged = dict(appworld_reference_from_step(code, output))
        merged.update(dict(catalog_reference_for_code(code, self.catalog)))
        return list(merged.items())

    def reference_for_text(self, text: str) -> list[tuple[str, str]]:
        return catalog_reference_for_text(text, self.catalog)

    def validate(self, code: str) -> str | None:
        shell = getattr(self.world, "shell", None)
        existing = set(shell.user_ns) if shell is not None else set()
        return validate_appworld_code(code, self.catalog, existing)

    def execute(self, code: str) -> str:
        self.steps += 1
        if not IS_WINDOWS:
            output = self.world.execute(code)
        else:
            timed_out, output = run_with_timeout(self.world.execute, code)
            if timed_out:
                output = TIMEOUT_MESSAGE
        return output + self._guard()

    def _guard(self) -> str:
        shell = getattr(self.world, "shell", None)
        if shell is None:
            return ""
        originals = {"apis": getattr(self.world, "apis", None), "requester": getattr(self.world, "requester", None)}
        return restore_protected(shell.user_ns, originals)

    def completed(self) -> bool:
        return bool(self.world.task_completed())

    @property
    def out_of_steps(self) -> bool:
        return self.steps >= self.max_steps

    def evaluate(self) -> dict[str, Any]:
        tracker = self.world.evaluate()
        return {
            "success": bool(tracker.success),
            "pass_percentage": float(tracker.pass_percentage),
            "pass_count": int(tracker.pass_count),
            "num_tests": int(tracker.num_tests),
        }

    def close(self) -> None:
        try:
            self.world.close()
        except Exception:
            pass
