# Natural-Language Image Policy Editor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a browser-based prototype that uploads real images, marks a region, routes a natural-language instruction to an image policy, executes the edit, and saves a dataset-ready trace.

**Architecture:** Keep policy logic in focused Python modules and reuse `composer.py` for operator orchestration. Serve a small HTML/CSS/JS UI from the Python standard library so the prototype has no network dependency. Store uploaded images, outputs, masks, and JSON traces under `data/image_policy_editor/`.

**Tech Stack:** Python 3 standard library, `unittest`, existing `composer.py`, browser canvas APIs, optional Pillow only when already available in the local environment.

---

## File Structure

- Create `image_policy/__init__.py`: package marker and public exports.
- Create `image_policy/region.py`: rectangle annotation value object and coordinate validation.
- Create `image_policy/router.py`: deterministic natural-language policy router with LLM-shaped output.
- Create `image_policy/traces.py`: trace IDs, trace dataclasses, and JSON persistence.
- Create `image_policy/operators.py`: operator functions used by policy pipelines.
- Create `image_policy/policies.py`: policy-level composer registrations and execution entrypoints.
- Create `image_policy/server.py`: standard-library HTTP server for uploads, routing, execution, and feedback.
- Create `image_policy/static/index.html`: browser UI shell.
- Create `image_policy/static/styles.css`: app styling.
- Create `image_policy/static/app.js`: upload, region marking, route preview, execute, and feedback client.
- Create `policies/image_editing/increment_number/README.md`: policy dataset contract.
- Create `policies/image_editing/redraw_region/README.md`: policy dataset contract.
- Create `tests/test_image_region.py`: region validation tests.
- Create `tests/test_image_policy_router.py`: router tests.
- Create `tests/test_image_traces.py`: trace persistence tests.
- Create `tests/test_image_policies.py`: policy execution and composer-order tests.

## Task 1: Region Model

**Files:**
- Create: `image_policy/__init__.py`
- Create: `image_policy/region.py`
- Test: `tests/test_image_region.py`

- [ ] **Step 1: Write the failing tests**

```python
import unittest

from image_policy.region import ImageRegion


class ImageRegionTests(unittest.TestCase):
    def test_rectangle_serializes_to_trace_shape(self):
        region = ImageRegion.rectangle(x=10, y=20, width=30, height=40)
        self.assertEqual(
            region.to_trace(),
            {"type": "rectangle", "x": 10, "y": 20, "width": 30, "height": 40},
        )

    def test_rejects_negative_coordinates(self):
        with self.assertRaisesRegex(ValueError, "x must be >= 0"):
            ImageRegion.rectangle(x=-1, y=20, width=30, height=40)

    def test_rejects_empty_region(self):
        with self.assertRaisesRegex(ValueError, "width must be > 0"):
            ImageRegion.rectangle(x=0, y=0, width=0, height=40)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m unittest tests.test_image_region -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'image_policy'`.

- [ ] **Step 3: Implement the region model**

Create `image_policy/__init__.py`:

```python
"""Natural-language image policy editor package."""
```

Create `image_policy/region.py`:

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ImageRegion:
    type: str
    x: int
    y: int
    width: int
    height: int

    @classmethod
    def rectangle(cls, *, x: int, y: int, width: int, height: int) -> "ImageRegion":
        values = {"x": x, "y": y, "width": width, "height": height}
        for name, value in values.items():
            if not isinstance(value, int):
                raise TypeError(f"{name} must be an int")
        if x < 0:
            raise ValueError("x must be >= 0")
        if y < 0:
            raise ValueError("y must be >= 0")
        if width <= 0:
            raise ValueError("width must be > 0")
        if height <= 0:
            raise ValueError("height must be > 0")
        return cls(type="rectangle", x=x, y=y, width=width, height=height)

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> "ImageRegion":
        if payload.get("type", "rectangle") != "rectangle":
            raise ValueError("only rectangle regions are supported")
        return cls.rectangle(
            x=int(payload["x"]),
            y=int(payload["y"]),
            width=int(payload["width"]),
            height=int(payload["height"]),
        )

    def to_trace(self) -> dict[str, object]:
        return {
            "type": self.type,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python3 -m unittest tests.test_image_region -v`

Expected: PASS, 3 tests.

- [ ] **Step 5: Commit**

```bash
git add image_policy/__init__.py image_policy/region.py tests/test_image_region.py
git commit -m "Add image region model"
```

## Task 2: Natural-Language Router

**Files:**
- Create: `image_policy/router.py`
- Test: `tests/test_image_policy_router.py`

- [ ] **Step 1: Write the failing tests**

```python
import unittest

from image_policy.router import route_instruction


class ImagePolicyRouterTests(unittest.TestCase):
    def test_routes_increment_language(self):
        result = route_instruction("make this number one higher")
        self.assertEqual(result.policy, "increment_number")
        self.assertEqual(result.args["amount"], 1)
        self.assertFalse(result.needs_user_confirmation)

    def test_routes_decrement_language(self):
        result = route_instruction("decrease this by 2")
        self.assertEqual(result.policy, "increment_number")
        self.assertEqual(result.args["amount"], -2)

    def test_routes_redraw_language(self):
        result = route_instruction("redraw this in pixel style")
        self.assertEqual(result.policy, "redraw_region")
        self.assertEqual(result.args["style"], "pixel style")

    def test_low_confidence_for_unknown_instruction(self):
        result = route_instruction("make it nicer")
        self.assertEqual(result.policy, "unknown")
        self.assertTrue(result.needs_user_confirmation)
        self.assertLess(result.confidence, 0.5)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m unittest tests.test_image_policy_router -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'image_policy.router'`.

- [ ] **Step 3: Implement the router**

Create `image_policy/router.py`:

```python
from __future__ import annotations

import re
from dataclasses import dataclass


NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


@dataclass(frozen=True)
class RouterResult:
    policy: str
    confidence: float
    args: dict[str, object]
    alternatives: tuple[str, ...] = ()
    needs_user_confirmation: bool = False

    def to_trace(self) -> dict[str, object]:
        return {
            "policy": self.policy,
            "confidence": self.confidence,
            "alternatives": list(self.alternatives),
        }


def route_instruction(instruction: str) -> RouterResult:
    text = instruction.strip().lower()
    if _looks_like_number_increment(text):
        return RouterResult(
            policy="increment_number",
            confidence=0.86,
            args={"amount": _extract_amount(text)},
            alternatives=("replace_text",),
        )
    if _looks_like_redraw(text):
        return RouterResult(
            policy="redraw_region",
            confidence=0.82,
            args={"style": _extract_style(text)},
            alternatives=("recolor_region",),
        )
    return RouterResult(
        policy="unknown",
        confidence=0.2,
        args={},
        alternatives=("increment_number", "redraw_region"),
        needs_user_confirmation=True,
    )


def _looks_like_number_increment(text: str) -> bool:
    number_terms = ("number", "digit", "score", "counter", "higher", "lower", "increase", "decrease", "increment", "decrement")
    return any(term in text for term in number_terms) and (
        "higher" in text or "lower" in text or "increase" in text or "decrease" in text or "increment" in text or "decrement" in text
    )


def _extract_amount(text: str) -> int:
    sign = -1 if any(term in text for term in ("decrease", "lower", "decrement", "down")) else 1
    match = re.search(r"\b(\d+)\b", text)
    if match:
        return sign * int(match.group(1))
    for word, value in NUMBER_WORDS.items():
        if re.search(rf"\b{word}\b", text):
            return sign * value
    return sign


def _looks_like_redraw(text: str) -> bool:
    return any(term in text for term in ("redraw", "draw this", "style", "pixel", "sketch", "watercolor", "sticker"))


def _extract_style(text: str) -> str:
    for marker in (" in ", " as ", " like "):
        if marker in text:
            return text.split(marker, 1)[1].strip()
    return text
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python3 -m unittest tests.test_image_policy_router -v`

Expected: PASS, 4 tests.

- [ ] **Step 5: Commit**

```bash
git add image_policy/router.py tests/test_image_policy_router.py
git commit -m "Add image policy router"
```

## Task 3: Trace Store

**Files:**
- Create: `image_policy/traces.py`
- Test: `tests/test_image_traces.py`

- [ ] **Step 1: Write the failing tests**

```python
import json
import tempfile
import unittest
from pathlib import Path

from image_policy.region import ImageRegion
from image_policy.router import RouterResult
from image_policy.traces import TraceStore


class TraceStoreTests(unittest.TestCase):
    def test_writes_dataset_ready_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = TraceStore(Path(tmp))
            path = store.write_trace(
                input_image="data/image_policy_editor/input/example.png",
                output_image="data/image_policy_editor/output/example.png",
                instruction="make this number one higher",
                region=ImageRegion.rectangle(x=1, y=2, width=3, height=4),
                router=RouterResult(policy="increment_number", confidence=0.86, args={"amount": 1}),
                policy_call={"name": "increment_number", "args": {"amount": 1, "detected_value": "7", "replacement_value": "8"}},
                operators=["crop_region", "detect_number", "erase_region", "render_text", "blend_region"],
                status="pending",
            )
            data = json.loads(path.read_text())
            self.assertEqual(data["instruction"], "make this number one higher")
            self.assertEqual(data["region"]["width"], 3)
            self.assertEqual(data["policy_call"]["args"]["replacement_value"], "8")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m unittest tests.test_image_traces -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'image_policy.traces'`.

- [ ] **Step 3: Implement trace persistence**

Create `image_policy/traces.py`:

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from image_policy.region import ImageRegion
from image_policy.router import RouterResult


@dataclass(frozen=True)
class TraceStore:
    root: Path

    def __post_init__(self) -> None:
        self.traces_dir.mkdir(parents=True, exist_ok=True)

    @property
    def traces_dir(self) -> Path:
        return self.root / "traces"

    def next_trace_id(self) -> str:
        prefix = datetime.now(ZoneInfo("America/Denver")).strftime("%Y%m%d")
        existing = sorted(self.traces_dir.glob(f"{prefix}-*.json"))
        return f"{prefix}-{len(existing) + 1:06d}"

    def write_trace(
        self,
        *,
        input_image: str,
        output_image: str | None,
        instruction: str,
        region: ImageRegion,
        router: RouterResult,
        policy_call: dict[str, object],
        operators: list[str],
        status: str,
        notes: str = "",
        error: str | None = None,
    ) -> Path:
        trace_id = self.next_trace_id()
        record = {
            "trace_id": trace_id,
            "created_at": datetime.now(ZoneInfo("America/Denver")).isoformat(timespec="seconds"),
            "input_image": input_image,
            "output_image": output_image,
            "instruction": instruction,
            "region": region.to_trace(),
            "router": router.to_trace(),
            "policy_call": policy_call,
            "operators": operators,
            "feedback": {"status": status, "notes": notes},
        }
        if error is not None:
            record["error"] = error
        path = self.traces_dir / f"{trace_id}.json"
        path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        return path
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python3 -m unittest tests.test_image_traces -v`

Expected: PASS, 1 test.

- [ ] **Step 5: Commit**

```bash
git add image_policy/traces.py tests/test_image_traces.py
git commit -m "Add image policy trace store"
```

## Task 4: Policy Operators and Composer Pipelines

**Files:**
- Create: `image_policy/operators.py`
- Create: `image_policy/policies.py`
- Test: `tests/test_image_policies.py`

- [ ] **Step 1: Write the failing tests**

```python
import tempfile
import unittest
from pathlib import Path

from image_policy.policies import execute_policy
from image_policy.region import ImageRegion


class ImagePolicyExecutionTests(unittest.TestCase):
    def test_increment_number_uses_composer_operator_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "input.txt"
            output_path = Path(tmp) / "output.txt"
            input_path.write_text("room 7")
            result = execute_policy(
                policy="increment_number",
                input_path=input_path,
                output_path=output_path,
                instruction="make this number one higher",
                region=ImageRegion.rectangle(x=0, y=0, width=10, height=10),
                args={"amount": 1, "current_value": "7"},
            )
            self.assertEqual(result["operators"], ["crop_region", "detect_number", "erase_region", "render_text", "blend_region"])
            self.assertEqual(result["policy_call"]["args"]["replacement_value"], "8")
            self.assertIn("8", output_path.read_text())

    def test_redraw_region_uses_mock_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "input.txt"
            output_path = Path(tmp) / "output.txt"
            input_path.write_text("horse")
            result = execute_policy(
                policy="redraw_region",
                input_path=input_path,
                output_path=output_path,
                instruction="redraw this in pixel style",
                region=ImageRegion.rectangle(x=0, y=0, width=10, height=10),
                args={"style": "pixel style"},
            )
            self.assertEqual(result["operators"], ["crop_region", "normalize_style", "mock_redraw", "blend_region"])
            self.assertIn("[redraw:pixel style]", output_path.read_text())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m unittest tests.test_image_policies -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'image_policy.policies'`.

- [ ] **Step 3: Implement mock-safe operators**

Create `image_policy/operators.py`:

```python
from __future__ import annotations

from pathlib import Path


def crop_region(context: dict[str, object]) -> dict[str, object]:
    return {"region_crop": Path(context["input_path"]).read_text(errors="ignore")}


def detect_number(context: dict[str, object]) -> dict[str, object]:
    args = dict(context["args"])
    value = str(args.get("current_value") or args.get("detected_value") or "")
    if not value:
        raise ValueError("current_value is required for the first increment_number prototype")
    return {"detected_value": value}


def erase_region(context: dict[str, object]) -> dict[str, object]:
    return {"erased_region": str(context["region_crop"]).replace(str(context["detected_value"]), "")}


def render_text(context: dict[str, object]) -> dict[str, object]:
    amount = int(dict(context["args"]).get("amount", 1))
    replacement = str(int(str(context["detected_value"])) + amount)
    return {"replacement_value": replacement, "rendered_region": replacement}


def blend_region(context: dict[str, object]) -> dict[str, object]:
    input_text = Path(context["input_path"]).read_text(errors="ignore")
    detected = str(context.get("detected_value", ""))
    replacement = str(context.get("rendered_region", context.get("redrawn_region", "")))
    output_text = input_text.replace(detected, replacement, 1) if detected else input_text + replacement
    Path(context["output_path"]).write_text(output_text)
    return {"output_image": str(context["output_path"])}


def normalize_style(context: dict[str, object]) -> dict[str, object]:
    style = str(dict(context["args"]).get("style", ""))
    return {"normalized_style": style.strip().lower() or "style transfer"}


def mock_redraw(context: dict[str, object]) -> dict[str, object]:
    return {"redrawn_region": f"[redraw:{context['normalized_style']}]"}
```

- [ ] **Step 4: Implement policy pipelines with the existing composer**

Create `image_policy/policies.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

from composer import Composer, FieldOperator
from image_policy import operators
from image_policy.region import ImageRegion


INCREMENT_OPERATORS = [
    FieldOperator("crop_region", ("input_path", "region"), ("region_crop",), operators.crop_region),
    FieldOperator("detect_number", ("region_crop", "args"), ("detected_value",), operators.detect_number),
    FieldOperator("erase_region", ("region_crop", "detected_value"), ("erased_region",), operators.erase_region),
    FieldOperator("render_text", ("detected_value", "args"), ("replacement_value", "rendered_region"), operators.render_text),
    FieldOperator("blend_region", ("input_path", "output_path", "detected_value", "rendered_region"), ("output_image",), operators.blend_region),
]

REDRAW_OPERATORS = [
    FieldOperator("crop_region", ("input_path", "region"), ("region_crop",), operators.crop_region),
    FieldOperator("normalize_style", ("args",), ("normalized_style",), operators.normalize_style),
    FieldOperator("mock_redraw", ("region_crop", "normalized_style"), ("redrawn_region",), operators.mock_redraw),
    FieldOperator("blend_region", ("input_path", "output_path", "redrawn_region"), ("output_image",), operators.blend_region),
]


def execute_policy(
    *,
    policy: str,
    input_path: Path,
    output_path: Path,
    instruction: str,
    region: ImageRegion,
    args: dict[str, object],
) -> dict[str, Any]:
    if policy == "increment_number":
        operator_defs = INCREMENT_OPERATORS
        targets = ("output_image", "detected_value", "replacement_value")
    elif policy == "redraw_region":
        operator_defs = REDRAW_OPERATORS
        targets = ("output_image", "normalized_style")
    else:
        raise ValueError(f"unknown policy: {policy}")

    composer = Composer(operator_defs)
    result = composer.run(
        targets,
        {
            "input_path": input_path,
            "output_path": output_path,
            "instruction": instruction,
            "region": region,
            "args": args,
        },
    )
    policy_args = dict(args)
    if "detected_value" in result:
        policy_args["detected_value"] = result["detected_value"]
    if "replacement_value" in result:
        policy_args["replacement_value"] = result["replacement_value"]
    if "normalized_style" in result:
        policy_args["normalized_style"] = result["normalized_style"]
    return {
        "output_image": result["output_image"],
        "operators": list(Composer(operator_defs).plan(("output_image",), ("input_path", "output_path", "instruction", "region", "args")).order),
        "policy_call": {"name": policy, "args": policy_args},
    }
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `python3 -m unittest tests.test_image_policies -v`

Expected: PASS, 2 tests.

- [ ] **Step 6: Commit**

```bash
git add image_policy/operators.py image_policy/policies.py tests/test_image_policies.py
git commit -m "Add image policy execution pipelines"
```

## Task 5: Policy Dataset Documentation

**Files:**
- Create: `policies/image_editing/increment_number/README.md`
- Create: `policies/image_editing/redraw_region/README.md`

- [ ] **Step 1: Add `increment_number` policy documentation**

```markdown
# increment_number

Changes a marked numeric text region by an integer amount.

## Inputs

- Uploaded raster image path.
- User-marked rectangle region.
- Natural-language instruction.
- Router argument `amount`.
- User-confirmed `current_value` when detection is unavailable.

## Operators

```text
crop_region -> detect_number -> erase_region -> render_text -> blend_region
```

## Trace Value

Accepted, corrected, and failed traces teach numeric grounding, value parsing, style preservation, and replacement rendering.
```

- [ ] **Step 2: Add `redraw_region` policy documentation**

```markdown
# redraw_region

Redraws a marked region using a style phrase extracted from natural language.

## Inputs

- Uploaded raster image path.
- User-marked rectangle region.
- Natural-language instruction.
- Router argument `style`.

## Operators

```text
crop_region -> normalize_style -> mock_redraw -> blend_region
```

## Trace Value

Accepted, corrected, and failed traces teach style normalization, preservation constraints, mask quality, and backend behavior.
```

- [ ] **Step 3: Commit**

```bash
git add policies/image_editing/increment_number/README.md policies/image_editing/redraw_region/README.md
git commit -m "Document image editing policy seeds"
```

## Task 6: HTTP API

**Files:**
- Create: `image_policy/server.py`

- [ ] **Step 1: Implement the standard-library server**

Create `image_policy/server.py`:

```python
from __future__ import annotations

import base64
import json
import shutil
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from image_policy.policies import execute_policy
from image_policy.region import ImageRegion
from image_policy.router import route_instruction
from image_policy.traces import TraceStore


ROOT = Path("data/image_policy_editor")
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"
STATIC_DIR = Path(__file__).parent / "static"


class ImagePolicyHandler(SimpleHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/":
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self) -> None:
        route = urlparse(self.path).path
        if route == "/api/upload":
            return self._upload_image()
        if route == "/api/route":
            return self._route_instruction()
        if route == "/api/execute":
            return self._execute_policy()
        self.send_error(404, "unknown route")

    def translate_path(self, path: str) -> str:
        if path.startswith("/data/"):
            return str(Path(path.lstrip("/")).resolve())
        return str((STATIC_DIR / path.lstrip("/")).resolve())

    def _read_json(self) -> dict[str, object]:
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _send_json(self, payload: dict[str, object], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _route_instruction(self) -> None:
        payload = self._read_json()
        result = route_instruction(str(payload["instruction"]))
        self._send_json({"policy": result.policy, "confidence": result.confidence, "args": result.args, "needs_user_confirmation": result.needs_user_confirmation})

    def _upload_image(self) -> None:
        payload = self._read_json()
        INPUT_DIR.mkdir(parents=True, exist_ok=True)
        filename = Path(str(payload["filename"])).name or "upload.png"
        encoded = str(payload["data_url"]).split(",", 1)[1]
        path = INPUT_DIR / filename
        path.write_bytes(base64.b64decode(encoded))
        self._send_json({"input_path": str(path), "url": f"/{path}"})

    def _execute_policy(self) -> None:
        payload = self._read_json()
        ROOT.mkdir(parents=True, exist_ok=True)
        INPUT_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        input_path = Path(str(payload["input_path"]))
        saved_input = INPUT_DIR / input_path.name
        if input_path.resolve() != saved_input.resolve():
            shutil.copyfile(input_path, saved_input)
        region = ImageRegion.from_payload(dict(payload["region"]))
        router = route_instruction(str(payload["instruction"]))
        output_path = OUTPUT_DIR / f"{saved_input.stem}-{router.policy}{saved_input.suffix}"
        result = execute_policy(
            policy=str(payload.get("policy") or router.policy),
            input_path=saved_input,
            output_path=output_path,
            instruction=str(payload["instruction"]),
            region=region,
            args=dict(payload.get("args") or router.args),
        )
        trace_path = TraceStore(ROOT).write_trace(
            input_image=str(saved_input),
            output_image=str(output_path),
            instruction=str(payload["instruction"]),
            region=region,
            router=router,
            policy_call=dict(result["policy_call"]),
            operators=list(result["operators"]),
            status="pending",
        )
        self._send_json({"output_image": str(output_path), "trace": str(trace_path), "policy_call": result["policy_call"], "operators": result["operators"]})


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 8765), ImagePolicyHandler)
    print("Serving image policy editor at http://127.0.0.1:8765")
    server.serve_forever()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-check imports**

Run: `python3 -m py_compile image_policy/server.py`

Expected: no output and exit code 0.

- [ ] **Step 3: Commit**

```bash
git add image_policy/server.py
git commit -m "Add image policy editor HTTP API"
```

## Task 7: Browser UI

**Files:**
- Create: `image_policy/static/index.html`
- Create: `image_policy/static/styles.css`
- Create: `image_policy/static/app.js`

- [ ] **Step 1: Create the UI shell**

Create `image_policy/static/index.html`:

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Image Policy Editor</title>
    <link rel="stylesheet" href="/styles.css">
  </head>
  <body>
    <main class="workspace">
      <section class="stage">
        <div class="toolbar">
          <label class="file-button">
            Upload image
            <input id="fileInput" type="file" accept="image/*">
          </label>
          <span id="imageStatus">No image loaded</span>
        </div>
        <canvas id="canvas" width="960" height="640"></canvas>
      </section>
      <aside class="panel">
        <label class="field">
          Instruction
          <textarea id="instruction" rows="4" placeholder="make this number one higher"></textarea>
        </label>
        <button id="routeButton" disabled>Interpret</button>
        <section class="card" id="interpretation">
          <h2>Policy</h2>
          <dl>
            <dt>Name</dt><dd id="policyName">-</dd>
            <dt>Confidence</dt><dd id="policyConfidence">-</dd>
            <dt>Arguments</dt><dd id="policyArgs">-</dd>
          </dl>
        </section>
        <label class="field">
          Current value
          <input id="currentValue" type="text" inputmode="numeric" placeholder="7">
        </label>
        <button id="executeButton" disabled>Run edit</button>
        <section class="card">
          <h2>Trace</h2>
          <pre id="traceOutput">{}</pre>
        </section>
      </aside>
    </main>
    <script src="/app.js"></script>
  </body>
</html>
```

- [ ] **Step 2: Create the UI styles**

Create `image_policy/static/styles.css`:

```css
:root {
  color-scheme: light;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: #f6f4ef;
  color: #1d2528;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
}

.workspace {
  min-height: 100vh;
  display: grid;
  grid-template-columns: minmax(0, 1fr) 360px;
}

.stage {
  padding: 20px;
  display: grid;
  grid-template-rows: 44px minmax(0, 1fr);
  gap: 12px;
}

.toolbar {
  display: flex;
  align-items: center;
  gap: 12px;
  min-height: 44px;
}

.file-button,
button {
  border: 1px solid #1d2528;
  background: #ffffff;
  color: #1d2528;
  border-radius: 6px;
  min-height: 36px;
  padding: 8px 12px;
  font: inherit;
  cursor: pointer;
}

.file-button input {
  display: none;
}

button:disabled {
  opacity: 0.45;
  cursor: not-allowed;
}

canvas {
  width: 100%;
  height: 100%;
  min-height: 480px;
  border: 1px solid #c8c1b5;
  background: #ffffff;
}

.panel {
  border-left: 1px solid #d7d0c4;
  background: #ffffff;
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.field {
  display: grid;
  gap: 6px;
  font-size: 13px;
  font-weight: 700;
}

textarea,
input {
  width: 100%;
  border: 1px solid #b8b0a6;
  border-radius: 6px;
  padding: 10px;
  font: inherit;
}

.card {
  border: 1px solid #d7d0c4;
  border-radius: 8px;
  padding: 12px;
}

.card h2 {
  margin: 0 0 10px;
  font-size: 15px;
}

dl {
  display: grid;
  grid-template-columns: 96px 1fr;
  gap: 6px;
  margin: 0;
  font-size: 13px;
}

dt {
  color: #566164;
}

dd {
  margin: 0;
  overflow-wrap: anywhere;
}

pre {
  min-height: 120px;
  max-height: 280px;
  overflow: auto;
  margin: 0;
  white-space: pre-wrap;
  font-size: 12px;
}

@media (max-width: 860px) {
  .workspace {
    grid-template-columns: 1fr;
  }

  .panel {
    border-left: 0;
    border-top: 1px solid #d7d0c4;
  }
}
```

- [ ] **Step 3: Create the client behavior**

Create `image_policy/static/app.js`:

```javascript
const fileInput = document.querySelector("#fileInput");
const imageStatus = document.querySelector("#imageStatus");
const canvas = document.querySelector("#canvas");
const ctx = canvas.getContext("2d");
const instruction = document.querySelector("#instruction");
const routeButton = document.querySelector("#routeButton");
const executeButton = document.querySelector("#executeButton");
const policyName = document.querySelector("#policyName");
const policyConfidence = document.querySelector("#policyConfidence");
const policyArgs = document.querySelector("#policyArgs");
const currentValue = document.querySelector("#currentValue");
const traceOutput = document.querySelector("#traceOutput");

let image = null;
let inputPath = null;
let region = null;
let dragStart = null;
let routeResult = null;

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (image) {
    const scale = Math.min(canvas.width / image.width, canvas.height / image.height);
    const width = image.width * scale;
    const height = image.height * scale;
    const x = (canvas.width - width) / 2;
    const y = (canvas.height - height) / 2;
    ctx.drawImage(image, x, y, width, height);
  }
  if (region) {
    ctx.strokeStyle = "#e23d28";
    ctx.lineWidth = 3;
    ctx.strokeRect(region.x, region.y, region.width, region.height);
    ctx.fillStyle = "rgba(226, 61, 40, 0.12)";
    ctx.fillRect(region.x, region.y, region.width, region.height);
  }
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json();
}

fileInput.addEventListener("change", async () => {
  const file = fileInput.files[0];
  if (!file) {
    return;
  }
  const reader = new FileReader();
  reader.addEventListener("load", async () => {
    const dataUrl = reader.result;
    const upload = await postJson("/api/upload", {filename: file.name, data_url: dataUrl});
    inputPath = upload.input_path;
    image = new Image();
    image.addEventListener("load", () => {
      imageStatus.textContent = file.name;
      region = null;
      routeButton.disabled = false;
      executeButton.disabled = true;
      draw();
    });
    image.src = dataUrl;
  });
  reader.readAsDataURL(file);
});

canvas.addEventListener("pointerdown", (event) => {
  if (!image) {
    return;
  }
  const rect = canvas.getBoundingClientRect();
  dragStart = {
    x: Math.round((event.clientX - rect.left) * (canvas.width / rect.width)),
    y: Math.round((event.clientY - rect.top) * (canvas.height / rect.height)),
  };
});

canvas.addEventListener("pointermove", (event) => {
  if (!dragStart) {
    return;
  }
  const rect = canvas.getBoundingClientRect();
  const x = Math.round((event.clientX - rect.left) * (canvas.width / rect.width));
  const y = Math.round((event.clientY - rect.top) * (canvas.height / rect.height));
  region = {
    type: "rectangle",
    x: Math.min(dragStart.x, x),
    y: Math.min(dragStart.y, y),
    width: Math.max(1, Math.abs(x - dragStart.x)),
    height: Math.max(1, Math.abs(y - dragStart.y)),
  };
  draw();
});

canvas.addEventListener("pointerup", () => {
  dragStart = null;
  executeButton.disabled = !(routeResult && region);
});

routeButton.addEventListener("click", async () => {
  routeResult = await postJson("/api/route", {instruction: instruction.value});
  policyName.textContent = routeResult.policy;
  policyConfidence.textContent = routeResult.confidence.toFixed(2);
  policyArgs.textContent = JSON.stringify(routeResult.args);
  executeButton.disabled = !(inputPath && region && routeResult.policy !== "unknown");
});

executeButton.addEventListener("click", async () => {
  const args = {...routeResult.args};
  if (routeResult.policy === "increment_number" && currentValue.value.trim()) {
    args.current_value = currentValue.value.trim();
  }
  const result = await postJson("/api/execute", {
    input_path: inputPath,
    instruction: instruction.value,
    region,
    policy: routeResult.policy,
    args,
  });
  traceOutput.textContent = JSON.stringify(result, null, 2);
});

draw();
```

- [ ] **Step 4: Run the server**

Run: `python3 -m image_policy.server`

Expected: terminal prints `Serving image policy editor at http://127.0.0.1:8765`.

- [ ] **Step 5: Browser smoke test**

Open `http://127.0.0.1:8765`, upload a small text fixture image or text-backed sample file, draw a region, enter `make this number one higher`, confirm the policy card shows `increment_number`, execute, and verify a trace JSON appears under `data/image_policy_editor/traces/`.

- [ ] **Step 6: Commit**

```bash
git add image_policy/static/index.html image_policy/static/styles.css image_policy/static/app.js
git commit -m "Add image policy editor UI"
```

## Task 8: End-to-End Verification

**Files:**
- Modify only files needed to fix failures found by verification.

- [ ] **Step 1: Run all unit tests**

Run: `python3 -m unittest discover -v`

Expected: all existing tests and new image policy tests pass.

- [ ] **Step 2: Run import compilation**

Run: `python3 -m py_compile image_policy/*.py`

Expected: no output and exit code 0.

- [ ] **Step 3: Verify trace artifacts**

Run: `find data/image_policy_editor -maxdepth 3 -type f | sort`

Expected: at least one file under `input`, one file under `output`, and one trace under `traces` after the manual smoke test.

- [ ] **Step 4: Final commit if fixes were needed**

```bash
git add image_policy tests policies data/image_policy_editor
git commit -m "Verify image policy editor vertical slice"
```

Skip this commit when verification required no code or fixture changes.

## Self-Review

- Spec coverage: Upload, region marking, natural-language routing, policy execution, trace capture, low-confidence routing, and the two starter policies are each covered by tasks.
- Placeholder scan: The plan intentionally uses a mock redraw backend as the first implementation, not an unspecified backend.
- Type consistency: `ImageRegion`, `RouterResult`, `TraceStore`, and `execute_policy` signatures are consistent across tests, server, and policy modules.
