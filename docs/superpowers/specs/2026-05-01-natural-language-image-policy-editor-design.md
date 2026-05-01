# Natural-Language Image Policy Editor Design

Date: 2026-05-01

## Goal

Build a first vertical slice of a natural-language image editor that turns real uploaded images into operator-policy training examples. The prototype should prove the loop:

```text
upload image -> mark region -> type instruction -> route to policy -> execute edit -> save trace
```

The first product should feel like a small editor, not a research notebook. The research value comes from the trace format and correction flow.

## Scope

The first prototype supports real uploaded raster images and two policy families:

- `increment_number`: change a marked numeric text region by an integer amount, preserving placement as much as possible.
- `redraw_region`: regenerate or restyle a marked region from a natural-language style phrase while preserving the surrounding image.

The first version will not train a custom model. It produces dataset-ready traces that can later train policy routing, region grounding, and operator execution.

## Core Experience

1. The user uploads an image.
2. The user types an instruction such as "make this number one higher" or "redraw this in pixel style".
3. The app asks the user to mark the affected region with a rectangle or brush mask.
4. The router converts the instruction and region into a structured policy call.
5. The app shows a compact interpretation card before execution:

```text
Policy: increment_number
Target: marked region
Arguments: amount = 1
Preserve: background, perspective, approximate style
```

6. The user runs the edit.
7. The app stores the before image, region, instruction, policy call, edit output, and feedback state as a trace.

If the router is uncertain, the app asks for a policy choice rather than guessing silently.

## Architecture

The system mirrors the existing operator-first spirit of the repository:

- `composer`: orchestrates a small DAG of edit operators.
- `policy_router`: maps natural language plus region metadata to a policy name and arguments.
- `image_region`: stores rectangle and mask annotations in image coordinates.
- `policies/image_editing/<policy_name>`: policy modules with a README, trace schema examples, and operator priors.
- `trace_store`: writes one JSON record per edit and keeps file references stable.
- `frontend`: provides upload, marking, instruction entry, interpretation card, execution, and feedback.

The policy router can start as deterministic heuristics with an LLM-shaped interface:

```json
{
  "policy": "increment_number",
  "confidence": 0.86,
  "args": {
    "amount": 1
  },
  "needs_user_confirmation": false
}
```

This keeps the prototype runnable without requiring a trained model.

## Policy Execution

### increment_number

Initial execution can be intentionally conservative:

1. Crop the marked region.
2. Run OCR or ask the user to confirm the detected value when OCR confidence is low.
3. Compute the replacement value.
4. Clear the marked region with a local fill.
5. Render the new number into the marked region.
6. Blend the result back into the image.

This policy prioritizes trace quality and clear failure modes over photorealistic perfection.

### redraw_region

Initial execution uses a local mock backend behind a pluggable interface:

1. Convert the marked rectangle or mask into a crop and alpha mask.
2. Normalize the style phrase from the instruction.
3. Generate a replacement region through the backend. The first backend applies a visible pixel-style or color-treatment mock so the UI and trace loop are testable without network access.
4. Composite the replacement into the original image.
5. Save the raw prompt, normalized style, mask, and output.

The backend boundary supports replacing the mock with a hosted image-editing API or a FLUX-based workflow without changing the router or trace schema.

## Trace Format

Each edit writes a JSON trace:

```json
{
  "trace_id": "20260501-000001",
  "created_at": "2026-05-01T00:00:00-06:00",
  "input_image": "data/images/input/20260501-000001.png",
  "output_image": "data/images/output/20260501-000001.png",
  "instruction": "make this number one higher",
  "region": {
    "type": "rectangle",
    "x": 120,
    "y": 48,
    "width": 96,
    "height": 54
  },
  "router": {
    "policy": "increment_number",
    "confidence": 0.86,
    "alternatives": ["replace_text"]
  },
  "policy_call": {
    "name": "increment_number",
    "args": {
      "amount": 1,
      "detected_value": "7",
      "replacement_value": "8"
    }
  },
  "operators": [
    "crop_region",
    "detect_number",
    "erase_region",
    "render_text",
    "blend_region"
  ],
  "feedback": {
    "status": "pending",
    "notes": ""
  }
}
```

Corrections append a linked correction trace. Failed attempts remain stored because they are useful negative examples.

## Error Handling

- If no image is loaded, disable marking and execution.
- If no region is marked, ask the user to mark the target region.
- If the router confidence is low, ask the user to choose between candidate policies.
- If `increment_number` cannot detect a value, ask the user to enter the current value.
- If a generation backend is unavailable, `redraw_region` uses the visible mock output and still saves the trace.
- If an edit fails, save a failed trace with the error message and no output image.

## Testing

Initial tests cover:

- Router classification for common instructions.
- Trace JSON validation.
- Region coordinate serialization.
- `increment_number` value calculation.
- Composer execution order for each policy.

Visual quality tests can wait until the rendering backend exists. The first acceptance test is that a user can complete one `increment_number` trace and one `redraw_region` trace from uploaded images.

## Decisions

- Start with uploaded real images only.
- Require region marking before execution.
- Support free-form natural language, but always resolve it into a known policy schema.
- Implement two policies first: one precise operator and one generative redraw operator.
- Treat trace capture as the durable product artifact.

## Implementation Planning Boundary

After this design is accepted, write an implementation plan for a small Python service plus a browser UI. The current repository is Python-heavy with only a minimal `package.json`, so this keeps the first build close to the existing codebase.
