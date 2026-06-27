# Architecture

How a style-transfer run flows through the package, from the Hydra entry point
to the written image. Each node names the module that owns it; GitHub renders
the diagram below natively (no build step). Keep it in sync by hand when the
pipeline changes — it is a map, not a generated artifact.

```mermaid
graph TD
    CLI["python -m style_transfer.run"] --> RUN["run.py · @hydra.main"]

    subgraph Config["Configuration (Hydra)"]
        GROUPS["configs/ groups<br/>backbone · data · transfer · runtime"]
        SCHEMA["config_schema.py<br/>typed Config (validated)"]
        GROUPS --> SCHEMA
    end
    RUN --> SCHEMA

    SCHEMA --> DEV["device.pick_device<br/>CUDA → MPS → CPU"]
    SCHEMA --> EXTRACT["models.build_extractor<br/>VGGFeatures / ResNetFeatures (frozen)"]
    SCHEMA --> LOAD["transforms.load_image<br/>content + style → tensors"]

    DEV -. device .-> EXTRACT
    DEV -. device .-> LOAD

    subgraph Optimize["transfer.run_style_transfer"]
        CAP["_capture targets<br/>content & style features"]
        OPT["_make_optimizer<br/>LBFGS / Adam"]
        LOOP["optimization loop<br/>only the image is trainable"]
        CAP --> LOOP
        OPT --> LOOP
    end

    EXTRACT --> CAP
    LOAD --> CAP

    subgraph Losses["losses.py · recomputed each step"]
        CL["content_loss<br/>(MSE on features)"]
        SL["style_loss<br/>(Gram matrix)"]
        TV["total_variation_loss"]
        TOTAL["total =<br/>cw·content + sw·style + tw·tv"]
        CL --> TOTAL
        SL --> TOTAL
        TV --> TOTAL
    end

    LOOP --> CL
    LOOP --> SL
    LOOP --> TV
    TOTAL -. backprop to image .-> LOOP

    LOOP --> RESULT["TransferResult<br/>image + loss history"]
    RESULT --> SAVE["transforms.save_image<br/>→ Hydra per-run output dir"]
```

## Reading the diagram

- **Entry & config** — [run.py](../src/style_transfer/run.py) is the Hydra entry
  point; `configs/` groups are composed and validated against the typed schema
  in [config_schema.py](../src/style_transfer/config_schema.py).
- **Setup** — the device is resolved once
  ([device.py](../src/style_transfer/device.py)), the frozen backbone is built
  ([models.py](../src/style_transfer/models.py)), and the content/style pair is
  loaded ([transforms.py](../src/style_transfer/transforms.py)).
- **Optimization** — [transfer.py](../src/style_transfer/transfer.py) captures
  the content/style targets once, then iterates: the generated image is the only
  trainable tensor, and each step recomputes the three losses
  ([losses.py](../src/style_transfer/losses.py)) and backpropagates into it.
- **Output** — the final image is written into Hydra's per-run output directory,
  so repeated runs and `--multirun` sweeps never collide.
