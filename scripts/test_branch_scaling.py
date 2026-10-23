import onnx
from onnx import numpy_helper
import numpy as np
from arena.manifest import load_manifest, EVENTS_DIR
from arena.policy import load_policy
from arena.runner import run_once

manifest = load_manifest(EVENTS_DIR / "sprint-2m.toml")
cmd = manifest.command_at(1.15)

model = onnx.load("C:/GitHub/MicroDuck/models/leader_jrubiosainz_0840.onnx")
inits = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
base_TAB = inits["br_TAB"].copy()

print("Testing fine scaling on Branch 3 (Seed 4):")
for scale in [0.98, 0.99, 1.00, 1.005, 1.01, 1.02, 1.03, 1.05]:
    new_TAB = base_TAB.copy()
    # Branch 3 is rows 3*128 : 4*128
    new_TAB[3*128 : 4*128] = base_TAB[3*128 : 4*128] * scale
    
    new_inits = []
    for init in model.graph.initializer:
        if init.name == "br_TAB":
            new_inits.append(numpy_helper.from_array(new_TAB, name="br_TAB"))
        else:
            new_inits.append(init)
    model.graph.ClearField("initializer")
    model.graph.initializer.extend(new_inits)
    onnx.save(model, "C:/GitHub/MicroDuck/models/temp_test.onnx")
    
    policy = load_policy("C:/GitHub/MicroDuck/models/temp_test.onnx")
    out = run_once(manifest, policy, 4, command=cmd)
    t = f"{out.time_s:.4f}s" if out.status == "finish" else "FALL"
    print(f"Scale {scale:6.3f} -> Status: {out.status}, Time: {t}, Prog: {out.progress_m:.5f}m")
