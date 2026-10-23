import onnx
from onnx import numpy_helper
import numpy as np
from arena.manifest import load_manifest, EVENTS_DIR
from arena.policy import load_policy
from arena.runner import run_once

manifest = load_manifest(EVENTS_DIR / "sprint-2m.toml")

# Load leader model
model = onnx.load("C:/GitHub/MicroDuck/models/leader_jrubiosainz_0840.onnx")
inits = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
base_OFF = inits["OFF"].copy()

print("Testing scaling factor on OFF table:")
for scale in [0.95, 0.98, 0.99, 1.00, 1.005, 1.01, 1.02, 1.05]:
    new_inits = []
    for init in model.graph.initializer:
        if init.name == "OFF":
            new_init = numpy_helper.from_array(base_OFF * scale, name="OFF")
            new_inits.append(new_init)
        else:
            new_inits.append(init)
    model.graph.ClearField("initializer")
    model.graph.initializer.extend(new_inits)
    onnx.save(model, "C:/GitHub/MicroDuck/models/temp_test.onnx")
    
    policy = load_policy("C:/GitHub/MicroDuck/models/temp_test.onnx")
    times = []
    finishes = 0
    cmd = manifest.command_at(1.15)
    for s in [1, 2, 3, 4, 5]:
        out = run_once(manifest, policy, s, command=cmd)
        if out.status == "finish":
            times.append(out.time_s)
            finishes += 1
        else:
            times.append(50.0)
    mean_val = np.mean(times) if finishes == 5 else 50.0
    time_strs = [f"{t:.3f}" if t < 50 else "FALL" for t in times]
    print(f"Scale {scale:6.3f} -> Finishes {finishes}/5, Times: {time_strs}, Mean: {mean_val:.4f}s")
