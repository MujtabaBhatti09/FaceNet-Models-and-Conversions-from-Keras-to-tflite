# 🧠 FaceNet Models — Keras → TFLite Conversions

> **Production-ready TFLite face embedding models for offline, on-device face recognition.**  
> Converted from H5 checkpoints with full `SELECT_TF_OPS` (Flex delegate) support.  
> Tested with React Native · fast-tflite · VisionCamera v4 · Android API 21+

---

## 📦 Repository Contents

```
.
├── notebooks/
│   └── GhostFaceNet_to_TFLite_v5.ipynb    # Colab notebook — H5 → TFLite conversion
│
├── models/
│   ├── ghostfacenet_float16.tflite         # FP16 weights, ~8-9 MB  ← recommended
│   └── ghostfacenet_dynamic_int8.tflite    # INT8 weights,   ~8 MB  ← widest device support
│
└── README.md
```

---

## 🏆 Model Reference

| Model | Input | Output | Size | Accuracy (LFW) | Use Case |
|---|---|---|---|---|---|
| `ghostfacenet_float16.tflite` | 112×112 RGB float32 | `Float32[512]` | ~8-9 MB | 99.78% | Mid/High-end Android, iOS |
| `ghostfacenet_dynamic_int8.tflite` | 112×112 RGB float32 | `Float32[512]` | ~8 MB | ~99.7% | Any Android API 21+ |

### Source Checkpoint

| Checkpoint | LFW Accuracy | Loss | Notes |
|---|---|---|---|
| `GhostFaceNet_W1.3_S1_ArcFace.h5` | **99.78%** | ArcFace | ✅ Used in this repo |
| `GhostFaceNet_W1.3_S2_ArcFace.h5` | 99.71% | ArcFace | — |
| `GhostFaceNet_W1.3_S1_CosFace.h5` | 99.75% | CosFace | — |

Original weights by [HamadYA/GhostFaceNets](https://github.com/HamadYA/GhostFaceNets).

---

## ⚠️ Important: Flex Delegate Requirement

GhostFaceNet's **ghost modules** use TF ops (`grouped Conv2D`, `GELU`, `extract_patches`)  
that are **not** in TFLite's standard builtin op set.

Both models are converted with `SELECT_TF_OPS` and **require the TFLite Flex delegate** at runtime.

> **Without it you will see:**  
> `TFLite: Failed to allocate memory for input/output tensors! Status: unresolved-ops`

---

## 📓 Conversion Notebook

**`GhostFaceNet_to_TFLite_v5.ipynb`** — Run in Google Colab (T4 GPU recommended)

### What it does

| Step | Description |
|---|---|
| **1** | Install deps (`tf_keras`, `deepface`, `huggingface_hub`) — restart runtime after |
| **2** | Imports + GPU check |
| **3** | Configuration (model choice, paths, dims) |
| **4** | Auto-download checkpoint (GitHub Releases → DeepFace → HuggingFace fallback chain) |
| **4b** | Manual fallback `wget` cell if Step 4 fails |
| **5** | Load H5 with `tf_keras` (legacy Keras 2.x) + sanity check |
| **6** | Export to SavedModel |
| **7** | Helper functions (`preprocess`, `run_tflite`, `cosine_sim`, `build_converter`) |
| **8** | **Convert** — generates `float16` + `dynamic_int8` variants |
| **9** | **Verify** — cosine similarity vs Keras baseline (🟢 >0.999 excellent) |
| **10** | Visualize accuracy + size chart |
| **11** | AMS integration test — enroll 3 synthetic employees, verify recognition |
| **12** | Summary + mobile integration instructions |

### Key converter config

```python
conv.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,   # ← required for GhostFaceNet ghost modules
]
conv._experimental_lower_tensor_list_ops = False
conv.experimental_new_converter = True
```

> **Model surgery was explicitly skipped** — `prepare_for_tflite()` causes a `3.68` max output diff  
> (model corruption) in TF 2.19. `SELECT_TF_OPS` is the correct solution.

### Expected output

```
──────────────────────────────────────────────────
🔄 [float16]
   ✅ ghostfacenet_float16.tflite (8.XX MB)

──────────────────────────────────────────────────
🔄 [dynamic_int8]
   ✅ ghostfacenet_dynamic_int8.tflite (8.11 MB)

==================================================
  2/2 variants converted
==================================================

Variant                 Cosine Sim        L2      MB  Status
──────────────────────────────────────────────────────────
float16                   0.999XXX   0.XXXXX    8.XX  🟢
dynamic_int8              0.999988   0.12045    8.11  🟢
```

---

## 📱 React Native Integration

### Stack

| Role | Package | Version |
|---|---|---|
| Camera | `react-native-vision-camera` | `4.x` |
| TFLite inference | `react-native-fast-tflite` | `1.x` |
| Frame resize | `vision-camera-resize-plugin` | `2.x` |
| Worklet bridge | `react-native-worklets-core` | `1.x` |
| Face detection | `react-native-vision-camera` built-in `useFaceDetector()` | v4 only |

---

### 1. Place model in assets

```
src/assets/ghostfacenet.tflite
```

Make sure `metro.config.js` bundles `.tflite` files:

```js
// metro.config.js
const { getDefaultConfig } = require('@react-native/metro-config');
const config = getDefaultConfig(__dirname);

config.resolver.assetExts.push('tflite');

module.exports = config;
```

---

### 2. Android — `android/app/build.gradle`

Add the Flex delegate dependency. **This is mandatory** — without it you get `unresolved-ops`.

```groovy
android {
    defaultConfig {
        ndk { abiFilters "arm64-v8a", "x86_64" }
    }
}

dependencies {
    implementation("com.facebook.react:react-android")

    // ── TFLite — Flex delegate required for GhostFaceNet SELECT_TF_OPS ──────
    implementation 'org.tensorflow:tensorflow-lite:2.14.0'
    implementation 'org.tensorflow:tensorflow-lite-select-tf-ops:2.14.0'  // ← fixes unresolved-ops
    implementation 'org.tensorflow:tensorflow-lite-gpu:2.14.0'
    implementation 'org.tensorflow:tensorflow-lite-gpu-delegate-plugin:0.4.4'
}
```

---

### 3. Load model

```ts
import { useTensorflowModel } from 'react-native-fast-tflite';

const model = useTensorflowModel(require('../../assets/ghostfacenet.tflite'));
// model.state: 'loading' | 'loaded' | 'error'
```

---

### 4. Preprocessing

GhostFaceNet expects **float32 input** normalized to `[-1, 1]`:

```ts
// vision-camera-resize-plugin — inside frame processor
const resized = resize(frame, {
  scale: { width: 112, height: 112 },
  crop: { x: cx, y: cy, width: cw, height: ch },
  pixelFormat: 'rgb',
  dataType: 'float32',   // ← float32, NOT uint8
});

// Manual normalization: pixel is already [0,1] from resize plugin
// Maps [0,1] → [-1, 1] matching GhostFaceNet training pipeline
const norm = new Float32Array(112 * 112 * 3);
for (let i = 0; i < resized.length; i++) {
  norm[i] = (resized[i] * 255 - 127.5) * 0.0078125;
}
```

---

### 5. Run inference (inside frame processor worklet)

```ts
const frameProcessor = useFrameProcessor((frame) => {
  'worklet';
  if (model.state !== 'loaded') return;

  const tflite = model.model;   // ← model.model, not model directly

  const faces = detectFaces(frame);
  if (!faces.length) return;

  // ... crop + resize + normalize (see above) ...

  const outputs = tflite.runSync([norm]);   // runSync inside worklet
  const rawOutput = outputs[0] as Float32Array;

  const arr: number[] = [];
  for (let i = 0; i < rawOutput.length; i++) arr.push(rawOutput[i]);

  runOnJS(setEmbedding)(l2Normalize(arr));
}, [model, detectFaces, resize]);
```

---

### 6. L2 normalize + cosine similarity

Always L2-normalize before storing **and** before comparing:

```ts
function l2Normalize(embedding: number[]): number[] {
  const norm = Math.sqrt(embedding.reduce((s, v) => s + v * v, 0));
  return norm === 0 ? embedding : embedding.map(v => v / norm);
}

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot; // Both are L2-normalized so dot == cosine similarity
}

// Thresholds (tuned for GhostFaceNet 512-dim)
const THRESHOLD_MATCH  = 0.60;  // above = same person
const THRESHOLD_STRICT = 0.70;  // above = high confidence
```

---

### 7. Multi-angle enrollment (recommended)

Capture 3–5 embeddings (front, slight left, slight right) and average them:

```ts
function buildCentroid(embeddings: number[][]): number[] {
  const avg = embeddings[0].map((_, i) =>
    embeddings.reduce((s, e) => s + e[i], 0) / embeddings.length
  );
  return l2Normalize(avg);
}
```

---

## 🔧 Model Config Reference

```ts
// config.ts
export const GHOSTFACENET_CONFIG = {
  modelFile:    'ghostfacenet.tflite',
  inputSize:    112,           // 112 × 112 px
  embeddingDim: 512,           // output dimensions
  dataType:     'float32',     // resize plugin dataType
  normalize:    (px: number) => (px * 255 - 127.5) * 0.0078125,
  threshold:    0.60,          // cosine similarity match threshold
} as const;
```

---

## ❓ Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `Status: unresolved-ops` | Missing Flex delegate on Android | Add `tensorflow-lite-select-tf-ops:2.14.0` to `build.gradle` |
| Output all zeros / NaN | Wrong normalization | Use `(pixel * 255 - 127.5) * 0.0078125` with `dataType: 'float32'` |
| `runSync is not a function` | Called `model.runSync` instead of `model.model.runSync` | Use `model.model.runSync(...)` inside worklets |
| `model.state === 'error'` silently | Wrong delegate for device | Try `'default'` delegate instead of `'gpu'` |
| Cosine sim always ~0.5 | Missing L2 normalize | Always normalize before storing AND before comparing |
| `Cannot find module '*.tflite'` | Metro not configured | Add `assetExts.push('tflite')` to `metro.config.js` |
| Worklet error on JS thread | Missing `'worklet'` directive | First line inside `useFrameProcessor` must be `'worklet';` |

---

## 🌍 Environment

| Tool | Version |
|---|---|
| Python | 3.12 |
| TensorFlow | 2.19.0 |
| tf_keras | 2.19.0 |
| React Native | 0.84+ |
| react-native-fast-tflite | 1.x |
| react-native-vision-camera | 4.x |
| Android min SDK | 21 |
| TFLite runtime | 2.14.0 |

---

## 📄 License

Model weights are released by [HamadYA](https://github.com/HamadYA/GhostFaceNets) under their original license.  
Conversion notebook and integration code in this repo are MIT licensed.

---

*Built for the [AMS — Offline-First Face Recognition Attendance System](https://github.com/MujtabaBhatti09) · React Native CLI · Android*
