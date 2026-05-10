# AnimeSeg-Next

<p>
		<img src="https://visitor-badge.laobi.icu/badge?page_id=suzukimain.AnimeSeg-Next" alt="Visitor Badge">
</p>

Mask2Former をベースにしたアニメキャラクターのセマンティック・セグメンテーションライブラリです。
`next-v1` / `next-v2` / `legacy-v1` のクラススキーマをサポートし、セマンティックな色付けと遅延オーバーレイ生成を提供します。

## sample image

<!-- 必要ならここに実際のサンプル画像を追加してください -->

## Installation

```bash
pip install -e .
```

## Usage

```python
from anime_seg_next import AnimeSegNextPipeline

pipe = AnimeSegNextPipeline.from_mask2former().to("cuda")
mask = pipe("path/to/image.jpg")
mask.save("output.png")

overlay = pipe("path/to/image.jpg", output_overlay=True)
overlay.overlay_map.save("overlay.png")
```

### AnimeSeg-Next

```python
from anime_seg_next import AnimeSegNextPipeline

pipe = AnimeSegNextPipeline.from_mask2former().to("cuda")
mask = pipe("path/to/image.jpg")
mask.save("output.png")
```

If the Hugging Face repo is private or gated, pass `token="hf_..."` or set `HF_TOKEN`.
`AnimeSegNextPipeline()` の直接呼び出しよりも `from_mask2former()` の利用を推奨します。

## Optional: output size

```python
# Same as input size (default)
mask_same = pipe("path/to/image.jpg")

# Fixed output size
mask_fixed = pipe("path/to/image.jpg", width=1024, height=1024)

# Width/height can be specified independently
mask_w = pipe("path/to/image.jpg", width=1024)
mask_h = pipe("path/to/image.jpg", height=1024)
```

## Advanced Usage

```python
from PIL import Image

img = Image.open("image.jpg")
mask = pipe(img)
mask.save("output.png")

# Overlay is available when keep_source=True (default)
result = pipe(img, keep_source=True)
result.overlay_map.save("overlay.png")
```

## Model Files

Models should follow the naming convention:

```
models/anime_seg_next_{architecture}_v{version}.safetensors
```

Example:
- `models/anime_seg_next_mask2former_v1.safetensors`

For AnimeSeg-Next, store Mask2Former metadata in `config.json` under `Config` with `num_classes`, `class_names`, and optionally `class_colors`.

```json
{
		"models": [
				{
						"FilePath": "models/anime_seg_next_mask2former_v1.safetensors",
						"TrainImageSize": 768,
						"Version": 1,
						"Architecture": "mask2former",
						"BaseModel": "facebook/mask2former-swin-large-ade-semantic",
						"Config": {
								"merged_full": true,
								"series": "next-v1",
								"num_classes": 31,
								"class_names": [
										"background",
										"back_hair",
										"bottomwear",
										"ears_left",
										"ears_right",
										"earwear_left",
										"earwear_right",
										"eyebrow_left",
										"eyebrow_right",
										"eyelash_left",
										"eyelash_right",
										"eyewear_left",
										"eyewear_right",
										"eyewhite_left",
										"eyewhite_right",
										"face",
										"footwear",
										"front_hair",
										"handwear",
										"headwear",
										"irides_left",
										"irides_right",
										"legwear",
										"mouth",
										"neck",
										"neckwear",
										"nose",
										"objects",
										"tail",
										"topwear",
										"wings"
								]
						}
				}
		]
}
```

If `num_classes` is omitted, the loader can infer it from the checkpoint's class head. If `class_colors` is omitted, it is generated deterministically.

## Segmentation Classes and Mask Colors

`next-v1` returns **31 classes** and `next-v2` returns **37 classes**.

### next-v1 highlights

- Face: `face`, `nose`, `mouth`, `eyebrow_left`, `eyebrow_right`, `eyelash_left`, `eyelash_right`
- Hair: `back_hair`, `front_hair`
- Clothes: `topwear`, `bottomwear`, `legwear`, `footwear`
- Accessories: `headwear`, `neckwear`, `earwear_left`, `earwear_right`, `eyewear_left`, `eyewear_right`, `handwear`

### next-v2 additions

- `body`, `lips`, `teeth`, `tongue`, `blush`, `hair_highlight`

## Troubleshooting

### `RuntimeError: "source image not available"`

`overlay_map` を使うには、`keep_source=True` のまま推論してください。

```python
result = pipe("path/to/image.jpg", keep_source=True)
result.overlay_map.save("overlay.png")
```

### `KeyError: "Unknown class"`

クラス名が正確か確認してください。クラス一覧は `SERIES_CLASS_MAP` または `docs/CLASSES.md` を参照してください。

## 技術仕様

- **バックボーン**: Mask2Former
- **出力**: `AnimeSegOutput`
- **クラス解決**: `series` → `num_classes` → `class_names`
- **色生成**: セマンティック HSV + 左右対称ルール

## Reference

- [ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [CLASSES.md](docs/CLASSES.md)
- [API_REFERENCE.md](docs/API_REFERENCE.md)
