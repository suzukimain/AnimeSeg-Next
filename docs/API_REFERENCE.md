# API リファレンス

## `anime_seg_next` 公開 API

### 最上位インポート

```python
from anime_seg_next import (
    AnimeSegNextPipeline,
    AnimeSegOutput,
    SERIES_CLASS_MAP,
    build_semantic_colors,
)
```

---

## `AnimeSegNextPipeline`

### クラス定義

```python
class AnimeSegNextPipeline(Mask2FormerAnimeSegPipeline):
    """Mask2Former ベースのアニメセグメンテーション・パイプライン
    
    事前学習済みモデルから推論を行い、AnimeSegOutput を返します。
    """
```

### クラスメソッド

#### `from_pretrained(model_id, **kwargs) -> AnimeSegNextPipeline`

事前学習済みモデルをロード・初期化します。

**パラメータ:**
- `model_id: str` — Hugging Face モデルリポジトリID
  - 例: `"suzukimain/AnimeSeg-Next"`
- `**kwargs` — 親クラス `Mask2FormerAnimeSegPipeline.from_pretrained()` に渡される追加引数
  - `device: str` — `"cpu"`, `"cuda"`, `"cuda:0"` など
  - `dtype: torch.dtype` — `torch.float32`, `torch.float16` など

**戻り値:**
- `AnimeSegNextPipeline` インスタンス

**例:**

```python
from anime_seg_next import AnimeSegNextPipeline

# 標準ロード（CPU）
pipeline = AnimeSegNextPipeline.from_pretrained("suzukimain/AnimeSeg-Next")

# GPU ロード
pipeline = AnimeSegNextPipeline.from_pretrained(
    "suzukimain/AnimeSeg-Next",
    device="cuda:0"
)
```

### インスタンスメソッド

#### `__call__(image, keep_source=True, **kwargs) -> AnimeSegOutput`

推論を実行してセグメンテーション結果を返します。

**パラメータ:**
- `image: Image.Image` — PIL Image（RGB または RGBA）
- `keep_source: bool = True` — ソース画像を保持
  - `True`: `overlay_map` の遅延生成に使用可能
  - `False`: メモリ節約（overlay_map はエラーになる）
- `**kwargs` — 親クラスのオプション引数

**戻り値:**
- `AnimeSegOutput` — セグメンテーション結果オブジェクト

**例:**

```python
from PIL import Image
from anime_seg_next import AnimeSegNextPipeline

pipeline = AnimeSegNextPipeline.from_pretrained("suzukimain/AnimeSeg-Next")
image = Image.open("character.png")

# 標準的な使用
output = pipeline(image)

# メモリ効率重視
output_no_source = pipeline(image, keep_source=False)
```

#### `to(device) -> AnimeSegNextPipeline`

パイプラインを指定デバイスに移動します。メソッドチェーン対応。

**パラメータ:**
- `device: str | torch.device` — `"cpu"`, `"cuda"`, など

**戻り値:**
- `AnimeSegNextPipeline` — self（チェーン用）

**例:**

```python
pipeline = (AnimeSegNextPipeline
    .from_pretrained("suzukimain/AnimeSeg-Next")
    .to("cuda:0"))
```

---

## `AnimeSegOutput`

### クラス定義

```python
@dataclass
class AnimeSegOutput:
    """セグメンテーション結果
    
    セグメンテーション map、色付けされたマスク、クラス定義を含みます。
    """
```

### 属性

#### `segmentation_map: np.ndarray`

- **型**: `np.ndarray[H, W]` (int32)
- **説明**: H×W のセグメンテーション ID マップ
  - 各ピクセルのクラス ID (0 ～ num_classes-1)
- **例**:
  ```python
  output.segmentation_map.shape  # (480, 640)
  output.segmentation_map[100, 50]  # 15 (face クラスID)
  ```

#### `color_map: Image.Image`

- **型**: `PIL.Image.Image` (RGB)
- **説明**: カラー化されたセグメンテーション結果
  - segmentation_map をクラス別に色付け
  - H×W の RGB 画像
- **例**:
  ```python
  output.color_map.save("segmentation_colored.png")
  ```

#### `class_names: List[str]`

- **型**: `List[str]`
- **説明**: クラス ID → クラス名のマッピング
  - `class_names[i]` は クラス ID `i` の名前
- **例**:
  ```python
  output.class_names[15]  # "face"
  output.class_names  # ['background', 'back_hair', ..., 'wings']
  ```

#### `id_to_color: Dict[int, Tuple[int, int, int]]`

- **型**: `Dict[int, Tuple[int, int, int]]`
- **説明**: クラス ID → RGB 色のマッピング
- **例**:
  ```python
  output.id_to_color[15]  # (242, 100, 120) # face の色
  ```

### プロパティ

#### `num_classes: int`

- **戻り値**: クラス数（通常 31, 37, または 12）
- **例**:
  ```python
  if output.num_classes == 31:
      print("next-v1 シリーズ")
  ```

#### `overlay_map: Image.Image`

- **戻り値**: `PIL.Image.Image` (RGB)
- **説明**: ソース画像と color_map の 60/40 ブレンド
  - ソース画像が 60%, color_map が 40%
  - 遅延計算（初回アクセス時に生成）
- **例**:
  ```python
  overlay = output.overlay_map
  overlay.save("overlay.png")
  ```
- **エラー**:
  - `RuntimeError`: ソース画像が保持されていない場合
    - 解決: `pipeline(image, keep_source=True)` を使用

### メソッド

#### `class_name_at(row: int, col: int) -> str`

座標 (row, col) のピクセルのクラス名を取得します。

**パラメータ:**
- `row: int` — 行インデックス (0 ～ H-1)
- `col: int` — 列インデックス (0 ～ W-1)

**戻り値:**
- `str` — クラス名（見つからない場合は `"class_N"`）

**例:**

```python
class_at_100_50 = output.class_name_at(100, 50)  # "face"
```

#### `class_mask(class_name_or_id: str | int) -> np.ndarray`

単一クラスのブール型マスクを取得します。

**パラメータ:**
- `class_name_or_id: str | int` — クラス名またはクラス ID
  - 文字列: `"face"`, `"back_hair"`
  - 整数: `15`, `1`

**戻り値:**
- `np.ndarray[H, W]` (bool) — `True` のピクセルが該当クラス

**例:**

```python
# クラス名で指定
face_mask = output.class_mask("face")
hair_mask = output.class_mask("back_hair")

# クラス ID で指定
hair_mask_alt = output.class_mask(1)

# マスク操作
import numpy as np
face_pixels = output.color_map.crop((0, 0, *output.color_map.size))
face_pixels[~face_mask] = 0  # 非 face を黒くする
```

#### `present_classes() -> List[str]`

セグメンテーション結果に実際に存在するクラスのリストを取得します。

**戻り値:**
- `List[str]` — 存在するクラス名（出現順）

**例:**

```python
classes_found = output.present_classes()
# ['background', 'face', 'back_hair', 'front_hair', 'topwear', ...]

if "wings" not in classes_found:
    print("翼は検出されなかった")
```

#### `__repr__() -> str`

オブジェクトの文字列表現。

**例:**

```python
str(output)
# 'AnimeSegOutput(size=640×480, num_classes=31, present=[background, face, ...])'
```

---

## `SERIES_CLASS_MAP`

### 定義

```python
SERIES_CLASS_MAP: Dict[str, List[str]]
```

シリーズごとのクラス名リスト。

### キー

- `"next-v1"` — 31 クラス（標準スキーマ）
- `"next-v2"` — 37 クラス（詳細版）
- `"legacy-v1"` — 12 クラス（過去スキーマ）

### 例

```python
from anime_seg_next import SERIES_CLASS_MAP

# next-v1 のクラス一覧
classes_v1 = SERIES_CLASS_MAP["next-v1"]
print(classes_v1[15])  # "face"

# クラス数で判定
num = len(SERIES_CLASS_MAP["next-v1"])  # 31
```

### スキーマ

```python
SERIES_CLASS_MAP = {
    "next-v1": [
        "background",    # 0
        "back_hair",     # 1
        ...
        "wings",         # 30
    ],
    "next-v2": [...],
    "legacy-v1": [...],
}
```

詳細は [CLASSES.md](CLASSES.md) を参照してください。

---

## `build_semantic_colors(class_names)`

### 関数定義

```python
def build_semantic_colors(class_names: List[str]) -> Dict[int, Tuple[int, int, int]]:
    """クラス名から セマンティック RGB 色を生成"""
```

クラス名のパターンと意味論に基づいて、知覚的に区別しやすい RGB 色を生成します。

### パラメータ

- `class_names: List[str]` — クラス名リスト（順序は ID に対応）

### 戻り値

- `Dict[int, Tuple[int, int, int]]` — クラス ID → (R, G, B)

### 色生成ルール

1. **background** → (0, 0, 0)
2. **ハードコード override** → 意味的に重要なクラス
3. **左右対称** → _left と _right は補色関係
4. **接頭辞調整** → back_* は暗く、front_* は明るく
5. **その他** → 黄金比を使った均一分布

### 例

```python
from anime_seg_next import build_semantic_colors

colors = build_semantic_colors([
    "background",
    "face",
    "back_hair",
    "front_hair",
    "irides_left",
    "irides_right",
])

print(colors)
# {
#     0: (0, 0, 0),                  # background
#     1: (242, 100, 120),            # face (warm skin tone)
#     2: (80, 40, 60),               # back_hair (darkened)
#     3: (180, 120, 200),            # front_hair (brightened)
#     4: (100, 200, 235),            # irides_left (blue)
#     5: (235, 100, 200),            # irides_right (complementary)
# }

for class_id, rgb in colors.items():
    print(f"{class_id}: RGB{rgb}")
```

---

## Core モジュール（高度な使用法）

### `anime_seg_next.core.resolve_series`

```python
def resolve_series(config_obj: Dict, num_classes: int) -> Optional[str]:
    """config と num_classes からシリーズ名を解決"""
```

**用途**: カスタムモデル対応時に、自動的にシリーズを判定

**例:**

```python
from anime_seg_next.core import resolve_series

config = {"series": "next-v1"}
series = resolve_series(config, 31)  # "next-v1"

# config に series がない場合、num_classes で判定
series = resolve_series({}, 31)  # "next-v1"
```

---

## Types モジュール（型情報のみ）

```python
from anime_seg_next.types import AnimeSegOutput
```

`AnimeSegOutput` の型情報は `types` に格納。通常は最上位の `anime_seg_next` からインポートしてください。

---

## 完全な使用例

```python
from PIL import Image
from anime_seg_next import (
    AnimeSegNextPipeline,
    SERIES_CLASS_MAP,
    build_semantic_colors,
)

# 1. パイプラインのロード
pipeline = AnimeSegNextPipeline.from_pretrained("suzukimain/AnimeSeg-Next")

# 2. 推論
image = Image.open("character.png")
output = pipeline(image)

# 3. 結果の確認
print(f"検出クラス数: {output.num_classes}")
print(f"実際に検出されたクラス: {output.present_classes()}")

# 4. クラス別マスク抽出
face_mask = output.class_mask("face")
hair_mask = output.class_mask("back_hair")

# 5. ピクセルレベルのクエリ
pixel_class = output.class_name_at(100, 150)
print(f"座標 (100, 150) のクラス: {pixel_class}")

# 6. 結果を画像として保存
output.color_map.save("segmentation.png")
output.overlay_map.save("overlay.png")

# 7. カスタムシリーズ対応
series_key = "next-v1"
classes = SERIES_CLASS_MAP[series_key]
colors = build_semantic_colors(classes)
```

---

**詳細は [ARCHITECTURE.md](ARCHITECTURE.md) と [CLASSES.md](CLASSES.md) を参照してください。**
