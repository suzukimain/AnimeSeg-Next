# AnimeSeg-Next

**高精度アニメキャラ分割パイプライン** — Mask2Former ベースの 31/37 クラスセマンティック・セグメンテーション

## 概要

`anime_seg_next` は、アニメキャラクターの詳細なパーツセグメンテーションを行う Python ライブラリです。
30+ のクラス（髪、顔、衣装、アクセサリーなど）を自動認識し、セマンティック・カラーマップや座標クエリを提供します。

### 主な特徴

- **複数シリーズ対応**: `next-v1` (31 クラス), `next-v2` (37 クラス), `legacy-v1` (12 クラス)
- **セマンティック・カラー生成**: 左右対称クラスと意味論的な色を自動生成
- **リッチな出力型**: 遅延生成オーバーレイ、ピクセルレベルのクエリ機能
- **モデル自動検出**: 重みファイルから class count で シリーズを自動判定

## クイックスタート

```python
from anime_seg_next import AnimeSegNextPipeline, AnimeSegOutput

# パイプラインのロード（デフォルト: Hugging Face）
pipeline = AnimeSegNextPipeline.from_pretrained("suzukimain/AnimeSeg-Next")

# 推論
image = Image.open("character.png")
output: AnimeSegOutput = pipeline(image)

# 結果にアクセス
print(f"クラス数: {output.num_classes}")
print(f"検出されたクラス: {output.present_classes()}")

# 特定クラスのマスク取得
face_mask = output.class_mask("face")
hair_mask = output.class_mask("back_hair")

# 座標クエリ
class_name = output.class_name_at(row=100, col=150)

# オーバーレイ表示（ソース画像を背後に）
overlay = output.overlay_map
overlay.save("result_overlay.png")
```

## パッケージ構成

```
anime_seg_next/
  __init__.py           # 公開 API
  mask2former/
	__init__.py
	mask2former_pipeline.py  # AnimeSegNextPipeline 実装
  
  types/                # 型定義
	__init__.py
	output.py           # AnimeSegOutput クラス
  
  core/                 # コア機能
	__init__.py
	series.py           # シリーズメタデータ & セマンティック色生成
```

## API リファレンス

### AnimeSegNextPipeline

```python
class AnimeSegNextPipeline(Mask2FormerAnimeSegPipeline):
	"""Mask2Former ベースのアニメセグメンテーション・パイプライン"""
    
	@classmethod
	def from_pretrained(cls, model_id: str, **kwargs) -> AnimeSegNextPipeline:
		"""事前学習済みモデルをロード"""
    
	def __call__(
		self,
		image: Image.Image,
		keep_source: bool = True,
	) -> AnimeSegOutput:
		"""推論実行"""
```

### AnimeSegOutput

```python
@dataclass
class AnimeSegOutput:
	"""セグメンテーション結果オブジェクト"""
    
	segmentation_map: np.ndarray          # H×W int32 クラス ID
	color_map: Image.Image                # RGB カラーマップ
	class_names: List[str]                # クラス名リスト
	id_to_color: Dict[int, Tuple[...]]   # クラス ID → RGB
    
	@property
	def overlay_map(self) -> Image.Image:
		"""ソース画像に 60/40 ブレンドしたオーバーレイ"""
    
	def class_name_at(self, row: int, col: int) -> str:
		"""(row, col) のピクセルのクラス名を取得"""
    
	def class_mask(self, class_name_or_id: str | int) -> np.ndarray:
		"""単一クラスのブール型マスク取得"""
    
	def present_classes(self) -> List[str]:
		"""セグメンテーション内に実際に存在するクラスの名前リスト"""
```

### シリーズとクラス定義

#### next-v1 (31 クラス)

```python
from anime_seg_next import SERIES_CLASS_MAP

next_v1_classes = SERIES_CLASS_MAP["next-v1"]
# ['background', 'back_hair', 'bottomwear', 'ears_left', 'ears_right', ...]
```

- **顔**: face, nose, mouth, 眉毛, 目（白・瞳）
- **髪**: back_hair, front_hair
- **衣装**: topwear, bottomwear, footwear
- **アクセサリー**: headwear, neckwear, earwear, eyewear, handwear
- **その他**: body, tail, wings, objects

#### next-v2 (37 クラス)

next-v1 に加え以下の詳細クラスを追加：
- lips, teeth, tongue, blush, hair_highlight

## セマンティック色生成

色は以下のルールで自動生成されます：

1. **background** → 黒 (0, 0, 0)
2. **ハードコード上書き** → 意味的に重要なクラス（irides: 青など）
3. **左右対**: 同じ hue、right は補色 (+180°)
4. **back_ 接頭辞** → 暗くなる (val × 0.55)
5. **front_ 接頭辞** → 明るくなる (val × 1.0)
6. **その他** → 黄金比間隔 HSV

```python
from anime_seg_next import build_semantic_colors

colors = build_semantic_colors(["background", "face", "back_hair", "front_hair"])
# {0: (0, 0, 0), 1: (242, 100, 120), 2: (80, 40, 60), 3: (180, 120, 200), ...}
```
