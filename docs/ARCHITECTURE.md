# アーキテクチャ

## パッケージ構成

`anime_seg_next` は以下の構成で整理されています：

```
anime_seg_next/
├── __init__.py                      # 公開 API
├── mask2former/
│   ├── __init__.py
│   └── mask2former_pipeline.py     # パイプライン実装
├── types/                           # 出力型定義
│   ├── __init__.py
│   └── output.py                   # AnimeSegOutput クラス
└── core/                            # コアメタデータ
    ├── __init__.py
    └── series.py                   # シリーズ定義 & 色生成
```

### 各モジュールの役割

#### `types/output.py`

アニメセグメンテーション結果を表すリッチなデータクラス。

**責務:**
- セグメンテーションマップ（クラス ID 配列）の保持
- カラーマップ（RGB 画像）の保持
- ピクセルレベルのクエリ（座標 → クラス名）
- 遅延オーバーレイ生成（ソース画像とのブレンド）

**主要メンバー:**
- `segmentation_map`: H×W の int32 配列
- `color_map`: PIL Image（RGB）
- `class_names`: クラス名リスト
- `id_to_color`: クラス ID → (R, G, B) マッピング

#### `core/series.py`

シリーズメタデータとセマンティック色生成ロジック。

**責務:**
- クラス名定義の保持（next-v1, next-v2, legacy-v1）
- モデル重みファイルからシリーズ自動検出
- 意味論的なカラーマップ生成（左右対称、セマンティック hue）

**主要コンポーネント:**
- `SERIES_CLASS_MAP`: {シリーズ名 → クラス名リスト}
- `resolve_series()`: config と num_classes からシリーズ決定
- `build_semantic_colors()`: クラス名リストから RGB 色を生成

#### `mask2former/mask2former_pipeline.py`

Mask2Former ベースのセグメンテーション・パイプライン。

**責務:**
- 事前学習済みモデルのロード
- 推論実行
- `types.AnimeSegOutput` 生成

**継承構造:**
```
Mask2FormerAnimeSegPipeline (anime_seg より)
    ↓
AnimeSegNextPipeline (本パッケージ)
    ├─ Series 自動検出オーバーライド
    └─ AnimeSegOutput 返却
```

## デザイン原則

### 1. 関心の分離

- **types**: データ構造と query API のみ
- **core**: メタデータと色生成ロジック
- **mask2former**: 推論パイプライン

各モジュールは独立し、変更の波及を最小化します。

### 2. 遅延評価

`overlay_map` は使用時に初めて計算されます。ソース画像の保持は `keep_source=True` で制御できます。

### 3. セマンティックなカラー

色はクラス名から自動生成され、以下のルールに従います：

- **背景**: 黒（特殊ケース）
- **体部**: 暖色（肌トーン）
- **髪**: 明暗対比（back_hair は暗く、front_hair は明るく）
- **左右対**: hue は同じ、right は +180°
- **その他**: 黄金比を使った均一分布

### 4. 複数シリーズ対応

モデルのクラス数から自動的にシリーズを判定し、対応するクラス名を割り当てます。
config に明示的に `series` を指定することも可能。

## 使用フロー

```
1. AnimeSegNextPipeline.from_pretrained() 
   ↓
2. パイプライン内部で series 自動検出
   ├─ resolve_series() でシリーズ確定
   └─ SERIES_CLASS_MAP[series] からクラス名取得
   ↓
3. pipeline(image) で推論
   ├─ mask2former で segmentation_map 生成
   ├─ build_semantic_colors() で color_map 生成
   └─ AnimeSegOutput 返却
   ↓
4. ユーザーが result にクエリ
   ├─ output.class_mask("face")
   ├─ output.overlay_map
   └─ output.class_name_at(r, c)
```

## 拡張ポイント

### シリーズの追加

`core/series.py` の `SERIES_CLASS_MAP` に新しいシリーズを追加：

```python
SERIES_CLASS_MAP["next-v3"] = ["background", "face", ...]
```

### 色生成ルールのカスタマイズ

`core/series.py` の `_SEMANTIC_HUE_OVERRIDES` または `build_semantic_colors()` ロジックを修正。

### パイプラインのカスタマイズ

`mask2former/mask2former_pipeline.py` で以下をオーバーライド：
- `_resolve_class_names()`
- `__call__()` の前後処理

## 今後の改善案

- [ ] テスト・スイート追加（types, core, 統合テスト）
- [ ] 性能ベンチマーク
- [ ] Batch 処理の最適化
- [ ] ONNX/TensorRT エクスポート
