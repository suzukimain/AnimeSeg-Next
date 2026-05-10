# クラス仕様書

## next-v1 (31 クラス)

世代1の標準スキーマ — 詳細なアニメキャラ分割用。

| ID  | クラス名        | カテゴリ      | 説明                           |
|-----|-----------------|--------------|--------------------------------|
| 0   | background      | 背景         | 背景ピクセル                   |
| 1   | back_hair       | 髪          | 後ろ髪（暗く描画）            |
| 2   | bottomwear      | 衣装         | ボトムス、スカート、ズボン    |
| 3   | ears_left       | 体          | 左耳                           |
| 4   | ears_right      | 体          | 右耳                           |
| 5   | earwear_left    | アクセ      | 左耳飾り                       |
| 6   | earwear_right   | アクセ      | 右耳飾り                       |
| 7   | eyebrow_left    | 顔          | 左眉                           |
| 8   | eyebrow_right   | 顔          | 右眉                           |
| 9   | eyelash_left    | 顔          | 左まつ毛                       |
| 10  | eyelash_right   | 顔          | 右まつ毛                       |
| 11  | eyewear_left    | アクセ      | 左眼鏡                         |
| 12  | eyewear_right   | アクセ      | 右眼鏡                         |
| 13  | eyewhite_left   | 顔          | 左の白目                       |
| 14  | eyewhite_right  | 顔          | 右の白目                       |
| 15  | face            | 顔          | 顔（肌）                       |
| 16  | footwear        | 衣装         | 靴、足首飾り                   |
| 17  | front_hair      | 髪          | 前髪（明るく描画）            |
| 18  | handwear        | アクセ      | グローブ、腕輪                 |
| 19  | headwear        | アクセ      | 帽子、ヘッドバンド             |
| 20  | irides_left     | 顔          | 左の瞳（虹彩）                |
| 21  | irides_right    | 顔          | 右の瞳（虹彩）                |
| 22  | legwear         | 衣装         | ニーハイ、ストッキング         |
| 23  | mouth           | 顔          | 口（輪郭）                     |
| 24  | neck            | 体          | 首                             |
| 25  | neckwear        | アクセ      | ネックレス、スカーフ           |
| 26  | nose            | 顔          | 鼻                             |
| 27  | objects         | その他      | 楽器、武器、その他小道具      |
| 28  | tail            | 体          | しっぽ                         |
| 29  | topwear         | 衣装         | トップス、ジャケット           |
| 30  | wings           | 体          | 羽（天使・悪魔など）          |

### カテゴリ別統計

- **顔** (9): eyebrow_*, eyelash_*, eyewhite_*, irides_*, face, mouth, nose
- **髪** (2): back_hair, front_hair
- **衣装** (4): topwear, bottomwear, legwear, footwear
- **アクセ** (7): earwear_*, eyewear_*, headwear, neckwear, handwear
- **体** (6): ears_*, neck, tail, wings
- **その他** (2): background, objects

## next-v2 (37 クラス)

拡張版 — next-v1 に詳細クラスを追加。

### next-v1 に対する追加クラス

| ID  | クラス名      | 説明                    |
|-----|----------------|------------------------|
| 31  | body           | 露出肌（腕・胴体）      |
| 32  | lips           | 唇                      |
| 33  | teeth          | 歯                      |
| 34  | tongue         | 舌                      |
| 35  | blush          | チーク・赤らみ          |
| 36  | hair_highlight | 髪の光沢・ハイライト    |

### 移行ガイド

- next-v1 のモデルを使用している場合、クラス 0-30 は同じ意味
- next-v2 を fine-tune する際、クラス 31-36 をアノテーションに追加
- クラス 31+ を自動生成する場合、back_hair を分割してハイライトを抽出

## legacy-v1 (12 クラス)

過去の早期学習スキーマ — 互換性のために保持。

| ID  | クラス名       | 対応 (next-v1)                      |
|-----|-----------------|-----------------------------------|
| 0   | background      | background                         |
| 1   | skin            | face, neck, ears_*, body           |
| 2   | clothes_top     | topwear, neckwear                  |
| 3   | clothes_bottom  | bottomwear, legwear, footwear     |
| 4   | hair_front      | front_hair                         |
| 5   | hair_back       | back_hair                          |
| 6   | face            | face (概要のみ)                     |
| 7   | eyes            | eyewhite_*, irides_*               |
| 8   | mouth           | mouth                              |
| 9   | accessories     | headwear, earwear_*, eyewear_*     |
| 10  | other_clothes   | handwear, neckwear                 |
| 11  | accessory       | objects                            |

**注**: legacy-v1 への新規使用は推奨されません。next-v1 への移行を検討してください。

## セマンティック色割り当て

各クラスは自動的に RGB 色が割り当てられます。

### ルール

1. **background**: 黒 (0, 0, 0)
2. **ハードコード override** → `_SEMANTIC_HUE_OVERRIDES`:
   - face, neck, body: 暖色肌トーン (hue ≈ 0.07)
   - irides (デフォルト): 鮮やかな青 (hue ≈ 0.58)
   - lips, blush, tongue: ピンク・赤系
   - teeth: ほぼ白
3. **左右対称**: _left と _right は同じ hue、_right は +180° (補色)
4. **接頭辞による調整**:
   - back_*: val を 0.55 倍（暗く）
   - front_*: val を 1.0 倍（明るく）
5. **その他**: 黄金比 (0.618...) を使った HSV 均一分布

### 例

```python
from anime_seg_next import build_semantic_colors

colors = build_semantic_colors([
    "background",
    "face",
    "back_hair",
    "front_hair",
    "irides_left",
    "irides_right"
])

# {
#     0: (0, 0, 0),                    # background: 黒
#     1: (242, 100, 120),              # face: 暖色肌
#     2: (80, 40, 60),                 # back_hair: 暗い（back_*）
#     3: (180, 120, 200),              # front_hair: 明るい（front_*）
#     4: (100, 200, 235),              # irides_left: 青
#     5: (235, 100, 200),              # irides_right: 補色（ピンク系）
# }
```

## クラス使用パターン

### 顔部分の詳細抽出

```python
face_classes = ["face", "nose", "mouth", "eyewhite_left", "eyewhite_right",
                "irides_left", "irides_right", "eyebrow_left", "eyebrow_right",
                "eyelash_left", "eyelash_right"]
face_mask = np.any([output.class_mask(c) for c in face_classes], axis=0)
```

### 髪の一括処理

```python
hair_classes = ["back_hair", "front_hair"]
if "hair_highlight" in output.class_names:
    hair_classes.append("hair_highlight")
hair_mask = np.any([output.class_mask(c) for c in hair_classes], axis=0)
```

### 衣装のみの抽出

```python
clothing_mask = np.any([
    output.class_mask(c) for c in 
    ["topwear", "bottomwear", "legwear", "footwear"]
], axis=0)
```

## トラブルシューティング

### Q: クラスが不完全に検出される

**A**: モデルの学習データによります。本番環境では以下を検討：
- 異なるモデルのアンサンブル
- クラス別の後処理（形態学的フィルタリング）
- コンテキスト情報の利用

### Q: カラーマップが予期しない色になった

**A**: `build_semantic_colors()` のハードコード override を確認：
```python
from anime_seg_next.core.series import _SEMANTIC_HUE_OVERRIDES
print(_SEMANTIC_HUE_OVERRIDES)
```

カスタム色を指定したい場合は、`build_semantic_colors()` の result を post-process してください。

### Q: 新しいクラスを追加したい

**A**: `SERIES_CLASS_MAP` に新シリーズを追加：
```python
SERIES_CLASS_MAP["custom"] = ["background", "face", "new_class_1", ...]
```

その後、モデルを fine-tune または該当する config で `series: "custom"` を指定。
