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

## next-v2 / v4 (37 クラス)

最新の標準スキーマ。マルチタスク学習（深度推定など）に対応し、手足のパーツ分割が追加されています。

| ID | クラス名 | カテゴリ | RGB (0-255) | Hex | Sample |
|---|---|---|---|---|---|
| 0 | background | 背景 | (0, 0, 0) | #000000 | ![](https://via.placeholder.com/15/000000/000000?text=+) |
| 1 | back_hair | 髪 | (40, 20, 60) | #28143c | ![](https://via.placeholder.com/15/28143c/000000?text=+) |
| 2 | bottomwear | 衣装 | (0, 102, 204) | #0066cc | ![](https://via.placeholder.com/15/0066cc/000000?text=+) |
| 3 | ears_left | 体 | (100, 180, 220) | #64b4dc | ![](https://via.placeholder.com/15/64b4dc/000000?text=+) |
| 4 | ears_right | 体 | (220, 120, 80) | #dc7850 | ![](https://via.placeholder.com/15/dc7850/000000?text=+) |
| 5 | earwear_left | アクセ | (70, 120, 200) | #4678c8 | ![](https://via.placeholder.com/15/4678c8/000000?text=+) |
| 6 | earwear_right | アクセ | (220, 100, 40) | #dc6428 | ![](https://via.placeholder.com/15/dc6428/000000?text=+) |
| 7 | eyebrow_left | 顔 | (50, 100, 200) | #3264c8 | ![](https://via.placeholder.com/15/3264c8/000000?text=+) |
| 8 | eyebrow_right | 顔 | (200, 80, 30) | #c8501e | ![](https://via.placeholder.com/15/c8501e/000000?text=+) |
| 9 | eyelash_left | 顔 | (40, 80, 180) | #2850b4 | ![](https://via.placeholder.com/15/2850b4/000000?text=+) |
| 10 | eyelash_right | 顔 | (180, 60, 20) | #b43c14 | ![](https://via.placeholder.com/15/b43c14/000000?text=+) |
| 11 | eyewear_left | アクセ | (100, 160, 240) | #64a0f0 | ![](https://via.placeholder.com/15/64a0f0/000000?text=+) |
| 12 | eyewear_right | アクセ | (240, 140, 60) | #f08c3c | ![](https://via.placeholder.com/15/f08c3c/000000?text=+) |
| 13 | eyewhite_left | 顔 | (200, 240, 255) | #c8f0ff | ![](https://via.placeholder.com/15/c8f0ff/000000?text=+) |
| 14 | eyewhite_right | 顔 | (255, 240, 200) | #fff0c8 | ![](https://via.placeholder.com/15/fff0c8/000000?text=+) |
| 15 | face | 顔 | (100, 150, 255) | #6496ff | ![](https://via.placeholder.com/15/6496ff/000000?text=+) |
| 16 | footwear | 衣装 | (32, 64, 96) | #204060 | ![](https://via.placeholder.com/15/204060/000000?text=+) |
| 17 | front_hair | 髪 | (50, 30, 80) | #321e50 | ![](https://via.placeholder.com/15/321e50/000000?text=+) |
| 18 | handwear | アクセ | (192, 192, 192) | #c0c0c0 | ![](https://via.placeholder.com/15/c0c0c0/000000?text=+) |
| 19 | headwear | アクセ | (200, 100, 50) | #c86432 | ![](https://via.placeholder.com/15/c86432/000000?text=+) |
| 20 | irides_left | 顔 | (80, 140, 220) | #508cdc | ![](https://via.placeholder.com/15/508cdc/000000?text=+) |
| 21 | irides_right | 顔 | (220, 180, 80) | #dcb450 | ![](https://via.placeholder.com/15/dcb450/000000?text=+) |
| 22 | legwear | 衣装 | (204, 51, 102) | #cc3366 | ![](https://via.placeholder.com/15/cc3366/000000?text=+) |
| 23 | mouth | 顔 | (255, 0, 150) | #ff0096 | ![](https://via.placeholder.com/15/ff0096/000000?text=+) |
| 24 | neck | 体 | (210, 170, 140) | #d2aa8c | ![](https://via.placeholder.com/15/d2aa8c/000000?text=+) |
| 25 | neckwear | アクセ | (100, 100, 100) | #646464 | ![](https://via.placeholder.com/15/646464/000000?text=+) |
| 26 | nose | 顔 | (255, 140, 0) | #ff8c00 | ![](https://via.placeholder.com/15/ff8c00/000000?text=+) |
| 27 | objects | その他 | (128, 128, 128) | #808080 | ![](https://via.placeholder.com/15/808080/000000?text=+) |
| 28 | tail | 体 | (200, 50, 50) | #c83232 | ![](https://via.placeholder.com/15/c83232/000000?text=+) |
| 29 | topwear | 衣装 | (0, 128, 0) | #008000 | ![](https://via.placeholder.com/15/008000/000000?text=+) |
| 30 | wings | 体 | (255, 255, 0) | #ffff00 | ![](https://via.placeholder.com/15/ffff00/000000?text=+) |
| 31 | handwear_L | 拡張 | (235, 229, 106) | #ebe56a | ![](https://via.placeholder.com/15/ebe56a/000000?text=+) |
| 32 | handwear_R | 拡張 | (191, 106, 235) | #bf6aeb | ![](https://via.placeholder.com/15/bf6aeb/000000?text=+) |
| 33 | legwear_L | 拡張 | (106, 235, 153) | #6aeb99 | ![](https://via.placeholder.com/15/6aeb99/000000?text=+) |
| 34 | legwear_R | 拡張 | (235, 116, 106) | #eb746a | ![](https://via.placeholder.com/15/eb746a/000000?text=+) |
| 35 | footwear_L | 拡張 | (106, 133, 235) | #6a85eb | ![](https://via.placeholder.com/15/6a85eb/000000?text=+) |
| 36 | footwear_R | 拡張 | (171, 235, 106) | #abeb6a | ![](https://via.placeholder.com/15/abeb6a/000000?text=+) |

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
