# NYC 街廓量體資料集：目前進度

專案：Generating Urban Massing in Context from Text and Block Graphs（10-x23）
更新日期：2026-11-16（PLUTO 版本 26v2）

---

## 一句話總結

**全紐約五個行政區的資料集已經做完，可以直接拿來訓練。**
一行程式就能載入樣本：每個樣本是一塊 512 m 見方的地圖，裡面遮住 1–4 個街廓，
同時附上所有街廓的數值屬性和街廓圖（graph）。
這些正好對應 proposal 對 Liu 模型的三個修改。

```python
from dataset import NYCMassing
ds = NYCMassing("train")   # 或 "val" / "test"
x = ds[0]
```

---

## 進度總覽

| 項目 | 狀態 | 說明 |
|---|---|---|
| 下載 PLUTO、Tax Lots、Building Footprints | ✅ 完成 | 全市 858k 筆地塊、1.08M 棟建物 |
| 試做區（Park Slope、Midtown、LIC）清理與 QA | ✅ 完成 | 先前已做好；擴到全市後重跑，結果完全相同 |
| 全市地塊／建物清理（Stage 1–2） | ✅ 完成 | 依行政區分別處理 |
| 街廓多邊形與目標值（Stage 3） | ✅ 完成 | 覆蓋率、FAR、中位樓層、住宅比例 |
| 全市 4 m 類別地圖（Stage 4） | ✅ 完成 | 一張全市大圖，樣本從中裁切 |
| 社區區劃分 split、街廓圖、樣本（Stage 5） | ✅ 完成 | 85k 個樣本 |
| 資料載入器 `dataset.py` | ✅ 完成 | 讀一個樣本約 3 ms |
| 數值詞彙的分位數（例如「dense」） | ✅ 完成 | `quantiles.json`，給 LLM 步驟用 |
| 文字請求生成 | ⬜ 未開始 | Chenyue 負責 |
| 轉成 GlobalMapper 的輸入格式 | ⬜ 未開始 | baseline 需要 |
| 團隊決定 val split 是否重選 | ⚠️ 待決定 | 見「待討論事項」 |

---

## 資料數量

| 項目 | 數量 |
|---|---|
| 建物 | 1,080,944 棟（保留原始 footprint 的 99.8%） |
| 街廓 | 33,337 個；其中 30,866 個可當目標（valid） |
| 樣本 | 84,981 個 |
| 每個街廓出現的樣本數 | 中位數 6，最多 21 |

### 各行政區

| | 曼哈頓 MN | 布朗克斯 BX | 布魯克林 BK | 皇后區 QN | 史坦頓島 SI | 全市 |
|---|---:|---:|---:|---:|---:|---:|
| 建物 | 44,916 | 103,917 | 329,637 | 460,250 | 142,224 | 1,080,944 |
| 街廓 | 2,942 | 4,211 | 8,398 | 12,767 | 5,019 | 33,337 |
| 可用街廓 | 2,439 | 3,819 | 7,799 | 12,083 | 4,726 | 30,866 |
| 可用比例 | 83% | 91% | 93% | 95% | 94% | 93% |
| 每層樓高（從資料估計） | 3.7 m | 4.0 m | 3.9 m | 3.9 m | 4.1 m | |

### Split（依社區區 community district 切，測試區的街坊沒在訓練時看過）

| | 社區區 | 可用街廓 | 樣本 | 預設使用的樣本＊ |
|---|---|---:|---:|---:|
| train | 44 個 | 22,519 | 61,621 | 55,813 |
| val | 108, 206, 307, 308, 403 | 1,445 | 3,980 | 3,825 |
| test | 103, 110, 210, 211, 305, 311, 312, 405, 407, 502 | 6,902 | 19,380 | 19,380 |

＊有些訓練樣本的周邊脈絡裡會看到 val/test 的街廓，預設排除這些樣本，避免資訊洩漏。

### 一個樣本遮住幾個街廓（k）

| k | train | val | test | 合計 |
|---:|---:|---:|---:|---:|
| 1 | 22,175 | 1,436 | 6,813 | 30,424 |
| 2 | 16,898 | 1,114 | 5,278 | 23,290 |
| 3 | 13,568 | 873 | 4,329 | 18,770 |
| 4 | 8,980 | 557 | 2,960 | 12,497 |

---

## 品質檢查

| 檢查 | 結果 | Proposal 原本寫的 |
|---|---|---|
| 建物連到 PLUTO 地塊的比例 | ≈ 100% | 94% |
| 街廓 FAR：footprint 計算值 vs PLUTO（可用街廓） | r = 0.978 | 0.88 |
| 街廓覆蓋率：向量 vs 4 m 網格 | r = 0.998，平均誤差 0.010 | — |

圖：
- [qa/city/01_split_map.png](qa/city/01_split_map.png)：各街廓屬於哪個 split
- [qa/city/02_samples.png](qa/city/02_samples.png)：隨機樣本，包含類別圖、遮罩（洋紅色）和街廓圖
- [qa/city/03_target_dists.png](qa/city/03_target_dists.png)：各 split 的目標值分布
- 完整數字：[qa/city/summary.md](qa/city/summary.md)

---

## 一個樣本裡有什麼

| 欄位 | 形狀 | 意思 | 對應 proposal |
|---|---|---|---|
| `cls` | [128,128] | 10 類：街道、空地、住宅／非住宅 × 4 個高度級（1–2、3–5、6–12、13+ 層） | x |
| `mask` | [128,128] | 目標街廓的格子（模型要生成的區域） | M |
| `node_map` | [128,128] | 每格屬於哪個街廓節點，-1 表示街道 | 修改 2：把節點 embedding 寫進自己的街廓 |
| `node_attr` | [n,5] | 覆蓋率、FAR、高度級、住宅比例、log 面積 | 修改 1：數值屬性；目標街廓的部分就是 T |
| `node_target` | [n] | 哪些節點是目標 | |
| `edge_index` | [2,E] | 隔著街道相對的街廓之間的邊（對角街廓不相連） | 修改 1：邊 |
| `edge_attr` | [E,2] | 相對長度、街道寬度（m） | |
| `water` | [128,128] | 水域格 | |

訓練時目標街廓用真實值；測試時把目標列換成 LLM 解析出來的表格。

---

## 資料夾結構

```
00b_download_city.py   下載全市資料並依行政區切開
01_lots.py             地塊 + PLUTO 清理與連結
02_buildings.py        建物：過濾、連到 PLUTO、樓層、用途
03_blocks.py           街廓多邊形 + 目標值 + 有效性檢查
04b_mosaic_city.py     全市 4 m 類別地圖
05_samples.py          split、街廓圖、目標群組、樣本視窗
dataset.py             載入器（numpy；有 torch 時也提供 NYCMassingTorch）
qa_city.py             全市 QA 報告
run_city.sh            一鍵重建：SKIP_DOWNLOAD=1 bash run_city.sh（約 30 分鐘）
config.py              所有可調參數；[TEAM DECISION] 標記的是待團隊決定的項目

data/processed/city/   ← 訓練要用的都在這裡（約 875 MB）
  cls.npy, block.npy, water.npy   全市地圖
  blocks.parquet                  街廓表（屬性、split、有效性）
  edges.parquet                   街廓圖的邊
  samples.parquet                 樣本清單
  splits.json, quantiles.json, grid.json
```

---

## 待討論事項

1. **val split 偏小，而且比較密。** 只有 1,445 個街廓，FAR 中位數 1.57，train 和 test 約 0.8。
   可以手動挑社區區，或改 `05_samples.py` 裡的 `SEED`。建議訓練前決定。
2. **27% 的候選群組被丟掉**，因為放不進 512 m 視窗（還要留 48 m 周邊脈絡）。
   被丟的多是皇后區、史坦頓島的大街廓，所以大街廓的樣本偏少。
3. **皇后區和史坦頓島的 footprint FAR 比 PLUTO 高約 23–26%。** 推測是斜屋頂的屋頂高度
   多換算出一層樓，還沒驗證。訓練不受影響，因為目標值和地圖用同一套算法；
   但跟 PLUTO 比較時要注意。
4. **Proposal 的數字要更新**：街廓數（約 28,800 → 33,337，被街道切開的 tax block
   會分成多塊）、連結率和 FAR 相關係數（見「品質檢查」）。
5. **其他 `[TEAM DECISION]` 設定**（`config.py`）：混合用途怎麼算、車庫和施工中建物
   要不要納入、FAR 不一致的街廓要不要排除。目前都用預設值。

---

## 下一步

- Chenyue：用 `blocks.parquet` 的真實值和 `quantiles.json` 生成文字請求
- James：用 `dataset.py` 接 Liu 模型的 2D stage，先在幾個 tile 上試 overfit
- 轉一份 GlobalMapper 格式，給 baseline 用
