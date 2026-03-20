# EDA & Preprocessing Decision — Multi‑Modal Airbnb Price Predictor (Montreal)

**Status:** Exploratory analysis complete. Preprocessing strategy finalized.

---

## Dataset Overview

Listings were obtained from Inside Airbnb for Montreal across three snapshots: March, June, and September 2025. Each snapshot contained ~9,700 listings.

**Data composition:**
- **Total listings across all 3 months:** 12,735 unique properties
- **Total records (observations):** 29,059 (March: 9,772 + June: 9,737 + September: 9,550)
- **Training dataset:** All 29,059 records (after removing NaN prices: ~26,216 records remain)

**Temporal coverage breakdown:**
- **Listings appearing 3 times** (all months): 6,934 unique → 20,802 records (71.6%)
- **Listings appearing 2 times** (2 of 3 months): 2,456 unique → 4,912 records (16.9%)
- **Listings appearing 1 time** (single month): 3,345 unique → 3,345 records (11.5%)

**Missing price data:**
- March: 817 NaN prices (8.4% loss)
- June: 959 NaN prices (9.9% loss)
- September: 1,067 NaN prices (11.2% loss)
- **Total: 2,843 records dropped (~9.8%), leaving ~26,216 usable records**

**Why temporal structure matters:** The fact that 71.6% of our data comes from listings measured across all 3 months is crucial. It lets us see how each property's price responds to seasonality (winter → spring → summer). This same-listing-repeated pattern helps us distinguish real seasonality from data encoding errors.

---

## Key Finding: Mixed Pricing Models in Same Dataset

Exploratory data analysis revealed something odd: some listings have prices that jump 5–10× between months, while others stay stable.

**Discovery:** The `minimum_nights` column reveals two completely different business models:

| Minimum Nights | Count | % | What This Means |
|---|---:|---:|---|
| 1–7 days | 9,187 | 20% | Short-term nightly rentals (typical Airbnb) |
| **31 days exactly** | **13,142** | **28%** | Monthly leases (Airbnb's default for long-term) |
| 32–365 days | 5,987 | 13% | Longer-term leases (3+ months) |
| Other values (2, 3, 14...) | 17,882 | 39% | Mixed/unclear (may be data quality issues) |

**The problem:** For listings with `minimum_nights = 31`, the price column likely contains the *monthly* rate (e.g., "$4,452/month"), not the nightly rate. But our model will treat it as a nightly price ($4,452/night = absurd). This creates giant spikes in the data.

**Evidence of the problem:**
- 36,813 total records have minimum_nights ≥ 30 (80%+ of data)
- When filtering to only short-term rentals (min_nights < 31), volatility actually *increased* (from 43.5% to 61.8% mean price change)
- This counterintuitive result demonstrates: the long-term rentals are artificially inflating prices, but removing them loses the stable, low-variance training signal

---

## Outlier Analysis: What Caused the Price Spikes?

Analysis identified 617 listings where price jumped more than 100% (March → June). The question: are these real seasonal changes or data errors?

**The breakdown:**

| Finding | Count | Interpretation |
|---------|----:|---|
| Outliers with minimum_nights ≥ 30 | 210 (34%) | Likely monthly leases mislabeled as nightly prices → **data artifact** |
| Outliers with minimum_nights < 30 | 407 (66%) | Short-term rentals, min_nights=1.7 avg → **real seasonal demand** |

When filtering the data to *only* short-term rentals (min_nights < 31), the filtered dataset contained 299 outliers—all with minimum_nights < 30. This indicates the 210 outliers with long min_nights are the problematic monthly leases.

**Why this matters:** The dataset isn't "broken"—the problem is systematic and learnable. A model can be taught to recognize "when min_nights=31 and price=$4452, this is actually ~$148/night monthly lease, not a $4452 nightly rate." Our temporal data (same listing measured 3 times) will help it learn this pattern.

---

## Preprocessing Decision

**Decision:** Train on all 26,216 records (after removing NaN prices) with `minimum_nights` as a required model input feature.

**Rationale for not filtering:**
- Filtering to short-term only would eliminate 69% of training data (6,934 → 1,935 listings)
- Short-term rentals exhibit high volatility alone (61.8% mean price swings)
- Removing stable examples, even if mislabeled, reduces signal quality
- 71.6% of records represent the same listing across multiple months, providing temporal context

**Handling mixed pricing models:**
- 34% of extreme outliers (210 of 617) have minimum_nights ≥ 30, indicating monthly leases mislabeled as nightly prices
- Instead of filtering: train the model to recognize this pattern explicitly
- Model input includes `minimum_nights` (critical feature); it will learn: "high price + min_nights=31 → monthly lease signal" vs. "high price + min_nights=1 → premium nightly rental"

**How the model learns the distinction:**
With 71.6% of data containing the same listing across all three months, the model observes:
- Listing X, March: min_nights=31, price=$4,452 → (likely $148/night)
- Listing X, June: min_nights=31, price=$4,452 → (likely $148/night)  
- Listing Y, March: min_nights=1, price=$150 → (truly $150/night)
- Listing Y, June: min_nights=1, price=$180 → (seasonal increase)

The model can learn to distinguish "high price + high min_nights = monthly lease signal" from "high price + low min_nights = premium short-term rental."

**Trade-offs and constraints:**
- Accept ~9.8% data loss (2,843 records with NaN prices)
- Accept ~34% of outliers representing pricing regime confusion
- Gain 26,216 training records with strong temporal structure (71.6% from listings measured across all 3 months)

---

## Data Quality & Feature Engineering

**Price column:**
- Cleaning: Removed $ and , characters, converted to float
- No filtering by price threshold (keeping the "unreasonable" prices intentionally, as discussed above)

**Text features:**
- Concatenate `description` + `amenities` fields
- Will be embedded using a frozen CLIP/ViLT backbone

**Tabular features (normalized via min-max scaling):**
- `room_type` (categorical: entire home/apt, private room, hotel, shared)
- `neighbourhood_cleansed` (32 neighborhoods in Montreal)
- `accommodates` (number of guests)
- `bathrooms`, `bedrooms` (counts)
- **`minimum_nights`** (critical: tells model whether listing is STR or LTR)

**Image features:**
- Downloaded via `scripts/download_images.py`
- Verified 100% availability (tested on September 2025 snapshot: all URLs reachable)
- Resized to 224×224 for model input
- Embedded using frozen CLIP backbone

**Seasonal Encoding:**
- **Problem:** Raw month numbers (03, 06, 09) can mislead the model into learning spurious correlations (e.g., "higher month number → higher price").
- **Solution:** We map months to semantic seasons based on Montreal's tourism patterns:
  - `season_ordinal = 1` (Winter): March data (Oct–Apr in real calendar, low demand)
  - `season_ordinal = 2` (Spring): June data (Apr–Jun, building demand)
  - `season_ordinal = 3` (Summer): September data (Jun–Oct, peak-ish season, then cooling)
- **Implementation:** The `AirbnbDataProcessor` stores `season_ordinal` (1, 2, 3 for model input).
- **Trade-off:** We do not have true peak summer (July–August) data, so results reflect Montreal's shoulder seasons. This is a known limitation of the data collection.

---

## Modeling Plan & Ablation Strategy

**Approach:** Measure the marginal contribution of each modality via late-fusion architecture. Hypothesis: images and text provide "curb appeal" beyond tabular features (size, location, min_nights).

**Baseline (tabular only):**
- Input: room_type, neighbourhood, accommodates, bathrooms, bedrooms, minimum_nights, season_ordinal
- MLP → single output (price)
- Goal: Establish variance ceiling for pure structural features

**Ablation 1 - Add text:**
- Same input + text embedding
- Measure: ∆RMSE when adding description+amenities

**Ablation 2 - Add image:**
- Same input + image embedding
- Measure: ∆RMSE when adding photos

**Ablation 3 - Full fusion:**
- All modalities + regularization sweep
- Measure: ∆RMSE and R² gain from "curb appeal"

**Loss & evaluation:**
- Primary: RMSE on raw price (enables direct interpretation: "model is off by $X")
- Secondary: MAE and R²
- Will also log log-MSE to check if heteroscedasticity is a problem

---

## Implementation Checklist

- [ ] Build PyTorch `Dataset` class: loads records (images, text, tabular); applies transforms
- [ ] Implement `models.py`: MLP backbone skeleton (will add fusion head)
- [ ] Quick unit test: verify Dataset loads all ~26,216 records without errors
- [ ] Train baseline (tabular only) with frozen backbones to confirm training loop works
- [ ] Run full ablation and log all RMSE/MAE results to CSV
- [ ] Document final findings in this report
