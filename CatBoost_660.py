import os, glob, warnings
import numpy as np
import pandas as pd
import polars as pl
import math
import catboost as cb

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ---- Paths & constants ----
DATA_ROOT  = "Your Path"
OUTPUT_DIR = "Your Path"
os.makedirs(OUTPUT_DIR, exist_ok=True)

STREAMING_COLLECT = True
K_FOLDS           = 5
TARGET_COL        = "clicked"
SUBMIT_LTR_PATH   = os.path.join(OUTPUT_DIR, "submission_ltr.csv")

ALPHA_UC = 200.0   # (user, content)
ALPHA_TC = 100.0   # (term, content)
ALPHA_C  = 50.0    # (content)

# ---- Logical file patterns (only those used by the pipeline) ----
LOGICAL_FILES = {
    "train_sessions": [
        "**/train_sessions.parquet",
        "**/*train*sessions*.parquet",
    ],
    "test_sessions": [
        "**/test_sessions.parquet",
        "**/*test*sessions*.parquet",
    ],
    "content_search_log": [
        "**/content/search_log.parquet",
        "**/content/*search*log*.parquet",
        "**/*content*search*log*.parquet",
    ],
    "content_metadata": [
        "**/content/metadata.parquet",
        "**/*content*metadata*.parquet",
    ],
    "content_price_rate": [
        "**/content/price_rate_review_data.parquet",
        "**/*price*rate*review*data*.parquet",
    ],
    "content_sitewide_log": [
        "**/content/sitewide_log.parquet",
        "**/*content*sitewide*log*.parquet"
    ],
    "content_top_terms_log": [
        "**/content/top_terms_log.parquet",
        "**/*content*top*terms*log*.parquet"
    ],
    "user_metadata": [
        "**/user/metadata.parquet",
        "**/*user*metadata*.parquet"
    ],
    "user_sitewide_log": [
        "**/user/sitewide_log.parquet",
        "**/*user*sitewide*log*.parquet"
    ],
    "user_top_terms_log": [
        "**/user/top_terms_log.parquet",
        "**/*user*top*terms*log*.parquet"
    ],
    "term_search_log": [
        "**/term/search_log.parquet",
        "**/*term*search*log*.parquet"
    ],
}

# ---- Helpers ----
def find_one(root, patterns):
    '''
	ENGLISH:
    •	Does: Scans root with multiple glob patterns and returns the first matching file path (sorted for determinism).
	•	Inputs: root: str, patterns: list[str].
	•	Returns: str | None.
	•	Notes: Stops at the first non-empty match list.

  	TURKISH:
   	•	Ne yapar: Bir kök klasörde, verilen çoklu glob pattern’lerinden ilk eşleşen dosya yolunu döndürür.
	•	Girdi: root (str), patterns (list[str]).
	•	Çıktı: Yol (str) ya da None.
	•	Not: İlk bulunanı döndürür (deterministik olması için sort var).
	'''
    for pat in patterns:
        paths = sorted(glob.glob(os.path.join(root, pat), recursive=True))
        if paths:
            return paths[0]
    return None

def require_file(logical_name):
    '''
    ENGLISH:
    •	Does: Looks up a logical name in LOGICAL_FILES, finds the first existing path, logs a message.
	•	Returns: str | None.
	•	Notes: Used to locate data files; prints a warning if missing.

	TÜRKÇE:
	•	Ne yapar: Bir kök klasörde, verilen çoklu glob pattern’lerinden ilk eşleşen dosya yolunu döndürür.
	•	Girdi: root (str), patterns (list[str]).
	•	Çıktı: Yol (str) ya da None.
	•	Not: İlk bulunanı döndürür (deterministik olması için sort var).
    '''
    path = find_one(DATA_ROOT, LOGICAL_FILES[logical_name])
    if path is None:
        print(f"[UYARI] '{logical_name}' için dosya bulunamadı. Bu sinyal atlanacak.")
    else:
        print(f"[OK] {logical_name}: {path}")
    return path

def schema_of(p):
    '''
    ENGLISH:
    •	Does: Loads the Polars schema (column names/types) from a Parquet path.
	•	Returns: pl.Schema.

	TÜRKÇE:
	•	Ne yapar: Parquet’in Polars schemasını (kolon adları/tipleri) döndürür.
	•	Girdi: Yol.
	•	Çıktı: pl.Schema.
    '''
    return pl.scan_parquet(p).collect_schema()

def make_session_strings(rank_df: pl.DataFrame) -> pl.DataFrame:
    '''
    ENGLISH:
    •	Does: From (session_id, content_id_hashed, score) builds one line per session by sorting items, deduping in order, and joining IDs with spaces into a prediction string.
	•	Returns: pl.DataFrame[session_id, prediction].
	•	Notes: Sorts by session_id asc, score desc, content_id_hashed asc.

	TÜRKÇE:
	•	Ne yapar: (session_id, content_id_hashed, score) tablosunu oturum bazında sıralayıp tek bir “prediction” string’ine çevirir.
	•	Girdi: pl.DataFrame.
	•	Çıktı: pl.DataFrame(session_id, prediction).
	•	Not: Skoru azalan, id’yi alfabetik sıralar; tekrarları unique(maintain_order=True) ile kırpar.

	'''
    out = (
        rank_df
        .with_columns([
            pl.col("score").cast(pl.Float64),
            pl.col("content_id_hashed").cast(pl.Utf8),
        ])
        .sort(["session_id", "score", "content_id_hashed"], descending=[False, True, False])
        .group_by("session_id")
        .agg(pl.col("content_id_hashed").unique(maintain_order=True).alias("_items"))
        .with_columns(pl.col("_items").list.join(" ").alias("prediction"))
        .select(["session_id", "prediction"])
    )
    return out

# ---- Locate files & create lazy frames ----
train_sessions_file = require_file("train_sessions")
test_sessions_file  = require_file("test_sessions")
content_file        = require_file("content_search_log")
content_md_file     = require_file("content_metadata")
content_pr_file     = require_file("content_price_rate")
content_sitewide_file   = require_file("content_sitewide_log")
content_top_terms_file  = require_file("content_top_terms_log")
user_md_file            = require_file("user_metadata")
user_sitewide_file      = require_file("user_sitewide_log")
user_top_terms_file     = require_file("user_top_terms_log")
term_search_file        = require_file("term_search_log")

content_sitewide_lf  = pl.scan_parquet(content_sitewide_file)  if content_sitewide_file  else None
content_top_terms_lf = pl.scan_parquet(content_top_terms_file) if content_top_terms_file else None
user_md_lf           = pl.scan_parquet(user_md_file)           if user_md_file           else None
user_sitewide_lf     = pl.scan_parquet(user_sitewide_file)     if user_sitewide_file     else None
user_top_terms_lf    = pl.scan_parquet(user_top_terms_file)    if user_top_terms_file    else None
term_search_lf       = pl.scan_parquet(term_search_file)       if term_search_file       else None
content_metadata_lf  = pl.scan_parquet(content_md_file)        if content_md_file        else None
content_price_lf     = pl.scan_parquet(content_pr_file)        if content_pr_file        else None

# ---- CTR helpers ----
def _smoothed_ctr_from_lf(lf: pl.LazyFrame, group_cols, alpha: float, prior: float,
                          score_name: str, n_name: str) -> pl.DataFrame:
    '''
    ENGLISH:
    •	Does: Computes Bayesian-smoothed CTR per group_cols using a label column clicked.
	•	Returns: pl.DataFrame[group_cols..., score_name, n_name].
	•	Notes: score = (clicks + alpha*prior) / (n + alpha); n_name holds group count.

	TÜRKÇE:
    •	Ne yapar: Bir LazyFrame üzerinde (clicked) hedefi ile grup CTR’ı Bayes smoothing ile hesaplar.
	•	Çıktı: pl.DataFrame([... group_cols, score_name, n_name]).
	•	Not: alpha ile prior karışımı; n_name grup örnek sayısıdır.
	'''
    lf2 = (
        lf.select(group_cols + [TARGET_COL])
          .filter(pl.col(TARGET_COL).is_not_null())
          .with_columns(pl.col(TARGET_COL).cast(pl.Float64))
          .group_by(group_cols)
          .agg([
              pl.sum(TARGET_COL).alias("__clicks__"),
              pl.len().alias("__n__"),
          ])
          .with_columns([
              ((pl.col("__clicks__") + pl.lit(alpha) * pl.lit(prior)) /
               (pl.col("__n__") + pl.lit(alpha))).alias(score_name),
              pl.col("__n__").alias(n_name),
          ])
          .select(group_cols + [score_name, n_name])
    )
    try:
        return lf2.collect(streaming=True)
    except Exception:
        return lf2.collect()

def _compute_global_mean_from_lf(lf: pl.LazyFrame) -> float:
    '''
    ENGLISH:
    •	Does: Global mean of the target (clicked) for a LazyFrame.
	•	Returns: float (prior).

	TÜRKÇE:
	•	Ne yapar: (clicked) kolonunun global ortalamasını döndürür (prior için).
	•	Çıktı: float.
	'''
    return (
        lf.select(TARGET_COL)
          .filter(pl.col(TARGET_COL).is_not_null())
          .with_columns(pl.col(TARGET_COL).cast(pl.Float64))
          .select(pl.mean(TARGET_COL).alias("gm"))
          .collect()
          .item()
    )

def _prior_from_click_imp(lf: pl.LazyFrame, click_col: str, imp_col: str) -> float:
    '''
    ENGLISH:
    •	Does: Derives a prior as sum(click)/sum(impression) safely (guards against zero).
	•	Returns: float.

	TÜRKÇE:
	•	Ne yapar: sum(click)/sum(impression) ile site/genel prior üretir.
	•	Çıktı: float.
	•	Not: impression=0 durumuna koruma var.
	'''
    return (
        lf.select([pl.col(click_col).sum().alias("__c"), pl.col(imp_col).sum().alias("__i")])
          .collect()
          .with_columns((pl.col("__c") / pl.when(pl.col("__i")>0).then(pl.col("__i")).otherwise(1.0)).alias("__p"))
          .select("__p").item()
    )

def _smoothed_ctr_from_click_imp(lf: pl.LazyFrame, group_cols,
                                 click_col: str, imp_col: str,
                                 alpha: float, prior: float,
                                 score_name: str, n_name: str) -> pl.DataFrame:
    '''
    ENGLISH:
    •	Does: Smoothed CTR using click/impression totals per group.
	•	Returns: pl.DataFrame[group_cols..., score_name, n_name].

	TÜRKÇE:
    •	Ne yapar: Click/Impression toplamlarından smoothed CTR hesaplar (örn. term veya content top terms log’ları).
	•	Çıktı: pl.DataFrame([... group_cols, score_name, n_name]).
	'''
    g = (
        lf.select(group_cols + [click_col, imp_col])
          .group_by(group_cols)
          .agg([
              pl.col(click_col).sum().alias("__clicks__"),
              pl.col(imp_col).sum().alias("__impr__"),
          ])
          .with_columns([
              ((pl.col("__clicks__") + pl.lit(alpha)*pl.lit(prior)) /
               (pl.col("__impr__")  + pl.lit(alpha))).alias(score_name),
              pl.col("__impr__").alias(n_name)
          ])
          .select(group_cols + [score_name, n_name])
    )
    try:
        return g.collect(streaming=True)
    except Exception:
        return g.collect()

def _term_idf_from_top_terms(lf: pl.LazyFrame) -> pl.DataFrame | None:
    '''
    ENGLISH:
    •	Does: Builds an IDF-like score for search_term_normalized:
        term_idf = log((N+1)/(df+1)), where df = # unique contents the term appears on; N = # unique contents overall.
	•	Needs: Columns search_term_normalized, content_id_hashed.
	•	Returns: pl.DataFrame[search_term_normalized, term_idf] | None.

	TÜRKÇE:
	•	Ne yapar: content_top_terms_log’tan IDF-benzeri skor üretir: log((N+1)/(df+1)).
	•	Girdi: LazyFrame’de search_term_normalized, content_id_hashed olmalı.
	•	Çıktı: pl.DataFrame(search_term_normalized, term_idf) veya None.
	'''
    names = set(lf.collect_schema().names())
    need = {"search_term_normalized","content_id_hashed"}
    if not need.issubset(names):
        return None
    df = (
        lf.select(["search_term_normalized","content_id_hashed"])
          .unique()
          .group_by("search_term_normalized")
          .agg(pl.col("content_id_hashed").n_unique().alias("__df__"))
    )
    N = lf.select(pl.col("content_id_hashed").n_unique().alias("__N__")).collect().item()
    idf = df.with_columns(
        ((pl.lit(N, dtype=pl.Float64)+1.0) / (pl.col("__df__").cast(pl.Float64)+1.0)).log().alias("term_idf")
    ).select(["search_term_normalized","term_idf"])
    return idf.collect()

# ---- Feature builders ----
def enrich_sessions_with_content_lazy(sessions_lf: pl.LazyFrame) -> pl.LazyFrame:
    '''
    ENGLISH:
    •	Does: Enriches sessions with content metadata and daily price/rating snapshot; also derives ts_date from ts_hour.
	•	Returns: pl.LazyFrame.
	•	Notes: Joins content_metadata_lf on content_id_hashed. If price data has update_date, it casts to ts_date and joins on [content_id_hashed, ts_date].

	TÜRKÇE:
	•	Ne yapar: Oturum satırlarına content metadata ve fiyat/review snapshot’larını join’ler; ts_hour → ts_date.
	•	Çıktı: LazyFrame (left join’lerle zenginleşmiş).
	•	Not: Fiyat datası günlük ise content_id_hashed, ts_date ile eşler.
	'''
    out = sessions_lf
    if "ts_hour" in sessions_lf.collect_schema().names():
        out = out.with_columns(pl.col("ts_hour").cast(pl.Datetime).dt.truncate("1h"))
        out = out.with_columns(pl.col("ts_hour").cast(pl.Date).alias("ts_date"))
    if content_metadata_lf is not None and "content_id_hashed" in content_metadata_lf.collect_schema().names():
        out = out.join(content_metadata_lf, on="content_id_hashed", how="left")
    if content_price_lf is not None:
        pr = content_price_lf
        sch = pr.collect_schema().names()
        if "update_date" in sch:
            pr = pr.with_columns(pl.col("update_date").cast(pl.Date).alias("ts_date"))
            keys = ["content_id_hashed","ts_date"]
            has_all = all(k in out.collect_schema().names() for k in keys)
            if has_all:
                out = out.join(pr, on=keys, how="left")
    return out

def fill_missing_values_polars(df: pl.LazyFrame) -> pl.LazyFrame:
    '''
    ENGLISH:
    •	Does: Fills common numeric nulls, creates missing flags, imputes content_rate_avg by category mean (fallback to global), normalizes cv_tags and creates tag_count.
	•	Returns: pl.LazyFrame.

	TÜRKÇE:
	•	Ne yapar: Eksikleri doldurur, eksik bayrakları çıkarır, kategori ortalamasıyla rating doldurur, cv_tags’ı normalize eder.
	•	Çıktı: LazyFrame.
	•	Not: Sayısal bazı kolonlar 0 ile doldurulur; tag sayısı çıkarılır.
	'''
    out = df
    if set(["content_rate_avg"]).issubset(out.collect_schema().names()):
        out = out.with_columns(pl.col("content_rate_avg").is_null().cast(pl.Int8).alias("rating_missing"))
    if "cv_tags" in out.collect_schema().names():
        out = out.with_columns(pl.col("cv_tags").is_null().cast(pl.Int8).alias("tags_missing"))
    if "content_creation_date" in out.collect_schema().names():
        out = out.with_columns(pl.col("content_creation_date").is_null().cast(pl.Int8).alias("creation_date_missing"))

    num_cols = [
        "content_review_count","content_review_wth_media_count","content_rate_count",
        "attribute_type_count","total_attribute_option_count","merchant_count",
        "filterable_label_count","original_price","selling_price","discounted_price"
    ]
    names = set(out.collect_schema().names())
    for c in num_cols:
        if c in names:
            out = out.with_columns(pl.col(c).fill_null(0).alias(c))

    if "content_rate_avg" in names and "level1_category_name" in names:
        cat_avg = (
            out.filter(pl.col("content_rate_avg").is_not_null())
               .group_by("level1_category_name")
               .agg(pl.col("content_rate_avg").mean().alias("cat_avg_rating"))
        )
        out = out.join(cat_avg, on="level1_category_name", how="left")
        glob_avg = out.select(pl.col("content_rate_avg").mean().alias("g")).collect().item()
        out = out.with_columns(
            pl.when(pl.col("content_rate_avg").is_null())
              .then(pl.coalesce([pl.col("cat_avg_rating"), pl.lit(glob_avg)]))
              .otherwise(pl.col("content_rate_avg")).alias("content_rate_avg")
        ).drop("cat_avg_rating")

    if "cv_tags" in names:
        out = out.with_columns([
            pl.col("cv_tags").fill_null("no_tags").alias("cv_tags"),
            pl.when(pl.col("cv_tags") == "no_tags").then(0)
              .otherwise(pl.col("cv_tags").str.split(",").list.len()).alias("tag_count"),
        ])
    return out

def add_advanced_features(df: pl.LazyFrame) -> pl.LazyFrame:
    '''
    ENGLISH:
    •	Does: Adds price/discount features (price_gap, discount_pct, has_discount, log_selling_price), category price stats (z, tiers, ratios), review ratios, rating/popularity tiers.
	•	Returns: pl.LazyFrame.

	TÜRKÇE:
	•	Ne yapar: Fiyata/indirime, popülerliğe ve rating’e bağlı türev özellikler (gap, discount_pct, tier’lar, z-score vb.).
	•	Çıktı: LazyFrame.
	'''
    names = set(df.collect_schema().names())
    def has(*cols): return all(c in names for c in cols)
    out = df
    if has("original_price", "selling_price"):
        out = out.with_columns([
            pl.max_horizontal(pl.lit(0), pl.col("original_price") - pl.col("selling_price")).alias("price_gap"),
            pl.when(pl.col("original_price") > 0)
            .then(((pl.col("original_price") - pl.col("selling_price")) / pl.col("original_price")).clip(0.0, 1.5))
            .otherwise(0.0).alias("discount_pct"),
            (pl.col("selling_price") <= 0).cast(pl.Int8).alias("is_zero_price"),
            (pl.col("selling_price") < pl.col("original_price")).cast(pl.Int8).alias("has_discount"),
            pl.when(pl.col("selling_price") > 0).then(pl.col("selling_price").log()).otherwise(None).alias("log_selling_price"),
        ])
    if has("level1_category_name", "selling_price"):
        _stats = (
            out.group_by("level1_category_name")
            .agg([
                pl.col("selling_price").mean().alias("__p_mean"),
                pl.col("selling_price").std(ddof=1).fill_null(0.0).alias("__p_std"),
                pl.col("selling_price").quantile(0.25, "nearest").alias("__p_q25"),
                pl.col("selling_price").quantile(0.50, "nearest").alias("__p_q50"),
                pl.col("selling_price").quantile(0.75, "nearest").alias("__p_q75"),
            ])
        )
        out = (
            out.join(_stats, on="level1_category_name", how="left")
            .with_columns([
                pl.when(pl.col("__p_std") > 0)
                .then((pl.col("selling_price") - pl.col("__p_mean")) / pl.col("__p_std"))
                .otherwise(0.0).alias("price_z_in_cat"),
                pl.when(pl.col("selling_price") <= pl.col("__p_q25")).then(0)
                .when(pl.col("selling_price") <= pl.col("__p_q50")).then(1)
                .when(pl.col("selling_price") <= pl.col("__p_q75")).then(2)
                .otherwise(3).alias("price_tier_cat"),
                pl.when(pl.col("__p_q50") > 0)
                .then(pl.col("selling_price") / pl.col("__p_q50"))
                .otherwise(1.0).alias("price_over_cat_median"),
            ])
            .drop(["__p_mean", "__p_std", "__p_q25", "__p_q50", "__p_q75"])
        )
    if has("content_rate_count","content_review_count"):
        out = out.with_columns(
            pl.when(pl.col("content_review_count") > 0)
              .then(pl.col("content_rate_count") / pl.col("content_review_count"))
              .otherwise(0.0).alias("rate_to_review_ratio")
        )
    if has("content_review_wth_media_count","content_review_count"):
        out = out.with_columns(
            pl.when(pl.col("content_review_count") > 0)
              .then(pl.col("content_review_wth_media_count") / pl.col("content_review_count"))
              .otherwise(0.0).alias("media_review_ratio")
        )
    if has("content_rate_avg"):
        out = out.with_columns([
            pl.when(pl.col("content_rate_avg") >= 4.5).then(4)
              .when(pl.col("content_rate_avg") >= 4.0).then(3)
              .when(pl.col("content_rate_avg") >= 3.5).then(2)
              .when(pl.col("content_rate_avg") >= 3.0).then(1)
              .otherwise(0).alias("rating_tier"),
        ])
    if has("content_rate_count"):
        out = out.with_columns([
            pl.when(pl.col("content_rate_count") >= 100).then(3)
              .when(pl.col("content_rate_count") >= 20).then(2)
              .when(pl.col("content_rate_count") >= 5).then(1)
              .otherwise(0).alias("popularity_tier"),
        ])
    return out

def add_in_session_ranks_lazy(df: pl.LazyFrame) -> pl.LazyFrame:
    '''
    ENGLISH:
    •	Does: Per-session ranks for price/discount/rating + normalized percentage ranks in each session.
	•	Returns: pl.LazyFrame.
	•	Notes: Uses rank("dense") and then divides by the session max to get 0–1 scales.

	TÜRKÇE:
	•	Ne yapar: Oturum içi fiyat/indirim/rating rank ve yüzde özelliklerini üretir.
	•	Çıktı: LazyFrame.
	•	Not: rank('dense'); sonrasında 0-1 ölçekli yüzde özellikleri.
	'''
    names = set(df.collect_schema().names())
    out = df
    if "session_id" in names and "selling_price" in names:
        out = out.with_columns(pl.col("selling_price").rank("dense").over("session_id").alias("price_rank_in_sess"))
    if "session_id" in names and "discount_pct" in names:
        out = out.with_columns((-pl.col("discount_pct")).rank("dense").over("session_id").alias("discount_rank_in_sess"))
    if "session_id" in names and "content_rate_avg" in names:
        out = out.with_columns((-pl.col("content_rate_avg")).rank("dense").over("session_id").alias("rating_rank_in_sess"))
    for col in ["price_rank_in_sess","discount_rank_in_sess","rating_rank_in_sess"]:
        if col in set(out.collect_schema().names()):
            out = out.with_columns(
                (pl.col(col) / pl.col(col).max().over("session_id")).alias(col.replace("_in_sess","_pct_in_sess"))
            )
    return out

def add_time_and_recency_features_lazy(df: pl.LazyFrame) -> pl.LazyFrame:
    '''
    ENGLISH:
    •	Does: Extracts time features: hour, weekday (dow), weekend flag, sin/cos encodings; derives ts_date; computes content_age_days if content_creation_date exists.
	•	Returns: pl.LazyFrame.

	TÜRKÇE:
	•	Ne yapar: Zaman özellikleri: saat, hafta günü, sin/cos encoding, ts_date, ve varsa içerik yaş (content_age_days).
	•	Çıktı: LazyFrame.
	'''
    names = set(df.collect_schema().names())
    out = df
    if "ts_hour" in names:
        out = out.with_columns([
            pl.col("ts_hour").cast(pl.Datetime).dt.hour().alias("hour"),
        ]).with_columns([
            pl.col("ts_hour").cast(pl.Datetime).dt.weekday().alias("dow"),
            pl.col("ts_hour").cast(pl.Datetime).dt.date().alias("ts_date"),
        ])
    elif "ts_date" in names:
        out = out.with_columns([pl.col("ts_date").cast(pl.Date).dt.weekday().alias("dow")])
    names = set(out.collect_schema().names())
    if "dow" in names:
        ang_d = pl.col("dow").cast(pl.Float64) * (2 * math.pi / 7.0)
        out = out.with_columns([
            pl.col("dow").is_in([5, 6]).cast(pl.Int8).alias("is_weekend"),
            ang_d.sin().alias("dow_sin"),
            ang_d.cos().alias("dow_cos"),
        ])
    if "hour" in names:
        ang_h = pl.col("hour").cast(pl.Float64) * (2 * math.pi / 24.0)
        out = out.with_columns([ang_h.sin().alias("hour_sin"), ang_h.cos().alias("hour_cos")])
    if {"ts_date", "content_creation_date"}.issubset(names):
        out = out.with_columns([
            (
                (pl.col("ts_date").cast(pl.Date).cast(pl.Datetime) -
                 pl.col("content_creation_date").cast(pl.Date).cast(pl.Datetime))
                .dt.total_days()
                .cast(pl.Int32)
                .clip(0, None)
            ).alias("content_age_days")
        ])
    return out

def add_session_aggregates_lazy(df: pl.LazyFrame) -> pl.LazyFrame:
    '''
    ENGLISH:
    •	Does: Session-level aggregates (median/min/max price, mean rating, max discount) and per-row deltas/ratios vs session stats.
	•	Returns: pl.LazyFrame.

	TÜRKÇE:
	•	Ne yapar: Oturum bazında medyan/min/max fiyat, ortalama rating, max indirim gibi agregalar ve fark/oran türevleri.
	•	Çıktı: LazyFrame.
	'''
    names = set(df.collect_schema().names())
    if "session_id" not in names:
        return df
    out = df
    aggs = [pl.len().alias("sess_n_items")]
    if "selling_price" in names:
        aggs += [
            pl.col("selling_price").median().alias("sess_price_med"),
            pl.col("selling_price").min().alias("sess_price_min"),
            pl.col("selling_price").max().alias("sess_price_max"),
        ]
    if "content_rate_avg" in names:
        aggs += [pl.col("content_rate_avg").mean().alias("sess_rating_mean")]
    if "discount_pct" in names:
        aggs += [pl.col("discount_pct").max().alias("sess_discount_max")]
    sess_stats = out.group_by("session_id").agg(aggs)
    out = out.join(sess_stats, on="session_id", how="left")
    if {"selling_price","sess_price_med"}.issubset(set(out.collect_schema().names())):
        out = out.with_columns([
            (pl.col("selling_price") - pl.col("sess_price_med")).alias("price_minus_sess_med"),
            pl.when(pl.col("sess_price_med") > 0)
              .then(pl.col("selling_price") / pl.col("sess_price_med"))
              .otherwise(1.0).alias("price_over_sess_med"),
        ])
    if {"content_rate_avg","sess_rating_mean"}.issubset(set(out.collect_schema().names())):
        out = out.with_columns([
            (pl.col("content_rate_avg") - pl.col("sess_rating_mean")).alias("rating_minus_sess_mean")
        ])
    return out

def add_cat_match_features_lazy(df: pl.LazyFrame) -> pl.LazyFrame:
    '''
    ENGLISH:
    •	Does: Tokenizes query and category names; computes Jaccard overlaps and any-match flags for L1/L2/leaf categories; legacy term_in_*_cat flags included.
	•	Needs: search_term_normalized (+ category columns if present).
	•	Returns: pl.LazyFrame.

	TÜRKÇE:
	•	Ne yapar: Sorgu token’ları ile kategori adlarının token’ları arasında Jaccard / any-match ve legacy term_in_*_cat bayrakları.
	•	Gereken: search_term_normalized ve ilgili kategori kolonları.
	•	Çıktı: LazyFrame.
    '''
    names = set(df.collect_schema().names())
    if "search_term_normalized" not in names:
        return df
    q_tokens = (
        pl.col("search_term_normalized")
          .str.to_lowercase()
          .str.replace_all(r"[^a-z0-9]+", " ")
          .str.strip_chars()
          .str.split(by=" ")
          .list.eval(pl.element().str.strip_chars())
          .list.eval(pl.element().filter(pl.element().str.len_chars() > 0))
          .alias("__q_tok")
    )
    out = df.with_columns(q_tokens)
    tmp_cols = ["__q_tok"]
    def cat_tokens(col: str, alias: str) -> pl.Expr:
        return (
            pl.col(col)
              .str.to_lowercase()
              .str.replace_all(r"[^a-z0-9]+", " ")
              .str.strip_chars()
              .str.split(by=" ")
              .list.eval(pl.element().str.strip_chars())
              .list.eval(pl.element().filter(pl.element().str.len_chars() > 0))
              .alias(alias)
        )
    for col, lvl in [("level1_category_name","l1"),
                     ("level2_category_name","l2"),
                     ("leaf_category_name","leaf")]:
        if col in names:
            ct_alias = f"__cat_tok_{lvl}"
            out = out.with_columns(cat_tokens(col, ct_alias))
            tmp_cols.append(ct_alias)
            inter = pl.col("__q_tok").list.set_intersection(pl.col(ct_alias)).list.len().alias(f"__inter_{lvl}")
            union = pl.col("__q_tok").list.set_union(pl.col(ct_alias)).list.len().alias(f"__union_{lvl}")
            out = out.with_columns([inter, union]).with_columns([
                pl.when(pl.col(f"__union_{lvl}") > 0)
                  .then(pl.col(f"__inter_{lvl}") / pl.col(f"__union_{lvl}"))
                  .otherwise(0.0).alias(f"qcat_jaccard_{lvl}"),
                (pl.col(f"__inter_{lvl}") > 0).cast(pl.Int8).alias(f"qcat_any_{lvl}"),
                (pl.col(f"__inter_{lvl}") > 0).cast(pl.Int8).alias(f"term_in_{lvl}_cat"),
            ])
            tmp_cols += [f"__inter_{lvl}", f"__union_{lvl}"]
    return out.drop([c for c in tmp_cols if c in set(out.collect_schema().names())])

def add_text_match_features_lazy(df: pl.LazyFrame) -> pl.LazyFrame:
    '''
    ENGLISH:
    •	Does: Tokenizes query and cv_tags (comma-separated), then computes Jaccard and any-match flags; also outputs token lengths.
	•	Needs: search_term_normalized, cv_tags.
	•	Returns: pl.LazyFrame.

	TÜRKÇE:
	•	Ne yapar: Sorgu token’ları ile cv_tags token’larının Jaccard ve eşleşme sayısı üzerinden özellik üretir.
	•	Gereken: search_term_normalized, cv_tags.
	•	Çıktı: LazyFrame.
	'''
    names = set(df.collect_schema().names())
    if "search_term_normalized" not in names or "cv_tags" not in names:
        return df
    out = df
    term_tokens = (
        pl.col("search_term_normalized")
          .str.to_lowercase()
          .str.replace_all(r"[^a-z0-9]+", " ")
          .str.strip_chars()
          .str.split(by=" ")
          .list.eval(pl.element().str.strip_chars())
          .list.eval(pl.element().filter(pl.element().str.len_chars() > 0))
          .alias("__term_tok")
    )
    tag_tokens = (
        pl.col("cv_tags")
          .str.to_lowercase()
          .str.replace_all(r"\s+", "")
          .str.split(by=",")
          .list.eval(pl.element().str.strip_chars())
          .list.eval(pl.element().filter(pl.element().str.len_chars() > 0))
          .alias("__tag_tok")
    )
    out = out.with_columns([term_tokens, tag_tokens]).with_columns([
        pl.col("__term_tok").list.len().alias("term_len"),
        pl.col("__tag_tok").list.len().alias("tag_len"),
        pl.col("__term_tok").list.set_intersection(pl.col("__tag_tok")).list.len().alias("__tok_inter"),
        pl.col("__term_tok").list.set_union(pl.col("__tag_tok")).list.len().alias("__tok_union"),
    ]).with_columns([
        pl.when(pl.col("__tok_union") > 0)
          .then(pl.col("__tok_inter") / pl.col("__tok_union"))
          .otherwise(0.0).alias("qtag_jaccard"),
        (pl.col("__tok_inter") > 0).cast(pl.Int8).alias("qtag_any_match"),
    ]).drop(["__term_tok","__tag_tok","__tok_inter","__tok_union"])
    return out

def add_ctr_flags_lazy(df: pl.LazyFrame) -> pl.LazyFrame:
    '''
    ENGLISH:
    •	Does: For every n_* column, adds log1p_n_* and cold_* (n<=0) flags; ensures n_* are floats.
	•	Returns: pl.LazyFrame.

	TÜRKÇE:
	•	Ne yapar: n_* kolonları için log1p_n_* ve cold-start bayrakları üretir; n_*’leri float’a çevirir.
	•	Çıktı: LazyFrame.
	'''
    names = set(df.collect_schema().names())
    out = df
    n_cols = [c for c in names if c.startswith("n_")]
    for n in n_cols:
        out = out.with_columns([
            pl.col(n).fill_null(0).cast(pl.Float64).alias(n),
            (pl.col(n).fill_null(0).cast(pl.Float64) + 1.0).log().alias(f"log1p_{n}"),
            (pl.col(n).fill_null(0) <= 0).cast(pl.Int8).alias(f"cold_{n[2:]}"),
        ])
    return out

def _daily_content_search_rollups():
    '''
    ENGLISH:
    •	Does: From content search logs, creates daily click/impression by content, then rolling windows (7/14/30d) of sums and CTRs, plus log1p impressions.
	•	Returns: pl.LazyFrame[content_id_hashed, d, cs_click_*d, cs_impr_*d, cs_ctr_*d, log1p_cs_impr_*].
	•	Notes: Uses group_by_dynamic with daily windows; output is sorted by content_id_hashed, d.

	TÜRKÇE:
	•	Ne yapar: content_search_log’dan günlük bazlı click/impr üretir; 7/14/30g rolling toplamlar ve CTR’lar çıkarır.
	•	Çıktı: LazyFrame(content_id_hashed, d, cs_click_*d, cs_impr_*d, cs_ctr_*d, log1p_cs_impr_*).
	•	Not: group_by_dynamic ile pencereler; sonraki as-of join için d sıralıdır.
    '''
    if not content_file:
        return None
    cs = (
        pl.scan_parquet(content_file)
        .with_columns(pl.col("date").cast(pl.Date).alias("d"))
        .group_by(["content_id_hashed","d"])
        .agg([
            pl.col("total_search_click").sum().alias("cs_click"),
            pl.col("total_search_impression").sum().alias("cs_impr"),
        ])
        .sort(["content_id_hashed","d"])
    )
    roll = (
        cs.group_by_dynamic(index_column="d", by="content_id_hashed",
                            every="1d", period="7d", closed="left")
          .agg([
              pl.col("cs_click").sum().alias("cs_click_7d"),
              pl.col("cs_impr").sum().alias("cs_impr_7d"),
          ])
          .join(
              cs.group_by_dynamic(index_column="d", by="content_id_hashed",
                                  every="1d", period="14d", closed="left")
                .agg([
                    pl.col("cs_click").sum().alias("cs_click_14d"),
                    pl.col("cs_impr").sum().alias("cs_impr_14d"),
                ]),
              on=["content_id_hashed","d"], how="left"
          )
          .join(
              cs.group_by_dynamic(index_column="d", by="content_id_hashed",
                                  every="1d", period="30d", closed="left")
                .agg([
                    pl.col("cs_click").sum().alias("cs_click_30d"),
                    pl.col("cs_impr").sum().alias("cs_impr_30d"),
                ]),
              on=["content_id_hashed","d"], how="left"
          )
          .with_columns([
              (pl.col("cs_click_7d")  / pl.when(pl.col("cs_impr_7d") > 0).then(pl.col("cs_impr_7d")).otherwise(1.0)).alias("cs_ctr_7d"),
              (pl.col("cs_click_14d") / pl.when(pl.col("cs_impr_14d")> 0).then(pl.col("cs_impr_14d")).otherwise(1.0)).alias("cs_ctr_14d"),
              (pl.col("cs_click_30d") / pl.when(pl.col("cs_impr_30d")> 0).then(pl.col("cs_impr_30d")).otherwise(1.0)).alias("cs_ctr_30d"),
              (pl.col("cs_impr_7d")+1.0).log().alias("log1p_cs_impr_7d"),
              (pl.col("cs_impr_30d")+1.0).log().alias("log1p_cs_impr_30d"),
          ])
    )
    return roll

def join_content_search_rollups_asof(sessions_lf: pl.LazyFrame) -> pl.LazyFrame:
    '''
    ENGLISH:
    •	Does: AS-OF join of sessions (by content_id_hashed and date) to the rolling content-search metrics (7/14/30d) as of the previous day.
	•	Returns: pl.LazyFrame.
	•	Important: For join_asof, both sides need to be sorted by the join key date; if you see errors, add explicit .sort(["content_id_hashed","ts_date"]) and .sort(["content_id_hashed","d"]).

	TÜRKÇE:
	•	Ne yapar: Oturum satırlarını (içerik, tarih) ile as-of biçimde 7/14/30g arkalı içerik-search metrikleriyle zenginleştirir.
	•	Çıktı: LazyFrame.
	•	Önemli: join_asof için left_on/right_on (tarih) kolonları sıralı olmalıdır. Gerekirse with_columns(...).sort("ts_date") ekleyin.
	'''
    roll = _daily_content_search_rollups()
    if roll is None:
        return sessions_lf
    lf = sessions_lf
    if "ts_date" not in lf.collect_schema().names():
        lf = lf.with_columns(pl.col("ts_hour").cast(pl.Datetime).dt.date().alias("ts_date"))
    lf = lf.with_columns((pl.col("ts_date") - pl.duration(days=1)).alias("__asof_d"))
    lf = lf.join_asof(
        roll, left_on="__asof_d", right_on="d",
        by="content_id_hashed", strategy="backward"
    ).drop(["__asof_d","d"])
    return lf

def join_user_affinity_oof_like(lf: pl.LazyFrame) -> pl.LazyFrame:
    '''
    ENGLISH:
    •	Does: Adds user-level affinity signals from sitewide logs: total clicks (log1p) and a price preference proxy (price_tier_cat - price_over_cat_median).
	•	Needs: user_sitewide_lf and price_tier_cat.
	•	Returns: pl.LazyFrame.

	TÜRKÇE:
	•	Ne yapar: Kullanıcı-genel tıklama hacmi (log1p) ve fiyat tier’ı ile relatif fiyatın farkı gibi kullanıcı eğilim sinyalleri ekler.
	•	Gereken: user_sitewide_lf ve price_tier_cat.
	•	Çıktı: LazyFrame.
    '''
    names = set(lf.collect_schema().names())
    if "price_tier_cat" not in names or user_sitewide_lf is None:
        return lf
    us = (
        user_sitewide_lf
        .with_columns(pl.col("ts_hour").cast(pl.Datetime).dt.date().alias("d"))
        .group_by("user_id_hashed").agg([pl.col("total_click").sum().alias("__u_click")])
        .select(["user_id_hashed","__u_click"])
    )
    out = lf.join(us.lazy(), on="user_id_hashed", how="left")
    out = out.with_columns([
        (pl.col("__u_click").fill_null(0.0)+1.0).log().alias("log1p_user_click_all"),
        (pl.col("price_tier_cat").cast(pl.Float64) - pl.col("price_over_cat_median").fill_null(1.0)).alias("tier_minus_relprice"),
    ])
    return out.drop(["__u_click"])

def add_query_intent_flags_lazy(lf: pl.LazyFrame) -> pl.LazyFrame:
    '''
    ENGLISH:
    •	Does: Regex-based intent flags from the query: gender (m/f), kids, color, size/number, discount intent.
	•	Returns: pl.LazyFrame.

	TÜRKÇE:
	•	Ne yapar: Sorguda cinsiyet, çocuk, renk, beden/numara, indirim niyeti gibi regex tabanlı intent bayrakları üretir.
	•	Çıktı: LazyFrame.
	'''
    names = set(lf.collect_schema().names())
    if "search_term_normalized" not in names:
        return lf
    q = pl.col("search_term_normalized").str.to_lowercase()
    return lf.with_columns([
        q.str.contains(r"\b(erkek|man)\b").cast(pl.Int8).alias("q_has_gender_m"),
        q.str.contains(r"\b(kad[ıi]n|woman)\b").cast(pl.Int8).alias("q_has_gender_f"),
        q.str.contains(r"\b(çocuk|k[ıi]d|kids)\b").cast(pl.Int8).alias("q_has_kids"),
        q.str.contains(r"\b(siyah|beyaz|k[ıi]rm[ıi]z[ıi]|mavi|lacivert|ye[sş]il|mor|pembe|bej|gri|kahverengi|turuncu)\b").cast(pl.Int8).alias("q_has_color"),
        q.str.contains(r"\b(\d{2,3}\s?(beden|numara))\b").cast(pl.Int8).alias("q_has_size"),
        q.str.contains(r"\b(indirim|ucuz|fiyat)\b").cast(pl.Int8).alias("q_is_discount_intent"),
    ])

def add_intent_interactions_lazy(lf: pl.LazyFrame) -> pl.LazyFrame:
    '''
    ENGLISH:
    •	Does: Cross-features between intents and price/discount/ranks/category/tag matches (e.g., discount-intent × discount_pct).
	•	Returns: pl.LazyFrame.
	•	Notes: Adds only when both operands exist, so it’s safe across varying schemas.

	TÜRKÇE:
	•	Ne yapar: Intent bayraklarını indirim/fiyat/rank ve kategori/etiket eşleşmeleriyle etkileşim (cross-feature) olarak genişletir.
	•	Çıktı: LazyFrame.
	•	Not: Sadece mevcut kolon kombinasyonlarını ekler (korumalı).
	'''
    names = set(lf.collect_schema().names())
    out = lf
    def has(*cols): return all(c in names for c in cols)
    terms: list[pl.Expr] = []
    if has("q_is_discount_intent","discount_pct"):
        terms.append((pl.col("q_is_discount_intent").cast(pl.Float32) * pl.col("discount_pct").cast(pl.Float32)).alias("int_disc_x_discount"))
    if has("q_is_discount_intent","has_discount"):
        terms.append((pl.col("q_is_discount_intent").cast(pl.Float32) * pl.col("has_discount").cast(pl.Float32)).alias("int_disc_x_hasdisc"))
    for rcol in ["discount_rank_pct_in_sess","discount_rank_in_sess_pct"]:
        if has("q_is_discount_intent", rcol):
            terms.append((pl.col("q_is_discount_intent").cast(pl.Float32) * (1.0 - pl.col(rcol).cast(pl.Float32))).alias("int_disc_x_bestdisc"))
    for lvl, legacy, anym, jacc in [
        ("l1","term_in_l1_cat","qcat_any_l1","qcat_jaccard_l1"),
        ("l2","term_in_l2_cat","qcat_any_l2","qcat_jaccard_l2"),
        ("leaf","term_in_leaf_cat","qcat_any_leaf","qcat_jaccard_leaf"),
    ]:
        if has("q_has_gender_f", legacy):
            terms.append((pl.col("q_has_gender_f") * pl.col(legacy)).cast(pl.Float32).alias(f"int_female_x_term_{lvl}"))
        elif has("q_has_gender_f", anym):
            terms.append((pl.col("q_has_gender_f").cast(pl.Float32) * pl.col(anym).cast(pl.Float32)).alias(f"int_female_x_catany_{lvl}"))
        elif has("q_has_gender_f", jacc):
            terms.append((pl.col("q_has_gender_f").cast(pl.Float32) * pl.col(jacc).cast(pl.Float32)).alias(f"int_female_x_catjacc_{lvl}"))
        if has("q_has_gender_m", legacy):
            terms.append((pl.col("q_has_gender_m") * pl.col(legacy)).cast(pl.Float32).alias(f"int_male_x_term_{lvl}"))
        elif has("q_has_gender_m", anym):
            terms.append((pl.col("q_has_gender_m").cast(pl.Float32) * pl.col(anym).cast(pl.Float32)).alias(f"int_male_x_catany_{lvl}"))
        elif has("q_has_gender_m", jacc):
            terms.append((pl.col("q_has_gender_m").cast(pl.Float32) * pl.col(jacc).cast(pl.Float32)).alias(f"int_male_x_catjacc_{lvl}"))
        if has("q_has_kids", legacy):
            terms.append((pl.col("q_has_kids") * pl.col(legacy)).cast(pl.Float32).alias(f"int_kids_x_term_{lvl}"))
        elif has("q_has_kids", anym):
            terms.append((pl.col("q_has_kids").cast(pl.Float32) * pl.col(anym).cast(pl.Float32)).alias(f"int_kids_x_catany_{lvl}"))
        elif has("q_has_kids", jacc):
            terms.append((pl.col("q_has_kids").cast(pl.Float32) * pl.col(jacc).cast(pl.Float32)).alias(f"int_kids_x_catjacc_{lvl}"))
    if has("q_has_color","qtag_any_match"):
        terms.append((pl.col("q_has_color").cast(pl.Float32) * pl.col("qtag_any_match").cast(pl.Float32)).alias("int_color_x_tagany"))
    if has("q_has_color","qtag_jaccard"):
        terms.append((pl.col("q_has_color").cast(pl.Float32) * pl.col("qtag_jaccard").cast(pl.Float32)).alias("int_color_x_tagjacc"))
    for c in ["attribute_type_count","filterable_label_count"]:
        if has("q_has_size", c):
            terms.append((pl.col("q_has_size").cast(pl.Float32) * pl.col(c).cast(pl.Float32)).alias(f"int_size_x_{c}"))
    for rcol in ["price_rank_pct_in_sess","price_rank_in_sess_pct"]:
        if has("q_is_discount_intent", rcol):
            terms.append((pl.col("q_is_discount_intent").cast(pl.Float32) * (1.0 - pl.col(rcol).cast(pl.Float32))).alias("int_disc_x_bestprice"))
    return out.with_columns(terms) if terms else out

# ---- Build OOF features (train/test) ----
def _build_oof_rank_features(train_path: str, test_path: str, k_folds: int = 5):
    '''
    ENGLISH:
    •	Does (core pipeline):
	    1.	Read minimal train/test columns with Polars scan.
	    2.	Compute external signals: term CTR, user×term CTR, content×term CTR (smoothed), content sitewide means, aggregated content search stats, user metadata features, term IDF.
	    3.	Apply the full feature pipeline (enrich_…, add_*, join_*).
	    4.	OOF folds (train): For each fold, compute UC/TC/C CTR tables on the other folds only, join to validation, build weighted blend and score_prior, add CTR flags, collect.
	    5.	Test: Recompute UC/TC/C CTR on full train, join to test, build the same blend and score_prior, add CTR flags.
	•	Returns: (train_oof_df: pl.DataFrame, test_df: pl.DataFrame, full_prior: float).
	•	Notes:
	    1.	Fold id: hash(session_id) % k.
	    2.	Weights: w = n / (n + alpha).
	    3.	Blend: (score_uc*w_uc + score_tc*w_tc + score_c*w_c) / (w_uc + w_tc + w_c), then fallback chain → score_prior.

    TÜRKÇE:
    •	Ne yapar (çekirdek):
	    1.	Train/test’i minimum kolonlarla scan eder.
	    2.	Dış sinyaller (term/user×term/content×term CTR, sitewide, content search, user md, term idf) hesaplanır/collect edilir.
	    3.	Feature pipeline (tüm add_* ve join_*) uygulanır.
	    4.	Train için OOF katman: her fold’da grup CTR’ları (UC/TC/C) leakage-safe hesaplanır ve valid’e join edilir; ağırlıklarla karıştırılıp score_prior üretilir.
	    5.	Test için full-train CTR’lar ile aynı karışım ve score_prior.
	•	Çıktı: train_oof_df (pl.DataFrame), test_df (pl.DataFrame), full_prior (float).
	•	Notlar:
	    1.	__fold__ = hash(session_id) % k.
	    2.	STREAMING_COLLECT destekli collect.
	    3.	Ağırlıklar: w = n/(n+alpha); miks: (uc, tc, c) → __mix__ → score_prior.
    '''
    sch_tr = schema_of(train_path)
    sch_te = schema_of(test_path)
    base_cols = ["row_id","session_id","user_id_hashed","content_id_hashed","ts_hour","search_term_normalized"]
    have_cols_tr = [c for c in base_cols + [TARGET_COL] if c in sch_tr.names()]
    have_cols_te = [c for c in base_cols if c in sch_te.names()]
    base_lf = (
        pl.scan_parquet(train_path)
          .select(have_cols_tr)
          .filter(pl.col(TARGET_COL).is_not_null())
          .with_columns(pl.col(TARGET_COL).cast(pl.Float64))
    )

    term_ctr_df = None
    if term_search_lf is not None:
        prior_t = _prior_from_click_imp(term_search_lf, "total_search_click", "total_search_impression")
        term_ctr_df = _smoothed_ctr_from_click_imp(
            term_search_lf, ["search_term_normalized"],
            "total_search_click","total_search_impression",
            alpha=200.0, prior=prior_t,
            score_name="score_term", n_name="n_term"
        )

    uterm_ctr_df = None
    if user_top_terms_lf is not None:
        prior_ut = _prior_from_click_imp(user_top_terms_lf, "total_search_click", "total_search_impression")
        uterm_ctr_df = _smoothed_ctr_from_click_imp(
            user_top_terms_lf, ["user_id_hashed","search_term_normalized"],
            "total_search_click","total_search_impression",
            alpha=150.0, prior=prior_ut,
            score_name="score_uterm", n_name="n_uterm"
        )

    cterm_ctr_df = None
    if content_top_terms_lf is not None:
        prior_ct = _prior_from_click_imp(content_top_terms_lf, "total_search_click", "total_search_impression")
        cterm_ctr_df = _smoothed_ctr_from_click_imp(
            content_top_terms_lf, ["content_id_hashed","search_term_normalized"],
            "total_search_click","total_search_impression",
            alpha=150.0, prior=prior_ct,
            score_name="score_cterm", n_name="n_cterm"
        )

    csite_df = None
    if content_sitewide_lf is not None:
        csite_df = (
            content_sitewide_lf
            .group_by("content_id_hashed")
            .agg([
                pl.col("total_click").mean().alias("sw_click_mean"),
                pl.col("total_cart").mean().alias("sw_cart_mean"),
                pl.col("total_fav").mean().alias("sw_fav_mean"),
                pl.col("total_order").mean().alias("sw_order_mean"),
            ]).collect()
        )
    csearch_df = None
    if content_file is not None:
        csearch_df = (
            pl.scan_parquet(content_file)
              .group_by("content_id_hashed")
              .agg([
                  pl.col("total_search_impression").sum().alias("__impr__"),
                  pl.col("total_search_click").sum().alias("__click__"),
              ])
              .with_columns([
                  pl.when(pl.col("__impr__")>0).then(pl.col("__click__")/pl.col("__impr__")).otherwise(0.0).alias("cs_ctr"),
                  (pl.col("__impr__")+1.0).log().alias("log1p_cs_impr"),
              ])
              .select(["content_id_hashed","cs_ctr","log1p_cs_impr"])
              .collect()
        )

    user_md_df = None
    if user_md_lf is not None:
        user_md_df = (
            user_md_lf
            .with_columns([
                pl.when(pl.col("user_birth_year").is_not_null())
                  .then((pl.lit(2025) - pl.col("user_birth_year").cast(pl.Float64)).clip(10, 100))
                  .otherwise(None).alias("user_age"),
                pl.col("user_tenure_in_days").cast(pl.Float64).fill_null(0.0).alias("user_tenure_in_days"),
                pl.when(pl.col("user_tenure_in_days").is_not_null())
                  .then((pl.col("user_tenure_in_days")+1.0).log())
                  .otherwise(0.0).alias("log1p_user_tenure"),
                pl.col("user_gender").fill_null("UNK").alias("user_gender"),
                (pl.col("user_gender").is_null()).cast(pl.Int8).alias("gender_missing"),
            ])
            .select(["user_id_hashed","user_age","log1p_user_tenure","user_gender","gender_missing"])
            .collect()
        )

    term_idf_df = None
    if content_top_terms_lf is not None:
        term_idf_df = _term_idf_from_top_terms(content_top_terms_lf)

    # base features
    base_lf = enrich_sessions_with_content_lazy(base_lf)
    base_lf = fill_missing_values_polars(base_lf)
    base_lf = add_advanced_features(base_lf)
    base_lf = add_in_session_ranks_lazy(base_lf)
    base_lf = add_time_and_recency_features_lazy(base_lf)
    base_lf = add_session_aggregates_lazy(base_lf)
    base_lf = add_query_intent_flags_lazy(base_lf)
    base_lf = add_text_match_features_lazy(base_lf)
    base_lf = add_cat_match_features_lazy(base_lf)
    base_lf = add_intent_interactions_lazy(base_lf)
    base_lf = join_content_search_rollups_asof(base_lf)
    base_lf = join_user_affinity_oof_like(base_lf)

    # --- join external signals ---
    if term_ctr_df is not None:
        base_lf = base_lf.join(term_ctr_df.lazy(), on=["search_term_normalized"], how="left")
    if uterm_ctr_df is not None:
        base_lf = base_lf.join(uterm_ctr_df.lazy(), on=["user_id_hashed", "search_term_normalized"], how="left")
    if cterm_ctr_df is not None:
        base_lf = base_lf.join(cterm_ctr_df.lazy(), on=["content_id_hashed", "search_term_normalized"], how="left")
    if csite_df is not None:
        base_lf = base_lf.join(csite_df.lazy(), on=["content_id_hashed"], how="left")
    if csearch_df is not None:
        base_lf = base_lf.join(csearch_df.lazy(), on=["content_id_hashed"], how="left")
    if user_md_df is not None:
        base_lf = base_lf.join(user_md_df.lazy(), on=["user_id_hashed"], how="left")
    if term_idf_df is not None:
        base_lf = base_lf.join(term_idf_df.lazy(), on=["search_term_normalized"], how="left")

    # --- OOF fold assignment ---
    if "session_id" not in have_cols_tr:
        raise RuntimeError("OOF için 'session_id' gerekiyor.")
    base_folded = base_lf.with_columns(
        ((pl.col("session_id").hash() % pl.lit(k_folds, dtype=pl.UInt64))
         .cast(pl.UInt8)).alias("__fold__")
    )

    uc_cols = ["user_id_hashed", "content_id_hashed"]
    tc_cols = ["search_term_normalized", "content_id_hashed"]
    c_cols = ["content_id_hashed"]

    have_uc = all(c in have_cols_tr for c in uc_cols)
    have_tc = all(c in have_cols_tr for c in tc_cols)

    # --- OOF loop (train side) ---
    oof_parts = []
    for k in range(k_folds):
        print(f"  -> OOF fold {k + 1}/{k_folds}")
        train_part = base_folded.filter(pl.col("__fold__") != k)
        valid_part = base_folded.filter(pl.col("__fold__") == k)

        # fold-level prior
        prior_k = _compute_global_mean_from_lf(train_part)

        # fold-level CTR tables
        if have_uc:
            ctr_uc_k = _smoothed_ctr_from_lf(train_part, uc_cols, ALPHA_UC, prior_k, "score_uc", "n_uc")
        else:
            ctr_uc_k = pl.DataFrame({"user_id_hashed": [], "content_id_hashed": [], "score_uc": [], "n_uc": []})

        if have_tc:
            ctr_tc_k = _smoothed_ctr_from_lf(train_part, tc_cols, ALPHA_TC, prior_k, "score_tc", "n_tc")
        else:
            ctr_tc_k = pl.DataFrame({"search_term_normalized": [], "content_id_hashed": [], "score_tc": [], "n_tc": []})

        ctr_c_k = _smoothed_ctr_from_lf(train_part, c_cols, ALPHA_C, prior_k, "score_c", "n_c")

        # join to valid fold
        vf = valid_part
        if have_uc:
            vf = vf.join(ctr_uc_k.lazy(), on=uc_cols, how="left")
        else:
            vf = vf.with_columns([pl.lit(None).alias("score_uc"), pl.lit(None).alias("n_uc")])

        if have_tc:
            vf = vf.join(ctr_tc_k.lazy(), on=tc_cols, how="left")
        else:
            vf = vf.with_columns([pl.lit(None).alias("score_tc"), pl.lit(None).alias("n_tc")])

        vf = vf.join(ctr_c_k.lazy(), on=c_cols, how="left")

        # blend + prior
        vf = (
            vf.with_columns([
                (pl.when(pl.col("n_uc").is_not_null())
                 .then(pl.col("n_uc") / (pl.col("n_uc") + pl.lit(ALPHA_UC)))
                 .otherwise(0.0)).alias("w_uc"),
                (pl.when(pl.col("n_tc").is_not_null())
                 .then(pl.col("n_tc") / (pl.col("n_tc") + pl.lit(ALPHA_TC)))
                 .otherwise(0.0)).alias("w_tc"),
                (pl.when(pl.col("n_c").is_not_null())
                 .then(pl.col("n_c") / (pl.col("n_c") + pl.lit(ALPHA_C)))
                 .otherwise(0.0)).alias("w_c"),
            ])
            .with_columns([
                (
                        (pl.coalesce([pl.col("score_uc")]) * pl.col("w_uc") +
                         pl.coalesce([pl.col("score_tc")]) * pl.col("w_tc") +
                         pl.coalesce([pl.col("score_c")]) * pl.col("w_c"))
                        / (pl.col("w_uc") + pl.col("w_tc") + pl.col("w_c"))
                ).alias("__mix__")
            ])
            .with_columns([
                pl.coalesce([
                    pl.col("__mix__"), pl.col("score_uc"), pl.col("score_tc"), pl.col("score_c")
                ]).fill_null(prior_k).alias("score_prior")
            ])
        )

        vf = add_ctr_flags_lazy(vf)
        vf = vf.collect(streaming=STREAMING_COLLECT)
        oof_parts.append(vf)

    train_oof_df = pl.concat(oof_parts)

    # --- Test side: full-train CTR ---
    full_prior = _compute_global_mean_from_lf(base_lf)
    if have_uc:
        ctr_uc_full = _smoothed_ctr_from_lf(base_lf, uc_cols, ALPHA_UC, full_prior, "score_uc", "n_uc")
    else:
        ctr_uc_full = pl.DataFrame({"user_id_hashed": [], "content_id_hashed": [], "score_uc": [], "n_uc": []})
    if have_tc:
        ctr_tc_full = _smoothed_ctr_from_lf(base_lf, tc_cols, ALPHA_TC, full_prior, "score_tc", "n_tc")
    else:
        ctr_tc_full = pl.DataFrame({"search_term_normalized": [], "content_id_hashed": [], "score_tc": [], "n_tc": []})
    ctr_c_full = _smoothed_ctr_from_lf(base_lf, c_cols, ALPHA_C, full_prior, "score_c", "n_c")

    test_lf = pl.scan_parquet(test_path).select(have_cols_te)
    test_lf = enrich_sessions_with_content_lazy(test_lf)
    test_lf = fill_missing_values_polars(test_lf)
    test_lf = add_advanced_features(test_lf)
    test_lf = add_in_session_ranks_lazy(test_lf)
    test_lf = add_time_and_recency_features_lazy(test_lf)
    test_lf = add_session_aggregates_lazy(test_lf)
    test_lf = add_query_intent_flags_lazy(test_lf)
    test_lf = add_text_match_features_lazy(test_lf)
    test_lf = add_cat_match_features_lazy(test_lf)
    test_lf = add_intent_interactions_lazy(test_lf)
    test_lf = join_content_search_rollups_asof(test_lf)
    test_lf = join_user_affinity_oof_like(test_lf)

    if term_ctr_df is not None:
        test_lf = test_lf.join(term_ctr_df.lazy(), on=["search_term_normalized"], how="left")
    if uterm_ctr_df is not None:
        test_lf = test_lf.join(uterm_ctr_df.lazy(), on=["user_id_hashed", "search_term_normalized"], how="left")
    if cterm_ctr_df is not None:
        test_lf = test_lf.join(cterm_ctr_df.lazy(), on=["content_id_hashed", "search_term_normalized"], how="left")
    if csite_df is not None:
        test_lf = test_lf.join(csite_df.lazy(), on=["content_id_hashed"], how="left")
    if csearch_df is not None:
        test_lf = test_lf.join(csearch_df.lazy(), on=["content_id_hashed"], how="left")
    if user_md_df is not None:
        test_lf = test_lf.join(user_md_df.lazy(), on=["user_id_hashed"], how="left")
    if term_idf_df is not None:
        test_lf = test_lf.join(term_idf_df.lazy(), on=["search_term_normalized"], how="left")

    if have_uc and all(c in have_cols_te for c in uc_cols):
        test_lf = test_lf.join(ctr_uc_full.lazy(), on=uc_cols, how="left")
    else:
        test_lf = test_lf.with_columns([pl.lit(None).alias("score_uc"), pl.lit(None).alias("n_uc")])
    if have_tc and all(c in have_cols_te for c in tc_cols):
        test_lf = test_lf.join(ctr_tc_full.lazy(), on=tc_cols, how="left")
    else:
        test_lf = test_lf.with_columns([pl.lit(None).alias("score_tc"), pl.lit(None).alias("n_tc")])
    test_lf = test_lf.join(ctr_c_full.lazy(), on=c_cols, how="left")

    test_lf = (
        test_lf.with_columns([
            (pl.when(pl.col("n_uc").is_not_null())
             .then(pl.col("n_uc") / (pl.col("n_uc") + pl.lit(ALPHA_UC)))
             .otherwise(0.0)).alias("w_uc"),
            (pl.when(pl.col("n_tc").is_not_null())
             .then(pl.col("n_tc") / (pl.col("n_tc") + pl.lit(ALPHA_TC)))
             .otherwise(0.0)).alias("w_tc"),
            (pl.when(pl.col("n_c").is_not_null())
             .then(pl.col("n_c") / (pl.col("n_c") + pl.lit(ALPHA_C)))
             .otherwise(0.0)).alias("w_c"),
        ])
        .with_columns([
            (
                    (pl.coalesce([pl.col("score_uc")]) * pl.col("w_uc") +
                     pl.coalesce([pl.col("score_tc")]) * pl.col("w_tc") +
                     pl.coalesce([pl.col("score_c")]) * pl.col("w_c"))
                    / (pl.col("w_uc") + pl.col("w_tc") + pl.col("w_c"))
            ).alias("__mix__")
        ])
        .with_columns([
            pl.coalesce([
                pl.col("__mix__"), pl.col("score_uc"), pl.col("score_tc"), pl.col("score_c")
            ]).fill_null(full_prior).alias("score_prior")
        ])
    )

    test_lf = add_ctr_flags_lazy(test_lf)
    test_df = test_lf.collect(streaming=STREAMING_COLLECT)

    print("OOF train örnek:")
    print(train_oof_df.head())
    print("Test örnek:")
    print(test_df.head())

    return train_oof_df, test_df, full_prior


# ---- Pandas split & model helpers ----
def _to_pandas_and_split_safe(train_df: pl.DataFrame, target_col: str):
    '''
    ENGLISH:
    •	Does: Converts Polars OOF train to Pandas, picks numeric features (excluding id-like columns), performs a group-aware 80/20 split by session (or user), and prepares arrays for ranking.
	•	Returns: X_tr, y_tr, grp_tr, X_va, y_va, grp_va, feature_cols, tr, va, group_key.
	•	Notes: If no inferred features, it falls back to a curated list; casts features/target to float32.

	TÜRKÇE:
	•	Ne yapar: Polars train OOF’u Pandas’a çevirir, grup (session_id ya da user) bazlı %20 validation böler; feature kolonlarını seçer/cast eder.
	•	Çıktı: (X_tr, y_tr, grp_tr, X_va, y_va, grp_va, feature_cols, tr, va, group_key).
	•	Not: Feature seçimi, tipleri sayısal olan ve id benzeri kolonlar dışındaki tüm kolonlardır (fallback listesi var).
    '''
    # group key
    if "session_id" in train_df.columns:
        group_key = "session_id"
    elif "user_id_hashed" in train_df.columns:
        group_key = "user_id_hashed"
    else:
        raise RuntimeError("Gruplama anahtarı yok (session_id / user_id_hashed).")

    # numeric features (exclude IDs/target)
    id_like = {
        "row_id", "session_id", "user_id_hashed", "content_id_hashed",
        "ts_hour", "search_term_normalized", "__fold__"
    }
    numeric_dtypes = {
        pl.Float64, pl.Float32,
        pl.Int64, pl.Int32, pl.Int16, pl.Int8,
        pl.UInt64, pl.UInt32, pl.UInt16, pl.UInt8,
        pl.Boolean,
    }
    schema = train_df.schema
    feature_cols = [
        c for c, dt in schema.items()
        if c not in id_like and c != target_col and dt in numeric_dtypes
    ]
    if not feature_cols:
        fallback_candidates = [
            "score_prior", "cs_ctr", "log1p_cs_impr",
            "price_over_cat_median", "price_z_in_cat", "discount_pct", "price_gap",
            "rating_tier", "popularity_tier", "media_review_ratio", "rate_to_review_ratio",
            "qtag_jaccard", "qtag_any_match", "term_in_l1_cat", "term_in_l2_cat", "term_in_leaf_cat",
            "price_rank_in_sess_pct", "discount_rank_in_sess_pct", "rating_rank_in_sess_pct",
            "sess_price_med", "sess_price_min", "sess_price_max", "sess_rating_mean", "sess_discount_max",
            "w_uc", "w_tc", "w_c", "score_uc", "score_tc", "score_c", "n_uc", "n_tc", "n_c",
            "log1p_n_uc", "log1p_n_tc", "log1p_n_c", "is_weekend", "hour_sin", "hour_cos", "dow_sin", "dow_cos"
        ]
        feature_cols = [c for c in fallback_candidates if c in train_df.columns]
    if not feature_cols:
        raise RuntimeError("Feature listesi boş. Feature üretimi çalışmamış olabilir.")

    sel_cols = [group_key, target_col] + feature_cols
    pldf_small = (
        train_df.select(sel_cols)
        .with_columns([pl.col(c).cast(pl.Float32) for c in feature_cols] +
                      [pl.col(target_col).cast(pl.Float32)])
    )
    pdf = pldf_small.to_pandas()

    uniq = pdf[group_key].unique()
    val_groups = pd.Series(uniq).sample(frac=0.20, random_state=42).values
    is_val = pdf[group_key].isin(val_groups)

    tr = pdf[~is_val].sort_values(group_key).reset_index(drop=True)
    va = pdf[is_val].sort_values(group_key).reset_index(drop=True)

    X_tr = tr[feature_cols]
    y_tr = tr[target_col]
    X_va = va[feature_cols]
    y_va = va[target_col]

    grp_tr = tr.groupby(group_key).size().values.tolist()
    grp_va = va.groupby(group_key).size().values.tolist()

    return X_tr, y_tr, grp_tr, X_va, y_va, grp_va, feature_cols, tr, va, group_key


def _train_cb_ranker(X_tr, y_tr, grp_tr, X_va, y_va, grp_va, feature_names, tr, va, group_key):
    '''
    ENGLISH:
    •	Does: Trains CatBoostRanker with YetiRank, evaluates NDCG:top=50, uses early stopping (od_type="Iter").
	•	Returns: cb.CatBoostRanker.
	•	Notes: Uses factorized group_id from group_key.

	TÜRKÇE:
	•	Ne yapar: CatBoost YetiRank ile sıralama modeli eğitir; eval metric NDCG:top=50.
	•	Çıktı: cb.CatBoostRanker modeli.
	•	Not: group_id olarak faktörize edilmiş grup kimliği kullanılır; use_best_model=True + od_wait.
    '''
    print("\n=== CatBoost Ranker ===")
    gtr = pd.factorize(tr[group_key])[0].astype(np.int64)
    gva = pd.factorize(va[group_key])[0].astype(np.int64)

    train_pool = cb.Pool(X_tr, label=y_tr, group_id=gtr, feature_names=feature_names)
    valid_pool = cb.Pool(X_va, label=y_va, group_id=gva, feature_names=feature_names)

    params = dict(
        loss_function="YetiRank",
        eval_metric="NDCG:top=50",
        iterations=1200,
        learning_rate=0.035,
        depth=8,
        l2_leaf_reg=3.0,
        random_seed=42,
        od_type="Iter",
        od_wait=200,
        task_type="CPU"
    )
    model = cb.CatBoostRanker(**params)
    model.fit(train_pool, eval_set=valid_pool, verbose=100, use_best_model=True)
    return model


def _predict_and_write_submission(model, feature_cols, test_df: pl.DataFrame, prior: float,
                                  out_path: str, batch_size: int = 200_000):
    '''
    ENGLISH:
    •	Does: Batched inference on test; applies sigmoid to raw scores; replaces non-finite with score_prior (row-wise) or prior (global); builds per-session ranked strings; writes CSV.
	•	Returns: None (writes to out_path).
	•	Notes: If session_id missing, falls back to user_id_hashed as session key.

	TÜRKÇE:
	•	Ne yapar: Test setini batch’ler halinde tahmin, logit → sigmoid çevirme, NaN/inf’te score_prior/prior fallback, submission strings oluşturma ve CSV yazma.
	•	Girdi: Trende kullanılan feature_cols.
	•	Çıktı: Dosyaya yazar; ayrıca kısa örnek log basar.
	•	Not: Oturum anahtarı yoksa user_id_hashed yedeği kullanılır.
	'''
    print("\n=== Predict & Submission (CatBoost) ===")
    if not isinstance(model, cb.CatBoost):
        raise TypeError("CatBoostRanker bekleniyor.")

    missing = [c for c in feature_cols if c not in test_df.columns]
    if missing:
        test_df = test_df.with_columns([pl.lit(0.0).cast(pl.Float32).alias(c) for c in missing])

    n = test_df.height
    preds = np.empty(n, dtype=np.float32)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        X_batch = (
            test_df
            .slice(start, end - start)
            .select([pl.col(c).fill_null(0.0).cast(pl.Float32) for c in feature_cols])
            .to_numpy()
        )
        p_raw = model.predict(X_batch)
        p_raw = np.asarray(p_raw, dtype=np.float32).reshape(-1)
        p = 1.0 / (1.0 + np.exp(-p_raw))
        bad = ~np.isfinite(p)
        if bad.any():
            if "score_prior" in test_df.columns:
                prior_batch = (
                    test_df.slice(start, end - start)
                    .select(pl.col("score_prior").fill_null(prior).cast(pl.Float32))
                    .to_numpy()
                    .reshape(-1)
                )
                p[bad] = prior_batch[bad]
            else:
                p[bad] = np.float32(prior)
        preds[start:end] = p

    if "session_id" in test_df.columns:
        sess_col = test_df["session_id"].cast(pl.Utf8)
    elif "user_id_hashed" in test_df.columns:
        sess_col = test_df["user_id_hashed"].cast(pl.Utf8)
    else:
        raise RuntimeError("Submission için 'session_id' (veya yedek olarak 'user_id_hashed') yok.")

    rank_df = pl.DataFrame({
        "session_id": sess_col,
        "content_id_hashed": test_df["content_id_hashed"].cast(pl.Utf8),
        "score": preds
    })

    submission_strings = make_session_strings(rank_df)
    submission_strings.write_csv(out_path)
    print(f"[KAYIT] submission -> {out_path}")
    print(submission_strings.head())


# ---- Orchestrator ----
def train_ltr_oof_and_write_submission(train_path, test_path, out_path, k_folds=5):
    '''
    ENGLISH:
    •	Does: End-to-end orchestration: build OOF features → split → train CatBoostRanker → predict on test → write submission.
	•	Returns: None (side-effects: training logs, submission file).

	TÜRKÇE:
	•	Ne yapar: Tüm orkestrasyon: OOF feature üret → Pandas split → CatBoostRanker eğit → Test tahmin → submission yaz.
	•	Çıktı: Yok (yan etki: dosya yazar).
	•	Not: Üst seviye çağrı; if train_sessions_file and test_sessions_file: bloğu burayı kullanır.
	'''
    train_df, test_df, global_prior = _build_oof_rank_features(train_path, test_path, k_folds)
    X_tr, y_tr, grp_tr, X_va, y_va, grp_va, feature_cols, tr, va, group_key = \
        _to_pandas_and_split_safe(train_df, TARGET_COL)
    model = _train_cb_ranker(X_tr, y_tr, grp_tr, X_va, y_va, grp_va, feature_cols, tr, va, group_key)
    _predict_and_write_submission(model, feature_cols, test_df, global_prior, out_path)


# ---- Run ----
if train_sessions_file and test_sessions_file:
    train_ltr_oof_and_write_submission(
        train_sessions_file, test_sessions_file, SUBMIT_LTR_PATH, k_folds=K_FOLDS
    )
else:
    print("[UYARI] OOF LTR için gerekli dosyalar yok (train/test).")
