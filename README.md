ENGLISH:
Trendyol Fashion Search — Learning to Rank (Clicks & Orders)

Overview

This repository trains a leakage-safe Learning-to-Rank model to reorder Trendyol fashion search results so that items most likely to be clicked and/or ordered surface first.
Stack: Polars for fast feature engineering, CatBoostRanker (YetiRank) for ranking.

What you get
	•	Auto file discovery under DATA_ROOT (train/test Parquet + optional logs/metadata).
	•	Rich features: price & discount, session stats, time signals, text/tag/category matches, user affinity, rolling content search metrics.
	•	OOF CTR features for (user,content), (term,content), (content) with Bayesian smoothing (no leakage).
	•	Group-aware training by session_id, submission writer per session.

Data layout (minimum)
	•	Train: row_id, session_id, user_id_hashed, content_id_hashed, ts_hour, search_term_normalized, clicked
(Add ordered if you’ll train on orders).
	•	Test:  row_id, session_id, user_id_hashed, content_id_hashed, ts_hour, search_term_normalized

Optional files (used if present): content/user sitewide logs, top-terms logs, term search log, content metadata, price/rating snapshots.
The script searches paths via predefined patterns; missing optional files are skipped gracefully.

QUICK START;

1) INSTALL:
   pip install polars==0.20.20 pandas numpy catboost

2) CONFIGURE:
    DATA_ROOT = "YOUR FILE PATH"
    OUTPUT_DIR = "YOUR FILE PATH"
    TARGET_COL = "clicked"   # or "ordered"
    K_FOLDS    = 5

3) RUN:
    python your_script.py

4) OUTPUT:
    OUTPUT_DIR/submission_ltr.csv with:
	•	session_id
	•	prediction → space-separated content_id_hashed ranked by model

Feature sketch (short)
	•	Price/Discount: discount_pct, gap, category z-score/tiers, has_discount.
	•	Session stats: median/min/max price, mean rating, max discount, per-item deltas.
	•	In-session ranks: normalized ranks by price/discount/rating.
	•	Time: hour/dow sine/cosine, weekend, content_age_days.
	•	Rolling search: 7/14/30-day content search clicks/impressions (as-of join).
	•	Text/Category/Tags: query vs L1/L2/leaf & cv_tags Jaccard/any-match.
	•	User affinity: sitewide user clicks (log1p), price-preference proxy.
	•	OOF CTRs: (u,c), (t,c), (c) blended with priors (no leakage).

Notes / Troubleshooting
	•	“[UYARI] ‘’ için dosya bulunamadı” → optional source missing; pipeline continues.
	•	“OOF için ‘session_id’ gerekiyor” → ensure session_id exists in train/test.
	•	Memory tight? Reduce K_FOLDS, lower CatBoost depth/iterations, keep STREAMING_COLLECT=True.


 TÜRKÇE:

Trendyol Moda Arama — Sıralama (Click & Order)

Özet

Bu repo, Trendyol moda arama sonuçlarını tıklanma ve sipariş olasılığına göre yeniden sıralayan sızıntı-güvenli bir LTR hattı sunar.
Teknoloji: Polars ile hızlı feature mühendisliği, CatBoostRanker (YetiRank) ile sıralama.

Sağlananlar
	•	DATA_ROOT altında otomatik dosya bulma (train/test Parquet + opsiyonel log/metadata).
	•	Zengin feature seti (fiyat/indirim, seans istatistikleri, zaman, metin/etiket/kategori, kullanıcı afinitesi, rolling arama metrikleri).
	•	(kullanıcı, içerik) / (terim, içerik) / (içerik) OOF CTR (Bayes yumuşatma, sızıntısız).
	•	session_id bazlı grup farkındalığı, submission yazımı.

Asgari veri
	•	Train: row_id, session_id, user_id_hashed, content_id_hashed, ts_hour, search_term_normalized, clicked
(Order eğitimi için ordered ekleyin).
	•	Test:  row_id, session_id, user_id_hashed, content_id_hashed, ts_hour, search_term_normalized

Opsiyonel kaynaklar varsa otomatik kullanılır; yoksa atlanır.

HIZLI KULLANIM;

1) KURULUM:
   pip install polars==0.20.20 pandas numpy catboost

2) AYAR:
    DATA_ROOT = "YOUR FILE PATH"
    OUTPUT_DIR = "YOUR FILE PATH"
    TARGET_COL = "clicked"   # or "ordered"
    K_FOLDS    = 5

3) ÇALIŞTIR:
    python your_script.py

4) ÇIKTI:
    OUTPUT_DIR/submission_ltr.csv:
	•	session_id
	•	prediction → skora göre sıralı content_id_hashed (boşlukla ayrılmış)

Özellik Özeti (kısa)
	•	Fiyat/İndirim: discount_pct, fiyat farkı (gap), kategori z-skoru/katmanları, has_discount.
	•	Seans istatistikleri: medyan/min/maks fiyat, ortalama puan, maksimum indirim, ürün-bazlı farklar.
	•	Seans içi sıralamalar: fiyat/indirim/puan normalize sıraları.
	•	Zaman: saat/gün sinüs/kosinüs, hafta sonu, content_age_days.
	•	Rolling arama: 7/14/30 günlük içerik arama tıklama/gösterimleri (as-of join).
	•	Metin/Kategori/Etiket: sorgu vs L1/L2/leaf ve cv_tags için Jaccard / “any-match”.
	•	Kullanıcı afinitesi: site-genel kullanıcı tıklamaları (log1p), fiyat-tercih temsili.
	•	OOF CTR’lar: (kullanıcı,içerik), (terim,içerik), (içerik) — prior ile harmanlanmış (sızıntı yok).

Notlar / Sorun Giderme
	•	“[UYARI] ‘’ için dosya bulunamadı” → opsiyonel kaynak eksik; pipeline devam eder.
	•	“OOF için ‘session_id’ gerekiyor” → train/test içinde session_id olduğundan emin olun.
	•	Bellek kısıtlı mı? K_FOLDS’u azaltın, CatBoost depth/iterations değerlerini düşürün, STREAMING_COLLECT=True bırakın.
