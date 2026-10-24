# Datasheet: What Makes a Video Go Viral? — 12,000 Videos with Engagement Metrics

## Motivation

**What is this dataset for?**
A synthetic YouTube-style dataset with engagement metrics and a clear viral target. Designed for classification (predict viral success), regression (predict views), and content strategy analysis.

**Why was it created?**
Real YouTube datasets require API access and often lack a clean binary target. This dataset provides a reproducible benchmark with realistic category-specific view distributions.

## Composition

**What do the instances represent?**
Each instance is a synthetic video with metadata, engagement metrics, and a viral outcome.

**How many instances?**
12,000 videos across 15 content categories.

**What data does each instance consist of?**
- **Identifiers:** video_id, category
- **Metadata:** duration_sec, title_length, thumbnail_face, upload_hour, upload_day, tags_count, description_length
- **Channel:** subscriber_count
- **Engagement:** views, likes, comments, shares
- **Target:** views (regression), viral (classification, top 20% of views)

**Is there a label or target?**
Yes. `viral` is the binary classification target. `views` is the regression target.

## Collection Process

**How was the data created?**
Synthetically generated via Python. Channel sizes and view counts follow lognormal distributions. Engagement metrics are computed as ratios of views, with realistic ranges per category. Viral status derived from top-20% threshold on views.

## Preprocessing

Data is generated in final format.

## Uses

- Binary classification: predict viral videos
- Regression: predict views (log-transform recommended)
- Analysis: which categories and features drive virality?
- Feature importance: subscriber count vs. engagement ratios
- Content strategy: does upload timing matter?

**Should NOT be used for:**
- Real YouTube analytics (synthetic data)
- Any commercial content strategy decisions

## Distribution

**License:** CC0 Public Domain

## Maintenance

**Maintainer:** Vedauv Rajkumar Leelavathi
**Contact:** https://www.kaggle.com/vedauv