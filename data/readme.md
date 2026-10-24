# What Makes a Video Go Viral? — 12,000 Videos with Engagement Metrics

## What is this?

A synthetic YouTube-style dataset with 12,000 videos across 15 categories. Includes metadata, channel size, engagement metrics, and a clear viral target.

## Why should you use it?

- **Clean, complete data** — no missing values
- **Defined target** — `viral` (top 20% of views) for classification, `views` for regression
- **Realistic distributions** — category-specific view and engagement patterns
- **Multiple tasks** — classification, regression, feature importance
- **Universal appeal** — anyone who watches YouTube can relate

## Columns

| Column | Description |
|--------|-------------|
| `video_id` | Unique identifier |
| `category` | Content category (15 categories) |
| `duration_sec` | Video length in seconds |
| `title_length` | Characters in title |
| `thumbnail_face` | 1 if thumbnail contains a face |
| `upload_hour` | Hour of day uploaded (0-23) |
| `upload_day` | Day of week uploaded |
| `tags_count` | Number of tags |
| `description_length` | Characters in description |
| `subscriber_count` | Channel subscriber count |
| `views` | Total views |
| `likes` | Like count |
| `comments` | Comment count |
| `shares` | Share count |
| `viral` | **Target:** top 20% of views |

## Suggested Tasks

- **Classification:** Predict `viral` from metadata + engagement
- **Regression:** Predict `views` (log-transform first)
- **Analysis:** Which categories have the highest viral rate?
- **Feature importance:** Does subscriber count dominate?
- **Timing:** Do certain upload hours/days correlate with virality?

## Quick Start

```python
import pandas as pd

df = pd.read_csv("viral_videos.csv")
print(df.shape)
print(f"Viral rate: {df['viral'].mean()*100:.1f}%")
print(df.head())
```

## License
CC0 Public Domain
