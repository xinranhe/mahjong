# Mahjong
ML for japanese mahjong

# Usage

## Crawling game data
```python -m crawler.main --start_date 20180101 --end_date 20181231 --output_data_dir data/raw```

## Prepare discarded Hai prediction training data in TFRecord
```python -m log_parser.discard_prediction_training_data --start_date 20180101 --end_date 20181231```

## Running training for discarded Hai prediction
```python -m trainer.task```
