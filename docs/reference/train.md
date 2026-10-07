# Fine-tuning

`laya.train` is imported on its own (`from laya.train import finetune, TrainConfig`); it is not
re-exported from the top-level package. `laya-train` is its command line. The
[fine-tuning guide](../finetune.md#fine-tune-with-laya-train) shows both in use.

## Running a fine-tune

::: laya.train.TrainConfig

::: laya.train.finetune

::: laya.train.dry_run

## The two halves of `finetune`

::: laya.train.train_model

::: laya.train.calibration_records

::: laya.train.calibration_report

## Reading data

::: laya.train.read_data

::: laya.train.rows_from_csv

## Building training items

::: laya.train.items_from_rows

::: laya.train.target_from_gold

::: laya.train.target_from_expected

::: laya.train.to_internal

::: laya.train.make_item

::: laya.train.encode_item

::: laya.train.draw_option_order

::: laya.train.split_calibration

## Losses

::: laya.train.soft_ce_loss

::: laya.train.rlcd_loss

## Checkpoints

::: laya.train.resolve_checkpoint_dir

::: laya.train.load_checkpoint

::: laya.train.save_checkpoint
