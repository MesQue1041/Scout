# Project Lighthouse: log anomaly detection

## Goal

Detect anomalous sessions in the HDFS log dataset without labels at training time.

## Approach

Parsed raw logs into event templates with Drain, then built per-session event-count vectors.

Baseline: Isolation Forest. F1 score 0.71 on the held-out test set.

Main model: LSTM autoencoder trained on normal sessions only. A session is flagged when its
reconstruction error exceeds the 99th percentile of errors on the validation set.
F1 score 0.86 on the same test set.

## Open issues

Concept drift: the threshold was fit once and will go stale as log patterns change.
Training on a single dataset means we don't know how well it generalises.

## Next steps

Try the BGL dataset. Look at a rolling threshold instead of a fixed percentile.
