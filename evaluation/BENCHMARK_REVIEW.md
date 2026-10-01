# Benchmark review

Reviewed against the 11 user-supplied PDFs on 2026-10-01. PDF page numbers are
1-based file pages, including cover pages, rather than printed article pages.
The benchmark contains 12 French questions supported by five documents.

| ID | Source | PDF pages | Evidence checked |
| --- | --- | --- | --- |
| ad-01 | DeepAnT | 6 | Figure 1 and architecture summary: convolution, pooling, 32 filters, ReLU, next-step output |
| ad-02 | DeepAnT | 6 | Anomaly detector: Euclidean prediction residual used as anomaly score |
| ad-03 | DeepAnT | 7 | Yahoo protocol: 40% training, 60% test, validation uses 10% of the training portion; no training labels |
| ad-04 | LSTM-AD | 2–4 | Stacked LSTM prediction, multivariate Gaussian error likelihood, threshold; normal and anomalous validation sets used to choose threshold |
| ad-05 | LSTM-AD | 3 | Motivation explicitly discusses abundant normal and rare anomalous examples |
| ad-06 | Schmidl et al. | 1, 7 | Evaluation scope: 71 algorithms, 976 real and synthetic time series |
| ad-07 | Schmidl et al. | 10 | Every family can be effective; no clear winner and no universally perfect algorithm |
| ad-08 | Choi et al. | 13 | Figure 8 and anomaly-criteria taxonomy: reconstruction, prediction, dissimilarity |
| ad-09 | Choi et al. | 12 | CNN section describes segmented patterns and difficulty with long-period behavior |
| ad-10 | SORO/TARCHOUNA report | 28 | Table 3.1: Rand Index on Trace, Gun Point, SyntheticControl |
| ad-11 | SORO/TARCHOUNA report | 29, 32 | Extraction bottleneck and complexity stated by the report; not an independently verified complexity derivation |
| ad-12 | SORO/TARCHOUNA report | 28–29 | Discussion of subsequence length and selection threshold; over/under-selection of shapelets |

Two references were clarified: DeepAnT's validation fraction applies to its
training portion; LSTM-AD's threshold calibration uses anomalous validation
examples, even though prediction-model training uses normal examples.
No unsupported entries were retained. No scores were inferred from references.

The corpus includes time-series clustering material in addition to anomaly
detection papers; the three U-Shapelets questions evaluate the supplied report,
not anomaly-detection performance. Sources and exact pages are recorded in each
JSONL entry. All 11 PDFs participate in retrieval, although only five have
reference questions. The small benchmark is exploratory, is not exhaustive,
and was reviewed by one assistant rather than an independent annotation panel.
The PDF files themselves are not uploaded to this public repository.
