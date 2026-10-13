# Warmup results

500-step single-seed packed optimizer states. Engineering memory/time measurements; no convergence or fused-kernel speedup claims.

|Optimizer|Variant|Steps|Final val loss|Seconds|Peak GiB|
|---|---|---:|---:|---:|---:|
|lion|L0|500|5.896123|661.6|2.667|
|lion|L1|500|5.902836|694.7|2.432|
|lion|L2|500|5.891776|730.8|2.316|
|signum|L0|500|6.055888|679.7|2.666|
|signum|L1|500|6.057379|687.6|2.435|
|signum|L2|500|6.050562|696.5|2.316|
|cadamw|A0|500|6.410835|734.5|3.146|
|cadamw|A1|500|6.413031|736.9|2.922|
|cadamw|A2|500|6.415615|732.1|2.919|
|cadamw|A3|500|6.413741|748.9|2.790|
|cadamw|A4|500|6.369463|756.7|2.790|
