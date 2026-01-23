# UdaciHeadline Optimization Report

## 1. Methodology
This project explored several optimization techniques to accelerate the inference of the Llama-3.2-1B model for news headline generation.

### Techniques Applied:
- **Baseline**: Standard FP16 inference.
- **KV-Caching**: Caching key/value states to avoid re-computation.
- **Pruning**: Unstructured magnitude-based pruning (30% sparsity).
- **Quantization**: 8-bit and 4-bit quantization using `bitsandbytes`.
- **Distributed Inference**: Tensor and Pipeline parallelism (if applicable).
- **Speculative Decoding**: Using a 4-bit draft model to accelerate the FP16 target model.

## 2. Benchmark Results

| Model                  |   Avg Latency (ms) |   Throughput (tok/s) |   Peak Memory (GB) | ROUGE-1   |
|:-----------------------|-------------------:|---------------------:|-------------------:|:----------|
| Baseline (No Cache)    |             485.28 |                38.95 |               2.4  | 0.2623    |
| KV-Cache               |             196.2  |                96.84 |               2.43 | 0.3087    |
| Pruned (30%)           |             173.18 |                97.01 |               2.44 | 0.1351    |
| Quantization (8-bit)   |             667.38 |                28.02 |               0.96 | 0.3379    |
| Quantization (4-bit)   |             286.73 |                62.08 |               0.77 | 0.2522    |
| Distributed (Auto)     |             226.5  |                74.17 |               1.51 | 0.1678    |
| Distributed (Balanced) |             256.92 |                75.12 |               1.51 | 0.2684    |
| Speculative Decoding   |             428.59 |                44.8  |               3.1  | N/A       |

## 3. Analysis of Trade-offs

### Performance vs. Quality
- **KV-Caching**: KV-Caching dramatically improved speed, reducing average latency by ~60% (from 485ms to 196ms) and increasing throughput by ~148% (38.95 to 96.84 tok/s). Quality (ROUGE-1) actually improved slightly to 0.3087, confirming it is a highly effective and safe optimization.
- **Quantization**: 4-bit quantization provided a good balance, reducing latency to 286ms (faster than baseline) while maintaining decent quality (ROUGE 0.2522). 8-bit quantization, however, increased latency to 667ms (slower than baseline) likely due to overhead, despite achieving the highest ROUGE score (0.3379).
- **Pruning**: Pruning (30% sparsity) achieved the lowest latency (173ms) and highest throughput (97 tok/s), but it came at a significant cost to quality, dropping the ROUGE-1 score to 0.1351. This indicates that unstructured pruning without retraining degrades the model's summarization capability too much to be viable.

### Resource Usage
- **Memory Footprint**: **4-bit Quantization** was the most memory-efficient method, reducing peak memory usage to just **0.77 GB**, which is less than one-third of the baseline usage (2.4 GB). 8-bit quantization also offered significant savings (0.96 GB).

## 4. Conclusion & Recommendation
Based on the results, the most effective strategy for this task is **KV-Caching** because it delivers the best overall performance (196ms latency, 96.84 tok/s) with high quality (0.3087 ROUGE) and no additional memory cost compared to the baseline.

For scenarios where memory is strictly limited (e.g., edge devices with < 1GB VRAM), **4-bit Quantization** is the recommended alternative, offering a massive memory reduction (to 0.77 GB) while still outperforming the baseline in speed.
