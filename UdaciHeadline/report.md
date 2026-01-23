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

| Model | Avg Latency (ms) | Throughput (tok/s) | Peak Memory (GB) | ROUGE-1 |
|-------|------------------|--------------------|------------------|---------|
| Model                |   Avg Latency (ms) |   Throughput (tok/s) |   Peak Memory (GB) | ROUGE-1   |
|:---------------------|-------------------:|---------------------:|-------------------:|:----------|
| Baseline (No Cache)  |             606.92 |                31.14 |               2.4  | 0.0       |
| KV-Cache             |             258.27 |                64.27 |               2.43 | 0.0       |
| Pruned (30%)         |             260.16 |                63.04 |               2.44 | 0.0       |
| Quantization (8-bit) |             870.85 |                20.9  |               1.55 | 0.0       |
| Quantization (4-bit) |             310.96 |                54.03 |               1.15 | 0.0       |
| Speculative Decoding |             424.08 |                42.92 |               3.48 | N/A       |

## 3. Analysis of Trade-offs

### Performance vs. Quality
- **KV-Caching**: [User to complete: Did it improve speed? Did quality change?]
- **Quantization**: [User to complete: Impact on latency and ROUGE scores?]
- **Pruning**: [User to complete: Did 30% sparsity affect performance?]

### Resource Usage
- **Memory Footprint**: [User to complete: Which method saved the most memory?]

## 4. Conclusion & Recommendation
Based on the results, the most effective strategy for this task is **[User to complete]** because...
