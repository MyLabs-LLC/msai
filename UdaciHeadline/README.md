# UdaciHeadline: LLM Inference Optimization

## Project Overview
This project focuses on optimizing the inference performance of a pre-trained Large Language Model (Llama-3.2-1B) fine-tuned for news headline generation using the News Category Dataset. The goal is to accelerate the headline generation pipeline by applying state-of-the-art LLM inference optimization techniques.

## Objectives
1.  **Baseline**: Establish a baseline inference pipeline and profile its performance.
2.  **Architectural Optimization**: Implement and evaluate KV-caching.
3.  **Model Compression**: Apply quantization (8-bit/4-bit) and pruning.
4.  **Distributed Inference**: Configure and benchmark Tensor and Pipeline Parallelism using DeepSpeed.
5.  **Advanced Decoding**: Apply speculative decoding.
6.  **Analysis**: Perform comprehensive benchmarking and produce a final report.

## Environment & Requirements
-   Python 3.10
-   PyTorch with CUDA support
-   Transformers, Datasets, Accelerate, Evaluate
-   ROUGE Score
-   Bitsandbytes, DeepSpeed
-   (Optional) TensorRT-LLM, Triton Inference Server

## Project Structure
-   `UdaciHeadline.ipynb`: Main notebook containing the implementation and benchmarking code.
-   `report.md`: Final report summarizing findings and trade-offs.

## Getting Started
1.  Install dependencies: `pip install -r requirements.txt`
2.  Open `UdaciHeadline.ipynb` and follow the steps.
