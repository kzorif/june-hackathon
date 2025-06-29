# filename: benchmark_perf_only.py
import torch
import numpy as np
import time
import os
from tqdm import tqdm

# Import components from the two versions
from pytorch_bench import load_yolo_model_from_pt as load_pytorch_model
from max_bench import (
    load_yolo_model_from_pt as load_max_model,
    engine,
    driver,
    INPUT_SIZE,
    MODEL_WEIGHTS,
)

# --- BENCHMARKING FUNCTIONS ---


def benchmark_pytorch(model, input_tensor, num_runs=200, warmup_runs=20):
    """Benchmarks a PyTorch model's inference latency."""
    print("\n--- Benchmarking PyTorch Model ---")
    device = input_tensor.device

    # Warm-up runs to cache kernels and stabilize system
    print(f"Performing {warmup_runs} warm-up runs...")
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Timed runs for stable measurement
    print(f"Performing {num_runs} timed runs...")
    timings = []
    with torch.no_grad():
        for _ in tqdm(range(num_runs), desc="PyTorch Inference"):
            # Synchronize before starting the timer for accurate GPU measurement
            if device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            # The actual inference call
            _ = model(input_tensor)

            # Synchronize after the operation to ensure it's complete
            if device.type == "cuda":
                torch.cuda.synchronize()
            end_time = time.perf_counter()

            # Store result in milliseconds
            timings.append((end_time - start_time) * 1000)

    avg_latency = np.mean(timings)
    print(f"PyTorch Average Latency: {avg_latency:.3f} ms")
    return avg_latency


def benchmark_max(max_model, input_tensor, num_runs=200, warmup_runs=20):
    """Benchmarks a compiled MAX model's inference latency."""
    print("\n--- Benchmarking Modular MAX Model ---")
    compiled_model = max_model.compiled_model

    # Warm-up runs
    print(f"Performing {warmup_runs} warm-up runs...")
    for _ in range(warmup_runs):
        _ = compiled_model.execute(input_tensor)

    # Timed runs
    print(f"Performing {num_runs} timed runs...")
    timings = []
    for _ in tqdm(range(num_runs), desc="MAX Inference    "):
        start_time = time.perf_counter()

        # The actual inference call using the MAX engine
        _ = compiled_model.execute(input_tensor)

        end_time = time.perf_counter()

        # Store result in milliseconds
        timings.append((end_time - start_time) * 1000)

    avg_latency = np.mean(timings)
    print(f"MAX Average Latency: {avg_latency:.3f} ms")
    return avg_latency


# --- MAIN BENCHMARKING LOGIC ---

if __name__ == "__main__":
    if not os.path.exists(MODEL_WEIGHTS):
        print(
            f"Error: Model weights not found at '{MODEL_WEIGHTS}'. Please download yolov8n.pt and place it here."
        )
        exit()

    print("\n" + "=" * 50)
    print("RUNNING PERFORMANCE-ONLY BENCHMARKS")
    print("=" * 50)

    # 1. Prepare a consistent input tensor
    print("Preparing input tensor...")
    # Using a random numpy array for input ensures both models get the exact same data.
    # The MAX model expects NHWC, PyTorch expects NCHW.
    input_np_nhwc = np.random.rand(1, INPUT_SIZE, INPUT_SIZE, 3).astype(np.float32)
    input_np_nchw = input_np_nhwc.transpose(0, 3, 1, 2)

    # 2. Setup PyTorch Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    pytorch_model = load_pytorch_model(MODEL_WEIGHTS, device)
    pytorch_input = torch.from_numpy(input_np_nchw).to(device)

    # 3. Setup MAX Model
    max_model_shell = load_max_model(MODEL_WEIGHTS)
    session = engine.InferenceSession()
    max_model_shell.compile(session)
    max_input = driver.Tensor.from_numpy(input_np_nhwc).to(max_model_shell.device)

    # 4. Run Benchmarks
    pytorch_latency = benchmark_pytorch(pytorch_model, pytorch_input)
    max_latency = benchmark_max(max_model_shell, max_input)

    # 5. Compare and Report Results
    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    print(f"PyTorch Average Latency: {pytorch_latency:.3f} ms")
    print(f"MAX Engine Average Latency: {max_latency:.3f} ms")

    if max_latency > 0:
        speedup = pytorch_latency / max_latency
        print(f"\nSpeedup (MAX vs PyTorch): {speedup:.2f}x")
        print(
            f"The MAX Engine is {speedup:.2f} times faster for this model on this hardware."
        )
    else:
        print("\nCould not calculate speedup due to zero or negative latency.")
