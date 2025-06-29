import torch
import numpy as np
import time
import os
import cv2
from tqdm import tqdm

# Import components from the two versions
from pytorch_bench import load_yolo_model_from_pt as load_pytorch_model
from max_bench import (
    load_yolo_model_from_pt as load_max_model,
    engine,
    driver,
    letterbox,
    INPUT_SIZE,
    MODEL_WEIGHTS,
)

# --- BENCHMARKING FUNCTIONS ---


def benchmark_pytorch(model, input_tensor, num_runs=200, warmup_runs=20):
    print("\n--- Benchmarking PyTorch Model ---")
    device = input_tensor.device

    # Warm-up runs
    print(f"Performing {warmup_runs} warm-up runs...")
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Timed runs
    print(f"Performing {num_runs} timed runs...")
    timings = []
    with torch.no_grad():
        for _ in tqdm(range(num_runs)):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            _ = model(input_tensor)

            if device.type == "cuda":
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            timings.append((end_time - start_time) * 1000)

    avg_latency = np.mean(timings)
    print(f"PyTorch Average Latency: {avg_latency:.3f} ms")
    return avg_latency


def benchmark_max(max_model, input_tensor, num_runs=200, warmup_runs=20):
    print("\n--- Benchmarking Modular MAX Model ---")
    compiled_model = max_model.compiled_model

    # Warm-up runs
    print(f"Performing {warmup_runs} warm-up runs...")
    for _ in range(warmup_runs):
        _ = compiled_model.execute(input_tensor)

    # Timed runs
    print(f"Performing {num_runs} timed runs...")
    timings = []
    for _ in tqdm(range(num_runs)):
        start_time = time.perf_counter()
        _ = compiled_model.execute(input_tensor)
        end_time = time.perf_counter()
        timings.append((end_time - start_time) * 1000)

    avg_latency = np.mean(timings)
    print(f"MAX Average Latency: {avg_latency:.3f} ms")
    return avg_latency


def accuracy_check(pytorch_output, max_output):
    print("\n--- Performing Accuracy Check ---")
    pytorch_np = pytorch_output.cpu().numpy()

    # The MAX __call__ returns (raw_outputs, processed_output)
    # The processed_output is what we need to compare
    max_np = max_output[1]

    # PyTorch raw output: (1, 84, 8400) -> transpose -> (1, 8400, 84)
    # MAX processed output: (1, 8400, 84)
    pytorch_np_transposed = pytorch_np.transpose(0, 2, 1)

    if pytorch_np_transposed.shape != max_np.shape:
        print(
            f"❌ ACCURACY CHECK FAILED: Shape mismatch! PyTorch: {pytorch_np_transposed.shape}, MAX: {max_np.shape}"
        )
        return False

    if np.allclose(pytorch_np_transposed, max_np, rtol=1e-3, atol=1e-4):
        print("✅ Accuracy Check PASSED: Outputs are numerically very close.")
        return True
    else:
        diff = np.abs(pytorch_np_transposed - max_np).max()
        print(f"❌ ACCURACY CHECK FAILED: Max absolute difference is {diff}.")
        return False


# --- MAIN BENCHMARKING LOGIC ---

if __name__ == "__main__":
    if not os.path.exists(MODEL_WEIGHTS):
        print(
            f"Error: Model weights not found at '{MODEL_WEIGHTS}'. Please download yolov8n.pt and place it here."
        )
        exit()

    # 1. Prepare a consistent input tensor
    print("Preparing input tensor...")
    # Use a random numpy array to ensure consistency, instead of reading an image
    # This removes CV2 processing from the equation entirely.
    # The MAX model expects NHWC, PyTorch expects NCHW.
    input_np_nhwc = np.random.rand(1, INPUT_SIZE, INPUT_SIZE, 3).astype(np.float32)
    input_np_nchw = input_np_nhwc.transpose(0, 3, 1, 2)

    # 2. Setup PyTorch Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    pytorch_model = load_pytorch_model(MODEL_WEIGHTS, device)
    pytorch_input = torch.from_numpy(input_np_nchw).to(device)

    # 3. Setup MAX Model
    max_model_shell = load_max_model(MODEL_WEIGHTS)
    session = engine.InferenceSession()
    max_model_shell.compile(session)
    max_input = driver.Tensor.from_numpy(input_np_nhwc).to(max_model_shell.device)

    # 4. Run Accuracy Check First
    print("\n" + "=" * 50)
    print("RUNNING ACCURACY AND PERFORMANCE BENCHMARKS")
    print("=" * 50)
    with torch.no_grad():
        pytorch_raw_output = pytorch_model(pytorch_input)
    max_raw_and_processed_output = max_model_shell(max_input)

    if not accuracy_check(pytorch_raw_output, max_raw_and_processed_output):
        print("\nAborting performance benchmark due to accuracy failure.")
        exit()

    # 5. Run Benchmarks
    pytorch_latency = benchmark_pytorch(pytorch_model, pytorch_input)
    max_latency = benchmark_max(max_model_shell, max_input)

    # 6. Compare Results
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
        print("\nCould not calculate speedup due to zero latency.")
