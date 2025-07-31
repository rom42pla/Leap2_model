import onnxruntime as ort
import numpy as np
import time
import gc
import psutil
import os
from onnxruntime.quantization import quantize_dynamic, QuantType

def get_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)  # MB

def run_inference(session, inputs, n_runs=100):
    times = []
    gc.collect()

    # Warm-up run
    session.run(None, inputs)

    start_mem = get_memory_mb()
    peak_mem = start_mem

    for _ in range(n_runs):
        start = time.time()
        session.run(None, inputs)
        end = time.time()
        times.append((end - start) * 1000)  # ms
        mem = get_memory_mb()
        if mem > peak_mem:
            peak_mem = mem

    avg_time_ms = sum(times) / len(times)
    return avg_time_ms, start_mem, peak_mem

def quantize_model(input_path, output_path):
    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QUInt8
    )
    print(f"Quantized model saved to {output_path}")

def prepare_dummy_inputs(session):
    input_names = [inp.name for inp in session.get_inputs()]
    input_shapes = [inp.shape for inp in session.get_inputs()]
    input_dtypes = [np.float32 if inp.type == 'tensor(float)' else np.uint8 for inp in session.get_inputs()]

    dummy_inputs = {}
    for name, shape, dtype in zip(input_names, input_shapes, input_dtypes):
        shape = [s if isinstance(s, int) else 1 for s in shape]
        dummy_inputs[name] = np.random.rand(*shape).astype(dtype)
    return dummy_inputs

def main():
    original_model = "model.onnx"
    quantized_model = "model_int8_dynamic.onnx"
    print("Available providers on this machine:", ort.get_available_providers())
    # Quantize model dynamically (no calibration)
    quantize_model(original_model, quantized_model)

    # Load FP32 model and profile
    session_fp32 = ort.InferenceSession(original_model, providers=['QNNExecutionProvider'])
    dummy_inputs_fp32 = prepare_dummy_inputs(session_fp32)
    avg_fp32, mem_start_fp32, mem_peak_fp32 = run_inference(session_fp32, dummy_inputs_fp32, n_runs=10)
    print("\nFP32 Model Performance:")
    print(f"Avg Inference Time: {avg_fp32:.2f} ms")
    print(f"Memory Start: {mem_start_fp32:.2f} MB")
    print(f"Memory Peak: {mem_peak_fp32:.2f} MB")
    print(f"Memory Delta: {mem_peak_fp32 - mem_start_fp32:.2f} MB")

    # Load INT8 quantized model and profile
    session_int8 = ort.InferenceSession(quantized_model)
    dummy_inputs_int8 = prepare_dummy_inputs(session_int8)
    avg_int8, mem_start_int8, mem_peak_int8 = run_inference(session_int8, dummy_inputs_int8, n_runs=10)
    print("\nINT8 Dynamic Quantized Model Performance:")
    print(f"Avg Inference Time: {avg_int8:.2f} ms")
    print(f"Memory Start: {mem_start_int8:.2f} MB")
    print(f"Memory Peak: {mem_peak_int8:.2f} MB")
    print(f"Memory Delta: {mem_peak_int8 - mem_start_int8:.2f} MB")

if __name__ == "__main__":
    main()
