import cv2
import numpy as np
import time
from cv2_fourier_mellin import(
    FourierMellinWithReference,
)
from threading import (
    Thread,
)

def time_it(function, *args, **kwargs):
    start_time = time.time()
    result = function(*args, **kwargs)
    end_time = time.time()
    return (end_time - start_time, result)

def benchmark_without_threads(reference, target, *, iterations=100):
    frameSize = reference.shape[:2][::-1]
    fm = FourierMellinWithReference(*frameSize)
    fm.set_reference(reference, -1)

    for _ in range(iterations):
        _,_ = fm.register_image(target)

def benchmark_with_threads(reference, target, *, maxThreads=8, iterations=100):
    frameSize = reference.shape[:2][::-1]
    fm = FourierMellinWithReference(*frameSize)
    fm.set_reference(reference, -1)

    threads = [None for _ in range(maxThreads)]

    for iteration in range(iterations):
        i = iteration % maxThreads
        if threads[i] is not None:
            threads[i].join()
        threads[i] = Thread(target=fm.register_image, args=(target.copy(),))
        threads[i].start()
    for i,th in enumerate(threads):
        if th is not None:
            th.join()

def benchmark_with_threads_and_batches(reference, target, *, maxThreads=8, iterations=100, batchSize=10):
    assert(iterations%batchSize==0)
    frameSize = reference.shape[:2][::-1]
    fm = FourierMellinWithReference(*frameSize)
    fm.set_reference(reference, -1)

    threads = [None for _ in range(maxThreads)]

    for iteration in range(iterations//batchSize):
        i = iteration % maxThreads
        if threads[i] is not None:
            threads[i].join()
        threads[i] = Thread(target=fm.register_image_batched, args=([target.copy() for _ in range(batchSize)],))
        threads[i].start()
    for i,th in enumerate(threads):
        if th is not None:
            th.join()

def benchmark_with_threads_no_image(reference, target, *, maxThreads=8, iterations=100):
    frameSize = reference.shape[:2][::-1]
    fm = FourierMellinWithReference(*frameSize)
    fm.set_reference(reference, -1)

    threads = [None for _ in range(maxThreads)]

    for iteration in range(iterations):
        i = iteration % maxThreads
        if threads[i] is not None:
            threads[i].join()
        threads[i] = Thread(target=fm.register_image_only_transform, args=(target.copy(),))
        threads[i].start()
    for i,th in enumerate(threads):
        if th is not None:
            th.join()

if __name__ == '__main__':
    reference = cv2.imread('../images/lenna.png', cv2.IMREAD_COLOR)
    target = cv2.imread('../images/lenna_transformed.png', cv2.IMREAD_COLOR)
    reference = reference.astype(np.float32)
    target = target.astype(np.float32)

    iterations = 512
    print(f"Benchmarking without threads.")
    time1,_ = time_it(benchmark_without_threads, reference, target, iterations=iterations)
    print(f"Benchmarking with threads.")
    time2,_ = time_it(benchmark_with_threads, reference, target, maxThreads=4, iterations=iterations)
    print(f"Benchmarking with threads and batches.")
    time3,_ = time_it(benchmark_with_threads_and_batches, reference, target, maxThreads=4, iterations=iterations, batchSize=64)
    print(f"Benchmarking with threads and no transformed image.")
    time4,_ = time_it(benchmark_with_threads_no_image, reference, target, maxThreads=4, iterations=iterations)

    print(f"No threads {time1:0.2f}, Threads {time2:0.2f}, Threads and Batches {time3:0.2f}, Threads No Image {time4:0.2f}.")
