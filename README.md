# ParallelAlgorithms

Project 1: Parallel Image Processing System

This project implements a multi-threaded and multi-process image processing system that applies various transformations to large image datasets. The system consists of:

1) Image and task registries for tracking processed images, applied filters, and their statuses.
2) A command-processing main thread that listens for user input and delegates tasks.
3) Worker threads that handle operations such as image addition, deletion, and processing.
4) Parallel image processing using Python’s multiprocessing module, ensuring task execution in separate processes for efficiency.
5) Implemented transformations include:
 - Grayscale conversion (computing pixel luminance using weighted sums of RGB components).
 - Gaussian Blur (applying a convolution with a Gaussian kernel for smooth blurring).
 - Brightness Adjustment (scaling pixel intensities based on computed average brightness).
6) Inter-thread communication via Queues and Condition variables, ensuring synchronization of tasks.
7) Task dependency management: images marked for deletion are only removed after dependent tasks complete.

This project demonstrates advanced parallel programming techniques in Python, utilizing the multiprocessing and threading modules to optimize performance in batch image processing.

Project 2: Color Histogram Analysis and Classification

This project focuses on functional programming and parallel data processing techniques to analyze images based on color histograms. The goal is to classify images into different categories by computing and comparing their histograms. The key components include:

1) Histogram computation using map() and reduce():
 - Each image’s histogram is calculated by counting pixel occurrences across RGB channels.
 - The computed histograms are normalized (dividing by the total number of pixels).
2) Class-based average histogram computation:
 - Histograms of all images in a class are aggregated using reduce(), and then averaged.
3) Cosine similarity computation:
 - Histograms are flattened into vectors and compared using cosine similarity to measure closeness.
4) A simple image classifier:
 - Given an input image, its histogram is compared to precomputed average histograms of each category.
 - The classifier assigns the image to the most similar category using cosine similarity.

The project enforces constraints such as avoiding explicit loops, instead requiring the use of functional programming techniques (map, reduce, lambda functions, and iterators). The dataset for classification is sourced from CIFAR-10 and all image processing is done using NumPy for optimized numerical computations.

Project 3: CUDA-Based Parallel Image Processing

This project explores high-performance parallel computing for image processing using CUDA and PyCUDA. The focus is on accelerating three fundamental image transformations:

1) Grayscale Conversion:
 - Uses a CUDA kernel where each thread computes the grayscale intensity of a single pixel.
 - The formula 0.299 * R + 0.587 * G + 0.114 * B is applied in parallel across all pixels.
2) Gaussian Blur:
 - Implements a convolution operation using a Gaussian kernel.
 - Each thread processes a pixel by applying a weighted sum of its neighboring pixels.
 - The Gaussian kernel is precomputed on the CPU and transferred to the GPU’s constant memory for efficiency.
 - CUDA shared memory is used to minimize redundant memory accesses.
3) Brightness Adjustment:
 - Computes the average pixel intensity across the image using parallel reduction.
 - Adjusts each pixel’s brightness proportionally to the computed average.
 - Uses two separate CUDA kernels: one for summing intensities and one for modifying pixel values.

Key optimizations in this project include:
 - Thread-level parallelism: Each thread processes one pixel independently.
 - Shared memory utilization: Reducing global memory accesses for performance improvement.
 - Efficient edge handling: Implementing strategies such as padding and mirror padding.

The project showcases the power of GPU computing with CUDA, demonstrating real-world applications of parallel image processing in high-performance computing environments.
