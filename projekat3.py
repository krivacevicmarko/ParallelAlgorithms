# -*- coding: utf-8 -*-
"""projekat3.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Hyx6mf8CiU51Xv2Ddm0rvYSHWYitYJwm
"""

!pip install pycuda

"""**GRAYSCALE**"""

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

image = Image.open("dolina.jpg")
image = image.convert("RGB")
rgb_array = np.array(image).astype(np.float32)
height, width, channels = rgb_array.shape

input_image_gpu = cuda.mem_alloc(rgb_array.nbytes)
output_image_gpu = cuda.mem_alloc(height * width * np.float32().nbytes)

cuda.memcpy_htod(input_image_gpu, rgb_array)

mod = SourceModule("""
    __global__ void grayscale(float *input, float *output, int width, int height) {
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;

        if (x < width && y < height) {
            int idx = (y * width + x) * 3; // Indeks u RGB nizu (svaki piksel ima 3 vrednosti)
            float r = input[idx];
            float g = input[idx + 1];
            float b = input[idx + 2];

            // Grayscale formula
            float gray = 0.299f * r + 0.587f * g + 0.114f * b;

            // Upisujemo grayscale vrednost
            output[y * width + x] = gray;
        }
    }
""")

block_dim = (32, 32, 1)
grid_dim = (int(np.ceil(width / block_dim[0])), int(np.ceil(height / block_dim[1])), 1)

grayscale_kernel = mod.get_function("grayscale")
grayscale_kernel(input_image_gpu, output_image_gpu, np.int32(width), np.int32(height), block=block_dim, grid=grid_dim)

output_image = np.empty((height, width), dtype=np.float32)
cuda.memcpy_dtoh(output_image, output_image_gpu)

output_image = (output_image * 255 / output_image.max()).astype(np.uint8)
grayscale_image = Image.fromarray(output_image)
grayscale_image.show()

grayscale_image.save("grayscale_output.jpg")

"""**BRIGHTNESS ADJUSTMENT**"""

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
from PIL import Image

image = Image.open("dolina.jpg")
image = image.convert("RGB")
rgb_array = np.array(image).astype(np.float32)
height, width, channels = rgb_array.shape
num_pixels = height * width


input_image_gpu = cuda.mem_alloc(rgb_array.nbytes)
output_image_gpu = cuda.mem_alloc(rgb_array.nbytes)
partial_sums_gpu = cuda.mem_alloc(256 * channels * np.float32().nbytes)

cuda.memcpy_htod(input_image_gpu, rgb_array)

mod = SourceModule("""
    __global__ void calculate_sum_rgb(float *input, float *partial_sums, int width, int height) {
        __shared__ float shared_data[256];
        int tid = threadIdx.x;
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        int channel_offset = width * height;

        for (int c = 0; c < 3; c++) {
            shared_data[tid] = (idx < channel_offset) ? input[idx + c * channel_offset] : 0;
            __syncthreads();

            for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    shared_data[tid] += shared_data[tid + s];
                }
                __syncthreads();
            }

            if (tid == 0) {
                partial_sums[blockIdx.x + c * gridDim.x] = shared_data[0];
            }
        }
    }
""")

mod_brightness_rgb = SourceModule("""
    __global__ void adjust_brightness_rgb(float *input, float *output, float *avg_intensity, float factor, int width, int height) {
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;

        int idx = y * width + x;
        int channel_offset = width * height;

        if (x < width && y < height) {
            for (int c = 0; c < 3; c++) {
                float value = input[idx + c * channel_offset];
                value = avg_intensity[c] + (value - avg_intensity[c]) * factor;
                output[idx + c * channel_offset] = fminf(fmaxf(value, 0.0f), 255.0f);
            }
        }
    }
""")

block_dim = (16, 16, 1)
grid_dim = (int(np.ceil(width / block_dim[0])), int(np.ceil(height / block_dim[1])), 1)

calculate_sum_rgb = mod.get_function("calculate_sum_rgb")
calculate_sum_rgb(input_image_gpu, partial_sums_gpu, np.int32(width), np.int32(height), block=(256, 1, 1), grid=(grid_dim[0], 1, 1))

partial_sums = np.empty((grid_dim[0] * 3), dtype=np.float32)
cuda.memcpy_dtoh(partial_sums, partial_sums_gpu)

total_sums = np.zeros(3, dtype=np.float32)
for i in range(3):
    total_sums[i] = np.sum(partial_sums[i * grid_dim[0]:(i + 1) * grid_dim[0]])

avg_intensity = total_sums / num_pixels
avg_intensity = avg_intensity.astype(np.float32)

avg_intensity_gpu = cuda.mem_alloc(3 * np.float32().nbytes)
cuda.memcpy_htod(avg_intensity_gpu, avg_intensity)

factor = 1.5

adjust_brightness_rgb = mod_brightness_rgb.get_function("adjust_brightness_rgb")
adjust_brightness_rgb(input_image_gpu, output_image_gpu, avg_intensity_gpu, np.float32(factor),
                      np.int32(width), np.int32(height), block=block_dim, grid=grid_dim)

output_image = np.empty_like(rgb_array)
cuda.memcpy_dtoh(output_image, output_image_gpu)

output_image = output_image.astype(np.uint8)
output_image_pil = Image.fromarray(output_image, mode="RGB")
output_image_pil.show()
output_image_pil.save("brightness_adjusted_rgb.jpg")

"""**GAUSSIAN BLUR**"""

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
from PIL import Image

def create_gaussian_kernel(size, sigma):
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)
    return kernel.astype(np.float32)

kernel_size = 5
sigma = 1.2
gaussian_kernel = create_gaussian_kernel(kernel_size, sigma)

mod = SourceModule(f"""
    __constant__ float gaussian_kernel[{kernel_size * kernel_size}];

    __global__ void gaussian_blur_shared(float *input, float *output, int width, int height, int channels) {{
        extern __shared__ float shared_mem[];

        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;
        int z = blockIdx.z;

        int shared_width = blockDim.x + {kernel_size - 1};
        int shared_x = threadIdx.x + {kernel_size // 2};
        int shared_y = threadIdx.y + {kernel_size // 2};

        // Kopiranje u deljenu memoriju
        if (x < width && y < height) {{
            shared_mem[shared_y * shared_width + shared_x] = input[(y * width + x) * channels + z];
        }} else {{
            shared_mem[shared_y * shared_width + shared_x] = 0.0f; // Padding za piksele van granica
        }}

        // Kopiranje dodatnih piksela za ivice bloka
        if (threadIdx.x < {kernel_size // 2}) {{
            int left_x = max(x - {kernel_size // 2}, 0);
            shared_mem[shared_y * shared_width + shared_x - {kernel_size // 2}] = input[(y * width + left_x) * channels + z];
        }}
        if (threadIdx.x >= blockDim.x - {kernel_size // 2}) {{
            int right_x = min(x + {kernel_size // 2}, width - 1);
            shared_mem[shared_y * shared_width + shared_x + {kernel_size // 2}] = input[(y * width + right_x) * channels + z];
        }}
        if (threadIdx.y < {kernel_size // 2}) {{
            int top_y = max(y - {kernel_size // 2}, 0);
            shared_mem[(shared_y - {kernel_size // 2}) * shared_width + shared_x] = input[(top_y * width + x) * channels + z];
        }}
        if (threadIdx.y >= blockDim.y - {kernel_size // 2}) {{
            int bottom_y = min(y + {kernel_size // 2}, height - 1);
            shared_mem[(shared_y + {kernel_size // 2}) * shared_width + shared_x] = input[(bottom_y * width + x) * channels + z];
        }}

        __syncthreads();

        // Primena Gaussian filtera
        if (x < width && y < height) {{
            float sum = 0.0f;

            for (int i = -{kernel_size // 2}; i <= {kernel_size // 2}; i++) {{
                for (int j = -{kernel_size // 2}; j <= {kernel_size // 2}; j++) {{
                    int kernel_idx = (i + {kernel_size // 2}) * {kernel_size} + (j + {kernel_size // 2});
                    int shared_idx = (shared_y + i) * shared_width + (shared_x + j);
                    sum += shared_mem[shared_idx] * gaussian_kernel[kernel_idx];
                }}
            }}

            output[(y * width + x) * channels + z] = sum;
        }}
    }}
""")

gaussian_kernel_gpu = mod.get_global("gaussian_kernel")[0]
cuda.memcpy_htod(gaussian_kernel_gpu, gaussian_kernel)

image = Image.open("dolina.jpg").convert("RGB")
rgb_array = np.array(image).astype(np.float32)
height, width, channels = rgb_array.shape

input_image_gpu = cuda.mem_alloc(rgb_array.nbytes)
output_image_gpu = cuda.mem_alloc(rgb_array.nbytes)
cuda.memcpy_htod(input_image_gpu, rgb_array)

block_dim = (16, 16, 1)
grid_dim = (int(np.ceil(width / block_dim[0])), int(np.ceil(height / block_dim[1])), channels)

shared_memory_size = (block_dim[0] + kernel_size - 1) * (block_dim[1] + kernel_size - 1) * np.float32().nbytes

gaussian_blur = mod.get_function("gaussian_blur_shared")
gaussian_blur(input_image_gpu, output_image_gpu, np.int32(width), np.int32(height), np.int32(channels),
              block=block_dim, grid=grid_dim, shared=shared_memory_size)

output_image = np.empty_like(rgb_array)
cuda.memcpy_dtoh(output_image, output_image_gpu)

output_image = output_image.astype(np.uint8)
output_image_pil = Image.fromarray(output_image, mode="RGB")
output_image_pil.show()
output_image_pil.save("gaussian_blurred_shared_memory.jpg")