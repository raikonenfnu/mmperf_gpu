### How to use this Repository

1. Clone repositories

```sh
cd ~/ && git clone https://github.com/ROCm/triton/ && cd triton && git checkout shared/triton-gfx950-launch
cd ~/ && git clone https://github.com/iree-org/wave && cd wave/
cd ~/ && git clone https://github.com/ROCm/aiter && cd aiter/ && git submodule sync && git submodule update --init --recursive
cd ~/ && git clone https://github.com/raikonenfnu/mmperf_gpu
```


2. Setup docker environment
```sh
docker image pull docker pull rocm/sgl-dev:v0.5.4.post2-rocm700-mi35x-20251104-srt
docker run --name "$USER"_"torch" -it -d --device=/dev/kfd --device=/dev/dri --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --network=host --ipc=host -v "$HOME":"$HOME" --workdir /home/$USER rocm/sgl-dev:v0.5.4.post2-rocm700-mi35x-20251104-srt /bin/bash
docker attach "$USER"_"torch"
export HOME=$PWD
apt update
```

3. Install libraries inside docker
```sh
cd ~/triton && pip install -e .
cd ~/aiter && python setup.py develop
cd ~/wave && pip install --no-cache-dir -r requirements-iree-pinned.txt --upgrade && pip install -r requirements.txt -e .
pip install matplotlib
pip install numpy==1.26.0
``` 

3. HipblasLT docker
```
#download hipblaslt
cd ~
git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-libraries.git
cd rocm-libraries
git sparse-checkout init --cone
git sparse-checkout set projects/hipblaslt shared/origami shared/rocroller shared/mxdatagenerator
git checkout develop

# Hipblas LT prerequisite package

sudo mkdir --parents --mode=0755 /etc/apt/keyrings
wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | \
  gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null

echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/7.0_rc1 jammy main" \
  | sudo tee /etc/apt/sources.list.d/rocm.list

echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/graphics/7.0_rc1/ubuntu jammy main" \
  | sudo tee /etc/apt/sources.list.d/rocm-graphics.list

echo -e 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600' \
  | sudo tee /etc/apt/preferences.d/rocm-pin-600
sudo apt update

pip uninstall cmake
pip install cmake==3.31.6

# Build hipblaslt
cd projects/hipblaslt
pip install -r tensilelite/requirements.txt
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib/llvm/lib
./install.sh --keep-build-tmp -idc -a gfx950

# Generate libraries
cd ~
mkdir hipblaslt-artifacts && cd ~/hipblaslt-artifacts
hipblaslt-bench --function matmul --transA T --transB N --a_type bf16_r --b_type bf16_r --c_type bf16_r --d_type bf16_r --scale_type f32_r --bias_type f32_r --compute_type f32_r --sizem 1536 --sizen 3072 --sizek 19776 --lda 19776 --ldb 19776 --ldc 3072 --ldd 3072 --initialization trig_float --alpha 1 --beta 0 --iters 100 --cold_iters 25 --use_gpu_timer --print_kernel_info --flush --rotating 512 --algo_method all --api_method cpp --device 0 | tee run_1536_3072_19776.log
```

4. Test runs
```sh
AMD_SERIALIZE_KERNEL=3 TRITON_HIP_ASYNC_COPY_BYPASS_PERMUTE=1 AMDGCN_USE_BUFFER_OPS=1 TRITON_HIP_ASYNC_FAST_SWIZZLE=1 TRITON_HIP_USE_ASYNC_COPY=1 TRITON_HIP_USE_BLOCK_PINGPONG=1 python ~/mmperf_gpu/benchmark_mm_f16.py --shape 1536 3072 19776 --backend triton

AMD_SERIALIZE_KERNEL=3 TRITON_HIP_ASYNC_COPY_BYPASS_PERMUTE=1 AMDGCN_USE_BUFFER_OPS=1 TRITON_HIP_ASYNC_FAST_SWIZZLE=1 TRITON_HIP_USE_ASYNC_COPY=1 TRITON_HIP_USE_BLOCK_PINGPONG=1 python ~/mmperf_gpu/benchmark_mm_f16.py --shape 1536 3072 19776 --backend asm

AMD_SERIALIZE_KERNEL=3 TRITON_HIP_ASYNC_COPY_BYPASS_PERMUTE=1 AMDGCN_USE_BUFFER_OPS=1 TRITON_HIP_ASYNC_FAST_SWIZZLE=1 TRITON_HIP_USE_ASYNC_COPY=1 TRITON_HIP_USE_BLOCK_PINGPONG=1 python ~/mmperf_gpu/benchmark_mm_f16.py --shape 1536 3072 19776 --backend hipblas

AMD_SERIALIZE_KERNEL=3 TRITON_HIP_ASYNC_COPY_BYPASS_PERMUTE=1 AMDGCN_USE_BUFFER_OPS=1 TRITON_HIP_ASYNC_FAST_SWIZZLE=1 TRITON_HIP_USE_ASYNC_COPY=1 TRITON_HIP_USE_BLOCK_PINGPONG=1 python ~/mmperf_gpu/benchmark_mm_f16.py --shape 1536 3072 19776 --backend wave
```
**NOTE: Above are e2e runs, we are more interested in kernel time, so we need to set up rocmProfileData for that**

5. set up rocmProfileData
```sh
cd ~/ && git clone https://github.com/ROCm/rocmProfileData
cd rocmProfileData
apt-get install sqlite3 libsqlite3-dev && apt-get install libfmt-dev && make; make install
cd ~/ && wget https://gist.githubusercontent.com/raikonenfnu/7d10e109a21a9c337a6f71f9f8a6b3eb/raw/1b0e3a6cf3508c7d555b7a9656c65623911ddc19/process_rpd.py
```

```sh

wget https://gist.githubusercontent.com/raikonenfnu/7d10e109a21a9c337a6f71f9f8a6b3eb/raw/1b0e3a6cf3508c7d555b7a9656c65623911ddc19/process_rpd.py

AMD_SERIALIZE_KERNEL=3 TRITON_HIP_ASYNC_COPY_BYPASS_PERMUTE=1 AMDGCN_USE_BUFFER_OPS=1 TRITON_HIP_ASYNC_FAST_SWIZZLE=1 TRITON_HIP_USE_ASYNC_COPY=1 TRITON_HIP_USE_BLOCK_PINGPONG=1 runTracer.sh -o triton.rpd python ~/mmperf_gpu/benchmark_mm_f16.py --shape 1536 3072 19776 --backend triton
python ~/process_rpd.py triton.rpd

AMD_SERIALIZE_KERNEL=3 TRITON_HIP_ASYNC_COPY_BYPASS_PERMUTE=1 AMDGCN_USE_BUFFER_OPS=1 TRITON_HIP_ASYNC_FAST_SWIZZLE=1 TRITON_HIP_USE_ASYNC_COPY=1 TRITON_HIP_USE_BLOCK_PINGPONG=1 runTracer.sh -o wave.rpd python ~/mmperf_gpu/benchmark_mm_f16.py --shape 1536 3072 19776 --backend wave
python ~/process_rpd.py wave.rpd


AMD_SERIALIZE_KERNEL=3 TRITON_HIP_ASYNC_COPY_BYPASS_PERMUTE=1 AMDGCN_USE_BUFFER_OPS=1 TRITON_HIP_ASYNC_FAST_SWIZZLE=1 TRITON_HIP_USE_ASYNC_COPY=1 TRITON_HIP_USE_BLOCK_PINGPONG=1 runTracer.sh -o asm.rpd python ~/mmperf_gpu/benchmark_mm_f16.py --shape 1536 3072 19776 --backend asm
python ~/process_rpd.py asm.rpd

AMD_SERIALIZE_KERNEL=3 TRITON_HIP_ASYNC_COPY_BYPASS_PERMUTE=1 AMDGCN_USE_BUFFER_OPS=1 TRITON_HIP_ASYNC_FAST_SWIZZLE=1 TRITON_HIP_USE_ASYNC_COPY=1 TRITON_HIP_USE_BLOCK_PINGPONG=1 runTracer.sh -o asm.rpd python ~/mmperf_gpu/benchmark_mm_f16.py --shape 1536 3072 19776 --backend hipblas
python ~/process_rpd.py hipblas.rpd

```
