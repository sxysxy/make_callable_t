#!env bash
run() { echo "$*"; "$@"; }
run g++ -c -std=c++20 -DBUILD_WITH_CUDA main.cpp -o main.o

CAPS=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader \
  | sed 's/\.//' \
  | sort -u)


if [ -z "$CAPS" ]; then
  echo "Error: No NVIDIA GPU detected." >&2
  exit 1
fi

GENCODES=""
for cc in $CAPS; do
  GENCODES+=" -gencode arch=compute_${cc},code=[sm_${cc},compute_${cc}]"
done

run nvcc --x cu --compile test_cuda_cap.cu $GENCODES --expt-relaxed-constexpr -std=c++20 -o test_cuda_cap.o
run nvcc test_cuda_cap.o main.o -lcudart -o main