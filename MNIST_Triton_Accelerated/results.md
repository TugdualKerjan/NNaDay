# MNIST using Triton to accelerate things

🎯 Implement TRITON JIT parts to go FAST based on https://github.com/triton-lang/triton/blob/main/python/tutorials/02-fused-softmax.py

🔢 Results:

Currently trying to follow the tutorials but the code isn't working - I suspect it has something to do with either the fact that I'm using a T4 that doesn't have enough lines or something. Have to check in tomorrow when I'll have more time.

🧩 Key Learnings:


    - Triton was initially optimized only for CUDA kernels. It seems that it's expanding out to AMD as CDNA is mentioned along with HIP.

⚠️ Challenges Faced:

    Seems that package versioning is giving me a tough time again. The example program doesn't seem to work correctly either, I'll have to investigate further. VSCODE gave me a tough time to set up as well which was a pain (problem with the python extension on the SSH of GCP)

🛠️ Improvements for Tomorrow

    Run the sample code and compare with my implementation