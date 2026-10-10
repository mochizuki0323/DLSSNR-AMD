# RDNA4 (RX 9000, gfx12): FP8 e4m3 WMMA, wave32. The network in windows/shaders/rdna4/.
# Sourced by build_package.sh, build_optiscaler_nr.sh and assemble_product_data.sh.
#
# The shaders and the host must agree: nr_graph refuses a shader directory whose
# shader-constants.txt disagrees with the defines below. The shaders are built by
# `python3 windows/build/build_network.py rdna4` from windows/shaders/rdna4/pipelines.json.
NR_GPU_ARCH=rdna4
NR_PRODUCT_SPV=build/windows/rdna4/network
NR_PRODUCT_DEFINES=(
    -DNR_GEMM_WIDE_MT=64 -DNR_GEMM_WIDE_NT=256
    -DNR_GEMM_PROJ_MT=32 -DNR_GEMM_PROJ_NT=128
    -DNR_PROJW_MIN_TOKENS=768 -DNR_GEMM_REMAP_HOST=1
    -DNR_FFWD_WGW=4 -DNR_FFWD_GMAJOR=1 -DNR_UPSVIEW_VEC=1 -DNR_REPACK_VEC=1
    -DNR_WIDE_UPS_MASK=7 -DNR_PERSIST_DF=1 -DNR_DS_FUSE=15
    -DNR_DECUPS_VEC=16 -DNR_DECQ_SMALL=1
    -DNR_PERSIST_DS_MASK=7 -DNR_PERSIST_UPS_MASK=7 -DNR_REPACK_FOLD=1 -DNR_STRAG_PCT=25
    -DNR_PERSIST_ONE_MASK=6 -DNR_NOISE_FIELD=1 -DNR_FFWD_FM2_MIN_TOKENS=2560 -DNR_POST_ALPHA=1
    -DNR_TCHAIN_TILES=1
)
# The model pack's file name inside dlssnr-amd/ (windows/src/core/nr_runtime.cpp).
NR_MODEL_NAME=dlssnr.bin
