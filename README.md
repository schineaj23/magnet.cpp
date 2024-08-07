# magnet.cpp
Implementation of [MAGNeT](https://arxiv.org/pdf/2401.04577) (Masked Audio Generation using a Single Non-Autoregressive Transformer) in C++ with the tensor library [GGML](https://github.com/ggerganov/ggml)
This is a **work in progress** and not considered production-stable. This project was inspired by [encodec.cpp](https://github.com/PABannier/encodec.cpp). ![Model architecture](./assets/model_arch.png)

## Usage
### Get the weights
Download MAGNeT's weights from [huggingface](https://huggingface.co/facebook/magnet-medium-30secs) and convert them into the [GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) format.
Ensure that you have the `state_dict.bin` as this is MAGNeT's model. Note that `compression_dict.bin` is the state dict for EnCodec.
```bash
python convert.py --dir-model=/path/to/hf/weights --out-dir=/path/to/gguf/weights
```


### Get the code
```bash
git clone --recurse-submodules https://github.com/schineaj23/magnet.cpp.git
```

### Build from source
```bash
mkdir build
cd build
cmake ..
make
```

### Run the executable
```bash
magnet /path/to/gguf/weights
```

## Roadmap
- [ ] Inference in FP32
- [ ] FP8/INT8 Quantization
