import torch
from omegaconf import OmegaConf
import argparse
from gguf import GGUFWriter
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--dir-model", type=str, required=True)
parser.add_argument("--out-dir", type=str, required=True)
parser.add_argument("--use-f16", action="store_true")

def parse_tensors(state_dict, writer: GGUFWriter):
    tensors = state_dict["best_state"]
    for name, tensor in tensors.items():
        # type = GGMLQuantizationType.F16
        # if tensor.dtype == torch.float32:
        #     type = GGMLQuantizationType.F32
        #if not torch.cuda.is_available():
        #    tensor = tensor.to(torch.float32)
        print(f"Adding {name}, Shape: {tensor.shape}, Type: {tensor.dtype}")
        writer.add_tensor(name, tensor.numpy(), tensor.shape)

def parse_hparams(state_dict, writer: GGUFWriter, use_f16: bool):
    config = OmegaConf.create(state_dict['xp.cfg'])

    transformer_hparams = config['transformer_lm']
    writer.add_int32("params.dim", transformer_hparams["dim"])
    writer.add_int32("params.num_heads", transformer_hparams["num_heads"])
    writer.add_int32("params.num_layers", transformer_hparams["num_layers"])
    writer.add_int32("params.n_q", transformer_hparams["n_q"])
    writer.add_int32("params.card", transformer_hparams["card"])
    writer.add_int32("params.hidden_scale", transformer_hparams["hidden_scale"])
    writer.add_int32("params.kv_repeat", transformer_hparams["kv_repeat"])
    writer.add_int32("params.subcodes_context", transformer_hparams["subcodes_context"])
    writer.add_int32("params.sample_rate", 32000)

    # TODO: add the default params for generation (key: "lm")


if __name__ == "__main__":
    args = parser.parse_args()

    dir_model = Path(args.dir_model)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    outfile = Path(out_dir / "ggml_model.bin")
    writer = GGUFWriter(outfile, "magnet")

    state_dict = torch.load(dir_model / "state_dict.bin", map_location="cpu")

    writer.add_architecture()

    # Append the hyperparameters to the model from the state dict
    parse_hparams(state_dict, writer, False)

    # Then add the tensors themselves
    # FIXME: may have to convert on the fly from F16 to F32
    parse_tensors(state_dict, writer)

    # Finish up and write everything to the file
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()

    writer.close()
    print(f"Converted tensors at {outfile}")
