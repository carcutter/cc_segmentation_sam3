import argparse
import glob
import os
import sys
from typing import List

import onnx

# Define a mapping from ONNX (numpy) data types to Triton's string representation
DTYPE_MAP = {
    "float32": "TYPE_FP32",
    "float16": "TYPE_FP16",
    "int64": "TYPE_INT64",
    "int32": "TYPE_INT32",
    "int8": "TYPE_INT8",
    "uint8": "TYPE_UINT8",
    "bool": "TYPE_BOOL",
    "string": "TYPE_STRING",
}


def generate_pbtxt_section(tensors, section_type):
    """
    Generates the string for an input or output section of the config.pbtxt.
    It reads the data type and dimensions of the tensors and generates the corresponding section.

    Args:
        tensors: A list of ONNX tensor objects (from model.graph.input or .output).
        section_type: A string, either "input" or "output".

    Returns:
        A formatted string for the entire section.
    """
    lines = [f"{section_type} ["]
    if not tensors:  # If no tensors, return an empty section
        lines.append("]")
        return "\n".join(lines)

    for i, tensor in enumerate(tensors):
        lines.append("  {")
        lines.append(f'    name: "{tensor.name}"')

        # Determine data type
        try:
            np_dtype = onnx.helper.tensor_dtype_to_np_dtype(
                tensor.type.tensor_type.elem_type
            )
            triton_dtype = DTYPE_MAP.get(str(np_dtype), "TYPE_INVALID")
            if triton_dtype == "TYPE_INVALID":
                print(
                    f"Warning: Unsupported data type '{np_dtype}' for tensor '{tensor.name}'.",
                    file=sys.stderr,
                )
            lines.append(f"    data_type: {triton_dtype}")
        except Exception as e:
            print(
                f"Warning: Could not determine data type for tensor '{tensor.name}': {e}",
                file=sys.stderr,
            )
            lines.append("    data_type: TYPE_INVALID")

        # Determine dimensions, skipping the first (batch) dimension
        dims = []
        if tensor.type.tensor_type.HasField("shape"):
            for dim in tensor.type.tensor_type.shape.dim:
                if dim.HasField("dim_param") and dim.dim_param == "batch_size":
                    # If the first dimension is the batch dimension
                    # it will be handled by Triton's `max_batch_size`.
                    continue
                if dim.HasField("dim_value"):
                    dims.append(str(dim.dim_value))
                else:  # Dynamic dimension (has dim_param or is unknown)
                    dims.append("-1")

            dims_str = ", ".join(dims)
            # Add the special 'reshape' field for scalar-like inputs per the example.
            # This is common for inputs like 'timestep' which have a shape like [batch_size, 1] in ONNX.
            if (
                section_type == "input"
                and len(tensor.type.tensor_type.shape.dim) == 1
                and tensor.type.tensor_type.shape.dim[0].dim_param == "batch_size"
            ):
                lines.append("    dims: [ 1 ]")
                lines.append("    reshape: { shape: [ ] }")
            else:
                lines.append(f"    dims: [ {dims_str} ]")

        # Add a comma for all but the last item in the list
        if i == len(tensors) - 1:
            lines.append("  }")
        else:
            lines.append("  },")

    lines.append("]")
    return "\n".join(lines)


def generate_warmup_dims(tensor, model_name: str) -> List[str]:
    """
    Generate concrete dimensions for warmup data, replacing dynamic dims with typical values.

    Args:
        tensor: ONNX tensor object
        model_name: Name of the model
    Returns:
        List of concrete dimension values as strings
    """
    dims = []
    if tensor.type.tensor_type.HasField("shape"):
        # Skip the first (batch) dimension as it's handled by batch_size
        for i, dim in enumerate(tensor.type.tensor_type.shape.dim):
            if i == 0 and dim.HasField("dim_param") and dim.dim_param == "batch":
                # If the first dimension is the batch dimension
                # it will be handled by Triton's `max_batch_size`.
                continue
            if dim.HasField("dim_value"):
                dims.append(str(dim.dim_value))
            else:  # Dynamic dimension - use typical values
                dims.append("-1")
    return dims


def generate_model_warmup_section(model: onnx.ModelProto, model_name: str) -> str:
    """
    Generates the model_warmup section for the config.pbtxt.

    Args:
        model: Loaded ONNX model
        model_name: Name of the model

    Returns:
        A formatted string for the model_warmup section.
    """
    lines = [
        "model_warmup [",
        "  {",
        f'    name: "warmup_{model_name}"',
        "    batch_size: 1",
        "    inputs [",
    ]

    for i, tensor in enumerate(model.graph.input):
        lines.append("      {")
        lines.append(f'        key: "{tensor.name}"')
        lines.append("        value: {")

        # Determine data type
        try:
            np_dtype = onnx.helper.tensor_dtype_to_np_dtype(
                tensor.type.tensor_type.elem_type
            )
            triton_dtype = DTYPE_MAP.get(str(np_dtype), "TYPE_FP16")
            lines.append(f"          data_type: {triton_dtype}")
        except Exception:
            lines.append("          data_type: TYPE_FP16")  # Default fallback

        # Generate concrete dimensions for warmup
        warmup_dims = generate_warmup_dims(tensor, model_name)
        dims_str = ", ".join(warmup_dims)

        # Handle special case for scalar-like tensors
        if (
            len(tensor.type.tensor_type.shape.dim) == 1
            and tensor.type.tensor_type.shape.dim[0].dim_param == "batch_size"
        ):
            lines.append("          dims: [ 1 ]")
        else:
            lines.append(f"          dims: [{dims_str}]")

        lines.append("          random_data: true")
        lines.append("        }")

        # Add comma for all but the last input
        if i == len(model.graph.input) - 1:
            lines.append("      }")
        else:
            lines.append("      },")

    lines.extend(["    ]", "  }", "]"])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Triton config.pbtxt files for ONNX models"
    )
    parser.add_argument(
        "--onnx_model_dir",
        default="onnx-models",
        help="Directory containing your model folders",
    )

    args = parser.parse_args()
    onnx_model_dir = args.onnx_model_dir

    for model_path in glob.glob(os.path.join(onnx_model_dir, "*.onnx")):
        model_name = os.path.basename(model_path).replace(".onnx", "")

        model = onnx.load(model_path)
        config_path = model_path.replace(".onnx", ".pbtxt")

        dynamic_batching = (
            model.graph.input[0].type.tensor_type.shape.dim[0].HasField("dim_param")
            and model.graph.input[0].type.tensor_type.shape.dim[0].dim_param
            == "batch_size"
        )
        max_batch_size = 8 if dynamic_batching else 0

        # Build the config file content as a list of strings
        config_content = [
            # Header
            f'name: "{model_name}"',
            'platform: "onnxruntime_onnx"',
            f"max_batch_size: {max_batch_size}",
            "",
            # Inputs
            generate_pbtxt_section(model.graph.input, "input"),
            "",
            # Outputs
            generate_pbtxt_section(model.graph.output, "output"),
            "",
            # Model warmup
            generate_model_warmup_section(model, model_name),
        ]

        # Write the complete config file at once
        with open(config_path, "w") as f:
            f.write("\n".join(config_content))
            f.write("\n")  # Add a final newline for POSIX compliance

        print(f"Successfully generated '{config_path}'")


if __name__ == "__main__":
    main()
