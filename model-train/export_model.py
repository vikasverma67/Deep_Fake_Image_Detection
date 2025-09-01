import os
import argparse
import torch
import json
import onnx
import numpy as np
from pathlib import Path

# Import custom modules
from src.models.efficientnet import create_model


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Export Deepfake Detection Model")
    
    # Model related arguments
    parser.add_argument("--model_name", type=str, default="efficientnet-b0",
                        help="Model architecture name")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint to export")
    parser.add_argument("--export_dir", type=str, default="./exported",
                        help="Directory to save exported model")
    parser.add_argument("--export_format", type=str, default="onnx",
                        choices=["onnx", "torchscript", "pytorch"],
                        help="Format to export the model to")
    parser.add_argument("--input_shape", type=str, default="1,3,224,224",
                        help="Input shape for the model (batch_size,channels,height,width)")
    
    # Parse arguments
    args = parser.parse_args()
    return args


def export_to_onnx(model, export_path, input_shape):
    """Export model to ONNX format"""
    # Create dummy input
    dummy_input = torch.randn(*input_shape)
    
    # Export the model
    torch.onnx.export(
        model,
        dummy_input,
        export_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['logits', 'probs', 'confidence', 'features'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'logits': {0: 'batch_size'},
            'probs': {0: 'batch_size'},
            'confidence': {0: 'batch_size'},
            'features': {0: 'batch_size'}
        }
    )
    
    # Verify the model
    onnx_model = onnx.load(export_path)
    onnx.checker.check_model(onnx_model)
    
    print(f"Model exported to ONNX format: {export_path}")
    return export_path


def export_to_torchscript(model, export_path, input_shape):
    """Export model to TorchScript format"""
    # Create dummy input
    dummy_input = torch.randn(*input_shape)
    
    # Trace the model
    traced_model = torch.jit.trace(model, dummy_input)
    
    # Save the traced model
    torch.jit.save(traced_model, export_path)
    
    print(f"Model exported to TorchScript format: {export_path}")
    return export_path


def export_to_pytorch(model, export_path):
    """Export model to PyTorch format (state_dict)"""
    # Save the model state dict
    torch.save(model.state_dict(), export_path)
    
    print(f"Model exported to PyTorch format: {export_path}")
    return export_path


def save_metadata(checkpoint, export_info, metadata_path):
    """Save model metadata for later use in inference"""
    # Extract metrics from checkpoint if available
    metrics = {}
    if 'metrics' in checkpoint:
        metrics = {k: float(v) if isinstance(v, (int, float)) else v 
                  for k, v in checkpoint['metrics'].items()}
    
    # Create metadata
    metadata = {
        'model_name': export_info['model_name'],
        'export_format': export_info['export_format'],
        'input_shape': export_info['input_shape'],
        'creation_date': export_info['creation_date'],
        'metrics': metrics,
        'labels': {
            0: "real",
            1: "fake"
        }
    }
    
    # Save metadata as JSON
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"Model metadata saved to: {metadata_path}")


def main():
    """Main export function"""
    # Parse arguments
    args = parse_args()
    
    # Convert input shape string to tuple of integers
    input_shape = tuple(map(int, args.input_shape.split(',')))
    
    # Create export directory if it doesn't exist
    os.makedirs(args.export_dir, exist_ok=True)
    
    # Load checkpoint
    checkpoint_path = args.checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"Loaded checkpoint from: {checkpoint_path}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
    
    # Create model
    model = create_model(
        model_name=args.model_name,
        pretrained=False
    )
    
    # Load model weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Set model to evaluation mode
    model.eval()
    
    # Get export path
    export_name = f"deepfake_detection_{args.model_name.replace('-', '_')}"
    export_path = os.path.join(
        args.export_dir, 
        f"{export_name}.{args.export_format}"
    )
    
    # Export based on format
    if args.export_format == 'onnx':
        exported_path = export_to_onnx(model, export_path, input_shape)
    elif args.export_format == 'torchscript':
        exported_path = export_to_torchscript(model, export_path, input_shape)
    elif args.export_format == 'pytorch':
        exported_path = export_to_pytorch(model, export_path)
    else:
        print(f"Unsupported export format: {args.export_format}")
        return
    
    # Save metadata
    from datetime import datetime
    
    export_info = {
        'model_name': args.model_name,
        'export_format': args.export_format,
        'input_shape': list(input_shape),
        'creation_date': datetime.now().isoformat()
    }
    
    metadata_path = os.path.join(
        args.export_dir,
        f"{export_name}_metadata.json"
    )
    
    save_metadata(checkpoint, export_info, metadata_path)
    
    print("Model export completed successfully!")


if __name__ == "__main__":
    main() 