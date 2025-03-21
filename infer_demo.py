import os
import argparse
import glob
import torch
from utils import utils_image as util


def select_model(model_id, device):
    """Load the selected super-resolution model."""
    if model_id == 0:
        # Baseline EFDN model
        from models.team00_EFDN import EFDN
        model_path = os.path.join('model_zoo', 'team00_EFDN.pth')
        model = EFDN()
        data_range = 1.0
    elif model_id == 38:
        # ESRNet model
        from models.team38_ESRNet import ESRNet
        model_path = os.path.join('model_zoo', 'team38_ESRNet.pth')
        model = ESRNet(3, 3, upscale=4, feature_channels=28)
        data_range = 1.0
    else:
        raise NotImplementedError(f"Model {model_id} is not implemented.")
    
    # Load model weights
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    
    # Disable gradient calculation
    for param in model.parameters():
        param.requires_grad = False
    
    model = model.to(device)
    return model, data_range


def forward(img_lq, model, tile=None, tile_overlap=32, scale=4):
    """Forward pass with optional tiling strategy."""
    if tile is None:
        # Process the image as a whole
        output = model(img_lq)
    else:
        # Process the image tile by tile
        b, c, h, w = img_lq.size()
        tile = min(tile, h, w)
        
        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h*scale, w*scale).type_as(img_lq)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx*scale:(h_idx+tile)*scale, w_idx*scale:(w_idx+tile)*scale].add_(out_patch)
                W[..., h_idx*scale:(h_idx+tile)*scale, w_idx*scale:(w_idx+tile)*scale].add_(out_patch_mask)
        
        output = E.div_(W)

    return output


def main():
    parser = argparse.ArgumentParser("Super Resolution Inference")
    parser.add_argument("--input_dir", required=False, type=str, help="Directory containing low-resolution images", default='../DIV2K_LSDIR_test_LR')
    parser.add_argument("--output_dir", required=False, type=str, help="Directory to save super-resolved images", default='../DIV2K_LSDIR_test_HR_infer')
    parser.add_argument("--model_id", default=0, type=int, help="Model ID to use for inference")
    parser.add_argument("--tile", default=None, type=int, help="Tile size for large images (None = no tiling)")
    parser.add_argument("--tile_overlap", default=32, type=int, help="Overlap size for tiling")
    parser.add_argument("--file_pattern", default="*.png", type=str, help="Pattern to match input files")
    
    args = parser.parse_args()
    
    # Setup device and load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    
    model, data_range = select_model(args.model_id, device)
    print(f"Model loaded (ID: {args.model_id})")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get all input images
    input_files = sorted(glob.glob(os.path.join(args.input_dir, args.file_pattern)))
    
    if not input_files:
        print(f"No files found matching pattern {args.file_pattern} in {args.input_dir}")
        return
    
    print(f"Found {len(input_files)} images. Starting inference...")
    
    # Process each image
    for i, img_path in enumerate(input_files):
        img_name = os.path.basename(img_path)
        output_path = os.path.join(args.output_dir, img_name).replace('x4', '')
        
        # Skip if already processed
        if os.path.exists(output_path):
            print(f"[{i+1}/{len(input_files)}] {img_name} already exists, skipping")
            continue
        
        # Load and preprocess input image
        img_lr = util.imread_uint(img_path, n_channels=3)
        img_lr = util.uint2tensor4(img_lr, data_range)
        img_lr = img_lr.to(device)
        
        # Run super-resolution
        with torch.no_grad():
            img_sr = forward(img_lr, model, args.tile, args.tile_overlap)
        
        # Convert back to uint8 and save
        img_sr = util.tensor2uint(img_sr, data_range)
        util.imsave(img_sr, output_path)
        
        print(f"[{i+1}/{len(input_files)}] Processed {img_name}")
    
    print(f"Done! All images saved to {args.output_dir}")


if __name__ == "__main__":
    main()