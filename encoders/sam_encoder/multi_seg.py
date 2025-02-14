import torch
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig, BitsAndBytesConfig
import numpy as np
import cv2
import replicate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.onnx import SamOnnxModel
import onnxruntime
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic
import warnings
import argparse
import os
from torch.nn import functional as F
from tqdm import tqdm
from PIL import Image
import re
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA


def fetch_points_with_molmo(processor, model, prompt, image):
    h,w = image.shape[:2]
    # Process the input for the model
    inputs = processor.process(
        images=[image],
        text="Point to the {} in the scene.".format(prompt),
    )
    inputs["images"] = inputs["images"].to(torch.bfloat16)

    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            output = model.generate_from_batch(
                inputs,
                GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
                tokenizer=processor.tokenizer
            )

            # Decode the generated tokens
            generated_tokens = output[0, inputs['input_ids'].size(1):]
            generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            # Print the generated text
            return extract_points(generated_text,w,h)

def extract_points(molmo_output, image_w, image_h):
    """Extract points from Molmo output."""
    all_points = []
    for match in re.finditer(r'x\d*="\s*([0-9]+(?:\.[0-9]+)?)"\s+y\d*="\s*([0-9]+(?:\.[0-9]+)?)"', molmo_output):
        try:
            point = [float(match.group(i)) for i in range(1, 3)]
        except ValueError:
            pass
        else:
            point = np.array(point)
            if np.max(point) > 100:
                continue
            point /= 100.0
            point = point * np.array([image_w, image_h])
            all_points.append(point)
    return all_points

parser = argparse.ArgumentParser(
    description=(
        "Segment images with prompt."
    )
)

parser.add_argument(
    "--prompt", 
    type=str,
    nargs="+",
    required=True,
    help="Text prompt for Molmo")

parser.add_argument(
    "--model-type",
    type=str,
    required=True,
    help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']",
)

parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="The path to the SAM checkpoint to use for mask generation.",
)

parser.add_argument(
    "--onnx_path",
    type=str,
    default='sam_onnx_example.onnx',
    # required=True,
    help="onnx model path",
)

parser.add_argument(
    "--onnx_quantized_path",
    type=str,
    default='',
    # required=True,
    help="onnx model quantized path",
)

parser.add_argument(
    "--data",
    type=str,
    help="Path including input images (and features)",
)

# parser.add_argument(
#     "--iteration",
#     type=int,
#     required=True,
#     help="Chosen number of iterations"
# )

parser.add_argument("--image", action="store_true", help="If true, encode feature from image") ###

parser.add_argument("--device", type=str, default="cuda", help="The device to run generation on.")

parser.add_argument('--point', 
                    type=int,
                    nargs='+',
                    help='two values x, y as a input point')

parser.add_argument('--box', 
                    type=int,
                    nargs='+',
                    help='four values x1, y1 as top left and x2, y2 corner as a bottom right corner')


def show_mask(mask, ax):
    color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))   

def main(args: argparse.Namespace) -> None:
    print("Loading model...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)

    # Set VRAM optimization parameters
    load_in_8bit = False
    load_in_4bit = True
    mixed_precision = "bf16"

    device_map = "auto"
    torch_dtype = torch.bfloat16

    quantization_config = BitsAndBytesConfig(
            load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit
    )

    # Load the processor
    processor = AutoProcessor.from_pretrained(
        'allenai/Molmo-7B-O-0924',
        trust_remote_code=True,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
    )
    print("Processor load complete.")

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        'allenai/Molmo-7B-O-0924',
        trust_remote_code=True,
        quantization_config=quantization_config,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
    )
    print("Model load complete.")

    sbert = SentenceTransformer("all-MiniLM-L6-v2")

    # use SBERT
    # prompt_embedding = sbert.encode(args.prompt, convert_to_tensor=True)
    # negative_embedding = -prompt_embedding

    prompt_embedding_list = []
    print(args.prompt)
    for prompt in args.prompt:
        prompt_embedding = sbert.encode(prompt, convert_to_tensor=True)
        prompt_embedding_list.append(prompt_embedding)
    negative_embedding_list = [-embedding for embedding in prompt_embedding_list]

    
    # onnx model setup
    onnx_model = SamOnnxModel(sam, return_single_mask=True)
    print("onnx model load complete.")

    dynamic_axes = {
        "point_coords": {1: "num_points"},
        "point_labels": {1: "num_points"},
    }

    embed_dim = sam.prompt_encoder.embed_dim
    embed_size = sam.prompt_encoder.image_embedding_size
    mask_input_size = [4 * x for x in embed_size]
    dummy_inputs = {
        "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
        "point_coords": torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float),
        "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float),
        "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float),
        "has_mask_input": torch.tensor([1], dtype=torch.float),
        "orig_im_size": torch.tensor([1500, 2250], dtype=torch.float),
    }
    output_names = ["masks", "iou_predictions", "low_res_masks"]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        with open(args.onnx_path, "wb") as f:
            torch.onnx.export(
                onnx_model,
                tuple(dummy_inputs.values()),
                f,
                export_params=True,
                verbose=False,
                opset_version= 12,
                do_constant_folding=True,
                input_names=list(dummy_inputs.keys()),
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )    
    
    if args.onnx_quantized_path != '':
        quantize_dynamic(
            model_input=args.onnx_path,
            model_output=args.onnx_quantized_path,
            optimize_model=True,
            per_channel=False,
            reduce_range=False,
            weight_type=QuantType.QUInt8,
        )
        args.onnx_path = args.onnx_quantized_path
    
    # # using an ONNX model
    ort_session = onnxruntime.InferenceSession(args.onnx_path)
    # sam.to(device=args.device)
    predictor = SamPredictor(sam)
    print("SAM predictor load complete.")

    # make seg dirs
    seg_path = os.path.join(args.data, "seg_molmo")
    os.makedirs(seg_path, exist_ok=True)

    # image directory setup
    # image_path = os.path.join(input_path, "ours_{}".format(args.iteration), "renders")
    image_path = os.path.join(args.data, "images")

    images = [
            f for f in sorted(os.listdir(image_path)) if not os.path.isdir(os.path.join(image_path, f))
        ]
    images = [os.path.join(image_path, f) for f in images]

    # feature directory setup
    if not args.image:
        # feature_path = os.path.join(input_path, "ours_{}".format(args.iteration), "saved_feature")
        feature_path = os.path.join(args.data, "sam_embeddings")
        features = [
                f for f in sorted(os.listdir(feature_path)) if not os.path.isdir(os.path.join(feature_path, f))
            ]
        features = [os.path.join(feature_path, f) for f in features]


    # output directory
    output_path = seg_path
    os.makedirs(output_path, exist_ok=True)


    # get image size
    img = cv2.imread(images[0])
    H, W = img.shape[:2]
    pool_size = 4
    embedding_dim = len(prompt_embedding_list[0])
    print(f"Setup: {embedding_dim}x{H}x{W} with pool size {pool_size}")

    # Create an empty mask input and an indicator for no mask.
    onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    onnx_has_mask_input = np.zeros(1, dtype=np.float32)


    for i, image in enumerate(tqdm(images, desc="Segment progress")):
        image_name = image.split("/")[-1].split(".")[0]
        
        # Initialize accumulation variables
        total_embedding_map = torch.zeros((embedding_dim, H, W), dtype=torch.float32)
        mask_overlay = np.zeros((H, W, 3), dtype=np.uint8)  # RGB overlay image
        
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert to RGB

        for idx, prompt in enumerate(args.prompt):
            embedding_map = torch.zeros((embedding_dim, H, W))

            # Input point
            input_point = np.array(fetch_points_with_molmo(processor, model, prompt, image))
            if input_point.size == 0:
                print(f"No points detected in image {image_name} for prompt '{prompt}'. Using fallback embeddings.")
                fallback_embeddings = negative_embedding_list[idx].unsqueeze(-1).unsqueeze(-1)
                fallback_embeddings = fallback_embeddings.repeat(1, H, W)
                fallback_embeddings = F.adaptive_avg_pool2d(fallback_embeddings.unsqueeze(0), (H // pool_size, W // pool_size)).squeeze(0)

                total_embedding_map += fallback_embeddings  # Accumulate for averaging
                continue  # Skip further processing for this prompt

            input_label = np.array([1 for _ in input_point])
            onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
            onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32)
            onnx_coord = predictor.transform.apply_coords(onnx_coord, image.shape[:2]).astype(np.float32)

            predictor.set_image(image)

            if args.image:
                embedding = predictor.get_image_embedding().cpu().numpy()
            else:
                embedding = torch.load(features[i], weights_only=True)[None]  
                _, _, fea_h, fea_w = embedding.shape
                embedding = F.pad(embedding, (0, 0, 0, fea_w - fea_h))
                embedding = embedding.cpu().numpy().astype(np.float32)

            ort_inputs = {
                "image_embeddings": embedding,
                "point_coords": onnx_coord,
                "point_labels": onnx_label,
                "mask_input": onnx_mask_input,
                "has_mask_input": onnx_has_mask_input,
                "orig_im_size": np.array(image.shape[:2], dtype=np.float32)
            }

            masks, _, low_res_logits = ort_session.run(None, ort_inputs)
            masks = masks > predictor.model.mask_threshold
            mask = masks[0]

            # Accumulate pixel-wise embeddings
            for k in range(H):
                for l in range(W):
                    if mask[0, k, l] == 1:
                        embedding_map[:, k, l] = prompt_embedding_list[idx]
                    else:
                        embedding_map[:, k, l] = negative_embedding_list[idx]
            total_embedding_map += embedding_map  # Accumulate for averaging

            # Overlay mask with a unique color
            color = np.random.randint(0, 255, (1, 3), dtype=np.uint8)  # Random RGB color
            mask_overlay[mask[0] > 0] = color

        # Compute final averaged embedding
        avg_embedding_map = total_embedding_map / len(prompt_embedding_list)
        avg_embedding_map = F.adaptive_avg_pool2d(avg_embedding_map.unsqueeze(0), (H // pool_size, W // pool_size)).squeeze(0)

        os.makedirs(os.path.join(args.data, "molmo_multi"), exist_ok=True)
        feature_save_path = os.path.join(args.data, "molmo_multi", f"{image_name}_fmap_CxHxW.pt")
        torch.save(avg_embedding_map, feature_save_path)

        # Save the overlaid mask image
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.imshow(mask_overlay, alpha=0.5)  # Blend the mask overlay with the image
        plt.axis('off')
        plt.savefig(os.path.join(output_path, image_name + '_overlay.png'), bbox_inches='tight', pad_inches=0)
        plt.close()
        


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)