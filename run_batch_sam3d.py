import os
import sys
import cv2
import argparse

# WINDOWS DOESN'T HAVE EGL
os.environ["PYOPENGL_PLATFORM"] = "win32"

from pathlib import Path
from utils import (
    setup_sam_3d_body, setup_visualizer,
    visualize_2d_results, visualize_3d_mesh, save_mesh_results
)

def iter_jpegs(folder: Path):
    # file ext to support
    exts = {".jpg", ".jpeg"}
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            yield p

 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Name of a folder inside ./inputs containing jpg/jpeg images")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="cpu or cuda")
    parser.add_argument("--hf_repo_id", type=str, default="facebook/sam-3d-body-dinov3")
    parser.add_argument("--save_mesh", action="store_true", help="Save mesh outputs (OBJ, PLY etc.)")
    parser.add_argument("--save_2d_vis", action="store_true", help="Save 2D overlay images")
    parser.add_argument("--render_3d_vis", action="store_true",
                        help="Try 3D rendering (may fail headless due to OpenGL)")
    args = parser.parse_args()

    # project root = where this script lives
    ROOT = Path(__file__).resolve().parent

    # inputs/<name>
    input_dir = ROOT / "inputs" / args.input_dir

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input dir not found: {input_dir}\n"
            f"Expected a folder inside ./inputs/"
        )

    # outputs/<name>
    out_root = ROOT / "outputs" / args.input_dir
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Input dir : {input_dir}")
    print(f"Output dir: {out_root}")

    # Setup models once
    print("Setting up SAM-3D-Body estimator...")
    estimator = setup_sam_3d_body(hf_repo_id=args.hf_repo_id, device=args.device)
    visualizer = setup_visualizer()

    jpgs = list(iter_jpegs(input_dir))
    print(f"Found {len(jpgs)} jpeg files in {input_dir}")

    for idx, file in enumerate(jpgs, start=1):
        print(f"\n[{idx}/{len(jpgs)}] Processing: {file.name}")
        image_path = file.resolve()

        try:
            img_cv2 = cv2.imread(str(image_path))
            if img_cv2 is None:
                print(f"cv2.imread failed, skipping: {image_path}")
                continue

            # Run model (use path string, since estimator likely expects that)
            outputs = estimator.process_one_image(str(image_path))

            if not outputs:
                print("No people detected.")
                continue

            print(f"People detected: {len(outputs)}")
            # print(f"  keys: {list(outputs[0].keys())}")

            image_stem = image_path.stem
            image_out_dir = out_root / image_stem
            image_out_dir.mkdir(parents=True, exist_ok=True)

            # 2D visualization
            if args.save_2d_vis:
                try:
                    vis_results = visualize_2d_results(img_cv2, outputs, visualizer)
                    # save each person’s 2D visualization
                    for i, vis_img in enumerate(vis_results):
                        out_path = image_out_dir / f"{image_stem}_person{i}_2d.png"
                        cv2.imwrite(str(out_path), vis_img)
                    print(f"Saved 2D visuals to {image_out_dir}")
                except Exception as e:
                    print(f"2D visualization failed: {e}")

            # 3D visualization (renderer) — optional, may fail headless
            if args.render_3d_vis:
                try:
                    mesh_results = visualize_3d_mesh(img_cv2, outputs, estimator.faces)
                    for i, mesh_img in enumerate(mesh_results):
                        out_path = image_out_dir / f"{image_stem}_person{i}_3dvis.png"
                        cv2.imwrite(str(out_path), mesh_img)
                    print(f"Saved 3D rendered visuals to {image_out_dir}")
                except Exception as e:
                    print(f"3D rendering failed (OpenGL/display issue likely): {e}")

            # Save meshes using your util
            if args.save_mesh:
                try:
                    # your util expects string output dir
                    print("n outputs:", len(outputs))
                    print("bboxes:", [o["bbox"] for o in outputs])
                    ply_files = save_mesh_results(
                        img_cv2, outputs, estimator.faces,
                        str(image_out_dir), image_stem
                    )
                    print(f"Saved {len(ply_files)} mesh file(s) to {image_out_dir}")
                    print("Meshes returned:", ply_files)
                    print("Files in dir:", [p.name for p in image_out_dir.glob("*.obj")])
                except Exception as e:
                    print(f"Mesh saving failed: {e}")

        except Exception as e:
            print(f"Unexpected failure on {file.name}: {e}")
            continue

    print("\nDone.")


if __name__ == "__main__":
    main()
