# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Yingshu Chen from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py

import argparse
import glob
import multiprocessing as mp
import os

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import tempfile
import time
import warnings

import cv2
import numpy as np
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from mask2former import add_maskformer2_config
from predictor import VisualizationDemo


# constants
WINDOW_NAME = "mask2former demo"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="maskformer2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        # help="A list of space separated input images; "
        # "or a single glob pattern such as 'directory/*.jpg'",
        help="A list of space separated directories containing input images.",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False

# PATCH_SIZE = 512
PATCH_SIZE = 1024

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    if args.input:
        # print(args.input, f"{args.input[0]}/*")
        dirlist = glob.glob(f"{args.input[0]}/*")
        dirlist = [args.input[0]] # uncomment when dir contains images
        dirlist.sort()
        # if len(args.input) == 1:
        #     args.input = glob.glob(os.path.expanduser(args.input[0]))
        #     assert args.input, "The input path(s) was not found"
        for dir in dirlist:
            if not os.path.isdir(dir):
                continue
            logger.info("Handling "+dir)
            os.makedirs(os.path.join(args.output, dir.split("/")[-1]+"_split%d"%PATCH_SIZE)) if not os.path.exists(os.path.join(args.output, dir.split("/")[-1]+"_split%d"%PATCH_SIZE)) else None
            os.makedirs(os.path.join(args.output, dir.split("/")[-1]+"_split%d"%PATCH_SIZE, "index")) if not os.path.exists(os.path.join(args.output, dir.split("/")[-1]+"_split%d"%PATCH_SIZE, "index")) else None
            input_path = glob.glob(f"{dir}/*")
            for path in tqdm.tqdm(input_path, disable=not args.output):
                # use PIL, to be consistent with evaluation
                img = read_image(path, format="BGR")
                # print("img.shape", img.shape) #H,W,3
                start_time = time.time()

                if (img.shape[0] > PATCH_SIZE) or (img.shape[1] > PATCH_SIZE):
                # if False:
                    predictions = np.zeros(img.shape[:2]).astype("uint8")
                    visualized_output = np.zeros(img.shape).astype("uint8")
                    for i in range(0, img.shape[0], PATCH_SIZE):
                        for j in range(0, img.shape[1], PATCH_SIZE):
                            crop_img = img[i:i+PATCH_SIZE, j:j+PATCH_SIZE, :]
                            crop_pred, crop_vis_output = demo.run_on_image(crop_img)
                            predictions[i:i+PATCH_SIZE, j:j+PATCH_SIZE] = crop_pred["sem_seg"].argmax(dim=0).cpu().numpy()
                            vis_np_img = crop_vis_output.get_image()
                            visualized_output[i:i+PATCH_SIZE, j:j+PATCH_SIZE, 0] = vis_np_img[:,:,2] #RGB ->BGR
                            visualized_output[i:i+PATCH_SIZE, j:j+PATCH_SIZE, 1] = vis_np_img[:,:,1] #RGB ->BGR
                            visualized_output[i:i+PATCH_SIZE, j:j+PATCH_SIZE, 2] = vis_np_img[:,:,0] #RGB ->BGR
                  
                    logger.info(
                        "{}: {} in {:.2f}s".format(
                            path,
                            "finished",
                            time.time() - start_time,
                        )
                    )

                    if args.output:
                        if os.path.isdir(args.output):
                            assert os.path.isdir(args.output), args.output
                            out_filename = os.path.join(args.output, dir.split("/")[-1]+"_split%d"%PATCH_SIZE, os.path.basename(path).replace('jpg','png').replace('JPG','png'))
                            out_idx_filename = os.path.join(args.output, dir.split("/")[-1]+"_split%d"%PATCH_SIZE, "index", os.path.basename(path).replace('jpg','png').replace('JPG','png'))
                            # Store png files without compression
                        else:
                            assert len(args.input) == 1, "Please specify a directory with args.output"
                            out_filename = args.output
                            out_idx_filename = os.path.join(args.output,"_index.png")
                        # print("visualized_output.shape", visualized_output.shape)
                        # print("predictions.shape", predictions.shape)
                        # print("predictions.min", predictions.min())
                        # print("predictions.max", predictions.max())
                        cv2.imwrite(out_filename, visualized_output) 
                        cv2.imwrite(out_idx_filename, predictions)
                else:
                    predictions, visualized_output = demo.run_on_image(img)
                    logger.info(
                        "{}: {} in {:.2f}s".format(
                            path,
                            "detected {} instances".format(len(predictions["instances"]))
                            if "instances" in predictions
                            else "finished",
                            time.time() - start_time,
                        )
                    )

                    if args.output:
                        if os.path.isdir(args.output):
                            assert os.path.isdir(args.output), args.output
                            out_filename = os.path.join(args.output, dir.split("/")[-1]+"_split", os.path.basename(path).replace('jpg','png').replace('JPG','png'))
                            out_idx_filename = os.path.join(args.output, dir.split("/")[-1]+"_split", "index", os.path.basename(path).replace('jpg','png').replace('JPG','png'))
                            # Store png files without compression
                        else:
                            assert len(args.input) == 1, "Please specify a directory with args.output"
                            out_filename = args.output
                            out_idx_filename = os.path.join(args.output,"_index.png")
                        visualized_output.save(out_filename)
                        index_pred = predictions["sem_seg"].argmax(dim=0).cpu().numpy()
                        print("index_pred.shape", index_pred.shape)
                        cv2.imwrite(out_idx_filename, index_pred)
                    else:
                        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                        cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                        if cv2.waitKey(0) == 27:
                            break  # esc to quit
    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        assert args.output is None, "output not yet supported with --webcam!"
        cam = cv2.VideoCapture(0)
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cam.release()
        cv2.destroyAllWindows()
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)
        codec, file_ext = (
            ("x264", ".mkv") if test_opencv_video_format("x264", ".mkv") else ("mp4v", ".mp4")
        )
        if codec == ".mp4v":
            warnings.warn("x264 codec not available, switching to mp4v")
        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + file_ext
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*codec),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()
