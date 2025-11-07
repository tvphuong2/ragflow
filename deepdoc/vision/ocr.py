#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import gc
import logging
import copy
import time
import os

from huggingface_hub import snapshot_download

from common.file_utils import get_project_base_directory
from common.misc_utils import pip_install_torch
from rag.settings import PARALLEL_DEVICES
from .operators import *  # noqa: F403
from . import operators
import numpy as np
import cv2
import onnxruntime as ort
from PIL import Image

# Pillow 10 removed the Image.ANTIALIAS alias while many dependencies (including
# some VietOCR transforms) still import it. Restore the alias when missing so the
# pipeline stays compatible across Pillow versions.
try:
    _lanczos = Image.Resampling.LANCZOS  # Pillow >= 9.1
except AttributeError:
    _lanczos = getattr(Image, "LANCZOS", None)

if _lanczos is not None and not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = _lanczos

from .postprocess import build_post_process

loaded_models = {}

def transform(data, ops=None):
    """ transform """
    if ops is None:
        ops = []
    for op in ops:
        data = op(data)
        if data is None:
            return None
    return data


def create_operators(op_param_list, global_config=None):
    """
    create operators based on the config

    Args:
        params(list): a dict list, used to create some operators
    """
    assert isinstance(
        op_param_list, list), ('operator config should be a list')
    ops = []
    for operator in op_param_list:
        assert isinstance(operator,
                          dict) and len(operator) == 1, "yaml format error"
        op_name = list(operator)[0]
        param = {} if operator[op_name] is None else operator[op_name]
        if global_config is not None:
            param.update(global_config)
        op = getattr(operators, op_name)(**param)
        ops.append(op)
    return ops


def load_model(model_dir, nm, device_id: int | None = None):
    model_file_path = os.path.join(model_dir, nm + ".onnx")
    model_cached_tag = model_file_path + str(device_id) if device_id is not None else model_file_path

    global loaded_models
    loaded_model = loaded_models.get(model_cached_tag)
    if loaded_model:
        logging.info(f"load_model {model_file_path} reuses cached model")
        return loaded_model

    if not os.path.exists(model_file_path):
        raise ValueError("not find model file path {}".format(
            model_file_path))

    def cuda_is_available():
        try:
            pip_install_torch()
            import torch
            target_id = 0 if device_id is None else device_id
            if torch.cuda.is_available() and torch.cuda.device_count() > target_id:
                return True
        except Exception:
            return False
        return False

    options = ort.SessionOptions()
    options.enable_cpu_mem_arena = False
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 2

    # https://github.com/microsoft/onnxruntime/issues/9509#issuecomment-951546580
    # Shrink GPU memory after execution
    run_options = ort.RunOptions()
    if cuda_is_available():
        gpu_mem_limit_mb = int(os.environ.get("OCR_GPU_MEM_LIMIT_MB", "2048"))
        arena_strategy = os.environ.get("OCR_ARENA_EXTEND_STRATEGY", "kNextPowerOfTwo")
        provider_device_id = 0 if device_id is None else device_id
        cuda_provider_options = {
            "device_id": provider_device_id, # Use specific GPU
            "gpu_mem_limit": max(gpu_mem_limit_mb, 0) * 1024 * 1024,
            "arena_extend_strategy": arena_strategy,  # gpu memory allocation strategy
        }
        sess = ort.InferenceSession(
            model_file_path,
            options=options,
            providers=['CUDAExecutionProvider'],
            provider_options=[cuda_provider_options]
            )
        run_options.add_run_config_entry("memory.enable_memory_arena_shrinkage", "gpu:" + str(provider_device_id))
        logging.info(f"load_model {model_file_path} uses GPU (device {provider_device_id}, gpu_mem_limit={cuda_provider_options['gpu_mem_limit']}, arena_strategy={arena_strategy})")
    else:
        sess = ort.InferenceSession(
            model_file_path,
            options=options,
            providers=['CPUExecutionProvider'])
        run_options.add_run_config_entry("memory.enable_memory_arena_shrinkage", "cpu")
        logging.info(f"load_model {model_file_path} uses CPU")
    loaded_model = (sess, run_options)
    loaded_models[model_cached_tag] = loaded_model
    return loaded_model


class TextRecognizer:
    def __init__(self, model_dir, device_id: int | None = None):
        try:
            from vietocr.tool.config import Cfg
            from vietocr.tool.predictor import Predictor
        except ImportError as exc:
            raise ImportError(
                "VietOCR is required for text recognition. Please install the 'vietocr' package."
            ) from exc

        pip_install_torch()
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "PyTorch is required for VietOCR text recognition."
            ) from exc

        target_device = "cpu"
        if torch.cuda.is_available():
            requested_id = 0 if device_id is None else device_id
            if requested_id < torch.cuda.device_count():
                target_device = f"cuda:{requested_id}"
            else:
                logging.warning(
                    "Requested CUDA device %s is not available. Falling back to CPU.",
                    requested_id,
                )
        self.rec_batch_num = max(1, int(os.environ.get("VIETOCR_BATCH_SIZE", "4")))

        config_name = os.environ.get("VIETOCR_CONFIG_NAME", "vgg_seq2seq")
        cfg = Cfg.load_config_from_name(config_name)
        cfg["device"] = target_device
        if "cnn" in cfg and isinstance(cfg["cnn"], dict):
            cfg["cnn"]["pretrained"] = False
        beamsearch_env = os.environ.get("VIETOCR_BEAMSEARCH")
        if beamsearch_env is not None and "predictor" in cfg and isinstance(cfg["predictor"], dict):
            cfg["predictor"]["beamsearch"] = beamsearch_env.lower() in {"1", "true", "yes", "y"}

        weights_path = os.environ.get("VIETOCR_WEIGHTS_PATH")
        if not weights_path and model_dir:
            candidate_path = os.path.join(model_dir, "vietocr.pth")
            if os.path.isfile(candidate_path):
                weights_path = candidate_path
        if weights_path:
            cfg["weights"] = weights_path

        self.predictor = Predictor(cfg)
        self._predict_batch = getattr(self.predictor, "predict_batch", None)
        self._device = target_device
        self._config_name = config_name
        self._weights_path = cfg.get("weights")

    def _to_pil(self, img):
        if isinstance(img, Image.Image):
            return img
        if img is None:
            raise ValueError("Image for OCR recognition is None")
        if len(img.shape) == 2:
            return Image.fromarray(img)
        if img.shape[2] == 1:
            return Image.fromarray(img[:, :, 0])
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def _normalize_prediction(self, prediction):
        if isinstance(prediction, tuple):
            text, prob = prediction
        else:
            text, prob = prediction, 1.0
        if isinstance(prob, (list, tuple)):
            prob = prob[0] if prob else 1.0
        if hasattr(prob, "item"):
            prob = float(prob.item())
        else:
            try:
                prob = float(prob)
            except (TypeError, ValueError):
                prob = 1.0
        return text, prob

    def _predict_single(self, pil_img):
        try:
            return self.predictor.predict(pil_img, return_prob=True)
        except TypeError:
            return self.predictor.predict(pil_img)

    def close(self):
        logging.info('Close text recognizer (VietOCR).')
        if hasattr(self, "predictor"):
            del self.predictor
        gc.collect()

    def __call__(self, img_list):
        st = time.time()
        results = []
        if not img_list:
            return results, time.time() - st

        pil_images = [self._to_pil(img) for img in img_list]

        if self._predict_batch and len(pil_images) > 1:
            for beg in range(0, len(pil_images), self.rec_batch_num):
                end = beg + self.rec_batch_num
                batch_predictions = self._predict_batch(pil_images[beg:end])
                for prediction in batch_predictions:
                    text, prob = self._normalize_prediction(prediction)
                    results.append([text, prob])
        else:
            for pil_img in pil_images:
                prediction = self._predict_single(pil_img)
                text, prob = self._normalize_prediction(prediction)
                results.append([text, prob])

        if not results and pil_images:
            logging.warning(
                "VietOCR returned no predictions for %d crops (device=%s, config=%s, weights=%s, batch_size=%d)",
                len(pil_images),
                getattr(self, "_device", "unknown"),
                getattr(self, "_config_name", "unknown"),
                getattr(self, "_weights_path", "<pretrained>"),
                self.rec_batch_num,
            )
        else:
            empty_candidates = sum(1 for text, _ in results if not str(text).strip())
            if empty_candidates == len(results):
                sample_preds = results[:min(3, len(results))]
                logging.warning(
                    "VietOCR produced only empty strings for %d crops before filtering (device=%s, config=%s, weights=%s, sample=%s)",
                    len(results),
                    getattr(self, "_device", "unknown"),
                    getattr(self, "_config_name", "unknown"),
                    getattr(self, "_weights_path", "<pretrained>"),
                    sample_preds,
                )

        return results, time.time() - st

    def __del__(self):
        self.close()

class TextDetector:
    def __init__(self, model_dir, device_id: int | None = None):
        pre_process_list = [{
            'DetResizeForTest': {
                'limit_side_len': 960,
                'limit_type': "max",
            }
        }, {
            'NormalizeImage': {
                'std': [0.229, 0.224, 0.225],
                'mean': [0.485, 0.456, 0.406],
                'scale': '1./255.',
                'order': 'hwc'
            }
        }, {
            'ToCHWImage': None
        }, {
            'KeepKeys': {
                'keep_keys': ['image', 'shape']
            }
        }]
        postprocess_params = {"name": "DBPostProcess", "thresh": 0.3, "box_thresh": 0.5, "max_candidates": 1000,
                              "unclip_ratio": 1.5, "use_dilation": False, "score_mode": "fast", "box_type": "quad"}

        self.postprocess_op = build_post_process(postprocess_params)
        self.predictor, self.run_options = load_model(model_dir, 'det', device_id)
        self.input_tensor = self.predictor.get_inputs()[0]

        img_h, img_w = self.input_tensor.shape[2:]
        if isinstance(img_h, str) or isinstance(img_w, str):
            pass
        elif img_h is not None and img_w is not None and img_h > 0 and img_w > 0:
            pre_process_list[0] = {
                'DetResizeForTest': {
                    'image_shape': [img_h, img_w]
                }
            }
        self.preprocess_op = create_operators(pre_process_list)

    def order_points_clockwise(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        tmp = np.delete(pts, (np.argmin(s), np.argmax(s)), axis=0)
        diff = np.diff(np.array(tmp), axis=1)
        rect[1] = tmp[np.argmin(diff)]
        rect[3] = tmp[np.argmax(diff)]
        return rect

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            if isinstance(box, list):
                box = np.array(box)
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def filter_tag_det_res_only_clip(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            if isinstance(box, list):
                box = np.array(box)
            box = self.clip_det_res(box, img_height, img_width)
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def close(self):
        logging.info("Close text detector.")
        if hasattr(self, "predictor"):
            del self.predictor
        gc.collect()

    def __call__(self, img):
        ori_im = img.copy()
        data = {'image': img}

        st = time.time()
        data = transform(data, self.preprocess_op)
        img, shape_list = data
        if img is None:
            return None, 0
        img = np.expand_dims(img, axis=0)
        shape_list = np.expand_dims(shape_list, axis=0)
        img = img.copy()
        input_dict = {}
        input_dict[self.input_tensor.name] = img
        for i in range(100000):
            try:
                outputs = self.predictor.run(None, input_dict, self.run_options)
                break
            except Exception as e:
                if i >= 3:
                    raise e
                time.sleep(5)

        post_result = self.postprocess_op({"maps": outputs[0]}, shape_list)
        dt_boxes = post_result[0]['points']
        dt_boxes = self.filter_tag_det_res(dt_boxes, ori_im.shape)

        return dt_boxes, time.time() - st

    def __del__(self):
        self.close()


class OCR:
    def __init__(self, model_dir=None):
        """
        If you have trouble downloading HuggingFace models, -_^ this might help!!

        For Linux:
        export HF_ENDPOINT=https://hf-mirror.com

        For Windows:
        Good luck
        ^_-

        """
        if not model_dir:
            try:
                model_dir = os.path.join(
                        get_project_base_directory(),
                        "rag/res/deepdoc")
                
                # Append muti-gpus task to the list
                if PARALLEL_DEVICES > 0:
                    self.text_detector = []
                    self.text_recognizer = []
                    for device_id in range(PARALLEL_DEVICES):
                        self.text_detector.append(TextDetector(model_dir, device_id))
                        self.text_recognizer.append(TextRecognizer(model_dir, device_id))
                else:
                    self.text_detector = [TextDetector(model_dir)]
                    self.text_recognizer = [TextRecognizer(model_dir)]

            except Exception:
                model_dir = snapshot_download(repo_id="InfiniFlow/deepdoc",
                                              local_dir=os.path.join(get_project_base_directory(), "rag/res/deepdoc"),
                                              local_dir_use_symlinks=False)
                
                if PARALLEL_DEVICES > 0:
                    self.text_detector = []
                    self.text_recognizer = []
                    for device_id in range(PARALLEL_DEVICES):
                        self.text_detector.append(TextDetector(model_dir, device_id))
                        self.text_recognizer.append(TextRecognizer(model_dir, device_id))
                else:
                    self.text_detector = [TextDetector(model_dir)]
                    self.text_recognizer = [TextRecognizer(model_dir)]

        self.drop_score = 0.5
        self.crop_image_res_index = 0

    def get_rotate_crop_image(self, img, points):
        '''
        img_height, img_width = img.shape[0:2]
        left = int(np.min(points[:, 0]))
        right = int(np.max(points[:, 0]))
        top = int(np.min(points[:, 1]))
        bottom = int(np.max(points[:, 1]))
        img_crop = img[top:bottom, left:right, :].copy()
        points[:, 0] = points[:, 0] - left
        points[:, 1] = points[:, 1] - top
        '''
        assert len(points) == 4, "shape of points must be 4*2"
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2])))
        pts_std = np.float32([[0, 0], [img_crop_width, 0],
                              [img_crop_width, img_crop_height],
                              [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            # Try original orientation
            rec_result = self.text_recognizer[0]([dst_img])
            text, score = rec_result[0][0]
            best_score = score
            best_img = dst_img

            # Try clockwise 90° rotation
            rotated_cw = np.rot90(dst_img, k=3)
            rec_result = self.text_recognizer[0]([rotated_cw])
            rotated_cw_text, rotated_cw_score = rec_result[0][0]
            if rotated_cw_score > best_score:
                best_score = rotated_cw_score
                best_img = rotated_cw

            # Try counter-clockwise 90° rotation
            rotated_ccw = np.rot90(dst_img, k=1)
            rec_result = self.text_recognizer[0]([rotated_ccw])
            rotated_ccw_text, rotated_ccw_score = rec_result[0][0]
            if rotated_ccw_score > best_score:
                best_img = rotated_ccw

            # Use the best image
            dst_img = best_img
        return dst_img

    def sorted_boxes(self, dt_boxes):
        """
        Sort text boxes in order from top to bottom, left to right
        args:
            dt_boxes(array):detected text boxes with shape [4, 2]
        return:
            sorted boxes(array) with shape [4, 2]
        """
        num_boxes = dt_boxes.shape[0]
        sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
        _boxes = list(sorted_boxes)

        for i in range(num_boxes - 1):
            for j in range(i, -1, -1):
                if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and \
                        (_boxes[j + 1][0][0] < _boxes[j][0][0]):
                    tmp = _boxes[j]
                    _boxes[j] = _boxes[j + 1]
                    _boxes[j + 1] = tmp
                else:
                    break
        return _boxes

    def detect(self, img, device_id: int | None = None):
        if device_id is None:
            device_id = 0

        time_dict = {'det': 0, 'rec': 0, 'cls': 0, 'all': 0}

        if img is None:
            return None, None, time_dict

        start = time.time()
        dt_boxes, elapse = self.text_detector[device_id](img)
        time_dict['det'] = elapse

        if dt_boxes is None:
            end = time.time()
            time_dict['all'] = end - start
            return None, None, time_dict

        return zip(self.sorted_boxes(dt_boxes), [
                   ("", 0) for _ in range(len(dt_boxes))])

    def recognize(self, ori_im, box, device_id: int | None = None):
        if device_id is None:
            device_id = 0

        img_crop = self.get_rotate_crop_image(ori_im, box)

        rec_res, elapse = self.text_recognizer[device_id]([img_crop])
        text, score = rec_res[0]
        if score < self.drop_score:
            return ""
        return text

    def recognize_batch(self, img_list, device_id: int | None = None):
        if device_id is None:
            device_id = 0
        rec_res, elapse = self.text_recognizer[device_id](img_list)
        texts = []
        for i in range(len(rec_res)):
            text, score = rec_res[i]
            if score < self.drop_score:
                text = ""
            texts.append(text)
        return texts

    def __call__(self, img, device_id = 0, cls=True):
        time_dict = {'det': 0, 'rec': 0, 'cls': 0, 'all': 0}
        if device_id is None:
            device_id = 0

        if img is None:
            return None, None, time_dict

        start = time.time()
        ori_im = img.copy()
        dt_boxes, elapse = self.text_detector[device_id](img)
        time_dict['det'] = elapse

        if dt_boxes is None:
            end = time.time()
            time_dict['all'] = end - start
            return None, None, time_dict

        img_crop_list = []

        dt_boxes = self.sorted_boxes(dt_boxes)

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = self.get_rotate_crop_image(ori_im, tmp_box)
            img_crop_list.append(img_crop)

        rec_res, elapse = self.text_recognizer[device_id](img_crop_list)

        time_dict['rec'] = elapse

        filter_boxes, filter_rec_res = [], []
        for box, rec_result in zip(dt_boxes, rec_res):
            text, score = rec_result
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_result)

        if not filter_rec_res:
            self._log_recognition_debug_info(device_id, img_crop_list, rec_res)
        end = time.time()
        time_dict['all'] = end - start

        # for bno in range(len(img_crop_list)):
        #    print(f"{bno}, {rec_res[bno]}")

        return list(zip([a.tolist() for a in filter_boxes], filter_rec_res))

    def _log_recognition_debug_info(self, device_id, img_crop_list, rec_res):
        recognizer = self.text_recognizer[device_id]
        recognizer_device = getattr(recognizer, "_device", device_id)
        config_name = getattr(recognizer, "_config_name", "unknown")
        weights_path = getattr(recognizer, "_weights_path", "<pretrained>")
        batch_size = getattr(recognizer, "rec_batch_num", len(img_crop_list))

        if not rec_res:
            logging.warning(
                "OCR recognize returned no candidate texts for %d crops (device=%s, config=%s, weights=%s, batch_size=%d, drop_score=%.2f)",
                len(img_crop_list),
                recognizer_device,
                config_name,
                weights_path,
                batch_size,
                self.drop_score,
            )
            return

        try:
            scores = [float(score) for _, score in rec_res]
            min_score = min(scores)
            max_score = max(scores)
        except Exception:
            scores = []
            min_score = max_score = None

        sample_preds = rec_res[:min(3, len(rec_res))]
        if scores:
            score_range = f"[{min_score:.4f}, {max_score:.4f}]"
        else:
            score_range = "unknown"

        logging.warning(
            "All %d recognition candidates were filtered out by drop_score %.2f (device=%s, config=%s, weights=%s, score_range=%s, sample=%s)",
            len(rec_res),
            self.drop_score,
            recognizer_device,
            config_name,
            weights_path,
            score_range,
            sample_preds,
        )
