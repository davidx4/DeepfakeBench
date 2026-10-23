# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-03-29
# description: Data pre-processing script for deepfake dataset.


"""Crop JPG images and save aligned faces, landmarks, and optional masks."""


import os
import sys
import time
import cv2
import dlib
import yaml
import logging
import datetime
import glob
import concurrent.futures
import numpy as np
from tqdm import tqdm
from pathlib import Path
from imutils import face_utils
from skimage import transform as trans


def create_logger(log_path):
    """
    Creates a logger object and saves all messages to a file.

    Args:
        log_path (str): The path to save the log file.

    Returns:
        logger: The logger object.
    """
    # Create logger object
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create file handler and set the formatter
    fh = logging.FileHandler(log_path)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(fh)

    # Add a stream handler to print to console
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def get_keypts(image, face, predictor, face_detector):
    # detect the facial landmarks for the selected face
    shape = predictor(image, face)
    
    # select the key points for the eyes, nose, and mouth
    leye = np.array([shape.part(37).x, shape.part(37).y]).reshape(-1, 2)
    reye = np.array([shape.part(44).x, shape.part(44).y]).reshape(-1, 2)
    nose = np.array([shape.part(30).x, shape.part(30).y]).reshape(-1, 2)
    lmouth = np.array([shape.part(49).x, shape.part(49).y]).reshape(-1, 2)
    rmouth = np.array([shape.part(55).x, shape.part(55).y]).reshape(-1, 2)
    
    pts = np.concatenate([leye, reye, nose, lmouth, rmouth], axis=0)

    return pts


def extract_aligned_face_dlib(face_detector, predictor, image, res=256, mask=None):
    def img_align_crop(img, landmark=None, outsize=None, scale=1.3, mask=None):
        """ 
        align and crop the face according to the given bbox and landmarks
        landmark: 5 key points
        """

        M = None
        target_size = [112, 112]
        dst = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)

        if target_size[1] == 112:
            dst[:, 0] += 8.0

        dst[:, 0] = dst[:, 0] * outsize[0] / target_size[0]
        dst[:, 1] = dst[:, 1] * outsize[1] / target_size[1]

        target_size = outsize

        margin_rate = scale - 1
        x_margin = target_size[0] * margin_rate / 2.
        y_margin = target_size[1] * margin_rate / 2.

        # move
        dst[:, 0] += x_margin
        dst[:, 1] += y_margin

        # resize
        dst[:, 0] *= target_size[0] / (target_size[0] + 2 * x_margin)
        dst[:, 1] *= target_size[1] / (target_size[1] + 2 * y_margin)

        src = landmark.astype(np.float32)

        # use skimage tranformation
        tform = trans.SimilarityTransform()
        tform.estimate(src, dst)
        M = tform.params[0:2, :]

        # M: use opencv
        # M = cv2.getAffineTransform(src[[0,1,2],:],dst[[0,1,2],:])

        img = cv2.warpAffine(img, M, (target_size[1], target_size[0]))

        if outsize is not None:
            img = cv2.resize(img, (outsize[1], outsize[0]))
        
        if mask is not None:
            mask = cv2.warpAffine(mask, M, (target_size[1], target_size[0]))
            mask = cv2.resize(mask, (outsize[1], outsize[0]))
            return img, mask
        else:
            return img, None

    # Image size
    height, width = image.shape[:2]

    # Convert to rgb
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect with dlib
    faces = face_detector(rgb, 1)
    if len(faces):
        # For now only take the biggest face
        face = max(faces, key=lambda rect: rect.width() * rect.height())
        
        # Get the landmarks/parts for the face in box d only with the five key points
        landmarks = get_keypts(rgb, face, predictor, face_detector)

        # Align and crop the face
        cropped_face, mask_face = img_align_crop(rgb, landmarks, outsize=(res, res), mask=mask)
        cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR)
        
        # Extract the all landmarks from the aligned face
        face_align = face_detector(cropped_face, 1)
        if len(face_align) == 0:
            return None, None, None
        landmark = predictor(cropped_face, face_align[0])
        landmark = face_utils.shape_to_np(landmark)

        return cropped_face, landmark, mask_face
    
    else:
        return None, None, None

def image_manipulate(image_paths, mask_paths, dataset_path, logger):
    """Crop every JPG image in a source folder and save its landmarks and optional mask."""
    face_detector = dlib.get_frontal_face_detector()
    predictor_path = './dlib_tools/shape_predictor_81_face_landmarks.dat'
    if not os.path.exists(predictor_path):
        logger.error(f"Predictor path does not exist: {predictor_path}")
        return
    face_predictor = dlib.shape_predictor(predictor_path)

    for source_path in image_paths:
        image = cv2.imread(str(source_path))
        if image is None:
            logger.warning(f"Failed to read image {source_path}")
            continue

        relative_path = source_path.relative_to(dataset_path)
        if relative_path.parts and relative_path.parts[0] == 'videos':
            relative_path = Path(*relative_path.parts[1:])
        output_group = relative_path.parent
        if output_group == Path('.'):
            output_group = Path(relative_path.stem)

        source_mask = next(
            (
                mask_paths.get(relative_path.with_suffix(suffix))
                for suffix in (source_path.suffix, '.jpg', '.jpeg', '.png')
                if mask_paths.get(relative_path.with_suffix(suffix)) is not None
            ),
            None,
        )
        mask = None
        if source_mask is not None:
            mask = cv2.imread(str(source_mask), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                logger.warning(f"Failed to read mask {source_mask}")
        elif mask_paths:
            logger.warning(f"No mask found for image {source_path}")

        cropped_face, landmarks, cropped_mask = extract_aligned_face_dlib(
            face_detector, face_predictor, image, mask=mask
        )
        if cropped_face is None:
            logger.warning(f"No face found in image {source_path}")
            continue
        if landmarks is None:
            logger.warning(f"No landmarks found in image {source_path}")
            continue

        frame_dir = dataset_path / 'frames' / output_group
        frame_dir.mkdir(parents=True, exist_ok=True)
        image_name = relative_path.with_suffix('.png').name
        output_image = frame_dir / image_name
        if not cv2.imwrite(str(output_image), cropped_face):
            logger.error(f"Failed to write cropped image {output_image}")
            continue

        landmark_path = dataset_path / 'landmarks' / output_group / relative_path.with_suffix('.npy').name
        landmark_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(landmark_path), landmarks)

        if source_mask is not None and cropped_mask is not None:
            mask_output = dataset_path / 'masks' / output_group / image_name
            mask_output.parent.mkdir(parents=True, exist_ok=True)
            _, binary_mask = cv2.threshold(cropped_mask, 1, 255, cv2.THRESH_BINARY)
            if not cv2.imwrite(str(mask_output), binary_mask):
                logger.error(f"Failed to write cropped mask {mask_output}")


def preprocess(dataset_path, mask_path, logger):
    dataset_path = Path(dataset_path)
    image_paths = sorted(
        Path(path) for path in glob.glob(
            os.path.join(dataset_path, '**/*.[jJ][pP][gG]'), recursive=True
        )
    )
    if not image_paths:
        logger.error(f"No JPG images found in {dataset_path}")
        return
    logger.info(f"{len(image_paths)} JPG images found in {dataset_path}")

    masks_by_relative_path = {}
    if mask_path is not None:
        mask_path = Path(mask_path)
        for mask_file in sorted(mask_path.rglob('*')):
            if mask_file.is_file() and mask_file.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                masks_by_relative_path[mask_file.relative_to(mask_path)] = mask_file
        logger.info(f"{len(masks_by_relative_path)} image masks found in {mask_path}")

    start_time = time.monotonic()
    images_by_directory = {}
    for image_path in image_paths:
        images_by_directory.setdefault(image_path.parent, []).append(image_path)

    max_workers = os.cpu_count() or 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(image_manipulate, paths, masks_by_relative_path, dataset_path, logger)
            for paths in images_by_directory.values()
        ]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            logger.info(f"Current time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error processing JPG images: {e}")

    duration_minutes = (time.monotonic() - start_time) / 60
    logger.info(f"Total time taken: {duration_minutes:.2f} minutes")

if __name__ == '__main__':
    # from config.yaml load parameters
    yaml_path = './config.yaml'
    # open the yaml file
    try:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.parser.ParserError as e:
        print("YAML file parsing error:", e)

    # Get the parameters
    dataset_name = config['preprocess']['dataset_name']['default']
    dataset_root_path = config['preprocess']['dataset_root_path']['default']
    comp = config['preprocess']['comp']['default']
    
    # use dataset_name and dataset_root_path to get dataset_path
    dataset_path = Path(os.path.join(dataset_root_path, dataset_name))

    # Create logger
    log_path = f'./logs/{dataset_name}.log'
    logger = create_logger(log_path)

    # Define dataset path based on the input arguments
    ## faceforensic++
    if dataset_name == 'FaceForensics++':
        sub_dataset_names = ["original_sequences/youtube","original_sequences/actors", \
                             "manipulated_sequences/Deepfakes", \
                            "manipulated_sequences/Face2Face", "manipulated_sequences/FaceSwap", \
                            "manipulated_sequences/NeuralTextures","manipulated_sequences/FaceShifter",\
                            "manipulated_sequences/DeepFakeDetection"]
        sub_dataset_paths = [Path(os.path.join(dataset_path, name, comp)) for name in sub_dataset_names]
        # mask
        mask_dataset_names = ["manipulated_sequences/Deepfakes", "manipulated_sequences/Face2Face", \
                            "manipulated_sequences/FaceSwap", "manipulated_sequences/NeuralTextures",\
                            "manipulated_sequences/DeepFakeDetection"]
        # mask_dataset_names = []
        mask_dataset_paths = [Path(os.path.join(dataset_path, name)) for name in mask_dataset_names]
    ## Celeb-DF-v1
    elif dataset_name == 'Celeb-DF-v1':
        sub_dataset_names = ['Celeb-real', 'Celeb-synthesis', 'YouTube-real']
        sub_dataset_paths = [Path(os.path.join(dataset_path, name)) for name in sub_dataset_names]
    
    ## Celeb-DF-v2
    elif dataset_name == 'Celeb-DF-v2':
        sub_dataset_names = ['Celeb-real', 'Celeb-synthesis', 'YouTube-real']
        sub_dataset_paths = [Path(os.path.join(dataset_path, name)) for name in sub_dataset_names]
    
    ## DFDCP
    elif dataset_name == 'DFDCP':
        sub_dataset_names = ['original_videos', 'method_A', 'method_B']
        sub_dataset_paths = [Path(os.path.join(dataset_path, name)) for name in sub_dataset_names]

    ## DFDC-test
    elif dataset_name == 'DFDC':
        sub_dataset_names = ['test', 'train']
        # train dataset is too large, so we split it into 50 parts
        sub_train_dataset_names = ["dfdc_train_part_" + str(i) for i in range(0,50)]
        sub_train_dataset_paths = [Path(os.path.join(dataset_path, 'train', name)) for name in sub_train_dataset_names]
        sub_dataset_paths = [Path(os.path.join(dataset_path, 'test'))] + sub_train_dataset_paths
   
   ## DeeperForensics-1.0
    elif dataset_name == 'DeeperForensics-1.0':
        real_sub_dataset_names = ['source_videos/' + name for name in os.listdir(os.path.join(dataset_path, 'source_videos'))]
        fake_sub_dataset_names = ['manipulated_videos/' + name for name in os.listdir(os.path.join(dataset_path, 'manipulated_videos'))]
        real_sub_dataset_names.extend(fake_sub_dataset_names)
        sub_dataset_names = real_sub_dataset_names
        sub_dataset_paths = [Path(os.path.join(dataset_path, name)) for name in sub_dataset_names]
        
    ## UADFV
    elif dataset_name == 'UADFV':
        sub_dataset_names = ['fake', 'real']
        sub_dataset_paths = [Path(os.path.join(dataset_path, name)) for name in sub_dataset_names]
    
    ## DeepFakeFace
    elif dataset_name == 'DeepFakeFace':
        fake_sub_dataset_names = ['fake/' + name for name in os.listdir(os.path.join(dataset_path, 'fake'))]
        sub_dataset_names = ['real'] + fake_sub_dataset_names
        sub_dataset_paths = [Path(os.path.join(dataset_path, name)) for name in sub_dataset_names]

    ## HiDF
    elif dataset_name == 'HiDF':
        sub_dataset_names = ['fake', 'real']
        sub_dataset_paths = [Path(os.path.join(dataset_path, name)) for name in sub_dataset_names]
    
    else:
        raise ValueError(f"Dataset {dataset_name} not recognized")
    
    # Check if dataset path exists
    if not Path(dataset_path).exists():
        logger.error(f"Dataset path does not exist: {dataset_path}")
        sys.exit()

    if 'sub_dataset_paths' in globals() and len(sub_dataset_paths) != 0:
        # Check if sub_dataset path exists
        for sub_dataset_path in sub_dataset_paths:
            if not Path(sub_dataset_path).exists():
                logger.error(f"Sub Dataset path does not exist: {sub_dataset_path}")
                sys.exit()
        # preprocess each sub_dataset
        for sub_dataset_path in sub_dataset_paths:
            # only part of FaceForensics++ has mask
            if dataset_name == 'FaceForensics++' and sub_dataset_path.parent in mask_dataset_paths:
                mask_dataset_path = os.path.join(sub_dataset_path.parent, "masks")
                preprocess(sub_dataset_path, mask_dataset_path, logger)
            else:
                preprocess(sub_dataset_path, None, logger)
    else:
        logger.error(f"Sub Dataset path does not exist: {sub_dataset_paths}")
        sys.exit()
    logger.info("Face cropping complete!")
