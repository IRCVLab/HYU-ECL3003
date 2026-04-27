# Filename: rectify_saved_images.py
# Description: Rectify saved stereo image pairs from multiple directories using a stereo calibration file and save the rectified images.
# (Argparse removed, configuration is set within the code)

import sys
import os
import glob
# import argparse # Argparse 제거
import numpy as np
import cv2
import yaml
from pathlib import Path
# from datetime import datetime # 파일 이름 처리에 필요할 수 있음 (현재 사용 안함)

# --- Helper Functions (Included for self-containment) ---

# stereo_preview.py 및 stereo_calibrator.py에서 사용된 로드 함수와 동일합니다.
def load_stereo_calibration_yaml_standard(yaml_file_path: Path):
    """
    표준 YAML 형식으로 저장된 스테레오 캘리브레이션 결과를 불러옵니다.
    """
    if not yaml_file_path.exists():
        print(f"Error: Calibration file not found at {yaml_file_path}")
        return None

    try:
        with open(yaml_file_path, 'r') as f:
            print(f"Attempting to load stereo YAML from {yaml_file_path} using yaml.safe_load...")
            calib_data = yaml.safe_load(f)

        # 렉티피케이션에 필요한 데이터 로드 및 numpy 배열로 변환
        cameraMatrix1 = np.array(calib_data.get('cameraMatrix1_result'))
        distCoeffs1 = np.array(calib_data.get('distCoeffs1_result'))
        cameraMatrix2 = np.array(calib_data.get('cameraMatrix2_result'))
        distCoeffs2 = np.array(calib_data.get('distCoeffs2_result'))
        R1 = np.array(calib_data.get('R1'))
        R2 = np.array(calib_data.get('R2'))
        P1 = np.array(calib_data.get('P1'))
        P2 = np.array(calib_data.get('P2'))

        img_width = calib_data.get('image_width')
        img_height = calib_data.get('image_height')
        img_size_calibrated = (img_width, img_height)

        validPixROI1 = calib_data.get('validPixROI1') # 튜플 또는 List 형태로 저장되어 있다면 그대로 로드
        validPixROI2 = calib_data.get('validPixROI2') # 튜플 또는 List 형태로 저장되어 있다면 그대로 로드


        # Check if essential data is loaded
        if cameraMatrix1 is None or distCoeffs1 is None or cameraMatrix2 is None or distCoeffs2 is None or \
           R1 is None or R2 is None or P1 is None or P2 is None or img_width is None or img_height is None:
             print("Error: Missing essential calibration data (matrices, image size) in YAML file for rectification.")
             missing_keys = [k for k in ['cameraMatrix1_result', 'distCoeffs1_result', 'cameraMatrix2_result', 'distCoeffs2_result',
                                         'R1', 'R2', 'P1', 'P2', 'image_width', 'image_height'] if calib_data.get(k) is None]
             print(f"Missing keys: {missing_keys}")

             return None

        print("Standard YAML calibration data loaded successfully.")

        # 렉티피케이션 맵 생성에 필요한 모든 데이터를 반환
        loaded_data = {
            'cameraMatrix1_result': cameraMatrix1,
            'distCoeffs1_result': distCoeffs1,
            'cameraMatrix2_result': cameraMatrix2,
            'distCoeffs2_result': distCoeffs2,
            'R1': R1,
            'R2': R2,
            'P1': P1,
            'P2': P2,
            'image_size_calibrated': img_size_calibrated, # 캘리브레이션 시 사용된 이미지 크기 반환
            'validPixROI1': validPixROI1,
            'validPixROI2': validPixROI2,
        }

        return loaded_data

    except Exception as e:
        print(f"Error loading calibration data from {yaml_file_path}: {e}")
        print("This error likely means the YAML file is not in the standard format saved by stereo_calibrator.py.")
        return None

# stereo_preview.py에서 사용된 맵 생성 함수와 동일합니다.
def create_rectification_maps(cameraMatrix1, distCoeffs1, R1, P1,
                              cameraMatrix2, distCoeffs2, R2, P2, img_size_output):
    """
    주어진 캘리브레이션 파라미터와 출력 이미지 크기로 렉티피케이션 매핑 테이블을 생성합니다.
    """
    print(f"Creating rectification maps for output size: {img_size_output}")
    try:
        map1_left, map2_left = cv2.initUndistortRectifyMap(
            cameraMatrix1, distCoeffs1, R1, P1, img_size_output, cv2.CV_32FC1
        )
        map1_right, map2_right = cv2.initUndistortRectifyMap(
            cameraMatrix2, distCoeffs2, R2, P2, img_size_output, cv2.CV_32FC1
        )
        print("Rectification maps created.")
        return map1_left, map2_left, map1_right, map2_right
    except Exception as e:
        print(f"Error creating rectification maps: {e}")
        return None, None, None, None


def derive_output_dir(input_dir: str, prefix: str = "rectified_"):
    """
    입력 디렉터리 경로로부터 출력 디렉터리 경로를 생성합니다.
    (예: /path/to/images_left -> /path/to/rectified_images_left)
    """
    input_path = Path(input_dir)
    parent_dir = input_path.parent
    dir_name = input_path.name
    output_dir_name = f"{prefix}{dir_name}"
    output_path = parent_dir / output_dir_name
    return str(output_path)


def process_image_pair_directory(left_input_dir: str, right_input_dir: str,
                                 left_output_dir: str, right_output_dir: str,
                                 calib_data, img_size_input, img_size_output,
                                 map1_left, map2_left, map1_right, map2_right,
                                 crop_to_valid_roi: bool):
    """
    하나의 좌/우 이미지 디렉터리 쌍에 대해 렉티피케이션을 수행합니다.
    """
    print(f"\nProcessing directory pair: {left_input_dir} and {right_input_dir}")
    print(f"Saving rectified images to: {left_output_dir} and {right_output_dir}")

    # --- 출력 디렉터리 생성 ---
    try:
        os.makedirs(left_output_dir, exist_ok=True)
        os.makedirs(right_output_dir, exist_ok=True)
    except Exception as e:
         print(f"Error creating output directories {left_output_dir}, {right_output_dir}: {e}")
         return 0 # 처리 실패

    # --- 이미지 파일 목록 가져오기 및 처리 ---
    left_images = sorted(glob.glob(os.path.join(left_input_dir, "*.png"))) # 예시로 png만
    right_images = sorted(glob.glob(os.path.join(right_input_dir, "*.png")))

    print(f"  Found {len(left_images)} left images and {len(right_images)} right images.")

    if len(left_images) == 0 or len(left_images) != len(right_images):
        print("  Error: Image counts do not match or no images found in input directories. Skipping this pair.")
        if len(left_images) > 0 and len(right_images) > 0:
             print(f"  Left: {len(left_images)}, Right: {len(right_images)}")
        return 0 # 처리 실패

    # validPixROI 정보 로드 (자르기 옵션 사용 시 필요)
    validPixROI1 = calib_data.get('validPixROI1')
    validPixROI2 = calib_data.get('validPixROI2')

    processed_count = 0
    total_images_in_pair = len(left_images)
    print(f"  Processing {total_images_in_pair} image pairs...")

    for i in range(total_images_in_pair):
        left_path = left_images[i]
        right_path = right_images[i]

        # 파일명 일치 여부 간단히 확인 (timestamps)
        left_filename = os.path.basename(left_path)
        right_filename = os.path.basename(right_path)
        # 'left_' 또는 'right_' 접두사를 제거하고 나머지 파일 이름이 동일한지 확인
        left_base_name = left_filename.split('_', 1)[-1] if '_' in left_filename else left_filename
        right_base_name = right_filename.split('_', 1)[-1] if '_' in right_filename else right_filename

        if left_base_name != right_base_name:
             print(f"  Warning: Filenames do not match for pair {i+1}: {left_filename} vs {right_filename}. Skipping pair.")
             continue

        # 이미지 로드 (원본 왜곡 이미지)
        img_left_raw = cv2.imread(left_path)
        img_right_raw = cv2.imread(right_path)

        if img_left_raw is None or img_right_raw is None:
            print(f"  Error: Could not read image pair {i+1} ({left_filename}). Skipping.")
            continue

        # 이미지 크기 확인 (지정된 입력 크기와 일치해야 함)
        if img_left_raw.shape[:2][::-1] != img_size_input:
            print(f"  Warning: Left image size mismatch for {left_filename}. Expected {img_size_input}, got {img_left_raw.shape[:2][::-1]}. Skipping pair.")
            continue
        if img_right_raw.shape[:2][::-1] != img_size_input:
             print(f"  Warning: Right image size mismatch for {right_filename}. Expected {img_size_input}, got {img_right_raw.shape[:2][::-1]}. Skipping pair.")
             continue

        # --- 이미지 렉티피케이션 적용 ---
        rectified_left = cv2.remap(img_left_raw, map1_left, map2_left, cv2.INTER_LINEAR)
        rectified_right = cv2.remap(img_right_raw, map1_right, map2_right, cv2.INTER_LINEAR)

        # --- 유효 영역으로 자르기 (옵션) ---
        if crop_to_valid_roi and validPixROI1 and validPixROI2:
             try:
                 # validPixROI는 튜플 또는 리스트로 로드됩니다.
                 x1, y1, w1, h1 = validPixROI1 if isinstance(validPixROI1, tuple) else tuple(validPixROI1)
                 x2, y2, w2, h2 = validPixROI2 if isinstance(validPixROI2, tuple) else tuple(validPixROI2)

                 # 자르기 영역 계산 (간단하게 ROI1 기준)
                 rectified_left = rectified_left[y1:y1+h1, x1:x1+w1]
                 rectified_right = rectified_right[y2:y2+h2, x2:x2+w2]

             except Exception as e:
                  print(f"  Warning: Failed to crop image pair {i+1} to valid ROI: {e}. Saving full rectified image.")


        # --- 렉티피케이션된 이미지 저장 ---
        # 출력 파일 이름 생성 (예: rectified_left_frame_timestamp.png)
        output_left_filename = os.path.join(left_output_dir, f"rectified_{left_filename}")
        output_right_filename = os.path.join(right_output_dir, f"rectified_{right_filename}")

        try:
            cv2.imwrite(output_left_filename, rectified_left)
            cv2.imwrite(output_right_filename, rectified_right)
            processed_count += 1
        except Exception as e:
            print(f"  Error: Failed to save rectified image pair {i+1}: {e}")

        # 진행 상황 표시 (선택 사항)
        if (i + 1) % 50 == 0 or (i + 1) == total_images_in_pair:
            print(f"  Processed {i + 1}/{total_images_in_pair} pairs.")


    print(f"Finished processing directory pair. Successfully rectified and saved {processed_count} pairs from {left_input_dir}/{right_input_dir}.")
    return processed_count


def main():
     # --- 설정 구간: 여기에 값을 직접 입력하세요 ---
    config = {
        # 원본(왜곡된) 왼쪽 이미지가 저장된 디렉터리 목록 (순서 중요)
        'right_dirs': [
           'stereo_calib_images/cam0',
           ],
        # 원본(왜곡된) 오른쪽 이미지가 저장된 디렉터리 목록 (left_dirs와 순서 및 개수 일치)
        'left_dirs': [
           'stereo_calib_images/cam1',
        ],
        # 스테레오 캘리브레이션 결과 YAML 파일 경로 (모든 폴더 쌍에 동일 적용)
        'stereo_calib_yaml': 'params/stereo_calibration_results.yaml', # 실제 파일 경로로 변경하세요

        # 이미지 너비와 높이 (캘리브레이션 및 원본 이미지 해상도와 일치해야 함)
        'width': 320, # 실제 해상도로 변경하세요
        'height': 256, # 실제 해상도로 변경하세요

        # 렉티피케이션된 이미지를 유효 영역(validPixROI)으로 자를지 여부 (True/False)
        'crop_to_valid_roi': False, # 필요에 따라 변경하세요

        # 출력 폴더 이름에 사용할 접두사 (예: rectified_calib_images_cam0_set1)
        'output_prefix': 'rectified_new_', # 필요에 따라 변경하세요
    }
    # --- 설정 구간 끝 ---


    # --- 설정 유효성 검사 ---
    if len(config['left_dirs']) != len(config['right_dirs']):
        print("Error: The number of left directories and right directories in the config must be the same.")
        sys.exit(1)

    # 각 입력 디렉터리가 실제로 존재하는지 사전 확인
    print("--- Checking Input Directories ---")
    for d in config['left_dirs'] + config['right_dirs']:
        if not os.path.isdir(d):
            print(f"Error: Input directory not found at {d}")
            sys.exit(1)
        print(f"Found input directory: {d}")

    # 캘리브레이션 파일 존재 확인
    if not os.path.exists(config['stereo_calib_yaml']):
        print(f"Error: Stereo calibration file not found at {config['stereo_calib_yaml']}")
        sys.exit(1)
    print(f"Found calibration file: {config['stereo_calib_yaml']}")


    # --- 캘리브레이션 데이터 로드 (한 번만) ---
    print("\n--- Loading Stereo Calibration Data ---")
    calib_data = load_stereo_calibration_yaml_standard(Path(config['stereo_calib_yaml']))

    if calib_data is None:
        print("Failed to load stereo calibration data. Cannot rectify images.")
        sys.exit(1)

    # 캘리브레이션 이미지 사이즈와 현재 지정된 이미지 사이즈 비교
    img_size_calibrated = calib_data.get('image_size_calibrated') # 로드된 튜플 형태
    img_size_input = (config['width'], config['height']) # 사용자가 지정한 입력 이미지 크기

    if img_size_input != img_size_calibrated:
         print(f"Error: Specified image size ({img_size_input}) in config does not match calibration file size ({img_size_calibrated}).")
         print("Calibration and image sizes must match for accurate rectification.")
         sys.exit(1)
    print(f"Image size validation successful: {img_size_input}")


    # --- 렉티피케이션 맵 생성 (한 번만) ---
    # 맵 생성 시 사용자가 지정한 입력 이미지 크기 (출력 이미지 크기) 사용
    map1_left, map2_left, map1_right, map2_right = create_rectification_maps(
        calib_data.get('cameraMatrix1_result'), calib_data.get('distCoeffs1_result'),
        calib_data.get('R1'), calib_data.get('P1'),
        calib_data.get('cameraMatrix2_result'), calib_data.get('distCoeffs2_result'),
        calib_data.get('R2'), calib_data.get('P2'),
        img_size_input # 맵 생성 시 사용될 출력 이미지 크기
    )

    if map1_left is None: # 맵 생성 실패 시
         print("Failed to create rectification maps. Cannot rectify images.")
         sys.exit(1)

    # --- 각 디렉터리 쌍 순차적으로 처리 ---
    total_processed_pairs_count = 0
    num_directory_pairs = len(config['left_dirs'])

    print(f"\n--- Starting Rectification of {num_directory_pairs} Directory Pairs ---")

    for i in range(num_directory_pairs):
        left_input_dir = config['left_dirs'][i]
        right_input_dir = config['right_dirs'][i]

        # 출력 디렉터리 경로 자동 생성
        left_output_dir = derive_output_dir(left_input_dir, config['output_prefix'])
        right_output_dir = derive_output_dir(right_input_dir, config['output_prefix'])

        print(f"\nProcessing pair {i+1}/{num_directory_pairs}:")

        # 현재 디렉터리 쌍 처리 함수 호출
        processed_count_in_pair = process_image_pair_directory(
            left_input_dir, right_input_dir,
            left_output_dir, right_output_dir,
            calib_data, img_size_input, img_size_input, # 입력 이미지 크기와 출력 이미지 크기는 같음
            map1_left, map2_left, map1_right, map2_right,
            config['crop_to_valid_roi']
        )
        total_processed_pairs_count += processed_count_in_pair

    print(f"\n--- Rectification Process Finished ---")
    print(f"Successfully processed {total_processed_pairs_count} image pairs in total across all directories.")


if __name__ == "__main__":
    main()