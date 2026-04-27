# Filename: stereo_calibrator.py
# Description: Perform stereo calibration from saved stereo image pairs and intrinsic calibration files.

import sys
import os
import glob
import argparse
import numpy as np
import cv2
import yaml
from pathlib import Path

# intrinsic_calibrator.py 스크립트에서 사용된 로드 함수를 포함합니다.
# 이는 좌우 intrinsic 데이터를 로드하기 위함입니다.
def load_intrinsic_calibration_yaml(yaml_file: Path):
    """
    Intrinsic calibration 결과를 YAML 파일에서 불러옵니다.
    """
    # (load_intrinsic_calibration_yaml 함수 코드는 이전 intrinsic_calibrator.py의
    # load_calibration_yaml 메서드와 동일합니다. 여기에 다시 포함시키거나
    # 별도 유틸리티 파일에서 임포트할 수 있습니다. 여기서는 편의상 포함시킵니다.)
    if not yaml_file.exists():
        print(f"Error: Intrinsic calibration file not found at {yaml_file}")
        return None, None, None, None

    try:
        with open(yaml_file, 'r') as f:
            # print(f"Attempting to load intrinsic YAML from {yaml_file} using yaml.safe_load...")
            calib_data = yaml.safe_load(f)

        camera_id = calib_data.get('camera_id') # 카메라 ID 로드
        camera_matrix = np.array(calib_data.get('camera_matrix'))
        distortion_coefficients = np.array(calib_data.get('distortion_coefficients'))
        image_width = calib_data.get('image_width')
        image_height = calib_data.get('image_height')
        # 그리드 정보 등 다른 intrinsic 정보도 필요에 따라 로드 가능

        if camera_matrix is None or distortion_coefficients is None or image_width is None or image_height is None:
             print(f"Error: Missing essential data in intrinsic calibration file {yaml_file}")
             return None, None, None, None

        if distortion_coefficients.ndim == 1:
             distortion_coefficients = distortion_coefficients.reshape(1, -1)

        # print(f"Intrinsic calibration data loaded successfully from {yaml_file}.")
        return camera_id, camera_matrix, distortion_coefficients, (image_width, image_height)

    except Exception as e:
        print(f"Error loading intrinsic calibration data from {yaml_file}: {e}")
        print("This error likely means the intrinsic YAML file is not in the format saved by intrinsic_calibrator.py.")
        return None, None, None, None


class StereoCalibrationCalculator:
    """
    스테레오 캘리브레이션 계산 과정을 캡슐화하는 클래스.
    """
    def __init__(self, left_image_dir: str, right_image_dir: str,
                 left_intrinsics_yaml: str, right_intrinsics_yaml: str,
                 pattern_width: int, pattern_height: int, square_size: float,
                 image_width: int, image_height: int, fix_intrinsics: bool = True):
        """
        StereoCalibrationCalculator 클래스를 초기화합니다.

        Args:
            left_image_dir (str): 왼쪽 카메라 캘리브레이션 이미지가 있는 디렉터리.
            right_image_dir (str): 오른쪽 카메라 캘리브레이션 이미지가 있는 디렉터리.
            left_intrinsics_yaml (str): 왼쪽 intrinsic YAML 파일 경로.
            right_intrinsics_yaml (str): 오른쪽 intrinsic YAML 파일 경로.
            pattern_width (int): 체커보드 내부 코너 x 개수.
            pattern_height (int): 체커보드 내부 코너 y 개수.
            square_size (float): 체커보드 한 칸의 실제 크기.
            image_width (int): 캘리브레이션 이미지의 너비 (해상도).
            image_height (int): 캘리브레이션 이미지의 높이 (해상도).
            fix_intrinsics (bool): stereoCalibrate 시 intrinsic 고정 여부 (CALIB_FIX_INTRINSIC 사용).
        """
        if not os.path.isdir(left_image_dir):
             raise FileNotFoundError(f"Left image directory not found: {left_image_dir}")
        if not os.path.isdir(right_image_dir):
             raise FileNotFoundError(f"Right image directory not found: {right_image_dir}")
        if not os.path.exists(left_intrinsics_yaml):
             raise FileNotFoundError(f"Left intrinsics file not found: {left_intrinsics_yaml}")
        if not os.path.exists(right_intrinsics_yaml):
             raise FileNotFoundError(f"Right intrinsics file not found: {right_intrinsics_yaml}")

        self.left_image_dir = left_image_dir
        self.right_image_dir = right_image_dir
        self.left_intrinsics_yaml = left_intrinsics_yaml
        self.right_intrinsics_yaml = right_intrinsics_yaml
        self.pattern_size = (pattern_width, pattern_height)
        self.square_size = square_size
        self.image_size = (image_width, image_height)
        self.fix_intrinsics = fix_intrinsics

        # 로드된 Intrinsic 결과 저장 변수
        self.left_cam_id = None
        self.left_intrinsic_mtx = None
        self.left_dist_coeffs = None
        self.left_img_size_calibrated = None

        self.right_cam_id = None
        self.right_intrinsic_mtx = None
        self.right_dist_coeffs = None
        self.right_img_size_calibrated = None

        # 수집된 데이터 저장 변수
        self.objpoints = []
        self.imgpoints_left = []
        self.imgpoints_right = []
        self.collected_count = 0

        # Stereo 캘리브레이션 결과 저장 변수
        self.stereo_reprojection_error = None
        self.R = None
        self.T = None
        self.E = None
        self.F = None
        self.R1 = None
        self.R2 = None
        self.P1 = None
        self.P2 = None
        self.Q = None
        self.validPixROI1 = None
        self.validPixROI2 = None
        # 최종 stereoCalibrate 결과 Intrinsic (fix 시 로드된 값, use_guess 시 최적화된 값)
        self.cameraMatrix1_result = None
        self.distCoeffs1_result = None
        self.cameraMatrix2_result = None
        self.distCoeffs2_result = None


    def _load_intrinsics(self):
        """Intrinsic calibration YAML 파일들을 로드합니다."""
        print("\n--- Loading Intrinsic Calibration Data ---")
        self.left_cam_id, self.left_intrinsic_mtx, self.left_dist_coeffs, self.left_img_size_calibrated = load_intrinsic_calibration_yaml(Path(self.left_intrinsics_yaml))
        self.right_cam_id, self.right_intrinsic_mtx, self.right_dist_coeffs, self.right_img_size_calibrated = load_intrinsic_calibration_yaml(Path(self.right_intrinsics_yaml))

        if self.left_intrinsic_mtx is None or self.right_intrinsic_mtx is None:
            print("Error: Failed to load one or both intrinsic calibration files.")
            return False # 로드 실패

        # Intrinsic 캘리브레이션 이미지 사이즈와 현재 지정된 이미지 사이즈 비교
        if self.image_size != self.left_img_size_calibrated or self.image_size != self.right_img_size_calibrated:
            print(f"Warning: Specified image size ({self.image_size}) does not match intrinsic calibration size (Left: {self.left_img_size_calibrated}, Right: {self.right_img_size_calibrated}).")
            print("Proceeding with stereo calibration using specified image size, but results may be inaccurate if sizes differ significantly.")
            # 이 경우, CALIB_FIX_INTRINSIC 보다는 CALIB_USE_INTRINSIC_GUESS가 권장됩니다.

        print("Intrinsic data loaded successfully.")
        return True # 로드 성공


    def _collect_stereo_data(self):
        """
        지정된 좌/우 이미지 디렉터리에서 스테레오 이미지 파일을 읽어 코너를 찾고 데이터를 수집합니다.
        """
        print(f"\n--- Collecting Stereo Data from Saved Images ---")
        print(f"Loading images from: {self.left_image_dir} and {self.right_image_dir}")
        print(f"Looking for chessboard pattern: {self.pattern_size[0]}x{self.pattern_size[1]}")
        print(f"Expected image size: {self.image_size}")

        # Assume filenames correspond based on sorting
        left_images = sorted(glob.glob(os.path.join(self.left_image_dir, '*.png'))) # 예시로 png만
        right_images = sorted(glob.glob(os.path.join(self.right_image_dir, '*.png')))

        print(f"Found {len(left_images)} left images and {len(right_images)} right images.")

        # Check file count and potential pairing issues
        if len(left_images) == 0 or len(left_images) != len(right_images):
            print("Error: Image counts do not match or no images found. Cannot proceed with data collection.")
            if len(left_images) > 0 and len(right_images) > 0:
                 print(f"Left: {len(left_images)}, Right: {len(right_images)}")
            self.collected_count = 0
            return False # 데이터 수집 실패

        # 3D object points 준비 (체커보드 좌표계)
        objp = np.zeros((self.pattern_size[0] * self.pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]].T.reshape(-1, 2) * self.square_size

        objpoints = []
        imgpoints_left = []
        imgpoints_right = []
        processed_count = 0

        # 체커보드 검출 플래그 (원본 이미지에서 코너 찾기)
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE

        print(f"Processing {len(left_images)} image pairs...")
        # (Optional) 디버그 창을 여기서 열어 코너 인식 실패 이미지 확인 가능
        # cv2.namedWindow("Corner Detection Debug", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("Corner Detection Debug", self.image_size[0] * 2, self.image_size[1])


        for i in range(len(left_images)):
            left_path = left_images[i]
            right_path = right_images[i]

            # 파일명 일치 여부 간단히 확인 (timestamps)
            left_id = os.path.basename(left_path)[len("left_"):]
            right_id = os.path.basename(right_path)[len("right_"):]
            # left_id = os.path.basename(left_path)[len("right_"):]
            # right_id = os.path.basename(right_path)[len("left_"):]

            if left_id != right_id:
                 # print(f"Warning: Filenames do not match for pair {i+1}: {os.path.basename(left_path)} vs {os.path.basename(right_path)}. Skipping pair.")
                 continue

            # 이미지 로드 (원본 왜곡 이미지)
            img_left_raw = cv2.imread(left_path)
            img_right_raw = cv2.imread(right_path)

            if img_left_raw is None or img_right_raw is None:
                # print(f"Error: Could not read image pair {i+1}. Skipping.")
                continue

            # 이미지 크기 확인 (모든 이미지가 동일 크기여야 함)
            if img_left_raw.shape[:2][::-1] != self.image_size:
                # print(f"Warning: Left image size mismatch for pair {i+1}. Expected {self.image_size}, got {img_left_raw.shape[:2][::-1]}. Skipping pair.")
                continue
            if img_right_raw.shape[:2][::-1] != self.image_size:
                 # print(f"Warning: Right image size mismatch for pair {i+1}. Expected {self.image_size}, got {img_right_raw.shape[:2][::-1]}. Skipping pair.")
                 continue

            # 체커보드 검출 (GRAY 이미지에서 수행)
            gray_left_raw = cv2.cvtColor(img_left_raw, cv2.COLOR_BGR2GRAY)
            gray_right_raw = cv2.cvtColor(img_right_raw, cv2.COLOR_BGR2GRAY)

            # findChessboardCornersSB 사용 시 (일반적으로 더 정확)
            # ret_left, corners_left = cv2.findChessboardCornersSB(gray_left_raw, self.pattern_size)
            # ret_right, corners_right = cv2.findChessboardCornersSB(gray_right_raw, self.pattern_size)

            # 일반 findChessboardCorners 사용 시 (더 많은 이미지에서 작동할 수 있음)
            ret_left, corners_left = cv2.findChessboardCorners(gray_left_raw, self.pattern_size, flags)
            ret_right, corners_right = cv2.findChessboardCorners(gray_right_raw, self.pattern_size, flags)

            if ret_left and ret_right:
                # 코너 정밀화 (원본 이미지에서 찾은 코너를 원본 이미지에서 정밀화)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 300, 1e-6)
                corners_left_refined = cv2.cornerSubPix(gray_left_raw, corners_left, (11, 11), (-1, -1), criteria)
                corners_right_refined = cv2.cornerSubPix(gray_right_raw, corners_right, (11, 11), (-1, -1), criteria)

                # 데이터 저장
                objpoints.append(objp)
                imgpoints_left.append(corners_left_refined)
                imgpoints_right.append(corners_right_refined)
                processed_count += 1
                # print(f"  Collected pair {processed_count}.")

                # (Optional) 디버그 창에 성공 이미지와 코너 표시
                # img_debug = np.hstack((cv2.drawChessboardCorners(img_left_raw.copy(), self.pattern_size, corners_left_refined, ret_left),
                #                        cv2.drawChessboardCorners(img_right_raw.copy(), self.pattern_size, corners_right_refined, ret_right)))
                # cv2.imshow("Corner Detection Debug", img_debug)
                # cv2.waitKey(50) # 짧게 대기

            # else:
                # (Optional) 디버그 창에 실패 이미지와 상태 표시
                # img_debug_fail = np.hstack((img_left_raw.copy(), img_right_raw.copy()))
                # cv2.putText(img_debug_fail, f"Pair {i+1}: Left Found: {ret_left}, Right Found: {ret_right}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                # cv2.imshow("Corner Detection Debug", img_debug_fail)
                # key = cv2.waitKey(0) # 실패 시 키 입력 대기
                # if key == ord('q') or key == 27: # ESC or 'q'
                #      break # 루프 중단


        # cv2.destroyAllWindows() # 디버그 창 닫기

        self.objpoints = objpoints
        self.imgpoints_left = imgpoints_left
        self.imgpoints_right = imgpoints_right
        self.collected_count = processed_count

        print(f"Finished collecting data. Successfully collected {self.collected_count} stereo pairs.")

        return self.collected_count > 0 # 데이터 수집 성공 여부 반환


    def _perform_calibration(self):
        """수집된 데이터와 로드된 intrinsic으로 스테레오 캘리브레이션을 수행합니다."""
        print("\n--- Performing Stereo Calibration Calculation ---")
        min_pairs_needed = 5
        if self.collected_count < min_pairs_needed:
            print(f"Error: Need at least {min_pairs_needed} captured pairs for stereo calibration, but only {self.collected_count} pairs collected. Cannot perform calculation.")
            return False # 캘리브레이션 실패

        if self.left_intrinsic_mtx is None or self.left_dist_coeffs is None or self.right_intrinsic_mtx is None or self.right_dist_coeffs is None:
             print("Error: Intrinsic calibration data not loaded for both cameras. Cannot perform stereo calibration.")
             return False # 캘리브레이션 실패

        # 캘리브레이션 플래그 설정
        calibration_flags = cv2.CALIB_ZERO_DISPARITY # 왼쪽 카메라의 주점을 (0,0)으로 설정 (렉티피케이션에 유용)

        if self.fix_intrinsics:
            calibration_flags |= cv2.CALIB_FIX_INTRINSIC
            # 만약 로드된 intrinsic 이미지 사이즈와 현재 지정 사이즈가 다르다면,
            # CALIB_FIX_INTRINSIC 사용 시 오류가 발생하거나 결과가 틀릴 수 있습니다.
            # 이때는 CALIB_USE_INTRINSIC_GUESS를 사용하는 것이 더 나을 수 있습니다.
            if self.image_size != self.left_img_size_calibrated or self.image_size != self.right_img_size_calibrated:
                 print("Warning: CALIB_FIX_INTRINSIC is used, but image sizes mismatch intrinsic calibration. Consider setting fix_intrinsics=False.")
            print("Stereo calibration flags: CALIB_FIX_INTRINSIC | CALIB_ZERO_DISPARITY")
        else:
            calibration_flags |= cv2.CALIB_USE_INTRINSIC_GUESS
            print("Stereo calibration flags: CALIB_USE_INTRINSIC_GUESS | CALIB_ZERO_DISPARITY")

        try:
            # cv2.stereoCalibrate 호출
            ret, cameraMatrix1_res, distCoeffs1_res, cameraMatrix2_res, distCoeffs2_res, R, T, E, F = cv2.stereoCalibrate(
                self.objpoints, self.imgpoints_left, self.imgpoints_right,
                self.left_intrinsic_mtx, self.left_dist_coeffs, # <-- 로드된 intrinsic을 초기값/고정값으로 사용
                self.right_intrinsic_mtx, self.right_dist_coeffs, # <-- 로드된 intrinsic을 초기값/고정값으로 사용
                self.image_size,
                flags=calibration_flags,
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 300, 1e-6) # 캘리브레이션 정밀도 기준
            )

            self.stereo_reprojection_error = ret
            print(f"Stereo calibration reprojection error: {self.stereo_reprojection_error}")
            if self.stereo_reprojection_error > 1.0: # 일반적인 기준 (1.0보다 크면 결과가 좋지 않다고 판단)
                 print("Warning: High reprojection error. Calibration results may not be accurate.")


            # cv2.stereoRectify 호출: 스테레오 이미지를 같은 평면에 오도록 회전시켜 렉티피케이션 변환을 계산
            R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
                cameraMatrix1_res, distCoeffs1_res, cameraMatrix2_res, distCoeffs2_res, # 최적화된(또는 고정된) intrinsic 사용
                self.image_size, R, T,
                flags=cv2.CALIB_ZERO_DISPARITY,
                alpha=0 # 0이면 유효 픽셀만 포함, 1이면 모든 픽셀 포함 (검은 테두리 생김)
            )

            # 결과 변수에 저장
            self.R = R
            self.T = T
            self.E = E
            self.F = F
            self.R1 = R1
            self.R2 = R2
            self.P1 = P1
            self.P2 = P2
            self.Q = Q
            # validPixROI는 튜플 형태이므로, YAML 저장을 위해 리스트로 변환 필요
            self.validPixROI1 = validPixROI1
            self.validPixROI2 = validPixROI2
            self.cameraMatrix1_result = cameraMatrix1_res
            self.distCoeffs1_result = distCoeffs1_res
            self.cameraMatrix2_result = cameraMatrix2_res
            self.distCoeffs2_result = distCoeffs2_res


            print("Stereo calibration calculation successful.")
            return True # 캘리브레이션 성공

        except Exception as e:
            print(f"An error occurred during stereo calibration calculation: {e}")
            import traceback
            traceback.print_exc()
            return False # 캘리브레이션 실패


    def calibrate(self):
        """
        Intrinsic 로드, 데이터 수집, 스테레오 캘리브레이션 계산의 전체 과정을 실행합니다.
        """
        print("\n--- Starting Stereo Calibration Process ---")

        # 1. Intrinsic 로드
        if not self._load_intrinsics():
            print("Failed to load intrinsic calibration data. Stereo calibration aborted.")
            return False # 전체 프로세스 실패

        # 2. 데이터 수집
        if not self._collect_stereo_data():
             print("Failed to collect enough stereo data from images. Stereo calibration aborted.")
             return False # 전체 프로세스 실패

        # 3. 캘리브레이션 계산
        if not self._perform_calibration():
             print("Stereo calibration calculation failed. Stereo calibration aborted.")
             return False # 전체 프로세스 실패

        print("\nStereo Calibration Process Completed Successfully.")
        return True # 전체 프로세스 성공


    def save_to_yaml(self, output_file: str = "stereo_calibration_results.yaml"):
        """
        클래스 내부에 저장된 스테레오 캘리브레이션 결과를 YAML 파일로 저장합니다.
        """
        if self.R is None or self.T is None or self.R1 is None or self.P1 is None:
            print("Error: No valid stereo calibration data to save. Run calibrate() first.")
            return False

        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        # validPixROI는 튜플이므로 .tolist()가 필요 없을 수 있습니다.
        # 확실하게 리스트로 변환하거나 튜플 상태로 저장합니다.
        # PyYAML은 기본적으로 튜플을 YAML 시퀀스로 저장합니다.
        # 따라서 그대로 저장해도 괜찮습니다.
        validPixROI1_serializable = self.validPixROI1 if self.validPixROI1 is None else list(self.validPixROI1)
        validPixROI2_serializable = self.validPixROI2 if self.validPixROI2 is None else list(self.validPixROI2)


        calib_data = {
            "stereo_reprojection_error": self.stereo_reprojection_error,
            "R": convert_numpy(self.R),
            "T": convert_numpy(self.T),
            "E": convert_numpy(self.E),
            "F": convert_numpy(self.F),
            "R1": convert_numpy(self.R1),
            "R2": convert_numpy(self.R2),
            "P1": convert_numpy(self.P1),
            "P2": convert_numpy(self.P2),
            "Q": convert_numpy(self.Q),
            "image_width": self.image_size[0],
            "image_height": self.image_size[1],
            "pattern_size_width": self.pattern_size[0],
            "pattern_size_height": self.pattern_size[1],
            "square_size_meters": self.square_size,
            "num_images_used": self.collected_count,
            "cameraMatrix1_result": convert_numpy(self.cameraMatrix1_result),
            "distCoeffs1_result": convert_numpy(self.distCoeffs1_result),
            "cameraMatrix2_result": convert_numpy(self.cameraMatrix2_result),
            "distCoeffs2_result": convert_numpy(self.distCoeffs2_result),
            "validPixROI1": validPixROI1_serializable, # 튜플 그대로 저장
            "validPixROI2": validPixROI2_serializable, # 튜플 그대로 저장
            # 로드된 intrinsic 파일 경로도 저장하면 유용할 수 있습니다.
            "left_intrinsics_yaml_used": str(Path(self.left_intrinsics_yaml).name), # 파일 이름만 저장
            "right_intrinsics_yaml_used": str(Path(self.right_intrinsics_yaml).name),
            # 로드된 intrinsic 파일의 카메라 ID도 저장
            "left_camera_id_used": self.left_cam_id,
            "right_camera_id_used": self.right_cam_id,
        }

        def convert_Coefficients(coeffs):
             if coeffs is None:
                 return None
             if isinstance(coeffs, np.ndarray):
                  # (1, N) 형태이면 N개짜리 리스트로 변환
                  return coeffs.flatten().tolist()
             return coeffs # 이미 리스트 등 다른 형태일 경우 그대로 반환


        output_path = Path(output_file)
        try:
            with open(output_path, 'w') as outfile:
                yaml.dump(calib_data, outfile, default_flow_style=False, indent=4, sort_keys=False)
            print(f"Stereo calibration results saved successfully to {output_path}")
            return True
        except Exception as e:
            print(f"Error saving stereo calibration data to {output_path}: {e}")
            return False

    # @classmethod
    # def load_from_yaml(cls, yaml_file: str):
    #     """
    #     YAML 파일에서 스테레오 캘리브레이션 결과를 불러와 클래스 객체를 생성합니다.
    #     (이 객체는 이미지 디렉터리 등 계산에 필요한 원본 정보는 포함하지 않습니다.)
    #     Previewer에서 로딩하는 것과는 역할이 다릅니다.
    #     """
    #     # 필요하다면 구현합니다. Previewer에서는 다른 로딩 함수를 사용합니다.
    #     pass


def main():
    """
    스크립트 단독 실행 시 스테레오 캘리브레이션 계산을 수행하는 main 함수.
    """
    parser = argparse.ArgumentParser(description="스테레오 카메라 캘리브레이션 계산 (YAML 입/출력)")

    parser.add_argument('--left_image_dir', type=str, default='stereo_calib_images_cam1',
                        help="왼쪽 카메라 캘리브레이션 이미지가 저장된 디렉터리")
    parser.add_argument('--right_image_dir', type=str, default='stereo_calib_images_cam0',
                        help="오른쪽 카메라 캘리브레이션 이미지가 저장된 디렉터리")
    parser.add_argument('--left_intrinsics_yaml', type=str, default='intrinsic_calibration_cam1.yaml',
                        help="왼쪽 카메라 intrinsic calibration YAML 파일 경로")
    parser.add_argument('--right_intrinsics_yaml', type=str, default='intrinsic_calibration_cam0.yaml',
                        help="오른쪽 카메라 intrinsic calibration YAML 파일 경로")
    parser.add_argument('--width', type=int, default=1280,
                        help="캘리브레이션 이미지의 너비 (해상도)")
    parser.add_argument('--height', type=int, default=720,
                        help="캘리브레이션 이미지의 높이 (해상도)")
    parser.add_argument('--pattern_width', type=int, default=10,
                        help="체커보드 내부 코너 수 (가로)")
    parser.add_argument('--pattern_height', type=int, default=7,
                        help="체커보드 내부 코너 수 (세로)")
    parser.add_argument('--square_size', type=float, default=0.025,
                        help="체커보드 한 칸의 실제 크기 (단위: intrinsic calib와 동일하게)")
    parser.add_argument('--fix_intrinsics', action='store_true', default=True,
                        help="Stereo calibration 시 intrinsic parameters를 고정 (기본값: True - CALIB_FIX_INTRINSIC). "
                             "False로 설정 시 intrinsic parameters도 함께 최적화 (CALIB_USE_INTRINSIC_GUESS 사용).")
    parser.add_argument('--output_yaml', type=str, default="stereo_calibration_results.yaml",
                        help="스테레오 캘리브레이션 결과 저장 YAML 파일 경로")


    args = parser.parse_args()

    # 필수 입력 파일/디렉터리 존재 확인은 __init__에서 수행됩니다.

    # StereoCalibrationCalculator 객체 생성
    calibrator = StereoCalibrationCalculator(
        left_image_dir=args.left_image_dir,
        right_image_dir=args.right_image_dir,
        left_intrinsics_yaml=args.left_intrinsics_yaml,
        right_intrinsics_yaml=args.right_intrinsics_yaml,
        pattern_width=args.pattern_width,
        pattern_height=args.pattern_height,
        square_size=args.square_size,
        image_width=args.width,
        image_height=args.height,
        fix_intrinsics=args.fix_intrinsics
    )

    # 캘리브레이션 전체 과정 실행 (로드, 수집, 계산)
    calibration_successful = calibrator.calibrate()

    # 결과 저장
    if calibration_successful: # 캘리브레이션 계산까지 성공했으면
        calibrator.save_to_yaml(args.output_yaml)
    else:
        print("\nStereo calibration process did not complete successfully.")
        # sys.exit(1) # 에러 발생 시 종료하려면 주석 해제


if __name__ == "__main__":
    main()