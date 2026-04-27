# Filename: intrinsic_calibrator.py
# Description: Intrinsic Calibration from captured single camera images, saves result to YAML.

import sys
import os
import glob
import argparse
import numpy as np
import cv2
import yaml
from pathlib import Path

class IntrinsicCalibrator:
    """
    단일 카메라의 intrinsic calibration을 수행하고 결과를 관리하는 클래스.
    """
    def __init__(self, image_dir: str, grid_x: int, grid_y: int, grid_size: float, camera_id: int):
        """
        IntrinsicCalibrator 클래스를 초기화합니다.

        Args:
            image_dir (str): 캘리브레이션 이미지가 저장된 디렉터리 경로.
            grid_x (int): 체커보드 내부 코너의 x 방향 개수.
            grid_y (int): 체커보드 내부 코너의 y 방향 개수.
            grid_size (float): 한 체커보드 정사각형의 실제 크기 (단위: 사용자 정의).
            camera_id (int): 이 캘리브레이션이 속한 카메라의 ID.
        """
        if not os.path.isdir(image_dir):
             raise FileNotFoundError(f"Image directory not found: {image_dir}")

        self.image_dir = image_dir
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.grid_size = grid_size
        self.camera_id = camera_id

        # 캘리브레이션 결과 저장 변수
        self.camera_matrix = None
        self.distortion_coefficients = None
        self.image_size = None
        self.num_images_used = 0
        self.reprojection_error = None

    def detect_corners(self, use_sb_alg=True):
        """
        초기화된 이미지 디렉터리에서 체커보드 코너를 검출합니다.

        Args:
            use_sb_alg (bool): findChessboardCornersSB 알고리즘 사용 여부.

        Returns:
            tuple: (objpoints, imgpoints, imgsize)
                   objpoints: 각 이미지에 대한 3D 월드 좌표 리스트.
                   imgpoints: 각 이미지에 대한 검출된 2D 이미지 좌표 리스트.
                   imgsize: 첫 번째 이미지 기준 해상도 (width, height).
        """
        objp = np.zeros((self.grid_x * self.grid_y, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.grid_x, 0:self.grid_y].T.reshape(-1, 2) * self.grid_size

        objpoints = []
        imgpoints = []

        images = sorted(glob.glob(os.path.join(self.image_dir, "*.png")))
        if not images:
            images = sorted(glob.glob(os.path.join(self.image_dir, "*.jpg")))

        if not images:
            print(f"Warning: No images found in directory: {self.image_dir}")
            return [], [], (0, 0)

        imgsize = (0, 0)
        print(f"Found {len(images)} images in {self.image_dir}. Processing...")

        for fname in images:
            # print(f"Processing {os.path.basename(fname)} ...", end=" ")
            img = cv2.imread(fname)
            if img is None:
                # print("Failed to load") # 너무 많이 출력될 수 있으므로 주석 처리
                continue

            if imgsize == (0, 0):
                 imgsize = (img.shape[1], img.shape[0])
                 print(f"\nDetected image size: {imgsize}")
            elif (img.shape[1], img.shape[0]) != imgsize:
                 # print(f"Warning: Image size mismatch for {os.path.basename(fname)}. Expected {imgsize}, skipping.")
                 continue # 크기 불일치 이미지 건너뛰기


            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 체커보드 코너 검출
            if use_sb_alg:
                ret, corners = cv2.findChessboardCornersSB(gray, (self.grid_x, self.grid_y))
            else:
                flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
                ret, corners = cv2.findChessboardCorners(gray, (self.grid_x, self.grid_y), flags)
                if ret:
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 300, 1e-6)
                    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            if ret:
                # print("Success")
                objpoints.append(objp)
                imgpoints.append(corners)
            # else:
                # print("Failed") # 너무 많이 출력될 수 있으므로 주석 처리

        print(f"Finished corner detection. Found corners in {len(objpoints)} images.")
        return objpoints, imgpoints, imgsize

    def calibrate_camera(self, objpoints, imgpoints, imgsize):
        """
        수집된 데이터로 단일 카메라 intrinsic calibration을 수행합니다.

        Args:
            objpoints: 3D 월드 좌표 리스트.
            imgpoints: 2D 이미지 좌표 리스트.
            imgsize: 이미지 해상도 (width, height).

        Returns:
            tuple: (mtx, dist, mean_error)
                   mtx: 카메라 매트릭스 (intrinsic matrix).
                   dist: 왜곡 계수.
                   mean_error: 평균 reprojection error.
        """
        if len(objpoints) == 0:
             print("Error: No points to calibrate.")
             return None, None, None

        print("Performing camera calibration...")
        flags = cv2.CALIB_RATIONAL_MODEL if len(imgpoints) > 0 and len(imgpoints[0]) is not None and len(imgpoints[0][0]) > 5 else cv2.CALIB_TILTED_MODEL
        # Add CALIB_ZERO_TANGENT_DIST, CALIB_FIX_PRINCIPAL_POINT if applicable
        # flags |= cv2.CALIB_ZERO_TANGENT_DIST # 접선 왜곡 무시
        # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT # 주점 고정 (보통 이미지 중앙)


        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, imgsize, None, None, flags=flags, criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 300, 1e-6))

        if not ret:
            raise RuntimeError("Calibration failed.")

        # 캘리브레이션 결과의 평균 reprojection error 계산
        mean_error = 0
        for i in range(len(objpoints)):
            # None 체크 추가: rvecs 또는 tvecs가 None일 수 있는 경우 대비
            if rvecs is not None and tvecs is not None and i < len(rvecs) and i < len(tvecs):
                imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
                # imgpoints[i]가 None이 아니고 imgpoints2도 None이 아닌 경우에만 norm 계산
                if imgpoints[i] is not None and imgpoints2 is not None and len(imgpoints[i]) > 0:
                     error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints[i])
                     mean_error += error
                # else:
                     # print(f"Warning: Skipping error calculation for image {i} due to empty points or projection failure.")
            # else:
                # print(f"Warning: Skipping error calculation for image {i} due to missing rvec/tvec.")


        if len(objpoints) > 0:
             mean_error /= len(objpoints)
        else:
             mean_error = float('inf') # 점이 없으면 에러 무한대

        print(f"\nAverage reprojection error: {mean_error}")
        if mean_error > 1.0:
             print("Warning: High reprojection error. Calibration results may not be accurate.")


        return mtx, dist, mean_error

    def calibrate(self, use_sb_alg=True):
        """
        코너 검출부터 캘리브레이션 계산까지 전체 과정을 실행하고 결과를 클래스 내부에 저장합니다.
        """
        print(f"--- Starting Intrinsic Calibration for camera {self.camera_id} from images in: {self.image_dir} ---")
        objpoints, imgpoints, imgsize = self.detect_corners(use_sb_alg)

        if len(objpoints) == 0:
            print("Error: No valid chessboard corners were detected in any image. Calibration skipped.")
            self.camera_matrix = None
            self.distortion_coefficients = None
            self.image_size = imgsize # 이미지 사이즈는 찾았을 수 있으므로 저장
            self.num_images_used = 0
            self.reprojection_error = None
            return # 캘리브레이션 수행 불가

        mtx, dist, mean_error = self.calibrate_camera(objpoints, imgpoints, imgsize)

        if mtx is not None:
            self.camera_matrix = mtx
            self.distortion_coefficients = dist
            self.image_size = imgsize
            self.num_images_used = len(objpoints)
            self.reprojection_error = mean_error
            print("\nCalibration results stored in the object.")
        else:
            print("\nCalibration calculation failed. Results not stored.")
            self.camera_matrix = None
            self.distortion_coefficients = None
            # 이미지 사이즈 등은 detect_corners에서 가져온 값 유지
            self.num_images_used = 0
            self.reprojection_error = None


    def save_to_yaml(self, output_file: str = "intrinsic_calibration.yaml"):
        """
        클래스 내부에 저장된 intrinsic calibration 결과를 YAML 파일로 저장합니다.
        """
        if self.camera_matrix is None or self.distortion_coefficients is None:
            print("Error: No valid calibration data to save. Run calibrate() first.")
            return False

        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        calib_data = {
            "camera_id": self.camera_id,
            "camera_matrix": convert_numpy(self.camera_matrix),
            "distortion_coefficients": convert_numpy(self.distortion_coefficients),
            "image_width": self.image_size[0],
            "image_height": self.image_size[1],
            "pattern_size_width": self.grid_x,
            "pattern_size_height": self.grid_y,
            "square_size_meters": self.grid_size, # 실제 단위는 사용자에 따라 다름
            "num_images_used": self.num_images_used,
            "reprojection_error": self.reprojection_error,
        }

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(output_path, 'w') as outfile:
                yaml.dump(calib_data, outfile, default_flow_style=False, indent=4, sort_keys=False)
            print(f"Intrinsic calibration results saved successfully to {output_path}")
            return True
        except Exception as e:
            print(f"Error saving intrinsic calibration data to {output_path}: {e}")
            return False

    @classmethod
    def load_from_yaml(cls, yaml_file: str):
        """
        YAML 파일에서 intrinsic calibration 결과를 불러와 IntrinsicCalibrator 객체를 생성합니다.
        (로드된 데이터만 포함하며, 이미지 디렉터리 정보 등은 포함되지 않음)
        """
        yaml_path = Path(yaml_file)
        if not yaml_path.exists():
            print(f"Error: Intrinsic calibration file not found at {yaml_file}")
            return None

        try:
            with open(yaml_path, 'r') as f:
                calib_data = yaml.safe_load(f)

            camera_id = calib_data.get('camera_id')
            camera_matrix = np.array(calib_data.get('camera_matrix'))
            distortion_coefficients = np.array(calib_data.get('distortion_coefficients'))
            image_width = calib_data.get('image_width')
            image_height = calib_data.get('image_height')
            grid_x = calib_data.get('pattern_size_width')
            grid_y = calib_data.get('pattern_size_height')
            grid_size = calib_data.get('square_size_meters')
            num_images_used = calib_data.get('num_images_used')
            reprojection_error = calib_data.get('reprojection_error')


            if camera_matrix is None or distortion_coefficients is None or image_width is None or image_height is None:
                 print(f"Error: Missing essential data in intrinsic calibration file {yaml_file}")
                 return None

            if distortion_coefficients.ndim == 1:
                 distortion_coefficients = distortion_coefficients.reshape(1, -1)

            print(f"Intrinsic calibration data loaded successfully from {yaml_file}.")

            # IntrinsicCalibrator 객체를 생성하고 로드된 데이터를 채웁니다.
            # 이미지 디렉터리는 이 시점에서는 알 수 없으므로 None 또는 빈 문자열로 설정합니다.
            loaded_calibrator = cls(
                image_dir="", # 이미지 디렉터리 정보는 파일에 저장되지 않으므로 빈 값
                grid_x=grid_x if grid_x is not None else 0, # 저장되지 않았다면 기본값 또는 0
                grid_y=grid_y if grid_y is not None else 0,
                grid_size=grid_size if grid_size is not None else 0.0,
                camera_id=camera_id if camera_id is not None else -1 # 저장되지 않았다면 -1
            )
            loaded_calibrator.camera_matrix = camera_matrix
            loaded_calibrator.distortion_coefficients = distortion_coefficients
            loaded_calibrator.image_size = (image_width, image_height)
            loaded_calibrator.num_images_used = num_images_used if num_images_used is not None else 0
            loaded_calibrator.reprojection_error = reprojection_error if reprojection_error is not None else float('inf')


            return loaded_calibrator

        except Exception as e:
            print(f"Error loading intrinsic calibration data from {yaml_file}: {e}")
            print("This error likely means the YAML file is not in the format saved by this script.")
            return None


def main():
    """
    스크립트 단독 실행 시 Intrinsic Calibration을 수행하는 main 함수.
    """
    parser = argparse.ArgumentParser(description="Intrinsic Calibration from Captured Images (YAML Output)")
    parser.add_argument('--image_dir', type=str, required=True,
                        help="Directory containing captured chessboard images")
    parser.add_argument('--camera_id', type=int, default=0,
                        help="ID of the camera these images belong to (saved in YAML)")
    parser.add_argument('--grid_x', type=int, default=10,
                        help="Number of internal corners in x dimension")
    parser.add_argument('--grid_y', type=int, default=7,
                        help="Number of internal corners in y dimension")
    parser.add_argument('--grid_size', type=float, default=0.025,
                        help="Size of one square on the chessboard (in real-world units)")
    parser.add_argument('--use_sb_alg', action='store_true',
                        help="Use the findChessboardCornersSB algorithm")
    parser.add_argument('--output_yaml', type=str, default="intrinsic_calibration.yaml",
                        help="Output YAML file to save calibration results")
    args = parser.parse_args()

    # Check if image directory exists
    if not os.path.isdir(args.image_dir):
         print(f"Error: Image directory not found at {args.image_dir}")
         sys.exit(1)

    # IntrinsicCalibrator 객체 생성 및 캘리브레이션 실행
    calibrator = IntrinsicCalibrator(
        image_dir=args.image_dir,
        grid_x=args.grid_x,
        grid_y=args.grid_y,
        grid_size=args.grid_size,
        camera_id=args.camera_id # camera_id 전달
    )

    calibrator.calibrate(args.use_sb_alg) # 캘리브레이션 수행

    # 결과 저장
    if calibrator.camera_matrix is not None: # 캘리브레이션 성공 시
         calibrator.save_to_yaml(args.output_yaml)


if __name__ == "__main__":
    main()