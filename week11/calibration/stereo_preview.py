# Filename: stereo_preview.py
# Description: Load stereo calibration results, display rectified stereo stream, and capture rectified images.

import sys
import os
import time
import argparse
import numpy as np
import cv2
import yaml
from pathlib import Path
from datetime import datetime # 캡쳐 시간 기록용

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0')
from gi.repository import Gst, GLib

# GStreamer 초기화는 main에서 수행합니다.

# --- Helper Functions (Included for self-containment) ---

def link_elements(*elements):
    """여러 GStreamer 요소를 순차적으로 연결하는 헬퍼 함수."""
    for i in range(len(elements) - 1):
            if not elements[i].link(elements[i+1]):
                print(f"Failed to link {elements[i].name} to {elements[i+1].name}")
                return False
    return True

# --- Stereo Calibration Data Load Function (Included for self-containment) ---
# stereo_calibrator.py에서 사용된 로드 함수와 동일합니다.
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

        # 필요한 데이터 로드 및 numpy 배열로 변환
        # stereo_calibrator.py의 save_to_yaml 메서드에서 저장된 키 이름과 일치해야 합니다.
        cameraMatrix1 = np.array(calib_data.get('cameraMatrix1_result'))
        distCoeffs1 = np.array(calib_data.get('distCoeffs1_result'))
        cameraMatrix2 = np.array(calib_data.get('cameraMatrix2_result'))
        distCoeffs2 = np.array(calib_data.get('distCoeffs2_result'))
        R = np.array(calib_data.get('R'))
        T = np.array(calib_data.get('T'))
        R1 = np.array(calib_data.get('R1'))
        R2 = np.array(calib_data.get('R2'))
        P1 = np.array(calib_data.get('P1'))
        P2 = np.array(calib_data.get('P2'))
        Q = np.array(calib_data.get('Q'))
        img_width = calib_data.get('image_width')
        img_height = calib_data.get('image_height')
        img_size = (img_width, img_height)

        validPixROI1 = calib_data.get('validPixROI1') # 튜플 또는 List 형태로 저장되어 있다면 그대로 로드
        validPixROI2 = calib_data.get('validPixROI2') # 튜플 또는 List 형태로 저장되어 있다면 그대로 로드

        # Check if essential data is loaded
        if cameraMatrix1 is None or distCoeffs1 is None or cameraMatrix2 is None or distCoeffs2 is None or \
           R1 is None or R2 is None or P1 is None or P2 is None or Q is None or img_width is None or img_height is None:
             print("Error: Missing essential calibration data (matrices, image size) in YAML file.")
             missing_keys = [k for k in ['cameraMatrix1_result', 'distCoeffs1_result', 'cameraMatrix2_result', 'distCoeffs2_result',
                                         'R1', 'R2', 'P1', 'P2', 'Q', 'image_width', 'image_height'] if calib_data.get(k) is None]
             print(f"Missing keys: {missing_keys}")
             return None

        print("Standard YAML calibration data loaded successfully.")

        # 반환 값 구조를 캘리브레이터 클래스에서 저장된 내용과 일치시킵니다.
        loaded_data = {
            'cameraMatrix1_result': cameraMatrix1,
            'distCoeffs1_result': distCoeffs1,
            'cameraMatrix2_result': cameraMatrix2,
            'distCoeffs2_result': distCoeffs2,
            'R': R,
            'T': T,
            'R1': R1,
            'R2': R2,
            'P1': P1,
            'P2': P2,
            'Q': Q,
            'image_width': img_width,
            'image_height': img_height,
            'validPixROI1': validPixROI1, # 튜플 그대로 저장
            'validPixROI2': validPixROI2, # 튜플 그대로 저장
            # 다른 저장된 정보 (reprojection_error, pattern_size, square_size 등)도 필요하면 로드하여 반환 가능
        }

        return loaded_data

    except Exception as e:
        print(f"Error loading calibration data from {yaml_file_path}: {e}")
        print("This error likely means the YAML file is not in the standard format saved by stereo_calibrator.py.")
        return None


class StereoPreviewer:
    """
    스테레오 캘리브레이션 결과를 사용하여 실시간 렉티피케이션 미리보기를 제공하고,
    렉티피케이션된 이미지를 캡쳐/저장하는 클래스.
    """
    def __init__(self, stereo_calib_yaml: str, left_camera_id: int, right_camera_id: int,
                 cam_mode: int, hflip: bool, vflip: bool, width: int, height: int, fps: int,
                 capture_mode: str = 'none', output_rectified_dir_left: str = None,
                 output_rectified_dir_right: str = None, capture_period_sec: float = 1.0):
        """
        StereoPreviewer 클래스를 초기화합니다.

        Args:
            stereo_calib_yaml (str): 스테레오 캘리브레이션 결과 YAML 파일 경로.
            left_camera_id (int): 왼쪽 카메라 sensor-id.
            right_camera_id (int): 오른쪽 카메라 sensor-id.
            cam_mode (int): GStreamer 카메라 센서 모드.
            hflip (bool): 수평 반전 적용 여부.
            vflip (bool): 수직 반전 적용 여부.
            width (int): 이미지/파이프라인 해상도 너비 (캘리브레이션 해상도와 일치해야 함).
            height (int): 이미지/파이플라인 해상도 높이 (캘리브레이션 해상도와 일치해야 함).
            fps (int): 파이프라인 프레임 레이트.
            capture_mode (str): 'none', 'manual' ('c' 키), 'periodic' (주기적). 기본값 'none'.
            output_rectified_dir_left (str): 렉티피케이션 왼쪽 이미지 저장 디렉터리.
            output_rectified_dir_right (str): 렉티피케이션 오른쪽 이미지 저장 디렉터리.
            capture_period_sec (float): 'periodic' 모드 시 캡쳐 주기 (초).
        """
        self.stereo_calib_yaml = stereo_calib_yaml
        self.left_camera_id = left_camera_id
        self.right_camera_id = right_camera_id
        self.cam_mode = cam_mode
        self.hflip = hflip
        self.vflip = vflip
        self.width = width
        self.height = height
        self.fps = fps
        self.img_size = (width, height)

        self.capture_mode = capture_mode.lower()
        self.output_rectified_dir_left = output_rectified_dir_left
        self.output_rectified_dir_right = output_rectified_dir_right
        self.capture_period_sec = capture_period_sec

        # 캘리브레이션 데이터 저장 변수
        self.calib_data = None
        self.map1_left = None
        self.map2_left = None
        self.map1_right = None
        self.map2_right = None

        # GStreamer 파이프라인 및 앱싱크 저장 변수
        self.left_pipeline = None
        self.left_appsink = None
        self.right_pipeline = None
        self.right_appsink = None

        # 캡쳐 디렉터리 생성 (캡쳐 모드일 경우)
        if self.capture_mode in ['manual', 'periodic']:
            if not self.output_rectified_dir_left or not self.output_rectified_dir_right:
                 print("Error: Capture mode requires output directories (--output_rectified_dir_left, --output_rectified_dir_right).")
                 # 이 시점에서 종료하는 대신, run_preview에서 오류를 처리합니다.
                 self.capture_mode = 'none' # 캡쳐 모드 비활성화
            else:
                try:
                    os.makedirs(self.output_rectified_dir_left, exist_ok=True)
                    os.makedirs(self.output_rectified_dir_right, exist_ok=True)
                    print(f"Rectified images will be saved to: {self.output_rectified_dir_left} and {self.output_rectified_dir_right}")
                except Exception as e:
                     print(f"Error creating output directories: {e}. Disabling capture mode.")
                     self.capture_mode = 'none' # 캡쳐 모드 비활성화


    def _load_calibration_data(self):
        """스테레오 캘리브레이션 YAML 파일을 로드합니다."""
        # (load_stereo_calibration_yaml_standard 함수는 클래스 외부에 정의되어 있다고 가정)
        print("\n--- Loading Stereo Calibration Data ---")
        calib_file_path = Path(self.stereo_calib_yaml)
        self.calib_data = load_stereo_calibration_yaml_standard(calib_file_path)

        if self.calib_data is None:
            print("Failed to load stereo calibration data.")
            return False # 로드 실패

        # 로드된 캘리브레이션 이미지 사이즈와 현재 미리보기 해상도 비교
        img_size_calibrated = (self.calib_data.get('image_width'), self.calib_data.get('image_height'))
        if self.img_size != img_size_calibrated:
             print(f"Warning: Current preview size ({self.img_size}) does not match calibration file size ({img_size_calibrated}).")
             print("Rectification maps will be created for the current preview size, but results may be inaccurate.")
             print(f"Consider running the preview with --width {img_size_calibrated[0]} --height {img_size_calibrated[1]}.")


        print("Stereo calibration data loaded successfully.")
        return True # 로드 성공


    def _build_pipelines(self):
        """양쪽 카메라의 GStreamer 파이프라인을 생성합니다."""
        print("\nBuilding GStreamer pipelines for preview...")
        # build_gst_pipeline 함수는 이전에 정의된 helper 함수입니다.
        # 여기에 포함시키거나 별도 유틸리티 파일에서 임포트합니다. 포함시키는 것으로 합니다.

        if Gst.ElementFactory.find('nvvideoconvert'):
            convert_plugin = 'nvvideoconvert'
            deepstream_available = True
        elif Gst.ElementFactory.find('nvvidconv'):
            convert_plugin = 'nvvidconv'
            deepstream_available = False
        else:
            print("Error: Neither 'nvvideoconvert' nor 'nvvidconv' plugins found! "
                    "Please install DeepStream or Jetson GStreamer extensions.")
            return False

        print(f"🔌 Video converter plugin: {convert_plugin} "
        f"(DeepStream available: {deepstream_available})")

        def build_single_pipeline(camera_id, cam_mode, hflip, vflip, width, height, fps):
             pipeline_name = f"pipeline_cam{camera_id}_preview"
             pipeline = Gst.Pipeline.new(pipeline_name)
             src = Gst.ElementFactory.make('nvarguscamerasrc', f'source_{camera_id}_preview')
             queue1 = Gst.ElementFactory.make('queue', f'queue1_{camera_id}_preview')
             caps_filter = Gst.ElementFactory.make('capsfilter', f'caps_filter_{camera_id}_preview')
             queue2 = Gst.ElementFactory.make('queue', f'queue2_{camera_id}_preview')
             video_convert = Gst.ElementFactory.make(convert_plugin, f'video_convert_{camera_id}_preview')
             queue3 = Gst.ElementFactory.make('queue', f'queue3_{camera_id}_preview')
             caps_filter2 = Gst.ElementFactory.make('capsfilter', f'caps_filter2_{camera_id}_preview')
             queue4 = Gst.ElementFactory.make('queue', f'queue4_{camera_id}_preview')
             appsink = Gst.ElementFactory.make('appsink', f'appsink_{camera_id}_preview')

             if not all([pipeline, src, queue1, caps_filter, queue2, video_convert, queue3, caps_filter2, queue4, appsink]):
                 print(f"GStreamer 요소 생성 실패 (카메라 ID: {camera_id}).")
                 return None, None

             Gst.util_set_object_arg(src, "sensor-id", f"{camera_id}")
             Gst.util_set_object_arg(src, "bufapi-version", "true")
             Gst.util_set_object_arg(src, "sensor-mode", f"{cam_mode}")

             caps_str = f"video/x-raw(memory:NVMM), width=(int){width}, height=(int){height}, framerate=(fraction){fps}/1"
             Gst.util_set_object_arg(caps_filter, "caps", caps_str)

             if hflip and vflip: flip_method = "2"
             elif hflip: flip_method = "4"
             elif vflip: flip_method = "6"
             else: flip_method = "0"
             Gst.util_set_object_arg(video_convert, "flip-method", flip_method)

             caps_str2 = "video/x-raw, format=(string)BGRx"
             Gst.util_set_object_arg(caps_filter2, "caps", caps_str2)

             appsink.set_property("emit-signals", False)
             appsink.set_property("max-buffers", 1)
             appsink.set_property("drop", True)
             appsink_caps = Gst.Caps.from_string(f"video/x-raw, format=(string)BGRx, width=(int){width}, height=(int){height}")
             appsink.set_property("caps", appsink_caps)

             pipeline.add(src)
             pipeline.add(queue1)
             pipeline.add(caps_filter)
             pipeline.add(queue2)
             pipeline.add(video_convert)
             pipeline.add(queue3)
             pipeline.add(caps_filter2)
             pipeline.add(queue4)
             pipeline.add(appsink)

             if not link_elements(src, queue1, caps_filter, queue2, video_convert, queue3, caps_filter2, queue4, appsink):
                 print(f"GStreamer 요소 연결 실패 (카메라 ID: {camera_id}).")
                 pipeline.set_state(Gst.State.NULL)
                 return None, None

             # print(f"GStreamer pipeline built for camera {camera_id}.")
             return pipeline, appsink


        self.left_pipeline, self.left_appsink = build_single_pipeline(
            self.left_camera_id, self.cam_mode, self.hflip, self.vflip,
            self.width, self.height, self.fps
        )
        self.right_pipeline, self.right_appsink = build_single_pipeline(
            self.right_camera_id, self.cam_mode, self.hflip, self.vflip,
            self.width, self.height, self.fps
        )

        if self.left_pipeline is None or self.right_pipeline is None:
            print("Failed to build one or both GStreamer pipelines for preview.")
            return False # 파이프라인 빌드 실패
        print("Pipelines built successfully.")
        return True # 파이프라인 빌드 성공


    def _start_pipelines(self):
        """양쪽 GStreamer 파이프라인을 PLAYING 상태로 시작합니다."""
        if self.left_pipeline is None or self.right_pipeline is None:
             print("Error: Pipelines not built. Cannot start.")
             return False

        print("Setting pipelines to PLAYING state...")
        ret_left = self.left_pipeline.set_state(Gst.State.PLAYING)
        ret_right = self.right_pipeline.set_state(Gst.State.PLAYING)

        if ret_left == Gst.StateChangeReturn.FAILURE or ret_right == Gst.StateChangeReturn.FAILURE:
             print("Error: Failed to set one or both pipelines to PLAYING state.")
             return False

        # PLAYING 상태로 전환될 때까지 약간 대기
        time.sleep(1.0)
        print("Pipelines are PLAYING.")
        return True

    def _stop_pipelines(self):
        """양쪽 GStreamer 파이프라인을 NULL 상태로 중지하고 해제합니다."""
        print("\nSetting GStreamer pipelines to NULL state...")
        if self.left_pipeline:
            self.left_pipeline.set_state(Gst.State.NULL)
            self.left_pipeline = None
            self.left_appsink = None
        if self.right_pipeline:
            self.right_pipeline.set_state(Gst.State.NULL)
            self.right_pipeline = None
            self.right_appsink = None
        print("GStreamer pipelines set to NULL state.")

    def _create_rectification_maps(self):
        """로드된 캘리브레이션 데이터로 렉티피케이션 매핑 테이블을 생성합니다."""
        if self.calib_data is None:
            print("Error: Calibration data not loaded. Cannot create rectification maps.")
            return False

        # calib_data에서 필요한 값 추출
        cameraMatrix1 = self.calib_data.get('cameraMatrix1_result')
        distCoeffs1 = self.calib_data.get('distCoeffs1_result')
        cameraMatrix2 = self.calib_data.get('cameraMatrix2_result')
        distCoeffs2 = self.calib_data.get('distCoeffs2_result')
        R1 = self.calib_data.get('R1')
        P1 = self.calib_data.get('P1')
        R2 = self.calib_data.get('R2')
        P2 = self.calib_data.get('P2')
        # 맵 생성은 현재 미리보기 해상도 (self.img_size) 기준으로 이루어집니다.

        if cameraMatrix1 is None or distCoeffs1 is None or cameraMatrix2 is None or distCoeffs2 is None or \
           R1 is None or P1 is None or R2 is None or P2 is None:
             print("Error: Missing essential calibration data for map creation.")
             return False

        print("Creating rectification maps for size:", self.img_size)
        try:
            # cv2.initUndistortRectifyMap(cameraMatrix, distCoeffs, R, P, newImageSize, m1type)
            self.map1_left, self.map2_left = cv2.initUndistortRectifyMap(
                cameraMatrix1, distCoeffs1, R1, P1, self.img_size, cv2.CV_32FC1
            )
            self.map1_right, self.map2_right = cv2.initUndistortRectifyMap(
                cameraMatrix2, distCoeffs2, R2, P2, self.img_size, cv2.CV_32FC1
            )
            print("Rectification maps created.")
            return True # 맵 생성 성공
        except Exception as e:
            print(f"Error creating rectification maps: {e}")
            return False # 맵 생성 실패

    def _save_rectified_pair(self, left_img, right_img):
        """렉티피케이션된 이미지 쌍을 파일로 저장합니다."""
        if not self.output_rectified_dir_left or not self.output_rectified_dir_right:
             print("Error: Output directories not specified for saving rectified images.")
             return False # 저장 실패

        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3] # 밀리초까지 포함
        left_filename = os.path.join(self.output_rectified_dir_left, f"rectified_left_{timestamp_str}.png")
        right_filename = os.path.join(self.output_rectified_dir_right, f"rectified_right_{timestamp_str}.png")

        try:
            cv2.imwrite(left_filename, left_img)
            cv2.imwrite(right_filename, right_img)
            print(f"Saved rectified pair: {os.path.basename(left_filename)}, {os.path.basename(right_filename)}")
            return True # 저장 성공
        except Exception as e:
            print(f"Error saving rectified image pair: {e}")
            return False # 저장 실패


    def run_preview_loop(self):
        """실시간으로 렉티피케이션된 스테레오 미리보기를 보여주고 이미지를 캡쳐/저장하는 루프."""
        if self.left_appsink is None or self.right_appsink is None:
            print("Error: Appsinks not available. Cannot run preview loop.")
            return

        if self.map1_left is None or self.map1_right is None:
             print("Error: Rectification maps not created. Cannot run preview.")
             return

        print(f"\n--- Starting Rectified Stream Preview ({self.capture_mode.capitalize()} capture) ---")
        print("Press 'q' or ESC to quit.")
        if self.capture_mode == 'manual':
             print("Press 'c' to capture a rectified image pair.")
        elif self.capture_mode == 'periodic':
             print(f"Capturing rectified images every {self.capture_period_sec:.2f} seconds.")
             print(f"Output directories: {self.output_rectified_dir_left}, {self.output_rectified_dir_right}")


        window_name_rect = "Rectified Stereo Preview"
        cv2.namedWindow(window_name_rect, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name_rect, self.img_size[0] * 2, self.img_size[1]) # 좌우 이미지 합쳐서 보여줌

        # validPixROI 정보를 calib_data에서 가져옵니다.
        validPixROI1 = self.calib_data.get('validPixROI1')
        validPixROI2 = self.calib_data.get('validPixROI2')

        last_capture_time = time.time() # 주기적 캡쳐를 위한 타이머 초기화
        captured_count = 0 # 캡쳐된 이미지 쌍 카운트

        try:
            while True:
                # 양쪽 appsink에서 최신 프레임을 가져옴
                left_sample = self.left_appsink.emit("pull-sample")
                right_sample = self.right_appsink.emit("pull-sample")

                if left_sample is None or right_sample is None:
                    if left_sample is None and right_sample is None:
                        time.sleep(0.001)
                    continue

                # GStreamer buffer에서 OpenCV 이미지로 변환 (BGRx -> BGR)
                try:
                    # 왼쪽 프레임
                    left_buffer = left_sample.get_buffer()
                    success_l, map_l = left_buffer.map(Gst.MapFlags.READ)
                    if not success_l: print("Error mapping left buffer"); left_buffer.unmap(map_l); continue
                    left_frame_bgr = np.frombuffer(map_l.data, dtype=np.uint8).reshape((self.img_size[1], self.img_size[0], 4))
                    left_frame_bgr = cv2.cvtColor(left_frame_bgr, cv2.COLOR_BGRA2BGR)
                    left_buffer.unmap(map_l)

                    # 오른쪽 프레임
                    right_buffer = right_sample.get_buffer()
                    success_r, map_r = right_buffer.map(Gst.MapFlags.READ)
                    if not success_r: print("Error mapping right buffer"); right_buffer.unmap(map_r); continue
                    right_frame_bgr = np.frombuffer(map_r.data, dtype=np.uint8).reshape((self.img_size[1], self.img_size[0], 4))
                    right_frame_bgr = cv2.cvtColor(right_frame_bgr, cv2.COLOR_BGRA2BGR)
                    right_buffer.unmap(map_r)

                except Exception as e:
                     print(f"Error processing GStreamer buffer: {e}")
                     continue


                # --- 이미지 렉티피케이션 적용 ---
                rectified_left = cv2.remap(left_frame_bgr, self.map1_left, self.map2_left, cv2.INTER_LINEAR)
                rectified_right = cv2.remap(right_frame_bgr, self.map1_right, self.map2_right, cv2.INTER_LINEAR)

                # 유효 픽셀 영역 표시 (선택 사항)
                # if validPixROI1 and validPixROI2:
                #     x1, y1, w1, h1 = validPixROI1 if isinstance(validPixROI1, tuple) else tuple(validPixROI1)
                #     x2, y2, w2, h2 = validPixROI2 if isinstance(validPixROI2, tuple) else tuple(validPixROI2)
                #     cv2.rectangle(rectified_left, (x1, y1), (x1+w1, y1+h1), (0, 0, 255), 2)
                #     cv2.rectangle(rectified_right, (x2, y2), (x2+w2, y2+h2), (255, 0, 0), 2)

                # 렉티피케이션된 이미지 합쳐서 미리보기
                preview_rectified = np.hstack((rectified_left, rectified_right))

                # 텍스트 오버레이
                cv2.putText(preview_rectified, "Rectified Stream (Press 'q' or ESC to quit)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                if self.capture_mode != 'none':
                     cv2.putText(preview_rectified, f"Captured: {captured_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                     if self.capture_mode == 'periodic':
                          cv2.putText(preview_rectified, f"Period: {self.capture_period_sec:.2f}s", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)


                cv2.imshow(window_name_rect, preview_rectified)

                # --- 이미지 캡쳐 로직 ---
                current_time = time.time()
                trigger_capture = False

                if self.capture_mode == 'manual':
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('c'):
                        trigger_capture = True
                    elif key == ord('q') or key == 27: # 'q' 또는 ESC
                         print("Quit signal received. Exiting rectified stream.")
                         break # 루프 종료
                elif self.capture_mode == 'periodic':
                    key = cv2.waitKey(1) & 0xFF # 주기 모드에서도 키 입력 확인
                    if key == ord('q') or key == 27: # 'q' 또는 ESC
                         print("Quit signal received. Exiting rectified stream.")
                         break # 루프 종료

                    if (current_time - last_capture_time) >= self.capture_period_sec:
                        trigger_capture = True
                        last_capture_time = current_time # 타이머 리셋

                # 캡쳐 트리거 발생 시 이미지 저장
                if trigger_capture:
                     if self._save_rectified_pair(rectified_left, rectified_right):
                          captured_count += 1


        except Exception as e:
            print(f"An error occurred during the rectified stream loop: {e}")
            import traceback
            traceback.print_exc()

        finally:
            cv2.destroyAllWindows()
            print("Rectified stream preview ended.")


    def run_preview(self):
        """
        미리보기 실행의 전체 과정을 조율합니다.
        """
        print(f"--- Starting Stereo Preview Process ---")

        # 1. 캘리브레이션 데이터 로드
        if not self._load_calibration_data():
            print("Failed to load calibration data. Preview aborted.")
            return # 전체 프로세스 실패

        # 2. GStreamer 파이프라인 빌드 및 시작
        if not self._build_pipelines():
            print("Failed to build pipelines. Preview aborted.")
            return # 전체 프로세스 실패

        if not self._start_pipelines():
            print("Failed to start pipelines. Preview aborted.")
            self._stop_pipelines()
            return # 전체 프로세스 실패

        # 3. 렉티피케이션 맵 생성
        if not self._create_rectification_maps():
             print("Failed to create rectification maps. Preview aborted.")
             self._stop_pipelines()
             return # 전체 프로세스 실패

        # 4. 미리보기 루프 실행 및 캡쳐 (루프 안에 캡쳐 로직 포함)
        self.run_preview_loop()

        # 5. 파이프라인 중지
        self._stop_pipelines()

        print("\nStereo Preview Process Completed.")


def main():
    """
    스크립트 단독 실행 시 스테레오 렉티피케이션 미리보기를 수행하며 선택적으로 이미지를 캡쳐하는 main 함수.
    """
    parser = argparse.ArgumentParser(description="스테레오 카메라 렉티피케이션 미리보기 및 이미지 캡쳐")
    Gst.init(sys.argv) # GStreamer 초기화

    parser.add_argument('--stereo_calib_yaml', type=str, default='params/stereo_calibration_results.yaml',
                        help="스테레오 캘리브레이션 결과 YAML 파일 경로")
    parser.add_argument('--left_camera', type=int, default=1, help="왼쪽 카메라 번호 (sensor-id)")
    parser.add_argument('--right_camera', type=int, default=0, help="오른쪽 카메라 번호 (sensor-id)")
    # 미리보기 해상도는 캘리브레이션 해상도와 일치해야 합니다.
    parser.add_argument('--width', type=int, default=320, help="미리보기/파이프라인 해상도 너비 (캘리브레이션 해상도와 일치해야 함)")
    parser.add_argument('--height', type=int, default=256, help="미리보기/파이프라인 해상도 높이 (캘리브레이션 해상도와 일치해야 함)")
    parser.add_argument('--fps', type=int, default=15, help="파이프라인 프레임 레이트")
    parser.add_argument('--camera_mode', type=int, default=2, help="카메라 센서 모드 (예: 2)") # 필요한 경우
    parser.add_argument('--hflip', action='store_true', help="수평 반전 활성화") # 필요한 경우
    parser.add_argument('--vflip', action='store_true', help="수직 반전 활성화") # 필요한 경우


    # --- 렉티피케이션 이미지 캡쳐 관련 인자 ---
    parser.add_argument('--capture_mode', type=str, default='manual',
                        choices=['none', 'manual', 'periodic'],
                        help="렉티피케이션 이미지 캡쳐 모드: 'none' (캡쳐 안함), 'manual' ('c' 키), 'periodic' (주기적).")
    parser.add_argument('--output_rectified_dir_left', type=str, default='rect_images_0',
                        help="[capture_mode 'manual' 또는 'periodic' 시 필수] 렉티피케이션 왼쪽 이미지 저장 디렉터리.")
    parser.add_argument('--output_rectified_dir_right', type=str,  default='rect_images_1',
                        help="[capture_mode 'manual' 또는 'periodic' 시 필수] 렉티피케이션 오른쪽 이미지 저장 디렉터리.")
    parser.add_argument('--capture_period_sec', type=float, default=0.3,
                        help="[capture_mode 'periodic' 시 사용] 이미지 캡쳐 주기 (초).")


    args = parser.parse_args()

    # 캡쳐 모드 시 출력 디렉터리 필수 확인은 클래스 __init__에서 수행합니다.
    # 여기서 기본값을 지정할 수도 있습니다. (예: --output_rectified_dir_left calib_rectified_cam0)

    # StereoPreviewer 객체 생성
    previewer = StereoPreviewer(
        stereo_calib_yaml=args.stereo_calib_yaml,
        left_camera_id=args.left_camera,
        right_camera_id=args.right_camera,
        cam_mode=args.camera_mode,
        hflip=args.hflip,
        vflip=args.vflip,
        width=args.width,
        height=args.height,
        fps=args.fps,
        capture_mode=args.capture_mode, # 캡쳐 관련 인자 전달
        output_rectified_dir_left=args.output_rectified_dir_left,
        output_rectified_dir_right=args.output_rectified_dir_right,
        capture_period_sec=args.capture_period_sec
    )

    # 미리보기 실행 (캡쳐 로직 포함)
    previewer.run_preview()

    print("\nStereo Preview and/or Capture process finished.")


if __name__ == "__main__":
    main()