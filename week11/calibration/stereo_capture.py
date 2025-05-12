# Filename: stereo_capture.py (Modified to auto-calibrate stereo after capture)

import sys
import os
import time
import argparse
from datetime import datetime
import glob # stereo_calibrator에서 필요하므로 미리 임포트
import yaml # stereo_calibrator에서 필요하므로 미리 임포트
from pathlib import Path # stereo_calibrator에서 필요하므로 미리 임포트

import cv2
import numpy as np

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0')
from gi.repository import Gst, GLib

# GStreamer 초기화는 main에서 수행합니다.

# stereo_calibrator.py 스크립트에서 필요한 함수들을 가져옵니다.
# 따라서 이 스크립트와 stereo_calibrator.py는 같은 디렉터리에 있어야 합니다.

try:
    from stereo_calibrator import load_intrinsic_calibration_yaml, StereoCalibrationCalculator
except ImportError:
    print("Error: Could not import load_intrinsic_calibration_yaml or StereoCalibrationCalculator from stereo_calibrator.py.")
    print("Please ensure stereo_calibrator.py is in the same directory and has no syntax errors.")
    load_intrinsic_calibration_yaml = None # 임포트 실패 시 None 설정
    StereoCalibrationCalculator = None # 임포트 실패 시 None 설정
    # 이 경우 자동 캘리브레이션은 불가능함을 알림
    print("Automatic stereo calibration after capture is disabled.")
print("Automatic stereo calibration after capture is disabled because stereo_calibrator.py could not be imported.")


def link_elements(*elements):
    """여러 GStreamer 요소를 순차적으로 연결하는 헬퍼 함수."""
    for i in range(len(elements) - 1):
            if not elements[i].link(elements[i+1]):
                print(f"Failed to link {elements[i].name} to {elements[i+1].name}")
                return False
    return True

def build_gst_pipeline(camera_id, cam_mode, hflip, vflip, width, height, fps):
    """
    주어진 카메라 ID와 설정으로 GStreamer 파이프라인을 생성합니다.
    Appsink를 포함하여 OpenCV로 프레임을 가져올 수 있도록 구성합니다.
    """
    """GStreamer 파이프라인을 생성하고 앱싱크를 반환합니다."""
    # 1) DeepStream 전용 nvvideoconvert 유무 검사
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
    try:
        # ... (build_gst_pipeline 함수 내용은 이전 stereo_capture.py와 동일)
        # GStreamer 요소 생성
        pipeline_name = f"pipeline_cam{camera_id}"
        pipeline = Gst.Pipeline.new(pipeline_name)

        src = Gst.ElementFactory.make('nvarguscamerasrc', f'source_{camera_id}')
        queue1 = Gst.ElementFactory.make('queue', f'queue1_{camera_id}')
        caps_filter = Gst.ElementFactory.make('capsfilter', f'caps_filter_{camera_id}')
        queue2 = Gst.ElementFactory.make('queue', f'queue2_{camera_id}')
        video_convert = Gst.ElementFactory.make(convert_plugin, f'video_convert_{camera_id}')
        queue3 = Gst.ElementFactory.make('queue', f'queue3_{camera_id}')
        caps_filter2 = Gst.ElementFactory.make('capsfilter', f'caps_filter2_{camera_id}')
        queue4 = Gst.ElementFactory.make('queue', f'queue4_{camera_id}')
        appsink = Gst.ElementFactory.make('appsink', f'appsink_{camera_id}')

        if not all([pipeline, src, queue1, caps_filter, queue2, video_convert, queue3, caps_filter2, queue4, appsink]):
            print(f"GStreamer 요소 생성에 실패했습니다 (카메라 ID: {camera_id}). 필요한 플러그인이 설치되었는지 확인하세요.")
            return None, None

        # src 설정
        Gst.util_set_object_arg(src, "sensor-id", f"{camera_id}")
        Gst.util_set_object_arg(src, "bufapi-version", "true")
        Gst.util_set_object_arg(src, "sensor-mode", f"{cam_mode}")

        # 첫번째 capsfilter: NVMM 메모리, 해상도, FPS 설정
        caps_str = f"video/x-raw(memory:NVMM), width=(int){width}, height=(int){height}, framerate=(fraction){fps}/1"
        Gst.util_set_object_arg(caps_filter, "caps", caps_str)

        # nvvideoconvert: flip 옵션 설정
        if hflip and vflip:
            Gst.util_set_object_arg(video_convert, "flip-method", "2")  # 180도 회전
        elif hflip:
            Gst.util_set_object_arg(video_convert, "flip-method", "4")  # 수평 반전
        elif vflip:
            Gst.util_set_object_arg(video_convert, "flip-method", "6")  # 수직 반전
        else:
            Gst.util_set_object_arg(video_convert, "flip-method", "0")  # no flip

        # 두번째 capsfilter: OpenCV와 호환되는 포맷 (BGRx)
        caps_str2 = "video/x-raw, format=(string)BGRx"
        Gst.util_set_object_arg(caps_filter2, "caps", caps_str2)

        # appsink 설정: 항상 최신 프레임만 보관 (max-buffers=1, drop=True)
        appsink.set_property("emit-signals", False)
        appsink.set_property("max-buffers", 1)
        appsink.set_property("drop", True)
         # appsink의 caps를 설정하여 원하는 최종 포맷을 보장 (필수)
        appsink_caps = Gst.Caps.from_string(f"video/x-raw, format=(string)BGRx, width=(int){width}, height=(int){height}")
        appsink.set_property("caps", appsink_caps)


        # 요소들을 파이프라인에 추가
        pipeline.add(src)
        pipeline.add(queue1)
        pipeline.add(caps_filter)
        pipeline.add(queue2)
        pipeline.add(video_convert)
        pipeline.add(queue3)
        pipeline.add(caps_filter2)
        pipeline.add(queue4)
        pipeline.add(appsink)


        # 요소들을 순차적으로 연결
        if not link_elements(src, queue1, caps_filter, queue2, video_convert, queue3, caps_filter2, queue4, appsink):
            print(f"GStreamer 요소들을 연결하지 못했습니다 (카메라 ID: {camera_id}).")
            pipeline.set_state(Gst.State.NULL)
            return None, None

        print(f"GStreamer pipeline built for camera {camera_id}.")
        return pipeline, appsink
    except Exception as e:
        print(f"파이프라인 생성 중 오류 발생 (카메라 ID: {camera_id}): {e}")
        return None, None


def run_stereo_capture_loop(left_pipeline, left_appsink, right_pipeline, right_appsink,
                          left_output_dir, right_output_dir,
                          img_size, pattern_size): # pattern_size는 미리보기 표시용
    """
    스테레오 이미지를 실시간으로 캡쳐하고 지정할 때 파일로 저장하는 루프.
    """
    window_name = "Stereo Capture Preview (Press 'c' to Save, 'q' or ESC to Quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, img_size[0] * 2, img_size[1]) # 좌우 이미지 합쳐서 보여줌

    captured_count = 0
    instructions = "Press 'c' to capture, 'q' or ESC to quit."
    print("\n--- Starting Stereo Capture ---")
    print(instructions)


    try:
        while True:
            # 양쪽 appsink에서 최신 프레임을 가져옴 (동기화되지 않을 수 있습니다)
            # 정확한 동기화를 위해서는 GStreamer 레벨에서 별도 처리가 필요할 수 있습니다.
            # 여기서는 단순히 각 파이프라인에서 준비된 최신 프레임을 가져옵니다.
            left_sample = left_appsink.emit("pull-sample")
            right_sample = right_appsink.emit("pull-sample")

            if left_sample is None or right_sample is None:
                # 프레임이 아직 준비되지 않았거나 오류 발생
                if left_sample is None and right_sample is None:
                     time.sleep(0.001)
                continue # 한쪽이라도 프레임이 있으면 계속 진행

            # 왼쪽 프레임 처리
            left_buffer = left_sample.get_buffer()
            success_l, map_l = left_buffer.map(Gst.MapFlags.READ)
            if not success_l:
                 print("Error mapping left buffer")
                 left_buffer.unmap(map_l) # 매핑 실패 시에도 언매핑 시도
                 continue
            try:
                # BGRx 포맷에서 BGR로 변환 (원본 이미지)
                left_frame_bgr_raw = np.frombuffer(map_l.data, dtype=np.uint8).reshape((img_size[1], img_size[0], 4))
                left_frame_bgr_raw = cv2.cvtColor(left_frame_bgr_raw, cv2.COLOR_BGRA2BGR)
            except Exception as e:
                print("Left frame processing error:", e)
                left_buffer.unmap(map_l)
                continue
            left_buffer.unmap(map_l)

            # 오른쪽 프레임 처리
            right_buffer = right_sample.get_buffer()
            success_r, map_r = right_buffer.map(Gst.MapFlags.READ)
            if not success_r:
                 print("Error mapping right buffer")
                 right_buffer.unmap(map_r) # 매핑 실패 시에도 언매핑 시도
                 continue
            try:
                # BGRx 포맷에서 BGR로 변환 (원본 이미지)
                right_frame_bgr_raw = np.frombuffer(map_r.data, dtype=np.uint8).reshape((img_size[1], img_size[0], 4))
                right_frame_bgr_raw = cv2.cvtColor(right_frame_bgr_raw, cv2.COLOR_BGRA2BGR)
            except Exception as e:
                print("Right frame processing error:", e)
                right_buffer.unmap(map_r)
                continue
            right_buffer.unmap(map_r)


            # --- 미리보기 화면 구성 ---
            # 미리보기에는 코너 찾기 결과 등을 표시하기 위해 원본 이미지를 복사하여 사용
            left_display = left_frame_bgr_raw.copy()
            right_display = right_frame_bgr_raw.copy()

            # 체커보드 검출 (미리보기 표시용 - 원본 이미지에서 빠르게 시도)
            # 실제 캘리브레이션 코너 찾기는 stereo_calibrator.py에서 수행
            left_gray_raw = cv2.cvtColor(left_frame_bgr_raw, cv2.COLOR_BGR2GRAY)
            right_gray_raw = cv2.cvtColor(right_frame_bgr_raw, cv2.COLOR_BGR2GRAY)

            flags_find_corners = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK # 미리보기는 빠르게
            ret_left_preview, corners_left_preview = cv2.findChessboardCorners(left_gray_raw, pattern_size, flags_find_corners)
            ret_right_preview, corners_right_preview = cv2.findChessboardCorners(right_gray_raw, pattern_size, flags_find_corners)

            if ret_left_preview:
                cv2.drawChessboardCorners(left_display, pattern_size, corners_left_preview, ret_left_preview)
            if ret_right_preview:
                cv2.drawChessboardCorners(right_display, pattern_size, corners_right_preview, ret_right_preview)

            # 코너 찾기 상태 텍스트 오버레이 (미리보기용)
            if ret_left_preview and ret_right_preview:
                 corner_status_text = "Chessboard: FOUND (Press 'c')"
                 corner_status_color = (0, 255, 0) # Green
            else:
                 corner_status_text = "Chessboard: NOT FOUND"
                 corner_status_color = (0, 0, 255) # Red

            # 좌/우 이미지를 좌우 결합하여 미리보기
            preview = np.hstack((left_display, right_display))

            # 텍스트 오버레이
            cv2.putText(preview, instructions, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(preview, f"Captured pairs: {captured_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(preview, corner_status_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, corner_status_color, 2)


            cv2.imshow(window_name, preview)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('c'):
                # 코너 찾기 여부와 무관하게 저장 (필요하다면 코너 찾았을 때만 저장하도록 수정 가능)
                # 여기서는 사용자가 원할 때 언제든 캡쳐하도록 구현
                print(f"Capturing pair {captured_count+1}...")

                # !!! 원본 프레임을 파일로 저장 !!!
                timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3] # 밀리초까지 포함
                left_filename = os.path.join(left_output_dir, f"left_{timestamp_str}.png")
                right_filename = os.path.join(right_output_dir, f"right_{timestamp_str}.png")

                try:
                    cv2.imwrite(left_filename, left_frame_bgr_raw) # <-- 원본 이미지 저장
                    cv2.imwrite(right_filename, right_frame_bgr_raw) # <-- 원본 이미지 저장
                    captured_count += 1
                    print(f"Saved pair {captured_count}. (Raw images saved)")
                except Exception as e:
                    print(f"Error saving image pair: {e}")


            elif key == ord('q') or key == 27: # 'q' 또는 ESC
                print("Quit signal received. Exiting capture loop.")
                break

    except Exception as e:
         print(f"An error occurred during the capture loop: {e}")
         import traceback
         traceback.print_exc()

    finally:
        cv2.destroyAllWindows()
        print("Capture loop finished.")

    return captured_count # 캡쳐된 이미지 쌍 개수 반환


def main():
    parser = argparse.ArgumentParser(description="스테레오 카메라 캡쳐 시스템 및 선택적 자동 캘리브레이션")
    Gst.init(sys.argv) # GStreamer 초기화

    parser.add_argument('--camera_mode', type=int, default=2, help="카메라 센서 모드 (예: 2)")
    parser.add_argument('--hflip', action='store_true', help="수평 반전 활성화")
    parser.add_argument('--vflip', action='store_true', help="수직 반전 활성화")
    parser.add_argument('--left_camera', type=int, default=1, help="왼쪽 카메라 번호 (sensor-id)")
    parser.add_argument('--right_camera', type=int, default=0, help="오른쪽 카메라 번호 (sensor-id)")
    parser.add_argument('--width', type=int, default=320, help="이미지/파이프라인 해상도 너비")
    parser.add_argument('--height', type=int, default=256, help="이미지/파이프라인 해상도 높이")
    parser.add_argument('--fps', type=int, default=10, help="파이프라인 프레임 레이트")
    parser.add_argument('--left_output_dir', type=str, default="stereo_calib_images/cam1", help="왼쪽 이미지 저장 디렉터리")
    parser.add_argument('--right_output_dir', type=str, default="stereo_calib_images/cam0", help="오른쪽 이미지 저장 디렉터리")
    parser.add_argument('--pattern_width', type=int, default=10, help="체커보드 내부 코너 수 (가로 - 미리보기 표시 및 캘리브레이션용)")
    parser.add_argument('--pattern_height', type=int, default=7, help="체커보드 내부 코너 수 (세로 - 미리보기 표시 및 캘리브레이션용)")
    parser.add_argument('--square_size', type=float, default=0.025, help="체커보드 한 칸의 실제 크기 (미터 단위 - 캘리브레이션용)")

    # --- 자동 캘리브레이션 관련 인자 ---
    parser.add_argument('--auto_calibrate', action='store_true', default=True,
                        help="캡쳐 종료 후 스테레오 캘리브레이션을 자동으로 수행합니다.")
    parser.add_argument('--left_intrinsics_yaml', type=str, default='params/intrinsic_param_cam1.yaml',
                        help="[--auto_calibrate 시 필수] 왼쪽 카메라 intrinsic calibration YAML 파일 경로")
    parser.add_argument('--right_intrinsics_yaml', type=str, default="params/intrinsic_param_cam0.yaml",
                        help="[--auto_calibrate 시 필수] 오른쪽 카메라 intrinsic calibration YAML 파일 경로")
    parser.add_argument('--stereo_calib_output_yaml', type=str, default="params/stereo_calibration_results.yaml",
                        help="[--auto_calibrate 시 사용] 스테레오 캘리브레이션 결과 저장 YAML 파일 경로")
    parser.add_argument('--fix_intrinsics', action='store_true', default=True,
                        help="[--auto_calibrate 시 사용] Stereo calibration 시 intrinsic parameters를 고정 (기본값: True - CALIB_FIX_INTRINSIC). "
                             "False로 설정 시 intrinsic parameters도 함께 최적화 (CALIB_USE_INTRINSIC_GUESS 사용).")


    args = parser.parse_args()

    # 출력 디렉터리 생성
    os.makedirs(args.left_output_dir, exist_ok=True)
    os.makedirs(args.right_output_dir, exist_ok=True)
    print(f"Left images will be saved to: {args.left_output_dir}")
    print(f"Right images will be saved to: {args.right_output_dir}")

    img_size = (args.width, args.height)
    pattern_size = (args.pattern_width, args.pattern_height)

    # GStreamer 파이프라인 생성
    print("\nBuilding GStreamer pipelines...")
    left_pipeline, left_appsink = build_gst_pipeline(
        args.left_camera, args.camera_mode, args.hflip, args.vflip,
        args.width, args.height, args.fps
    )
    right_pipeline, right_appsink = build_gst_pipeline(
        args.right_camera, args.camera_mode, args.hflip, args.vflip,
        args.width, args.height, args.fps
    )

    if left_pipeline is None or right_pipeline is None:
        print("Failed to build one or both GStreamer pipelines. Exiting.")
        sys.exit(1)
    print("Pipelines built.")

    # 파이프라인 실행
    print("Setting pipelines to PLAYING state...")
    left_pipeline.set_state(Gst.State.PLAYING)
    right_pipeline.set_state(Gst.State.PLAYING)

    # PLAYING 상태로 전환될 때까지 약간 대기
    time.sleep(1.0)
    print("Pipelines are PLAYING.")

    # 캡쳐 루프 실행
    captured_count = run_stereo_capture_loop(
        left_pipeline, left_appsink, right_pipeline, right_appsink, # right_pipeline은 루프에서 직접 사용 안 함
        args.left_output_dir, args.right_output_dir,
        img_size, pattern_size
    )

    # GStreamer 파이프라인 정리
    print("\nSetting GStreamer pipelines to NULL state...")
    if left_pipeline:
        left_pipeline.set_state(Gst.State.NULL)
    if right_pipeline:
        right_pipeline.set_state(Gst.State.NULL)
    print("GStreamer pipelines set to NULL state.")

    # --- 캡쳐 종료 후 스테레오 캘리브레이션 자동 실행 ---
    if args.auto_calibrate:
        print(f"\n--- Capture finished. Automatically starting Stereo Calibration ---")
        min_pairs_needed = 5
        # 임포트 성공 여부 및 캡쳐 이미지 수 확인
        if StereoCalibrationCalculator is None: # <-- 임포트 실패했는지 확인
            print("Cannot perform automatic calibration: StereoCalibrationCalculator class not found.")
        elif captured_count < min_pairs_needed:
            print(f"Not enough captured pairs ({captured_count}) for stereo calibration. Need at least {min_pairs_needed}. Skipping automatic calibration.")
        else:
            try:
                # StereoCalibrationCalculator 객체 생성 및 실행
                # 이 객체는 캡쳐된 이미지가 있는 디렉터리, intrinsic 파일 경로, 체커보드 정보를 사용합니다.
                calibrator = StereoCalibrationCalculator(
                    left_image_dir=args.left_output_dir,
                    right_image_dir=args.right_output_dir,
                    left_intrinsics_yaml=args.left_intrinsics_yaml, # 기본값 또는 지정된 경로 사용
                    right_intrinsics_yaml=args.right_intrinsics_yaml, # 기본값 또는 지정된 경로 사용
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
                    calibrator.save_to_yaml(args.stereo_calib_output_yaml) # 기본값 또는 지정된 경로 사용
                else:
                    print("Automatic stereo calibration calculation failed.")

            except FileNotFoundError as e:
                # StereoCalibrationCalculator 초기화 중 발생한 FileNotFoundError 처리
                print(f"Error during automatic calibration setup: {e}")
                print("Please ensure intrinsic YAML files and image directories exist.")
            except Exception as e:
                print(f"An unexpected error occurred during automatic stereo calibration: {e}")
                import traceback
                traceback.print_exc()

    else:
        print("\nAutomatic stereo calibration is disabled. To run, use --auto_calibrate.")



if __name__ == "__main__":
    main()