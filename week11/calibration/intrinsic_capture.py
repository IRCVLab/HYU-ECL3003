# Filename: intrinsic_capture.py
# Description: Capture images for intrinsic calibration from a single camera and auto-calibrate.

import sys
import os
import time
import argparse
from datetime import datetime

import cv2
import numpy as np

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0')
from gi.repository import Gst, GLib
from pathlib import Path

# GStreamer 초기화는 main에서 수행합니다.

# intrinsic_calibrator.py 스크립트에서 IntrinsicCalibrator 클래스를 가져옵니다.
# 이 스크립트와 intrinsic_calibrator.py는 같은 디렉터리에 있어야 합니다.
try:
    from intrinsic_calibrator import IntrinsicCalibrator
except ImportError:
    print("Error: Could not import IntrinsicCalibrator from intrinsic_calibrator.py.")
    print("Please ensure intrinsic_calibrator.py is in the same directory.")
    IntrinsicCalibrator = None # 임포트 실패 시 클래스를 None으로 설정


class IntrinsicCapturer:
    """
    단일 카메라에서 intrinsic calibration용 이미지를 캡쳐하고 저장하는 클래스.
    """
    def __init__(self, camera_id: int, cam_mode: int, hflip: bool, vflip: bool,
                 width: int, height: int, fps: int, output_dir: str,
                 pattern_width: int, pattern_height: int, grid_size: float):
        """
        IntrinsicCapturer 클래스를 초기화합니다.

        Args:
            camera_id (int): 사용할 카메라의 sensor-id.
            cam_mode (int): GStreamer 카메라 센서 모드.
            hflip (bool): 수평 반전 적용 여부.
            vflip (bool): 수직 반전 적용 여부.
            width (int): 이미지/파이프라인 해상도 너비.
            height (int): 이미지/파이프라인 해상도 높이.
            fps (int): 파이프라인 프레임 레이트.
            output_dir (str): 캡쳐 이미지 저장 디렉터리 경로.
            pattern_width (int): 체커보드 내부 코너 x 개수 (미리보기 표시용).
            pattern_height (int): 체커보드 내부 코너 y 개수 (미리보기 표시용).
            grid_size (float): 체커보드 한 칸의 실제 크기 (캘리브레이터 전달용).
        """
        self.camera_id = camera_id
        self.cam_mode = cam_mode
        self.hflip = hflip
        self.vflip = vflip
        self.width = width
        self.height = height
        self.fps = fps
        self.output_dir = output_dir
        self.pattern_size = (pattern_width, pattern_height)
        self.grid_size = grid_size # 캘리브레이터에 전달할 값
        self.img_size = (width, height)

        self.pipeline = None
        self.appsink = None

    def _link_elements(self, *elements):
        """여러 GStreamer 요소를 순차적으로 연결하는 헬퍼 함수."""
        for i in range(len(elements) - 1):
            if not elements[i].link(elements[i+1]):
                print(f"Failed to link {elements[i].name} to {elements[i+1].name}")
                return False
        return True

    def build_pipeline(self):
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
            # GStreamer 요소 생성
            pipeline_name = f"pipeline_cam{self.camera_id}_capture_intrinsic"
            self.pipeline = Gst.Pipeline.new(pipeline_name)

            src          = Gst.ElementFactory.make('nvarguscamerasrc', f'source_{self.camera_id}')
            queue1       = Gst.ElementFactory.make('queue', f'queue1_{self.camera_id}')
            caps_filter  = Gst.ElementFactory.make('capsfilter', f'caps_filter_{self.camera_id}')
            queue2       = Gst.ElementFactory.make('queue', f'queue2_{self.camera_id}')
            # 여기서 검사한 플러그인 이름을 사용
            video_convert = Gst.ElementFactory.make(convert_plugin, f'video_convert_{self.camera_id}')
            queue3       = Gst.ElementFactory.make('queue', f'queue3_{self.camera_id}')
            caps_filter2 = Gst.ElementFactory.make('capsfilter', f'caps_filter2_{self.camera_id}')
            queue4       = Gst.ElementFactory.make('queue', f'queue4_{self.camera_id}')
            self.appsink = Gst.ElementFactory.make('appsink', f'appsink_{self.camera_id}')


            # src 설정
            Gst.util_set_object_arg(src, "sensor-id", f"{self.camera_id}")
            Gst.util_set_object_arg(src, "bufapi-version", "true")
            Gst.util_set_object_arg(src, "sensor-mode", f"{self.cam_mode}")

            # 첫번째 capsfilter: NVMM 메모리, 해상도, FPS 설정
            caps_str = f"video/x-raw(memory:NVMM), width=(int){self.width}, height=(int){self.height}, framerate=(fraction){self.fps}/1"
            Gst.util_set_object_arg(caps_filter, "caps", caps_str)

            # nvvideoconvert: flip 옵션 설정
            if self.hflip and self.vflip:
                Gst.util_set_object_arg(video_convert, "flip-method", "2")  # 180도 회전
            elif self.hflip:
                Gst.util_set_object_arg(video_convert, "flip-method", "4")  # 수평 반전
            elif self.vflip:
                Gst.util_set_object_arg(video_convert, "flip-method", "6")  # 수직 반전
            else:
                Gst.util_set_object_arg(video_convert, "flip-method", "0")  # no flip

            # 두번째 capsfilter: OpenCV와 호환되는 포맷 (BGRx)
            caps_str2 = "video/x-raw, format=(string)BGRx"
            Gst.util_set_object_arg(caps_filter2, "caps", caps_str2)

            # appsink 설정: 항상 최신 프레임만 보관 (max-buffers=1, drop=True)
            self.appsink.set_property("emit-signals", False)
            self.appsink.set_property("max-buffers", 1)
            self.appsink.set_property("drop", True)
            # appsink의 caps를 설정하여 원하는 최종 포맷을 보장 (필수)
            appsink_caps = Gst.Caps.from_string(f"video/x-raw, format=(string)BGRx, width=(int){self.width}, height=(int){self.height}")
            self.appsink.set_property("caps", appsink_caps)


            # 요소들을 파이프라인에 추가
            self.pipeline.add(src)
            self.pipeline.add(queue1)
            self.pipeline.add(caps_filter)
            self.pipeline.add(queue2)
            self.pipeline.add(video_convert)
            self.pipeline.add(queue3)
            self.pipeline.add(caps_filter2)
            self.pipeline.add(queue4)
            self.pipeline.add(self.appsink)


            # 요소들을 순차적으로 연결
            if not self._link_elements(src, queue1, caps_filter, queue2, video_convert, queue3, caps_filter2, queue4, self.appsink):
                print(f"GStreamer 요소들을 연결하지 못했습니다 (카메라 ID: {self.camera_id}).")
                self.pipeline.set_state(Gst.State.NULL)
                self.pipeline = None
                self.appsink = None
                return False # 파이프라인 빌드 실패

            print(f"GStreamer pipeline built for camera {self.camera_id}.")
            return True # 파이프라인 빌드 성공
        except Exception as e:
            print(f"파이프라인 생성 중 오류 발생 (카메라 ID: {self.camera_id}): {e}")
            self.pipeline = None
            self.appsink = None
            return False # 파이프라인 빌드 실패


    def start_pipeline(self):
        """GStreamer 파이프라인을 PLAYING 상태로 시작합니다."""
        if self.pipeline is None:
            print("Error: Pipeline not built. Cannot start.")
            return False
        print(f"Setting pipeline {self.pipeline.get_name()} to PLAYING state...")
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
             print(f"Error: Failed to set pipeline {self.pipeline.get_name()} to PLAYING state.")
             return False
        # PLAYING 상태로 전환될 때까지 약간 대기
        time.sleep(1.0)
        print(f"Pipeline {self.pipeline.get_name()} is PLAYING.")
        return True

    def stop_pipeline(self):
        """GStreamer 파이프라인을 NULL 상태로 중지하고 해제합니다."""
        if self.pipeline:
            print(f"Setting pipeline {self.pipeline.get_name()} to NULL state...")
            self.pipeline.set_state(Gst.State.NULL)
            self.pipeline = None
            self.appsink = None
            print(f"Pipeline set to NULL state.")
        else:
            print("Pipeline is already stopped or not built.")


    def run_capture_loop(self):
        """이미지 캡쳐 및 미리보기 루프를 실행합니다."""
        if self.appsink is None:
            print("Error: Appsink not available. Cannot run capture loop.")
            return 0

        window_name = "Intrinsic Capture Preview (Press 'c' to Save, 'q' or ESC to Quit)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.img_size[0], self.img_size[1]) # 단일 이미지 보여줌

        captured_count = 0
        instructions = "Press 'c' to capture, 'q' or ESC to quit."
        print("\n--- Starting Intrinsic Capture Loop ---")
        print(instructions)

        try:
            while True:
                sample = self.appsink.emit("pull-sample")
                if sample is None:
                    time.sleep(0.001)
                    continue

                buffer = sample.get_buffer()
                success, map_info = buffer.map(Gst.MapFlags.READ)
                if not success:
                    print("Error mapping buffer")
                    buffer.unmap(map_info)
                    continue

                try:
                    # BGRx 포맷에서 BGR로 변환 (원본 프레임)
                    frame_bgr_raw = np.frombuffer(map_info.data, dtype=np.uint8).reshape((self.img_size[1], self.img_size[0], 4))
                    frame_bgr_raw = cv2.cvtColor(frame_bgr_raw, cv2.COLOR_BGRA2BGR)
                except Exception as e:
                    print("Frame processing error:", e)
                    buffer.unmap(map_info)
                    continue
                buffer.unmap(map_info)

                # --- 미리보기 화면 구성 ---
                frame_display = frame_bgr_raw.copy()

                # 체커보드 검출 (미리보기 표시용 - 원본 이미지에서 빠르게 시도)
                gray_raw = cv2.cvtColor(frame_bgr_raw, cv2.COLOR_BGR2GRAY)
                flags_find_corners = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
                ret_preview, corners_preview = cv2.findChessboardCorners(gray_raw, self.pattern_size, flags_find_corners)

                if ret_preview:
                    cv2.drawChessboardCorners(frame_display, self.pattern_size, corners_preview, ret_preview)

                # 코너 찾기 상태 텍스트 오버레이 (미리보기용)
                corner_status_text = "Chessboard: FOUND" if ret_preview else "Chessboard: NOT FOUND"
                corner_status_color = (0, 255, 0) if ret_preview else (0, 0, 255)

                # 텍스트 오버레이
                cv2.putText(frame_display, instructions, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                cv2.putText(frame_display, f"Captured: {captured_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                cv2.putText(frame_display, corner_status_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, corner_status_color, 2)

                cv2.imshow(window_name, frame_display)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('c'):
                    print(f"Capturing frame {captured_count+1}...")
                    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                    filename = os.path.join(self.output_dir, f"frame_{timestamp_str}.png")

                    try:
                        cv2.imwrite(filename, frame_bgr_raw) # <-- 원본 이미지 저장
                        captured_count += 1
                        print(f"Saved frame to {filename}. Total captured: {captured_count}")
                    except Exception as e:
                        print(f"Error saving image: {e}")

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

        return captured_count

    def capture_and_calibrate(self, output_yaml_file: str):
        """
        이미지 캡쳐를 실행하고, 완료 후 자동으로 캘리브레이션을 수행합니다.
        """
        print(f"--- Intrinsic Capture and Calibration for Camera {self.camera_id} ---")
        
        # 1. 파이프라인 빌드 및 시작
        if not self.build_pipeline():
            print("Failed to build pipeline. Cannot proceed with capture or calibration.")
            return

        if not self.start_pipeline():
            print("Failed to start pipeline. Cannot proceed with capture or calibration.")
            self.stop_pipeline()
            return

        # 2. 캡쳐 루프 실행
        captured_count = self.run_capture_loop()

        # 3. 파이프라인 중지
        self.stop_pipeline()

        # 4. 자동 캘리브레이션 실행
        if captured_count > 0: # 캡쳐된 이미지가 하나라도 있을 경우에만 캘리브레이션 시도
            print(f"\n--- Capture finished. Automatically starting Intrinsic Calibration ---")
            if IntrinsicCalibrator is None:
                 print("Cannot perform automatic calibration: IntrinsicCalibrator class not found (check intrinsic_calibrator.py).")
                 return

            try:
                # IntrinsicCalibrator 객체 생성 및 실행
                # 캡쳐된 이미지가 있는 디렉터리, 체커보드 정보, 카메라 ID 사용
                calibrator = IntrinsicCalibrator(
                    image_dir=self.output_dir,
                    grid_x=self.pattern_size[0], # 캘리브레이터는 grid_x/y를 사용
                    grid_y=self.pattern_size[1],
                    grid_size=self.grid_size,
                    camera_id=self.camera_id
                )
                calibrator.calibrate() # 캘리브레이션 수행

                # 캘리브레이션 결과 저장
                if calibrator.camera_matrix is not None: # 캘리브레이션 성공 시
                     calibrator.save_to_yaml(output_yaml_file)
                else:
                     print("Automatic intrinsic calibration calculation failed.")

            except Exception as e:
                print(f"An unexpected error occurred during automatic intrinsic calibration: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("\nNo images captured. Skipping automatic intrinsic calibration.")


def main():
    parser = argparse.ArgumentParser(description="단일 카메라 Intrinsic Calibration 이미지 캡쳐 및 캘리브레이션")

    Gst.init(sys.argv)

    parser.add_argument('--camera_id', type=int, default=1, help="사용할 카메라 번호 (sensor-id)")
    parser.add_argument('--mode', type=int, default=2, help="카메라 센서 모드 (예: 2)")
    parser.add_argument('--hflip', action='store_true', help="수평 반전 활성화")
    parser.add_argument('--vflip', action='store_true', help="수직 반전 활성화")
    parser.add_argument('--width', type=int, default=320, help="이미지/파이프라인 해상도 너비")
    parser.add_argument('--height', type=int, default=256, help="이미지/파이프라인 해상도 높이")
    parser.add_argument('--fps', type=int, default=15, help="파이프라인 프레임 레이트")

    parser.add_argument('--output_dir', type=str, default="intrinsic_calib_images", help="캡쳐 이미지 저장 디렉터리 (모든 카메라 공통)")

    parser.add_argument('--output_yaml_base', type=str, default="intrinsic_param", help="Intrinsic 캘리브레이션 결과 YAML 파일 기본 이름")

    # 체커보드 패턴 인자
    parser.add_argument('--pattern_width', type=int, default=10, help="체커보드 내부 코너 수 (가로)")
    parser.add_argument('--pattern_height', type=int, default=7, help="체커보드 내부 코너 수 (세로)")
    parser.add_argument('--grid_size', type=float, default=0.025, help="체커보드 한 칸의 실제 크기 (미터 단위 - 캘리브레이션 전달용)")

    args = parser.parse_args()

    final_output_dir = args.output_dir
    final_output_dir = Path(final_output_dir) / f'cam{args.camera_id}'

    # 카메라별 캘리브레이션 결과가 저장될 최종 YAML 파일 이름
    final_output_yaml = f"params/{args.output_yaml_base}_cam{args.camera_id}.yaml"

    # 출력 디렉터리 생성 (모든 카메라가 이 디렉토리에 이미지 저장)
    os.makedirs(final_output_dir, exist_ok=True)
    print(f"Capture images will be saved to: {final_output_dir}")
    print(f"Calibration result for Camera {args.camera_id} will be saved to: {final_output_yaml}")


    # IntrinsicCapturer 객체 생성 시 파싱된 args 값을 사용
    capturer = IntrinsicCapturer(
        camera_id=args.camera_id,
        cam_mode=args.mode,
        hflip=args.hflip,
        vflip=args.vflip,
        width=args.width,
        height=args.height,
        fps=args.fps,
        output_dir=final_output_dir, # 이미지는 공통 디렉토리에 저장
        pattern_width=args.pattern_width,
        pattern_height=args.pattern_height,
        grid_size=args.grid_size
    )

    # 캡쳐 및 자동 캘리브레이션 실행, 결과는 카메라별 고유 YAML 파일에 저장
    capturer.capture_and_calibrate(final_output_yaml)

    print(f"\n✅ Intrinsic Capture and Calibration process finished for Camera {args.camera_id}.")


if __name__ == "__main__":
    main()

