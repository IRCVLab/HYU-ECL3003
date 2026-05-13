"""
듀얼 CSI 카메라 실시간 화면 출력
--------------------------------
Jetson 의 CSI 포트 0, 1 에 연결된 카메라 두 대를 동시에 열고
한 창에 좌(cam0) / 우(cam1) 로 붙여서 보여줍니다.

조작:
  q       : 종료
  s       : 두 프레임을 각각 snapshot_camN_*.jpg 로 저장

실행:
  $ python3 camera_live_dual.py

주의 사항:
  - 두 카메라는 같은 capture_width / capture_height / fps 로 여는 게 가장 무난합니다.
    해상도가 다르면 hstack 전에 같은 크기로 resize 해야 합니다.
  - 두 대를 동시에 돌리면 대역폭/CPU 부담이 커지므로 1280x720 정도가 안정적입니다.
  - 카메라 인식 확인:
      $ ls /dev/video*
      -> /dev/video0, /dev/video1 두 개가 보여야 정상.
  - "Resource busy" 에러가 나면 이전 프로세스가 카메라를 잡고 있는 것입니다:
      $ sudo systemctl restart nvargus-daemon
"""

import time
import cv2
import numpy as np
from jetcam.csi_camera import CSICamera


def open_camera(device_id):
    """device_id 번 CSI 포트의 카메라를 동일한 설정으로 연다."""
    return CSICamera(
        capture_device=device_id,    # 0 또는 1
        capture_width=1280,
        capture_height=720,
        downsample=2,                # 출력 640 x 360
        capture_fps=30,              # 두 대 동시 캡처라 30fps 로 약간 낮춤
    )


def to_bgr(frame_rgb):
    """jetcam 은 RGB 로 주므로 imshow 용 BGR 로 변환."""
    return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)


def label(frame, text):
    """프레임 좌상단에 카메라 이름 표시."""
    cv2.putText(
        frame, text, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA,
    )
    return frame


def main():
    print("Opening camera 0 ...")
    cam0 = open_camera(0)
    print("Opening camera 1 ...")
    cam1 = open_camera(1)

    window_name = "Dual CSI Camera (q: quit, s: save)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    prev_t = time.time()
    fps = 0.0
    snapshot_idx = 0

    try:
        while True:
            f0 = cam0.read()
            f1 = cam1.read()
            if f0 is None or f1 is None:
                print("프레임을 읽지 못했습니다. 종료합니다.")
                break

            f0 = to_bgr(f0)
            f1 = to_bgr(f1)

            # 혹시 두 카메라의 출력 크기가 다르면 cam0 기준으로 cam1 을 맞춤
            if f1.shape[:2] != f0.shape[:2]:
                f1 = cv2.resize(f1, (f0.shape[1], f0.shape[0]))

            # FPS 측정
            now = time.time()
            inst_fps = 1.0 / max(now - prev_t, 1e-6)
            fps = inst_fps if fps == 0.0 else 0.9 * fps + 0.1 * inst_fps
            prev_t = now

            label(f0, f"cam0  {fps:5.1f} FPS")
            label(f1, "cam1")

            # 좌우로 이어붙이기
            combined = np.hstack([f0, f1])
            cv2.imshow(window_name, combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                n0 = f"snapshot_cam0_{snapshot_idx:03d}.jpg"
                n1 = f"snapshot_cam1_{snapshot_idx:03d}.jpg"
                cv2.imwrite(n0, f0)
                cv2.imwrite(n1, f1)
                print(f"저장: {n0}, {n1}")
                snapshot_idx += 1
    finally:
        cam0.cap.release()
        cam1.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
