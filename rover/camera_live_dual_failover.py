"""
듀얼 CSI 카메라 실시간 출력 (failover)

좌(cam0) / 우(cam1) 로 두 대를 띄우되, 어느 한 대가 안 잡히거나 도중에
실패하면 그 자리만 "UNAVAILABLE" 로 채우고 나머지 카메라는 계속 동작합니다.

상태:
  - 두 대 OK   : 정상적으로 좌우 출력
  - 한 대 FAIL : 실패한 쪽만 빨간 placeholder, 다른 쪽은 정상
  - 두 대 FAIL : 메시지 출력 후 종료

조작:
  q : 종료
  s : 살아있는 카메라(들) 의 현재 프레임을 snapshot 으로 저장

실행:
  $ python3 camera_live_dual_failover.py
"""

import time
import cv2
import numpy as np
from jetcam.csi_camera import CSICamera


# ====== 캡처 설정 ======
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
DOWNSAMPLE = 2          # 출력은 640 x 360
CAPTURE_FPS = 30

# 표시용 한 프레임의 크기 (capture / downsample)
DISPLAY_W = CAPTURE_WIDTH // DOWNSAMPLE
DISPLAY_H = CAPTURE_HEIGHT // DOWNSAMPLE

# 연속 read 실패가 이 횟수에 도달하면 해당 카메라를 dead 처리하고 다시 시도하지 않음
MAX_CONSECUTIVE_FAILS = 5


def try_open(device_id):
    """
    카메라 객체 생성 + 첫 프레임 read 까지 성공해야 OK 로 간주.
    실패 시 None 을 돌려준다.
    (주의: jetcam.CSICamera 는 카메라가 실제로 데이터를 못 줘도
     생성자에서는 예외를 안 던지므로 read 까지 해봐야 한다.)
    """
    try:
        cam = CSICamera(
            capture_device=device_id,
            capture_width=CAPTURE_WIDTH,
            capture_height=CAPTURE_HEIGHT,
            downsample=DOWNSAMPLE,
            capture_fps=CAPTURE_FPS,
        )
    except Exception as e:
        print(f"[cam{device_id}] 객체 생성 실패: {e}")
        return None

    try:
        frame = cam.read()
        if frame is None:
            raise RuntimeError("첫 프레임이 None")
        print(f"[cam{device_id}] OK  shape={frame.shape}")
        return cam
    except Exception as e:
        print(f"[cam{device_id}] 첫 read 실패: {e}")
        try:
            cam.cap.release()
        except Exception:
            pass
        return None


def make_unavailable_placeholder(name):
    """카메라가 죽었을 때 좌우 레이아웃을 유지하기 위한 검은 패널."""
    img = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)
    cv2.putText(img, name, (20, DISPLAY_H // 2 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "UNAVAILABLE", (20, DISPLAY_H // 2 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
    return img


def safe_read(cam, device_id, fails):
    """
    한 프레임 읽기 시도.
      성공: (BGR 프레임, 0) 반환 (fails 카운터 리셋)
      실패: (None, fails+1) 반환
    """
    if cam is None:
        return None, fails
    try:
        frame_rgb = cam.read()
        if frame_rgb is None:
            return None, fails + 1
        return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), 0
    except Exception as e:
        new_fails = fails + 1
        # 로그가 너무 시끄럽지 않도록 처음과 30회 단위로만 출력
        if new_fails == 1 or new_fails % 30 == 0:
            print(f"[cam{device_id}] read 실패 #{new_fails}: {e}")
        return None, new_fails


def label_frame(frame, text):
    cv2.putText(frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)


def main():
    print("Opening cameras ...")
    cam0 = try_open(0)
    cam1 = try_open(1)

    if cam0 is None and cam1 is None:
        print("\n양쪽 카메라 모두 사용 불가. 종료합니다.")
        print("  - 케이블/슬롯 연결 점검")
        print("  - sudo systemctl restart nvargus-daemon")
        return

    print(f"\n초기 상태: cam0={'OK' if cam0 else 'DEAD'}, "
          f"cam1={'OK' if cam1 else 'DEAD'}")
    print("창에서  q: 종료  s: 스냅샷\n")

    # 처음부터 죽은 카메라는 곧장 dead 처리
    dead0 = cam0 is None
    dead1 = cam1 is None
    fails0 = 0
    fails1 = 0

    # 마지막으로 성공한 프레임 (간헐적 실패 동안 잠깐 보여주는 용도)
    last_good0 = None
    last_good1 = None

    window_name = "Dual CSI (q: quit, s: save) [failover]"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    prev_t = time.time()
    fps = 0.0
    snapshot_idx = 0

    try:
        while True:
            # ---- cam0 ----
            if not dead0:
                frame0, fails0 = safe_read(cam0, 0, fails0)
                if frame0 is not None:
                    last_good0 = frame0
                if fails0 >= MAX_CONSECUTIVE_FAILS:
                    print(f"[cam0] 연속 {MAX_CONSECUTIVE_FAILS}회 실패 -> dead 처리")
                    dead0 = True
                    try:
                        cam0.cap.release()
                    except Exception:
                        pass

            # ---- cam1 ----
            if not dead1:
                frame1, fails1 = safe_read(cam1, 1, fails1)
                if frame1 is not None:
                    last_good1 = frame1
                if fails1 >= MAX_CONSECUTIVE_FAILS:
                    print(f"[cam1] 연속 {MAX_CONSECUTIVE_FAILS}회 실패 -> dead 처리")
                    dead1 = True
                    try:
                        cam1.cap.release()
                    except Exception:
                        pass

            # 양쪽 다 죽었으면 종료
            if dead0 and dead1:
                print("두 카메라 모두 사용 불가 상태가 되어 종료합니다.")
                break

            # ---- 표시할 프레임 결정 ----
            if dead0:
                disp0 = make_unavailable_placeholder("cam0")
            elif last_good0 is not None:
                disp0 = last_good0.copy()
            else:
                disp0 = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)

            if dead1:
                disp1 = make_unavailable_placeholder("cam1")
            elif last_good1 is not None:
                disp1 = last_good1.copy()
            else:
                disp1 = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)

            # 두 프레임 크기가 어쩌다 다르면 cam0 기준으로 맞춤
            if disp1.shape[:2] != disp0.shape[:2]:
                disp1 = cv2.resize(disp1, (disp0.shape[1], disp0.shape[0]))

            # FPS (지수 이동 평균)
            now = time.time()
            inst_fps = 1.0 / max(now - prev_t, 1e-6)
            fps = inst_fps if fps == 0.0 else 0.9 * fps + 0.1 * inst_fps
            prev_t = now

            label_frame(disp0, f"cam0 {'DEAD' if dead0 else 'OK'}  {fps:5.1f} FPS")
            label_frame(disp1, f"cam1 {'DEAD' if dead1 else 'OK'}")

            combined = np.hstack([disp0, disp1])
            cv2.imshow(window_name, combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                saved_any = False
                if not dead0 and last_good0 is not None:
                    n0 = f"snapshot_cam0_{snapshot_idx:03d}.jpg"
                    cv2.imwrite(n0, last_good0)
                    print(f"저장: {n0}")
                    saved_any = True
                if not dead1 and last_good1 is not None:
                    n1 = f"snapshot_cam1_{snapshot_idx:03d}.jpg"
                    cv2.imwrite(n1, last_good1)
                    print(f"저장: {n1}")
                    saved_any = True
                if saved_any:
                    snapshot_idx += 1
                else:
                    print("저장할 살아있는 카메라가 없습니다.")
    finally:
        for cam in (cam0, cam1):
            if cam is not None:
                try:
                    cam.cap.release()
                except Exception:
                    pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
