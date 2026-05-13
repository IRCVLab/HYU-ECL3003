"""
듀얼 CSI 카메라 진단 스크립트
-----------------------------
cam0, cam1 을 한 대씩 따로 열어서 read() 가 성공하는지 확인합니다.
어느 쪽 카메라가 문제인지 빠르게 가려낼 때 사용하세요.

실행:
  $ python3 camera_check.py
"""

import os
import traceback
from jetcam.csi_camera import CSICamera


def check(device_id):
    print(f"\n=== camera {device_id} 테스트 ===")

    dev_path = f"/dev/video{device_id}"
    if not os.path.exists(dev_path):
        print(f"  [FAIL] {dev_path} 가 존재하지 않습니다.")
        print(f"         → 카메라 {device_id} 가 H/W 레벨에서 인식되지 않은 상태입니다.")
        print(f"         → 케이블 방향, 결합 상태, 부팅 시점 연결 여부 확인 필요.")
        return False

    print(f"  {dev_path} 존재 확인 OK")

    cam = None
    try:
        cam = CSICamera(
            capture_device=device_id,
            capture_width=1280,
            capture_height=720,
            downsample=2,
            capture_fps=30,
        )
        print(f"  CSICamera 객체 생성 OK")

        frame = cam.read()
        if frame is None:
            print(f"  [FAIL] read() 가 None 을 반환.")
            return False

        print(f"  read() OK, shape = {frame.shape}")
        return True

    except Exception as e:
        print(f"  [FAIL] 예외 발생: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

    finally:
        # 다음 카메라 테스트 전에 반드시 해제
        if cam is not None:
            try:
                cam.cap.release()
            except Exception:
                pass


def main():
    ok0 = check(0)
    ok1 = check(1)

    print("\n=== 결과 ===")
    print(f"  cam0: {'OK' if ok0 else 'FAIL'}")
    print(f"  cam1: {'OK' if ok1 else 'FAIL'}")

    if ok0 and ok1:
        print("두 카메라 모두 정상 — camera_live_dual.py 를 실행해도 됩니다.")
    elif ok0 and not ok1:
        print("cam1 만 문제. 케이블/슬롯 위치를 바꿔서 다시 테스트해보세요.")
        print("(같은 케이블/카메라를 cam0 슬롯에 꽂아 OK 가 나오면 슬롯 문제,")
        print(" cam0 슬롯에서도 FAIL 이면 카메라/케이블 자체 문제)")
    elif not ok0 and ok1:
        print("cam0 만 문제. 위와 같은 방법으로 슬롯/카메라를 분리해서 진단하세요.")
    else:
        print("둘 다 실패. 먼저 `sudo systemctl restart nvargus-daemon` 후 재시도.")


if __name__ == "__main__":
    main()
