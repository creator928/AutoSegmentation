# -*- coding: utf-8 -*-
"""PyInstaller 훅 경고를 피하기 위한 로컬 qt_material 스텁 패키지입니다."""

# 이 프로젝트는 qt_material을 실제로 사용하지 않지만,
# 설치된 외부 패키지의 훅이 빌드 중 import qt_material을 수행하면서 경고를 출력합니다.
# 로컬 스텁 패키지를 우선 탐색시키면 불필요한 경고 없이 훅 분석만 통과시킬 수 있습니다.


def get_hook_dirs() -> list[str]:
    """PyInstaller 엔트리 포인트가 요구하는 훅 디렉터리 목록을 빈 배열로 반환합니다."""
    return []
