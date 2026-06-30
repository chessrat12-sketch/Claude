"""
예제 5: matplotlib 시각화 — 8종 그래프 생성

생성 파일 (plots/ 디렉터리):
  01_geodesics.png          — AdS₃ 측지선
  02_ee_comparison.png      — RT vs CFT 얽힘 엔트로피
  03_phase_transition.png   — 홀로그래픽 위상 전이
  04_page_curve.png         — Page 곡선 (Island 공식)
  05_mera_network.png       — MERA 텐서 네트워크
  06_rg_flow.png            — 홀로그래픽 RG 흐름
  07_bulk_profile.png       — 벌크 장 프로파일
  08_btz_thermodynamics.png — BTZ 블랙홀 열역학
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ads_cft_ml.visualization import (
    plot_geodesics,
    plot_ee_comparison,
    plot_phase_transition,
    plot_page_curve,
    plot_mera_network,
    plot_rg_flow,
    plot_bulk_profile,
    plot_btz_thermodynamics,
    OUTDIR,
    plot_all,
)


if __name__ == "__main__":
    print("=" * 60)
    print("AdS/CFT 홀로그래피 + ML — matplotlib 시각화")
    print("=" * 60)
    print(f"\n  저장 경로: {OUTDIR}/\n")

    paths = plot_all()

    print()
    print("=" * 60)
    print(f"완료. {len(paths)}개 그래프 생성됨.")
    print("=" * 60)
    for p in paths:
        size_kb = os.path.getsize(p) / 1024
        print(f"  {os.path.basename(p):45s}  {size_kb:6.1f} KB")
