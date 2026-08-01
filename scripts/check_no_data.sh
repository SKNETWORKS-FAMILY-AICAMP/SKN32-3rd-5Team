#!/usr/bin/env bash
# data/ 아래에 **올려도 되는 것 외의** 파일이 스테이징/커밋되었는지 확인한다 (D-29).
#
# 공개 저장소이고 코퍼스에는 이용약관 제약이 있는 자료가 섞여 있다.
# 한 번 푸시되면 히스토리에서 지우기 어려우므로 커밋 전에 막는다.
#
# 올려도 되는 것 (.gitignore 의 예외와 **같은 집합**이어야 한다):
#   data/manifests/**   대장 4종
#   data/facts/*.csv    사실 표 — **우리 산출물이다** (D-37).
#                       원문이 아니라 사실이고, 스키마·분류·문장화는 우리가 만든 것이다.
#   data/facts/*.md     검수 목록·작성 지침
#   data/README.md · data/*/.gitkeep · data/*/README.md
#
# ⚠️ **.gitignore 와 여기가 어긋나면 안 된다.** 2026-08-01 에 어긋났다 —
#    .gitignore 에 data/facts/ 예외를 넣어 888행 사실 표를 커밋 가능하게 만들었는데
#    이 스크립트의 허용 목록에는 안 넣어서 `make verify` 와 CI 가 빨간불이 됐다.
#    한쪽을 고치면 **반드시 다른 쪽도** 본다 (D-22 단일 출처).
#
# ⚠️ git 저장소가 아니면 **통과가 아니라 실패**다.
#    검사를 하나도 못 한 것과 통과한 것은 다르다.
set -euo pipefail

if ! git rev-parse --git-dir >/dev/null 2>&1; then
  echo "✗ git 저장소가 아니다 — 자료 유출 검사를 수행할 수 없다."
  echo "  검사를 건너뛴 것을 '통과'로 기록하지 않는다."
  exit 2
fi

ALLOWED='^data/(manifests/|facts/[^/]+\.(csv|md)$|README\.md$|\.gitkeep$|[a-z]+/(\.gitkeep|README\.md)$)'

# `-c core.quotepath=false` 가 없으면 git 이 한글 경로를 따옴표로 감싸고 8진 이스케이프한다 —
#   "data/facts/\352\262\200\354\210\230..."
# 그러면 맨 앞 따옴표 때문에 `^data/` 앵커가 안 걸려 **허용 목록이 통째로 무시된다.**
# 이 저장소는 한글 파일명이 많아 반드시 꺼야 한다 (2026-08-01 발견).
GIT='git -c core.quotepath=false'

staged=$($GIT diff --cached --name-only --diff-filter=A | grep '^data/' | grep -vE "$ALLOWED" || true)
tracked=$($GIT ls-files 'data/**' | grep -vE "$ALLOWED" || true)

bad=$(printf '%s\n%s\n' "$staged" "$tracked" | sed '/^$/d' | sort -u)

if [ -n "$bad" ]; then
  echo "✗ 자료 파일이 커밋 대상에 있다 — D-29 위반:"
  echo "$bad" | sed 's/^/    /'
  echo
  echo "  data/ 에는 대장(manifests/)과 사실 표(facts/)만 올린다. 되돌리려면:"
  echo "    git rm --cached <파일>"
  echo
  echo "  자료가 아니라 산출물인데 걸렸다면 **.gitignore 와 이 스크립트를 함께** 고친다."
  exit 1
fi

echo "✓ data/ 에는 대장과 사실 표만 있다"
