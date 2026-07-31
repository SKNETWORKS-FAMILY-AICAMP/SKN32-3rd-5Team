#!/usr/bin/env bash
# data/ 아래에 매니페스트 외의 파일이 스테이징/커밋되었는지 확인한다 (D-29).
#
# 공개 저장소이고 코퍼스에는 이용약관 제약이 있는 자료가 섞여 있다.
# 한 번 푸시되면 히스토리에서 지우기 어려우므로 커밋 전에 막는다.
set -euo pipefail

ALLOWED='^data/(manifests/|README\.md$|\.gitkeep$)'

staged=$(git diff --cached --name-only --diff-filter=A 2>/dev/null | grep '^data/' | grep -vE "$ALLOWED" || true)
tracked=$(git ls-files 'data/**' | grep -vE "$ALLOWED" || true)

bad=$(printf '%s\n%s\n' "$staged" "$tracked" | sed '/^$/d' | sort -u)

if [ -n "$bad" ]; then
  echo "✗ 자료 파일이 커밋 대상에 있다 — D-29 위반:"
  echo "$bad" | sed 's/^/    /'
  echo
  echo "  data/ 에는 매니페스트만 올린다. 되돌리려면:"
  echo "    git rm --cached <파일>"
  exit 1
fi

echo "✓ data/ 에는 매니페스트만 있다"
