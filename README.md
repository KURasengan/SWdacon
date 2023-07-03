# SWdacon

SW중심대학 공동 AI 경진대회 2023
2023.07.03 ~ 2023.07.28 09:59

## 주의사항

- 전체 데이터와 같이 용량이 큰 파일은 올리지 않는다. (.gitignore에 png 확장자 추가)
- Commit convention을 지켜 어떤 것이 업데이트 되었는지 모두가 볼 수 있게 한다.
- Branch 전략을 지켜 협업한다.

## Commit convention

- 파일 추가
  [ADD] Commit 내용
- 파일 삭제
  [DEL] Commit 내용
- 파일 수정
  [FIX] Commit 내용

## Branch 전략

- main
  메인 Branch로 항상 오류가 없도록 유지한다.
- 각자의 branch
  모든 개발은 각자의 branch에서 한다. 기능 개발이 끝나면 main branch로 merge하여 main branch를 업데이트한다.
  - main branch에 merge하기 위해서는 반드시 pull request 기능을 사용하여 오류가 없음을 교차검증한다.
