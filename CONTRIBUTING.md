# Contributing

How we work on this project. Keep it simple — the goal is that four people can
work in parallel without stepping on each other.

## Workflow

`main` is protected: it always works, and changes reach it through a pull request.

```bash
git switch main && git pull          # start from latest
git switch -c feat/siamese-training  # branch per piece of work
# ...work, commit...
git push -u origin feat/siamese-training
gh pr create                         # or open the PR on GitHub
```

Approvals are not required, so you can merge your own PR once it's ready. Still
open one — it gives everyone a place to see and comment on the change. Branches
delete themselves after merge.

## Branch names

`<type>/<short-description>`, for example:

- `feat/data-augmentation`
- `fix/bbox-coordinate-order`
- `docs/report-methodology`

## Commit messages

[Conventional Commits](https://www.conventionalcommits.org/):

```
<type>: <short summary in the imperative>
```

Types we use: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`.

```
feat: add Siamese network training loop
fix: correct bb_width parsing in MOT16 ground truth
docs: add methodology section to report
```

## What not to commit

Model weights, datasets, and generated video are gitignored deliberately — they
are too large for the repo and for Canvas. Share them over Drive instead, and
link from the report.

Clear notebook outputs before committing (`Edit > Clear all outputs` in Colab):
they make diffs unreadable and bloat the repo.

## Splitting up work

Use [Issues](../../issues) to claim a piece of work so two people don't build the
same thing. Assign yourself, and reference the issue from your PR (`Closes #12`).
