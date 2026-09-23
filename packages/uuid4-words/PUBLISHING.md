# Publishing `uuid4-words` to PyPI

This documents the concrete, one-time setup and the steady-state release
process. Nothing here has been run yet — it needs a human with a PyPI
account to do the account-side setup once, after which releases are just
"push a tag."

## One-time setup (do this once, as the repo/package owner)

1. **Create a PyPI account** at https://pypi.org/account/register/ if you
   don't have one, and turn on 2FA (PyPI requires it for anyone who
   publishes a project).

2. **(Recommended) Do a dry run on TestPyPI first.** Create a separate
   account at https://test.pypi.org/account/register/ (TestPyPI is a fully
   separate instance, not a mode of pypi.org) and repeat steps 3-4 below
   against it before touching the real index. This catches metadata/build
   problems (broken README rendering, missing files in the sdist, etc.)
   without burning a real release.

3. **Register a Trusted Publisher** — this replaces the old
   "generate an API token and paste it into a repo secret" flow. It lets
   GitHub Actions publish via short-lived OIDC tokens, no secret to manage
   or rotate:
   - On PyPI: **Your account → Publishing → Add a new pending publisher**
     (you can do this *before* the project exists — PyPI reserves the name
     and links it to this publisher the first time it successfully
     publishes).
   - Fill in:
     - PyPI project name: `uuid4-words`
     - Owner: `EdOlson-Morgan`
     - Repository name: `shared-embeddings-space-experiments`
     - Workflow filename: `publish-uuid4-words.yml`
     - Environment name: `pypi` (matches the workflow in this repo)
   - Repeat on TestPyPI with environment name `testpypi` if you did step 2.

4. **Confirm the package name is still free.** It was free as of this
   writing (`https://pypi.org/pypi/uuid4-words/json` → 404). PyPI names are
   first-come-first-served and can't be meaningfully reused once taken by
   someone else, so if it's since been claimed, the project name in
   `packages/uuid4-words/pyproject.toml` and the Trusted Publisher config
   above both need to change together.

## Steady-state release process

1. Bump `version` in `packages/uuid4-words/pyproject.toml` (and
   `uuid_words.__version__` in `src/uuid_words/__init__.py`) — follow
   [SemVer](https://semver.org/). Add an entry to `CHANGELOG.md`.
2. Commit that on `main` (normal PR review applies).
3. Tag the release commit and push the tag:
   ```bash
   git tag uuid4-words-v0.1.0
   git push origin uuid4-words-v0.1.0
   ```
   The tag prefix (`uuid4-words-v*`) is what the workflow in
   `.github/workflows/publish-uuid4-words.yml` matches on — this lets other
   packages in this monorepo (if any are added later) tag and release
   independently without colliding.
4. The `publish-uuid4-words` GitHub Actions workflow runs automatically on
   that tag push: builds the sdist + wheel with `uv build`, then publishes
   via `pypa/gh-action-pypi-publish` using the `pypi` environment's OIDC
   trust relationship set up above. No manual `twine upload` needed.
5. Watch the **Actions** tab for the run; then verify at
   `https://pypi.org/project/uuid4-words/`.

## Manual fallback (no CI)

If you ever need to publish by hand instead:

```bash
cd packages/uuid4-words
uv build
uv publish  # prompts for a PyPI API token if no trusted publisher applies
```
