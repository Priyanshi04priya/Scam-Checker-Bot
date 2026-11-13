"""Static checks of the image definition."""

import re
import tomllib

from scambot.core.settings import PROJECT_ROOT

DOCKERFILE = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")


def test_dockerfile_is_multi_stage_and_keeps_the_editable_layout() -> None:
    assert re.search(r"^FROM .* AS builder$", DOCKERFILE, re.M)
    assert re.search(r"^FROM .* AS runtime$", DOCKERFILE, re.M)
    assert "COPY --from=builder /opt/venv /opt/venv" in DOCKERFILE
    assert "COPY --from=builder /app /app" in DOCKERFILE
    assert "--no-editable" not in DOCKERFILE
    assert "uv pip install --no-deps -e ." in DOCKERFILE


def test_dockerfile_builds_the_project_as_a_non_root_user() -> None:
    assert "COPY config ./config" in DOCKERFILE
    runtime = DOCKERFILE.split(" AS runtime", 1)[1]
    assert "USER scambot" in runtime
    assert "USER root" not in DOCKERFILE
    assert "--uid 10001" in runtime
    assert runtime.rindex("RUN ") < runtime.index("USER scambot")
    assert "MEDIA_DIR=" in DOCKERFILE
    assert "OMP_THREAD_LIMIT=1" in DOCKERFILE
    assert "ENV HF_HOME=/models/huggingface" in DOCKERFILE


def test_dockerfile_installs_the_locked_dependencies_for_the_chosen_groups() -> None:
    assert "uv export --locked --no-dev" in DOCKERFILE
    assert DOCKERFILE.count("$UV_GROUPS") == 1
    assert "--group embeddings" in DOCKERFILE
    assert DOCKERFILE.count("--group media") == 1
    assert DOCKERFILE.count("--group c2pa") == 1


def test_dockerfile_uses_the_cpu_torch_index_without_cuda_packages() -> None:
    assert "TORCH_INDEX=https://download.pytorch.org/whl/cpu" in DOCKERFILE
    assert "nvidia-" in DOCKERFILE
    assert "--no-deps" in DOCKERFILE


def test_dockerfile_installs_system_packages_cleanly() -> None:
    apt_block = DOCKERFILE.split("apt-get update", 1)[1].split("\n\n", 1)[0]
    assert "--no-install-recommends" in apt_block
    for package in ("ffmpeg", "tesseract-ocr", "tesseract-ocr-hin"):
        assert package in apt_block
    assert "rm -rf /var/lib/apt/lists/*" in apt_block


def test_dockerignore_keeps_secrets_and_deployment_files_out_of_the_image() -> None:
    ignored = (PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8").split()
    assert {".env", ".venv", "tests", "eval", "deploy", "Caddyfile", "backups"} <= set(ignored)
    assert "!README.md" in ignored


def test_project_files_the_build_needs_exist() -> None:
    for name in ("pyproject.toml", "uv.lock", "README.md", "alembic.ini"):
        assert (PROJECT_ROOT / name).is_file()
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "embeddings" in pyproject["dependency-groups"]


def test_dockerignore_excludes_every_env_file() -> None:
    lines = (PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8").split()
    assert ".env" in lines
    assert ".env.*" in lines
