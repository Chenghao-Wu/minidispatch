from __future__ import annotations

from abc import ABC, abstractmethod


class BaseContext(ABC):
    """Abstract base context for file transfer and command execution."""

    _registry: dict[str, type[BaseContext]] = {}

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        name = cls.__name__.lower()
        BaseContext._registry[name] = cls
        # Also register without "context" suffix
        short = name.replace("context", "")
        if short:
            BaseContext._registry[short] = cls

    @classmethod
    def create(
        cls,
        context_type: str,
        *,
        local_root: str,
        remote_root: str,
        remote_profile: dict | None = None,
    ) -> BaseContext:
        key = context_type.lower()
        if key not in cls._registry:
            raise ValueError(
                f"Unknown context_type {context_type!r}. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[key](
            local_root=local_root,
            remote_root=remote_root,
            remote_profile=remote_profile or {},
        )

    @abstractmethod
    def bind_submission(self, work_base: str, submission_hash: str) -> None: ...

    @abstractmethod
    def upload(self, tasks: list, forward_common_files: list[str]) -> None: ...

    @abstractmethod
    def download(self, tasks: list) -> None: ...

    @abstractmethod
    def write_file(self, fname: str, content: str) -> None: ...

    @abstractmethod
    def read_file(self, fname: str) -> str: ...

    @abstractmethod
    def check_file_exists(self, fname: str) -> bool: ...

    @abstractmethod
    def block_call(self, cmd: str) -> tuple[int, str, str]: ...

    @abstractmethod
    def clean(self) -> None: ...

    def rename_file(self, src: str, dst: str) -> None:
        """Rename a file in the remote root. Default: write+delete via read."""
        content = self.read_file(src)
        self.write_file(dst, content)

    def serialize(self) -> dict:
        return {
            "context_type": self.__class__.__name__,
            "local_root": str(self.init_local_root),  # type: ignore[attr-defined]
            "remote_root": str(self.init_remote_root),  # type: ignore[attr-defined]
            "remote_profile": getattr(self, "remote_profile", {}),
        }
