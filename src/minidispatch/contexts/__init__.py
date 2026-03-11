from minidispatch.contexts.local import LocalContext

__all__ = ["LocalContext"]

try:
    from minidispatch.contexts.ssh import SSHContext  # noqa: F401

    __all__.append("SSHContext")
except ImportError:
    pass
