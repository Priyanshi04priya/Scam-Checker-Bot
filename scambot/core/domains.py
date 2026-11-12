"""Host and registrable-domain helpers backed by the bundled public suffix list."""

import ipaddress
from typing import Final
from urllib.parse import urlsplit

import tldextract

# The bundled suffix snapshot only: never fetch the list over the network, never write a cache.
_EXTRACT: Final = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)


def absolute_url(url: str) -> str:
    """``url`` with ``https://`` added when it has no scheme."""
    return url if "://" in url else f"https://{url}"


def url_host(url: str) -> str | None:
    """Lower-case host of an http(s) URL (``https://`` is assumed when there is no scheme)."""
    candidate = absolute_url(url)
    try:
        parts = urlsplit(candidate)
        host = parts.hostname
    except ValueError:
        return None
    if parts.scheme.lower() not in {"http", "https"} or not host:
        return None
    return host.lower().rstrip(".")


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        return False
    return True


def registrable_domain(host: str) -> str:
    """The domain directly under the public suffix (``a.b.sbi.co.in`` → ``sbi.co.in``)."""
    if _is_ip(host):
        return ""
    return _EXTRACT(host).top_domain_under_public_suffix


def domain_label(host: str) -> str:
    """The label directly under the public suffix (``a.sbi.co.in`` → ``sbi``)."""
    if _is_ip(host):
        return ""
    return _EXTRACT(host).domain


def has_public_suffix(host: str) -> bool:
    """True when ``host`` ends in a known public suffix and has a label under it."""
    if _is_ip(host):
        return False
    parts = _EXTRACT(host)
    return bool(parts.suffix and parts.domain)
