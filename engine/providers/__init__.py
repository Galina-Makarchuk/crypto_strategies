"""Market-data providers: the registry, dispatch and per-provider validation.

A strategy/run picks a provider via DataSpec.provider (default "bybit").
load_data() dispatches through make_provider; validation is per-provider so each
source declares its own supported intervals and product categories. Adding a
provider = implement the DataProvider contract (see base.py) and register it here.
"""

from __future__ import annotations

from .base import CONTRACT_COLUMNS, DataProvider, finalize_ohlcv
from .bybit import BybitFetcher
from .yahoo import YahooProvider

# Registry: DataSpec.provider value -> provider class. The only place concrete
# provider classes are named.
PROVIDERS: dict[str, type] = {
    BybitFetcher.NAME: BybitFetcher,   # "bybit"
    YahooProvider.NAME: YahooProvider,  # "yahoo"
}

PROVIDER_NAMES = tuple(PROVIDERS)


def provider_class(name: str) -> type:
    """The provider class for a name, or raise with the known set."""
    try:
        return PROVIDERS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown provider {name!r}. Known: {sorted(PROVIDERS)}"
        ) from exc


def make_provider(name: str) -> DataProvider:
    """Instantiate the provider registered under ``name``."""
    return provider_class(name)()


def provider_intervals(name: str) -> frozenset:
    """Canonical interval codes a provider supports."""
    return provider_class(name).VALID_INTERVALS


def provider_categories(name: str):
    """Product categories a provider supports, or None if it has no taxonomy."""
    return provider_class(name).VALID_CATEGORIES


def default_category(name: str):
    """The provider's default category (None for providers without a taxonomy)."""
    return provider_class(name).DEFAULT_CATEGORY


def resolve_category(name: str, category):
    """The category actually used for fetch/cache/signature.

    None for providers with no product taxonomy (Yahoo) — the DataSpec.category
    value is simply ignored there. For providers that do have categories (Bybit),
    return the given category, or the provider default when None.
    """
    if provider_categories(name) is None:
        return None
    return category if category is not None else default_category(name)


def validate_spec(name: str, interval: str, category) -> None:
    """Provider-aware validation of a (provider, interval, category) triple.

    Raises ValueError naming the offending field — so a bad combo fails loudly at
    DataSpec construction / load_data, the same way the global validators did.
    """
    if name not in PROVIDERS:
        raise ValueError(
            f"provider {name!r} is invalid: must be one of {sorted(PROVIDERS)}"
        )
    intervals = provider_intervals(name)
    if interval not in intervals:
        raise ValueError(
            f"interval {interval!r} is invalid for provider {name!r}: "
            f"must be one of {sorted(intervals)}"
        )
    categories = provider_categories(name)
    if categories is not None:
        cat = category if category is not None else default_category(name)
        if cat not in categories:
            raise ValueError(
                f"category {cat!r} is invalid for provider {name!r}: "
                f"must be one of {sorted(categories)}"
            )
    # providers with no taxonomy (categories is None) ignore category entirely.


__all__ = [
    "CONTRACT_COLUMNS",
    "DataProvider",
    "finalize_ohlcv",
    "BybitFetcher",
    "YahooProvider",
    "PROVIDERS",
    "PROVIDER_NAMES",
    "provider_class",
    "make_provider",
    "provider_intervals",
    "provider_categories",
    "default_category",
    "resolve_category",
    "validate_spec",
]
