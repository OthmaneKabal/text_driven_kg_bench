"""Registry and factories for the graph encoders supported by TDG-Bench.

To add a model, implement a small builder with the common signature and
register it once with :func:`register_model`. Benchmark code then only needs
the registered name; it never imports model classes directly.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


EncoderBuilder = Callable[..., Any]


class ModelRegistry:
    """Name-to-builder registry for TDG-Bench encoders."""

    def __init__(self) -> None:
        self._builders: dict[str, EncoderBuilder] = {}

    def register(self, name: str, builder: EncoderBuilder) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("A model name must be a non-empty string")
        if name in self._builders:
            raise ValueError(f"A model is already registered as {name!r}")
        self._builders[name] = builder

    def build(
        self,
        name: str,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_relations: int,
        **kwargs: Any,
    ) -> Any:
        try:
            builder = self._builders[name]
        except KeyError as exc:
            raise ValueError(
                f"Unknown model_name: {name}. "
                f"Supported models: {', '.join(self.names())}"
            ) from exc
        return builder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_relations=num_relations,
            **kwargs,
        )

    def names(self) -> tuple[str, ...]:
        return tuple(self._builders)


MODEL_REGISTRY = ModelRegistry()


def register_model(name: str) -> Callable[[EncoderBuilder], EncoderBuilder]:
    """Register an encoder builder under a unique public model name.

    A builder receives ``in_channels``, ``hidden_channels``, ``out_channels``
    and ``num_relations`` plus model-specific keyword arguments.
    """

    def decorator(builder: EncoderBuilder) -> EncoderBuilder:
        MODEL_REGISTRY.register(name, builder)
        return builder

    return decorator


@register_model("GCN")
def _build_gcn(
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    num_relations: int,
    **kwargs: Any,
):
    from models.GCNEncoder import GCNEncoder

    return GCNEncoder(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        **kwargs,
    )


@register_model("RGCN")
def _build_rgcn(
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    num_relations: int,
    **kwargs: Any,
):
    from models.RGCNEncoder import RGCNEncoder

    return RGCNEncoder(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_relations=num_relations,
        **kwargs,
    )


@register_model("GAT")
def _build_gat(
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    num_relations: int,
    **kwargs: Any,
):
    from models.GATEncoder import GATEncoder

    return GATEncoder(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        **kwargs,
    )


def _build_transgcn(
    *,
    kg_score_fn: str,
    variant: str,
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    num_relations: int,
    **kwargs: Any,
):
    from models.TransGCNEncoder import TransGCNEncoder

    return TransGCNEncoder(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        kg_score_fn=kg_score_fn,
        variant=variant,
        **kwargs,
    )


@register_model("TransEGCN_conv")
def _build_transe_conv(
    in_channels: int, hidden_channels: int, out_channels: int, num_relations: int, **kwargs: Any
):
    return _build_transgcn(
        kg_score_fn="TransE", variant="conv", in_channels=in_channels,
        hidden_channels=hidden_channels, out_channels=out_channels, num_relations=num_relations, **kwargs,
    )


@register_model("RotatEGCN_conv")
def _build_rotate_conv(
    in_channels: int, hidden_channels: int, out_channels: int, num_relations: int, **kwargs: Any
):
    return _build_transgcn(
        kg_score_fn="RotatE", variant="conv", in_channels=in_channels,
        hidden_channels=hidden_channels, out_channels=out_channels, num_relations=num_relations, **kwargs,
    )


@register_model("TransEGCN_attn")
def _build_transe_attn(
    in_channels: int, hidden_channels: int, out_channels: int, num_relations: int, **kwargs: Any
):
    return _build_transgcn(
        kg_score_fn="TransE", variant="attn", in_channels=in_channels,
        hidden_channels=hidden_channels, out_channels=out_channels, num_relations=num_relations, **kwargs,
    )


@register_model("RotatEGCN_attn")
def _build_rotate_attn(
    in_channels: int, hidden_channels: int, out_channels: int, num_relations: int, **kwargs: Any
):
    return _build_transgcn(
        kg_score_fn="RotatE", variant="attn", in_channels=in_channels,
        hidden_channels=hidden_channels, out_channels=out_channels, num_relations=num_relations, **kwargs,
    )


def available_model_names() -> tuple[str, ...]:
    """Return all public names currently registered for benchmark use."""
    return MODEL_REGISTRY.names()


# Kept for code which imported the old constant.
SUPPORTED_MODEL_NAMES = available_model_names()


def build_encoder(
    model_name: str,
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    num_relations: int,
    **kwargs: Any,
):
    """Build a registered encoder from its public name."""
    return MODEL_REGISTRY.build(
        name=model_name,
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_relations=num_relations,
        **kwargs,
    )


def get_default_model_kwargs(model_name: str) -> dict[str, Any]:
    """Return the default hyperparameters for one registered model."""
    if model_name not in available_model_names():
        raise ValueError(
            f"Unknown model_name: {model_name}. "
            f"Supported models: {', '.join(available_model_names())}"
        )

    defaults = {
        "GCN": {"num_layers": 2, "dropout": 0.5, "batch_norm": True},
        "RGCN": {
            "num_layers": 2, "dropout": 0.5, "num_bases": 50, "batch_norm": True,
        },
        "GAT": {
            "num_layers": 2, "dropout": 0.5, "heads": 4, "batch_norm": True, "concat": False,
        },
        "TransEGCN_conv": {
            "num_layers": 2, "dropout": 0.5, "batch_norm": True,
            "use_edges_info": True, "activation": "relu", "bias": True,
        },
        "RotatEGCN_conv": {
            "num_layers": 2, "dropout": 0.5, "batch_norm": True,
            "use_edges_info": True, "activation": "relu", "bias": True,
        },
        "TransEGCN_attn": {
            "num_layers": 2, "dropout": 0.5, "batch_norm": True,
            "use_edges_info": True, "activation": "relu", "bias": True,
        },
        "RotatEGCN_attn": {
            "num_layers": 2, "dropout": 0.5, "batch_norm": True,
            "use_edges_info": True, "activation": "relu", "bias": True,
        },
    }
    return dict(defaults.get(model_name, {}))
