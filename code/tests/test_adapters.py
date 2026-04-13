from peft import PeftModel
from transformers import LlamaConfig, LlamaForCausalLM

from dora_repro.adapters import attach_adapter, build_lora_config
from dora_repro.config import AdapterPreset


def test_build_lora_config_enables_dora() -> None:
    config = build_lora_config(AdapterPreset(method="dora", scope="full"))
    target_modules = config.target_modules
    assert config.use_dora is True
    assert target_modules is not None
    assert "q_proj" in set(target_modules)


def test_attach_adapter_returns_peft_model() -> None:
    config = LlamaConfig()
    config.vocab_size = 64
    config.hidden_size = 32
    config.intermediate_size = 64
    config.num_hidden_layers = 1
    config.num_attention_heads = 4
    config.num_key_value_heads = 4

    model = LlamaForCausalLM(config)
    adapted = attach_adapter(model, AdapterPreset(method="lora", scope="attention_only"))
    assert isinstance(adapted, PeftModel)
    assert any(parameter.requires_grad for parameter in adapted.parameters())
