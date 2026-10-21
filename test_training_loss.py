"""Check that memory-saving supervision matches full causal LM loss and gradients."""
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM, LlamaConfig, LlamaForCausalLM
from peft import LoraConfig, TaskType, get_peft_model
from experiment import answer_only_loss

torch.set_num_threads(2)
for config_type, model_type in [(Qwen2Config, Qwen2ForCausalLM), (LlamaConfig, LlamaForCausalLM)]:
    torch.manual_seed(42)
    config = config_type(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64,
                         attention_dropout=0.0)
    model = get_peft_model(model_type(config), LoraConfig(task_type=TaskType.CAUSAL_LM,
                          target_modules=['q_proj', 'v_proj'], r=2, lora_alpha=4, lora_dropout=0))
    ids = torch.randint(0, 32, (2, 8))
    labels = ids.clone()
    labels[:, :5] = -100
    labels[1, 7] = -100
    inputs = dict(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels)
    full = model(**inputs).loss
    full.backward()
    gradients = {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}
    model.zero_grad()
    selected = answer_only_loss(model, inputs)
    selected.backward()
    torch.testing.assert_close(full, selected, atol=1e-6, rtol=1e-6)
    for n, p in model.named_parameters():
        if p.requires_grad:
            torch.testing.assert_close(gradients[n], p.grad, atol=1e-6, rtol=1e-5)
    print(model_type.__name__, 'loss and all LoRA gradients match')
