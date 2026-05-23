# model.py
# SKG-KT model: loading, LoRA config, loss computation

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel, get_peft_model

from utils import get_checkpoint_path


# -----------------------------
# Tokenizer Helpers
# -----------------------------
def get_true_false_tokens(tokenizer):
    true = tokenizer("True").input_ids[-1]
    false = tokenizer("False").input_ids[-1]
    return true, false


# -----------------------------
# Model Loading
# -----------------------------
def get_base_model(base_model_name: str, tokenizer, local_rank: int, quantize: bool = False):
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        pad_token_id=tokenizer.pad_token_id,
        torch_dtype=torch.bfloat16 if not quantize else torch.float32,
        device_map={"": local_rank},
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    return model


def get_skgkt_model(
    base_model_name: str,
    test: bool,
    local_rank: int,
    model_name: str = None,
    pt_model_name: str = None,
    r: int = None,
    lora_alpha: int = None,
    quantize: bool = False,
):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, padding_side="right")
    tokenizer.pad_token = tokenizer.bos_token

    model = get_base_model(base_model_name, tokenizer, local_rank, quantize=quantize)

    if test and model_name:
        model = PeftModel.from_pretrained(model, get_checkpoint_path(model_name))
        model.eval()
    elif not test:
        if pt_model_name:
            model = PeftModel.from_pretrained(
                model, get_checkpoint_path(pt_model_name), is_trainable=True, adapter_name="default"
            )
        else:
            peft_config = LoraConfig(
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=0.05,
                task_type="CAUSAL_LM",
                inference_mode=False,
            )
            model = get_peft_model(model, peft_config)

    return model, tokenizer


# -----------------------------
# Loss Computation
# -----------------------------
def _unwrap_causal_lm(m):
    """Unwrap DDP / PEFT wrappers to get the underlying CausalLM."""
    if hasattr(m, "module"):
        m = m.module
    if hasattr(m, "base_model") and hasattr(m.base_model, "model"):
        m = m.base_model.model
    return m


def get_skgkt_loss_packed(model, batch, true_token, false_token, args):
    device = model.device

    input_ids = batch["input_ids"].to(device)
    position_ids = batch["position_ids"].to(device)
    last_idxs = batch["last_idxs"].to(device)
    num_kcs = batch["num_kcs"].to(device).clamp(min=1)
    labels = batch["labels"].to(device).to(torch.float32)

    # Additive attention mask
    attention_mask = batch["attention_mask"].to(device).clone()
    min_dtype = torch.finfo(model.dtype).min
    attention_mask[attention_mask == 0] = min_dtype
    attention_mask[attention_mask == 1] = 0
    attention_mask = attention_mask.to(model.dtype)

    # Avoid full vocab logits: use backbone + selective lm_head projection
    hf_lm = _unwrap_causal_lm(model)
    backbone = hf_lm.model
    lm_head = hf_lm.lm_head

    out = backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
        return_dict=True,
    )
    hidden = out.last_hidden_state  # [B, L, H]

    B = hidden.size(0)
    K = last_idxs.size(1)

    # Gather hidden states at KC positions -> [B, K, H]
    hidden_sel = hidden[
        torch.arange(B, device=device).unsqueeze(1),
        last_idxs.clamp(min=0)
    ]

    # Project only True/False tokens: [B, K, 2]
    w = lm_head.weight.index_select(0, torch.tensor([true_token, false_token], device=device))
    logits_2 = torch.matmul(hidden_sel, w.t())

    # KC logits = logit(True) - logit(False): [B, K]
    kc_logits = logits_2[..., 0] - logits_2[..., 1]

    pad_mask = (last_idxs == 0)
    kc_logits = kc_logits.masked_fill(pad_mask, 0.0)

    # Aggregation
    if args.agg == "prod":
        turn_logits = kc_logits.sum(dim=1)
    else:
        turn_logits = kc_logits.sum(dim=1) / num_kcs

    loss = torch.nn.BCEWithLogitsLoss()(turn_logits, labels)
    corr_probs = torch.sigmoid(turn_logits)

    kc_probs = torch.sigmoid(kc_logits)
    kc_probs_grouped = [
        probs[:n].detach().cpu().tolist()
        for probs, n in zip(kc_probs, num_kcs)
    ]

    return loss, kc_probs_grouped, corr_probs
