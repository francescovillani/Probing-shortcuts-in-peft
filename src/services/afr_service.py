"""
Service for Automatic Feature Reweighting (AFR).

This service implements the second stage of the AFR method, which involves
re-training the classifier head on a held-out portion of the training data
with a custom weighted loss.
"""

import hashlib
import re
from torch.utils.data import DataLoader, TensorDataset
import logging
from typing import Dict, Optional, Tuple, Union
from transformers import PreTrainedModel
from tqdm import tqdm
from contextlib import contextmanager
import torch
import torch.nn.functional as F
from config import TrainingConfig
from services.evaluation_service import EvaluationService
from pathlib import Path
import json
import io
import os
logger = logging.getLogger(__name__)

class AFRService:
    def __init__(self, config: TrainingConfig, device: torch.device):
        self.config = config
        self.afr_config = config.afr
        self.device = device
        self.evaluation_service = EvaluationService(device=device)
        # --- CACHE DIR
        self.cache_dir = Path(self.config.outputdir, "afr_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- Cache keys/paths ----------------
    def _cache_key(self, model_id: str, split_name: str = "drw", pooling: str = "cls", dtype: str = "fp16") -> str:
        v = "v1"
        # rimuovi caratteri pericolosi (slash, due punti, spazi…) e compatta
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(model_id)).strip("-_.")
        # opzionale: append un hash corto per unicità
        short = hashlib.sha1(str(model_id).encode("utf-8")).hexdigest()[:8]
        return f"afr_{v}_{safe_id}_{short}_{split_name}_{pooling}_{dtype}"
    
    def _cache_paths(self, key: str):
        base = self.cache_dir / key
        return {
            "emb":   base.with_suffix(".emb.pt"),
            "logits":base.with_suffix(".logits.pt"),
            "labels":base.with_suffix(".y.pt"),
            "index": base.with_suffix(".indices.pt"),
            "meta":  base.with_suffix(".meta.json"),
        }

    def _load_cached(self, key: str):
        paths = self._cache_paths(key)
        E = torch.load(paths["emb"], map_location="cpu")
        Z = torch.load(paths["logits"], map_location="cpu")
        Y = torch.load(paths["labels"], map_location="cpu")
        I = torch.load(paths["index"], map_location="cpu")
        with io.open(paths["meta"], "r", encoding="utf-8") as f:
            meta = json.load(f)
        return E, Z, Y, I, meta

    # ---------------- Individuare la head e congelare la base ----------------
    def _get_head_and_freeze_base(self, model: PreTrainedModel):
        """
        Restituisce (head_module, head_params) e congela TUTTO tranne la head.
        Riusiamo direttamente la RobertaClassificationHead già presente nel modello.
        """
        # Debug: Check model type and structure
        logger.debug(f"[AFR] Model type: {type(model)}")
        logger.debug(f"[AFR] Model config: {getattr(model, 'config', 'No config')}")
        if hasattr(model, 'config'):
            logger.debug(f"[AFR] Model hidden size: {getattr(model.config, 'hidden_size', 'Unknown')}")
            logger.debug(f"[AFR] Model num labels: {getattr(model.config, 'num_labels', 'Unknown')}")
        
        # Debug: Check all possible head names
        possible_head_names = ["classifier", "classification_head", "score", "lm_head", "fc", "cls"]
        logger.debug(f"[AFR] Checking for classification head in model...")
        for name in possible_head_names:
            if hasattr(model, name):
                head_module = getattr(model, name)
                logger.debug(f"[AFR] Found '{name}': {head_module} (type: {type(head_module)})")
                if hasattr(head_module, 'parameters'):
                    param_count = sum(p.numel() for p in head_module.parameters())
                    logger.debug(f"[AFR] '{name}' has {param_count:,} parameters")
                    
                    # Debug: Check individual parameters in the head
                    for i, param in enumerate(head_module.parameters()):
                        logger.debug(f"[AFR]   Param {i}: shape={param.shape}, numel={param.numel()}")
        
        # Debug: Check if this is a PEFT model
        if hasattr(model, 'peft_config'):
            logger.debug(f"[AFR] This is a PEFT model with config: {model.peft_config}")
        if hasattr(model, 'base_model'):
            logger.debug(f"[AFR] This model has a base_model: {type(model.base_model)}")
        
        # trova la head
        head = getattr(model, "classifier", None)
        if head is None:
            for name in ["classification_head", "score", "lm_head", "fc"]:
                if hasattr(model, name):
                    head = getattr(model, name)
                    logger.info(f"[AFR] Using head '{name}': {head}")
                    break
        if head is None:
            raise RuntimeError("AFR: classifier head non trovata (es. model.classifier).")

        # congela tutto
        for p in model.parameters():
            p.requires_grad = False
        # sblocca solo la head
        for p in head.parameters():
            p.requires_grad = True
        head_params = [p for p in head.parameters() if p.requires_grad]
        return head, head_params


    # ---------------- Estrazione features per la head ----------------
    @torch.no_grad()
    def _extract_features_for_head(self, model: PreTrainedModel, batch, pooling: str = "cls"):
        """
        SOLO TESTO (RoBERTa). Ritorna [B, D] per la head.
        - pooling="cls": hidden[:,0,:]
        - pooling="mean": media masked sui token validi
        """
        model.eval()
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # backbone roberta
        if hasattr(model, "roberta"):
            backbone = model.roberta
        elif hasattr(model, "base_model"):
            backbone = model.base_model
        else:
            backbone = model

        outputs = backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = outputs.hidden_states[-1]  # [B, T, D]

        if pooling == "cls":
            feats = last_hidden[:, 0, :]  # <s>/CLS
        elif pooling == "mean":
            if attention_mask is None:
                feats = last_hidden.mean(dim=1)
            else:
                mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)  # [B,T,1]
                feats = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        else:
            raise ValueError("pooling deve essere 'cls' o 'mean'")
        return feats  # [B, D]


    @torch.no_grad()
    def _build_or_load_cache_text(
        self,
        model: PreTrainedModel,
        drw_loader: DataLoader,     # deve essere shuffle=False + collate con "indices"
        model_id: str,
        pooling: str = "cls",
        dtype: str = "fp32",
        split_name: str = "drw",
    ):
        key = self._cache_key(model_id=model_id, split_name=split_name, pooling=pooling, dtype=dtype)
        paths = self._cache_paths(key)
        if all(Path(p).exists() for p in paths.values()):
            E = torch.load(paths["emb"],    map_location="cpu")
            Z = torch.load(paths["logits"], map_location="cpu")
            Y = torch.load(paths["labels"], map_location="cpu")
            I = torch.load(paths["index"],  map_location="cpu")
            return key, (E, Z, Y, I)

        model.eval()
        head = getattr(model, "classifier", None)
        if head is None:
            raise RuntimeError("AFR: classifier head non trovata.")

        embs, logits, labels = [], [], []
        idx_list = []  # <--- lista per gli indici (nome diverso)
        use_fp16 = (dtype == "fp16")

        for batch in tqdm(drw_loader, desc="AFR cache (RoBERTa: E,Z,y,idx)"):
            idx_batch = batch.get("indices", None)   # <--- tensore [B], opzionale
            y         = batch["labels"]

            if use_fp16:
                with torch.autocast(self.device.type, enabled=True, dtype=torch.float16):
                    feats = self._extract_features_for_head(model, batch, pooling=pooling)  # [B,D]
                    z     = head(feats.unsqueeze(1).to(self.device))                        # [B,1,D] -> [B,C]
            else:
                feats = self._extract_features_for_head(model, batch, pooling=pooling)
                z     = head(feats.unsqueeze(1).to(self.device))

            embs.append(feats.detach().cpu())
            logits.append(z.detach().cpu())
            labels.append(y.detach().cpu())
            if idx_batch is not None:
                idx_list.append(idx_batch.detach().cpu())

        E = torch.cat(embs,   dim=0)              # [N,D]
        Z = torch.cat(logits, dim=0)              # [N,C]
        Y = torch.cat(labels, dim=0)              # [N]
        I = torch.cat(idx_list, dim=0) if idx_list else torch.arange(E.size(0))  # [N]

        paths["emb"].parent.mkdir(parents=True, exist_ok=True)
        torch.save(E, paths["emb"])
        torch.save(Z, paths["logits"])
        torch.save(Y, paths["labels"])
        torch.save(I, paths["index"])
        meta = {"N": int(E.size(0)), "D": int(E.size(1)), "C": int(Z.size(1)),
                "pooling": pooling, "dtype": dtype, "model_id": model_id, "split": split_name}
        with io.open(paths["meta"], "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return key, (E, Z, Y, I)
    
    def _calculate_weights(
        self,
        erm_logits: torch.Tensor,
        class_label: torch.Tensor,
        sum_to: str = "N"
    ) -> torch.Tensor:
        """
        w_i ∝ β_{y_i} * exp(-γ * p_true_i), poi normalizzati.
        - erm_logits: [N, C]
        - class_label: [N] (long)
        - sum_to: "N" (coerente con CE mean) oppure "1"
        """
        device, dtype = erm_logits.device, erm_logits.dtype
        class_label = class_label.to(device=device, dtype=torch.long)
        N, C = erm_logits.shape

        with torch.no_grad():
            p = F.softmax(erm_logits, dim=-1)
            p_true = p.gather(1, class_label.view(-1, 1)).squeeze(1)   # [N]

            # exp(-gamma * p_true)
            raw = torch.exp(-float(self.afr_config.gamma) * p_true)                    # [N]

            # β_y = N / n_y (gestisce classi assenti)
            counts = torch.bincount(class_label, minlength=C).to(device=device, dtype=dtype)
            counts = torch.clamp(counts, min=1)
            beta = counts.sum() / counts                                # [C]

            w = raw * beta[class_label]                                 # [N]

            denom = w.sum().clamp_min(torch.finfo(dtype).tiny)
            if sum_to == "N":
                w = w * (N / denom)
            elif sum_to == "1":
                w = w / denom
            else:
                raise ValueError("sum_to must be 'N' or '1'")
        return w
    
    
    def group_logits_3to2(logits, ent_idx=0, nonent_idx=(1, 2)):
        """
        Converte logit 3-class (MNLI) in 2-class (HANS) raggruppando:
        - classe 0: entailment
        - classe 1: non-entailment = neutral ∪ contradiction
        Ritorna nuovi logit a 2 classi (in log-prob, ma vanno benissimo per argmax).
        """
        logp = F.log_softmax(logits, dim=-1)                 # [B,3] -> log-prob
        ent = logp[..., ent_idx]                             # [B]
        non = logp[..., list(nonent_idx)].logsumexp(dim=-1)  # [B]
        grouped_logp = torch.stack([ent, non], dim=-1)       # [B,2]
        return grouped_logp 

    @torch.no_grad()
    def _worst_class_accuracy_fast(
        self,
        model: PreTrainedModel,
        loader_or_dict,
        device: torch.device,
        num_labels: int | None = None,
        max_batches: int | None = None,  
        subsample: int | None = None,     
        hans_index: int | None = None,
    ):
        """
        Ritorna (WCA, MCA, acc_per_class) calcolati correttamente come:
        - WCA = min_c acc_c
        - MCA = mean_c acc_c
        Implementazione vettoriale con bincount. Impossibile ottenere WCA > MCA.
        """
        if hans_index is None:
            hans_index = -1 
        
        model.eval()
        if isinstance(loader_or_dict, dict):
            loaders = loader_or_dict.values()
        else:
            loaders = [loader_or_dict]

        # Se non passato, prova a inferirlo dal modello
        if num_labels is None:
            num_labels = int(getattr(getattr(model, "config", None), "num_labels", 0)) or 2

        correct_per_class = torch.zeros(num_labels, dtype=torch.long)
        total_per_class   = torch.zeros(num_labels, dtype=torch.long)

        seen = 0
        for l_idx, loader in enumerate(loaders):
            if l_idx == hans_index:
                logger.debug("[AFR] Applicazione conversione logit 3->2 per HANS nel calcolo WCA/MCA.")
            for b_idx, batch in enumerate(tqdm(loader,
                                        total=len(loader),
                                        desc=f"fast_wca_computation",
                                        dynamic_ncols=True,
                                        leave=False,
                                        mininterval=0.5,
                                        smoothing=0.1,
                                        position=0,
                                       )
                                    ):
                if max_batches is not None and b_idx >= max_batches:
                    break
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attn      = batch.get("attention_mask")
                if attn is not None:
                    attn = attn.to(device, non_blocking=True)
                labels    = batch["labels"].to(device, non_blocking=True)

                logits = model(input_ids=input_ids, attention_mask=attn).logits
                if l_idx == hans_index:
                    logits = self.group_logits_3to2(logits, ent_idx=0, nonent_idx=(1, 2))
                preds  = logits.argmax(dim=-1)

                # Conta totali per classe
                tot = torch.bincount(labels, minlength=num_labels if l_idx != hans_index else 2)
                total_per_class[:len(tot)] += tot.cpu()

                # Conta corretti per classe: filtra dove pred == label e binconta le label
                mask = (preds == labels)
                if mask.any():
                    cor = torch.bincount(labels[mask], minlength=num_labels)
                    correct_per_class[:len(cor)] += cor.cpu()

                seen += labels.numel()
                if subsample is not None and seen >= subsample:
                    break

        # Evita divisioni per zero: ignora classi mai viste
        valid = total_per_class > 0
        if not valid.any():
            return 0.0, 0.0, {}

        acc_per_class = torch.zeros_like(total_per_class, dtype=torch.float)
        acc_per_class[valid] = correct_per_class[valid].float() / total_per_class[valid].float()

        wca = float(acc_per_class[valid].min().item())
        mca = float(acc_per_class[valid].mean().item())

        # Guard-rail: WCA non può superare MCA (entro un eps numerico)
        assert wca <= mca + 1e-7, f"Incoerenza WCA({wca}) > MCA({mca})"

        # Converte in dict pulito solo per le classi viste
        acc_dict = {int(i): float(acc_per_class[i].item()) for i in torch.where(valid)[0].tolist()}
        return wca, mca, acc_dict


    def train_drw(
        self,
        model: PreTrainedModel,
        drw_loader: DataLoader,                      # per il cache: deve essere shuffle=False, include_idx=True
        drw_dataset_raw: torch.utils.data.Dataset,
        *,
        model_id: str | None = None,
        pooling: str = "cls",
        cache_dtype: str = "fp16",
        val_loaders: Optional[Union[DataLoader, Dict[str, DataLoader]]] = None,  # <-- facoltativo (per WCA)
        wca_tol: float = 0,                        # tolleranza per miglioramento WCA
        eval_every: int = 5,                 
        val_max_batches: int | None = None,  
        val_subsample: int | None = None,    
    ) -> PreTrainedModel:
        """
        AFR Stage 2 (testo): head-only training con pesi fissi su DRW, early stopping su Worst-Class Accuracy (WCA)
        calcolata su validation (se fornita). Se `val_loaders` è None, fa solo guardrail su DRW accuracy.
        """
        import torch
        import torch.nn.functional as F
        from copy import deepcopy

        logger.info("AFR Stage 2 (text): head-only training con cache e early stopping su Worst-Class Acc.")

        # ---------- Identità modello ----------
        if model_id is None:
            model_id = getattr(model.config, "_name_or_path", "roberta") + f"_{sum(p.numel() for p in model.parameters())}"

        # ---------- 1) Cache (E,Z,y,indices) ----------
        _, (E_cpu, Z_cpu, y_cpu, indices_cpu) = self._build_or_load_cache_text(
            model=model,
            drw_loader=drw_loader,
            model_id=model_id,
            pooling=pooling,
            dtype=cache_dtype,
            split_name="drw",
        )

        def _check_finite(name, t):
            ok = torch.isfinite(t).all()
            if not ok:
                n_nan = (~torch.isfinite(t)).sum().item()
                logger.error(f"[AFR] {name} contiene {n_nan} valori non finiti.")
            return bool(ok)

        if not (_check_finite("E", E_cpu) and _check_finite("Z", Z_cpu) and _check_finite("y", y_cpu)):
            logger.error("[AFR] Cache corrotto (NaN/Inf). Rigenera il cache in fp32.")
            raise RuntimeError("AFR: invalid cache tensors")

        device = self.device

        # ---------- 2) Pesi AFR dai logits ERM (fissi) ----------
        logits = Z_cpu.to(device, non_blocking=True)          # [N,C]
        labels = y_cpu.to(device, non_blocking=True).long()   # [N]
        w_all  = self._calculate_weights(logits, labels, sum_to="N")  # [N]

        # ---------- 3) Prepara head e L2 verso ϕ̂ ----------
        head, head_params = self._get_head_and_freeze_base(model)
        phi_init = [p.detach().clone() for p in head_params]
        total_params = float(max(1, sum(p.numel() for p in head_params)))
        
        # Debug: Verify head parameters are trainable
        trainable_params = sum(1 for p in head_params if p.requires_grad)
        total_head_params = sum(p.numel() for p in head_params)
        logger.debug(f"[AFR] Head parameters: {len(head_params)} tensors, {trainable_params} trainable, {total_head_params:,} total elements")
        
        # Debug: Check head structure
        logger.debug(f"[AFR] Head module: {head}")
        logger.debug(f"[AFR] Head type: {type(head)}")
        if hasattr(head, '__dict__'):
            logger.debug(f"[AFR] Head attributes: {list(head.__dict__.keys())}")
        
        # Debug: Check if any head parameters have gradients enabled
        model_trainable = sum(1 for p in model.parameters() if p.requires_grad)
        model_trainable_elements = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.debug(f"[AFR] Model trainable parameters: {model_trainable} tensors, {model_trainable_elements:,} elements")

        # ---------- 4) Tensori full-batch ----------
        E     = E_cpu.to(device, non_blocking=True)     # [N,D]
        y     = labels                                  # alias
        w     = w_all
        w_norm = w.sum().detach()

        epochs   = int(getattr(self.afr_config, "epochs", self.config.epochs))
        lr       = float(getattr(self.afr_config, "lr", self.config.lr))
        lam_l2   = float(getattr(self.afr_config, "lambda_l2", 0.0))
        patience = int(getattr(self.afr_config, "patience", 20))

        # Only optimize parameters that actually have gradients (modules_to_save)
        trainable_head_params = [p for p in head_params if p.requires_grad]
        logger.info(f"[AFR] Optimizer will train {len(trainable_head_params)}/{len(head_params)} head parameters")
        optimizer = torch.optim.SGD(trainable_head_params, lr=lr, weight_decay=0.0)

        # ---------- Sicurezza: disattiva dropout nella head (ripristino a fine) ----------
        orig_dropout_p = None
        if hasattr(head, "dropout"):
            orig_dropout_p = getattr(head.dropout, "p", None)
            if orig_dropout_p is not None:
                head.dropout.p = 0.0

        # ---------- Sanity: pesi head finiti ----------
        for k, v in head.state_dict().items():
            if not torch.isfinite(v).all():
                logger.error(f"[AFR] Pesi head non finiti prima di AFR: {k}")
                raise RuntimeError("Head NaN prima di AFR: ripristina checkpoint.")

        # ---------- Baseline per guardrail  se servono ----------
        # with torch.no_grad():
        #     # NB: usa la stessa forma di input che usi nel train (qui feats->head); nel tuo setup: head(E.unsqueeze(1))
        #     logits0 = head(E.unsqueeze(1))
        #     acc_drw0 = (logits0.argmax(1) == y).float().mean().item()

        # ---------- Baseline WCA su validation (se disponibile) ----------
        best_state = deepcopy(head.state_dict())
        best_epoch = 0
        no_improve = 0
        best_wca = -1.0  # vogliamo massimizzare
        best_mca = -1.0  # vogliamo massimizzare anche MCA
        base_id_acc = None

        # Baseline WCA/MCA (se val esiste) una sola volta
        if val_loaders is not None:
            wca0, mca0, _ = self._worst_class_accuracy_fast(
                model=model, loader_or_dict=val_loaders, device=device,
                num_labels=getattr(model.config, "num_labels", None),
                max_batches=val_max_batches,
                subsample=val_subsample,
                hans_index=getattr(self.afr_config, "hpo_is_hans_index", None),
                
            )
            base_id_acc = mca0
            best_wca = wca0
            best_mca = mca0
            logger.info(f"[AFR] Baseline validation: WCA={wca0:.4f}, MCA={mca0:.4f}")
        else:
            logger.info("[AFR] Nessun validation loader passato: userò solo guardrail su DRW accuracy.")

        # ---------- 5) Loop di training ----------
        for epoch in range(1, epochs + 1):
            head.train()
            optimizer.zero_grad(set_to_none=True)

            logits_head = head(E.unsqueeze(1))                 # [N,C]
            ce = F.cross_entropy(logits_head, y, reduction="none")
            loss_ce = (w * ce).sum() / w_norm

            if lam_l2 > 0.0:
                reg = 0.0
                # Only regularize the trainable parameters
                trainable_phi_init = [p for p in phi_init if p.requires_grad]
                for p0, p in zip(trainable_phi_init, trainable_head_params):
                    reg = reg + (p - p0).pow(2).sum()
                reg = 0.5 * reg / len(trainable_head_params)  # Normalize by number of trainable params
                loss = loss_ce + lam_l2 * reg
            else:
                loss = loss_ce

            loss.backward()
            
            # Debug: Check if gradients exist and calculate norm
            grad_norm = 0.0
            params_with_grads = [p for p in head_params if p.grad is not None]
            
            if params_with_grads:
                grad_norm = torch.sqrt(sum((p.grad.detach()**2).sum() for p in params_with_grads)).item()
                logger.debug(f"[AFR] Gradients exist on {len(params_with_grads)}/{len(head_params)} parameters, norm={grad_norm:.6f}")
            else:
                logger.warning(f"[AFR] No gradients found for any head parameters!")
                # Check if any parameters have gradients
                for i, p in enumerate(head_params):
                    if p.grad is not None:
                        logger.debug(f"[AFR] Parameter {i} has gradient: {p.grad.norm().item():.6f}")
                    else:
                        logger.debug(f"[AFR] Parameter {i} has NO gradient")
            
            # Only clip gradients for parameters that actually have gradients
            if params_with_grads:
                torch.nn.utils.clip_grad_norm_(params_with_grads, max_norm=1.0)
            optimizer.step()
            
            # Calculate parameter shift after optimization step (only for trainable params)
            with torch.no_grad():
                import math
                trainable_phi_init = [p for p in phi_init if p.requires_grad]
                num = sum((p - p0).pow(2).sum() for p, p0 in zip(trainable_head_params, trainable_phi_init))
                den = sum(p0.pow(2).sum() for p0 in trainable_phi_init)
                rel_shift = math.sqrt(num / max(den, 1e-12))
            
            logger.debug(f"[AFR] head_rel_shift={rel_shift:.3e}, grad_norm={grad_norm:.3e}")
            head.eval()
            #--TODEL
            # --- monitor training (facoltativo) ---
            with torch.no_grad():
                logits_eval = head(E.unsqueeze(1))
                ce_eval = F.cross_entropy(logits_eval, y, reduction="none")
                afr_ce = (w * ce_eval).sum() / w_norm
                acc_drw = (logits_eval.argmax(1) == y).float().mean().item()

            # ---------- Valutazione per early stopping e model selection ----------
            do_eval = (val_loaders is not None) and (epoch % eval_every == 0 or epoch == 1)
            if do_eval:
                # Debug: Check if model is in eval mode and head parameters are still trainable
                logger.debug(f"[AFR] Before validation: model.training={model.training}, head.training={head.training}")
                
                # Debug: Check if head parameters have actually changed (only trainable ones)
                with torch.no_grad():
                    trainable_phi_init = [p for p in phi_init if p.requires_grad]
                    param_change = sum((p - p0).abs().sum().item() for p, p0 in zip(trainable_head_params, trainable_phi_init))
                    logger.debug(f"[AFR] Total parameter change from initial: {param_change:.6f}")
                
                wca, mca, _ = self._worst_class_accuracy_fast(
                    model=model, loader_or_dict=val_loaders, device=device,
                    num_labels=getattr(model.config, "num_labels", None),
                    max_batches=val_max_batches,
                    subsample=val_subsample,
                    hans_index=getattr(self.afr_config, "hpo_is_hans_index", None)
                    
                )
                
                # Debug: Log validation results
                logger.info(f"[AFR] Validation results: WCA={wca:.6f}, MCA={mca:.6f}")
                logger.info(f"[AFR][Ep {epoch}/{epochs}] loss={loss.item():.6f} AFR_CE={afr_ce.item():.6f} "
                            f"DRW_acc={acc_drw:.4f} | VAL WCA={wca:.4f} MCA={mca:.4f}")

                # Consider improvement if either WCA or MCA improves
                wca_improved = (wca > best_wca + wca_tol)
                mca_improved = (mca > best_mca + wca_tol)
                improved = wca_improved or mca_improved
                
                logger.info(f"[AFR] Improvement check: WCA {wca:.6f} vs best {best_wca:.6f} (improved: {wca_improved}), "
                           f"MCA {mca:.6f} vs best {best_mca:.6f} (improved: {mca_improved})")

                if improved:
                    if wca_improved:
                        best_wca = wca
                    if mca_improved:
                        best_mca = mca
                    best_state = deepcopy(head.state_dict())
                    best_epoch = epoch
                    no_improve = 0
                    logger.info(f"[AFR] New best model at epoch {epoch}: WCA={best_wca:.4f}, MCA={best_mca:.4f}")
                else:
                    no_improve += 1

                if no_improve >= patience:
                    logger.info(f"[AFR] Early stopping a epoca {epoch} (best @ {best_epoch}, WCA={best_wca:.4f}, MCA={best_mca:.4f}).")
                    break
            else:
                # log leggero senza validazione
                if epoch % eval_every == 0 or epoch == 1:
                    logger.info(f"[AFR][Ep {epoch}/{epochs}] loss={loss.item():.6f} AFR_CE={afr_ce.item():.6f} DRW_acc={acc_drw:.4f}")

        # ---------- 6) Ripristina best_state e cleanup ----------
        head.load_state_dict(best_state)
        if hasattr(head, "dropout") and orig_dropout_p is not None:
            head.dropout.p = orig_dropout_p

        model.eval()
        return model

    
    @contextmanager
    def _patch_afr_config(self, **overrides):
        old = {}
        try:
            for k, v in overrides.items():
                old[k] = getattr(self.afr_config, k)
                setattr(self.afr_config, k, v)
            yield
        finally:
            for k, v in old.items():
                setattr(self.afr_config, k, v)

    def grid_search(
        self,
        model: PreTrainedModel,
        drw_loader: DataLoader,                   # deve essere quello shuffle=False usato per il cache
        drw_dataset_raw: torch.utils.data.Dataset,
        *,
        param_grid: dict,                         # es: {"gamma":[2,4,6], "lambda_l2":[0.1,1.0], "lr":[3e-3], "epochs":[200], "patience":[20]}
        model_id: str | None = None,
        pooling: str = "cls",
        cache_dtype: str = "fp32",
        val_loaders: Optional[Union[DataLoader, Dict[str, DataLoader]]] = None,   # uno o più loader
        # --- NUOVI CONTROLLI DI VELOCITÀ (propagati a train_drw) ---
        eval_every: int = 5,
        val_max_batches: Optional[int] = None,
        val_subsample: Optional[int] = None,
        # --- VALUTAZIONE FINALE DOPO OGNI COMBINAZIONE ---
        final_eval_max_batches: Optional[int] = None,   # None => full val
        final_eval_subsample: Optional[int] = None,
    ):
        """
        Grid-search per AFR Stage 2 con selezione iperparametri basata su Worst-Class Accuracy (WCA),
        calcolata su `val_loaders` (uno o più). Nessun uso dei pesi su validation.
        Usa valutazione WCA/MCA veloce (_worst_class_accuracy_fast) e passa i controlli di velocità
        dentro train_drw (eval_every/val_max_batches/val_subsample).
        """
        from copy import deepcopy
        import itertools

        device = self.device

        # 0) Warm cache UNA volta (accelererà i train_drw successivi)
        if model_id is None:
            model_id = getattr(model.config, "_name_or_path", "roberta") + f"_{sum(p.numel() for p in model.parameters())}"
        _ = self._build_or_load_cache_text(
            model=model,
            drw_loader=drw_loader,
            model_id=model_id,
            pooling=pooling,
            dtype=cache_dtype,
            split_name="drw",
        )

        # 1) Salva lo stato iniziale della head per isolare gli esperimenti
        head, _ = self._get_head_and_freeze_base(model)
        head_init = deepcopy(head.state_dict())

        # 2) Crea tutte le combinazioni
        keys = list(param_grid.keys())
        combos = list(itertools.product(*[param_grid[k] for k in keys]))

        results = []
        for i, combo in enumerate(combos, 1):
            params = dict(zip(keys, combo))
            # ripristina head allo stato iniziale (isolamento fra run)
            head.load_state_dict(head_init)
            head.eval()

            # patch temporanea della config AFR
            with self._patch_afr_config(**params):
                if hasattr(self.config, "logger"):
                    self.config.logger.info(f"[AFR Grid] {i}/{len(combos)}: {params}")

                # 3) Esegui AFR Stage 2 con ES su WCA (se val_loaders passato) e valutazioni veloci
                self.train_drw(
                    model=model,
                    drw_loader=drw_loader,
                    drw_dataset_raw=drw_dataset_raw,
                    model_id=model_id,
                    pooling=pooling,
                    cache_dtype=cache_dtype,
                    val_loaders=val_loaders,            # singolo loader o dict
                    wca_tol=1e-4,
                    eval_every=eval_every,
                    val_max_batches=val_max_batches,
                    val_subsample=val_subsample,
                )

                # 4) Valutazione finale (WCA, MCA) su validation per questa combinazione
                if val_loaders is not None:
                    # Se vuoi la valutazione finale FULL, lascia final_* = None (default)
                    wca, mca, per_class = self._worst_class_accuracy_fast(
                        model=model,
                        loader_or_dict=val_loaders,
                        device=device,
                        num_labels=getattr(model.config, "num_labels", None),
                        max_batches=final_eval_max_batches,
                        subsample=final_eval_subsample,
                        hans_index=getattr(self.afr_config, "hpo_is_hans_index", None)
                    )
                else:
                    # se non hai val loader: logga 0 per coerenza, ma idealmente passa sempre una val
                    wca, mca, per_class = 0.0, 0.0, {}

            results.append({
                **params,
                "worst_class_acc": float(wca),
                "mean_class_acc":  float(mca),
                "acc_per_class":   {int(k): float(v) for k, v in per_class.items()},
            })

            # 5) Ripristina la head allo stato iniziale prima della prossima combinazione
            head.load_state_dict(head_init)
            head.eval()

        # 6) Ordina: massimizza WCA e, a parità, MCA
        results.sort(key=lambda r: (-r["worst_class_acc"], -r["mean_class_acc"]))
        return results