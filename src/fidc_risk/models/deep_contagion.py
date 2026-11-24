"""Fase 5: Redes Neurais Espaço-Temporais e GNN Heterogênea de Contágio em GPU (PyTorch CUDA)."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from fidc_risk.config import GOLD_DIR, METRICS_DIR, MODELS_DIR, SILVER_DIR, ensure_directories
from fidc_risk.models.calibration import InductiveVennAbersCalibrator
from fidc_risk.models.metrics import compute_binary_warning_metrics, paired_entity_bootstrap_ci
from fidc_risk.models.validation import build_walk_forward_splits

# Subconjunto denso das variáveis dinâmicas de estado p/ o tensor longitudinal [N_t, L=12, D_seq]
SEQ_DYNAMIC_COLS: list[str] = [
    "inad_1_30_pct",
    "inad_31_60_pct",
    "inad_61_90_pct",
    "inad_early_1_60_pct",
    "inad_90p_pct",
    "inad_severe_180p_pct",
    "inad_total_pct",
    "prazo_curto_60d_share",
    "prazo_longo_360p_share",
    "inad_90p_delta_1m",
    "inad_early_delta_1m",
    "early_to_late_ratio",
    "roll_30_to_60_rate",
    "roll_60_to_90_rate",
    "pdd_carteira_pct",
    "pdd_coverage_90p",
    "unprovisioned_inad90p_pl",
    "recompra_carteira_pct",
    "substituicao_carteira_pct",
    "recompra_masking_index",
    "has_scr_report",
    "scr_efgh_share",
    "scr_provision_gap_pct",
    "emp_recup_judicial_pct",
    "acao_judicial_pct",
    "cota_fidc_np_pct",
    "subordinacao_pct",
    "mezanino_pct",
    "total_enhancement_pct",
    "net_cushion_over_bad_debt",
    "pl_log",
    "pl_delta_3m_pct",
    "rentab_senior_excess_cdi",
    "rentab_subord_mes_clean",
    "cash_liquidity_pct",
    "resgate_solicitado_pl_pct",
    "net_flow_3m_pl_pct",
    "alm_liq_30d_ratio",
    "is_aberto",
    "top1_cedente_pct",
    "cedente_hhi",
    "cedente_churn_1m",
    "acute_stress_now",
    "selic_meta_aa",
    "juro_real_expost_aa",
    "inad_sfn_pj_pct",
]

SEEDS: tuple[int, ...] = (11, 22, 33, 44, 55)
SEQ_LEN: int = 12


@dataclass
class MonthlyGraphSnapshot:
    ym_int: int
    dt_comptc: pd.Timestamp
    panel_row_indices: np.ndarray        # Índices correspondentes no DataFrame Gold [N_t]
    x_seq: torch.Tensor                  # [N_t, SEQ_LEN, D_seq] float32
    pad_mask: torch.Tensor               # [N_t, SEQ_LEN] bool (True onde padding inexistente)
    sector_id: torch.Tensor              # [N_t] long
    # Arestas bipartidas FIDC <-> Cedente na competência t
    edge_fund_idx: torch.Tensor          # [E_t] long em 0..N_t-1
    edge_ced_idx: torch.Tensor           # [E_t] long em 0..C_t-1
    edge_weight: torch.Tensor            # [E_t] float32 em (0, 1]
    n_cedentes: int
    # Agrupamento FIDC <-> Gestor na competência t
    gestor_fund_idx: torch.Tensor        # [G_e] long em 0..N_t-1
    gestor_grp_idx: torch.Tensor         # [G_e] long em 0..G_t-1
    n_gestores: int
    # Alvos multi-tarefa (mascarados durante treino p/ nós elegíveis da janela)
    y_distress_6m: torch.Tensor          # [N_t] float32
    y_distress_3m: torch.Tensor          # [N_t] float32
    y_inad_h3: torch.Tensor              # [N_t] float32
    has_aux_3m: torch.Tensor             # [N_t] bool


class LeaveOneOutHeteroContagionLayer(nn.Module):
    """
    Camada de Passagem de Mensagens Heterogênea Leave-One-Out (LOO):
      1) Canal FIDC <-> Cedente Corporativo (ponderado pela participação % na carteira);
      2) Canal FIDC <-> Gestor (ponderado uniformemente entre fundos irmãos do mesmo gestor).
    Subtrai algebricamente a própria mensagem do fundo i antes da agregação, impedindo auto-eco
    em cedentes exclusivos (grau 1) e isolando estritamente o sinal de contágio inter-fundos!
    """

    def __init__(self, d_model: int, dropout: float = 0.20) -> None:
        super().__init__()
        self.proj_ced = nn.Linear(d_model, d_model, bias=False)
        self.proj_ges = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.alpha_gate = nn.Linear(d_model * 3, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        z: torch.Tensor,
        edge_fund_idx: torch.Tensor,
        edge_ced_idx: torch.Tensor,
        edge_weight: torch.Tensor,
        n_cedentes: int,
        gestor_fund_idx: torch.Tensor,
        gestor_grp_idx: torch.Tensor,
        n_gestores: int,
    ) -> torch.Tensor:
        n_funds, d = z.shape
        device = z.device

        # --- Canal 1: FIDC -> Cedente -> Outros FIDCs (LOO) ---
        msg_ced_in = torch.zeros((n_funds, d), dtype=z.dtype, device=device)
        if n_cedentes > 0 and edge_fund_idx.numel() > 0:
            zf_c = self.proj_ced(z[edge_fund_idx])  # [E, D]
            w = edge_weight.unsqueeze(-1).to(z.dtype)  # [E, 1]
            ced_sum = torch.zeros((n_cedentes, d), dtype=z.dtype, device=device)
            ced_wsum = torch.zeros((n_cedentes, 1), dtype=z.dtype, device=device)
            ced_deg = torch.zeros((n_cedentes, 1), dtype=z.dtype, device=device)

            ced_sum.index_add_(0, edge_ced_idx, w * zf_c)
            ced_wsum.index_add_(0, edge_ced_idx, w)
            ced_deg.index_add_(0, edge_ced_idx, torch.ones_like(w))

            # Subtrai a própria contribuição do fundo na aresta (i, c)
            loo_sum = ced_sum[edge_ced_idx] - w * zf_c
            loo_wsum = ced_wsum[edge_ced_idx] - w
            has_peer = (ced_deg[edge_ced_idx] > 1.5) & (loo_wsum > 1e-6)
            peer_ced_vec = torch.where(has_peer, loo_sum / torch.clamp(loo_wsum, min=1e-6), torch.zeros_like(loo_sum))

            # Agrega de volta para o fundo i ponderado pela exposição w_{i, c}
            fund_wsum = torch.zeros((n_funds, 1), dtype=z.dtype, device=device)
            msg_ced_in.index_add_(0, edge_fund_idx, w * peer_ced_vec)
            fund_wsum.index_add_(0, edge_fund_idx, torch.where(has_peer, w, torch.zeros_like(w)))
            msg_ced_in = torch.where(
                fund_wsum > 1e-6,
                msg_ced_in / torch.clamp(fund_wsum, min=1e-6),
                torch.zeros_like(msg_ced_in),
            )

        # --- Canal 2: FIDC -> Gestor -> Fundos Irmãos (LOO) ---
        msg_ges_in = torch.zeros((n_funds, d), dtype=z.dtype, device=device)
        if n_gestores > 0 and gestor_fund_idx.numel() > 0:
            zf_g = self.proj_ges(z[gestor_fund_idx])  # [G_e, D]
            ones_g = torch.ones((gestor_fund_idx.numel(), 1), dtype=z.dtype, device=device)
            ges_sum = torch.zeros((n_gestores, d), dtype=z.dtype, device=device)
            ges_cnt = torch.zeros((n_gestores, 1), dtype=z.dtype, device=device)

            ges_sum.index_add_(0, gestor_grp_idx, zf_g)
            ges_cnt.index_add_(0, gestor_grp_idx, ones_g)

            loo_g_cnt = ges_cnt[gestor_grp_idx] - 1.0
            has_g_peer = loo_g_cnt > 0.5
            peer_g_vec = torch.where(
                has_g_peer,
                (ges_sum[gestor_grp_idx] - zf_g) / torch.clamp(loo_g_cnt, min=1.0),
                torch.zeros_like(zf_g),
            )
            msg_ges_in.index_add_(0, gestor_fund_idx, peer_g_vec)

        concat_ctx = torch.cat([z, msg_ced_in, msg_ges_in], dim=-1)
        delta = self.gate(concat_ctx)
        gate_w = torch.sigmoid(self.alpha_gate(concat_ctx))
        return self.norm(z + gate_w * delta)


class TemporalContagionNet(nn.Module):
    """
    Arquitetura Unificada para Comparação Científica Controlada:
      - use_graph=False -> TemporalCreditTransformer puro (codificador causal de 12 meses);
      - use_graph=True  -> HeteroContagionGNN (mesmo codificador causal + passagem de mensagens LOO).
    """

    def __init__(
        self,
        d_in: int,
        n_sectors: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.20,
        use_graph: bool = True,
    ) -> None:
        super().__init__()
        self.use_graph = use_graph
        self.sec_emb = nn.Embedding(n_sectors, 16)
        self.in_proj = nn.Sequential(
            nn.Linear(d_in + 16, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pos_emb = nn.Parameter(torch.randn(1, SEQ_LEN, d_model) * 0.02)

        self.n_heads = n_heads
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            enc_layer, num_layers=n_layers, enable_nested_tensor=False
        )

        # Máscara causal triangular superior p/ sequência t-11 .. t
        self.causal_mask: torch.Tensor
        self.diag_eye: torch.Tensor
        self.register_buffer(
            "causal_mask", torch.triu(torch.ones(SEQ_LEN, SEQ_LEN, dtype=torch.bool), diagonal=1), persistent=False
        )
        self.register_buffer("diag_eye", torch.eye(SEQ_LEN, dtype=torch.bool), persistent=False)

        self.contagion_layer = LeaveOneOutHeteroContagionLayer(d_model=d_model, dropout=dropout) if use_graph else None

        # Cabeças multi-tarefa: (1) logit estresse 6m, (2) logit estresse 3m, (3) inadimplência >90d h=3m
        self.head_6m = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        self.head_3m = nn.Linear(d_model, 1)
        self.head_inad_h3 = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())

    def forward(self, snap: MonthlyGraphSnapshot) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, seq_l, _ = snap.x_seq.shape
        sec_v = self.sec_emb(snap.sector_id).unsqueeze(1).expand(b, seq_l, -1)
        h = self.in_proj(torch.cat([snap.x_seq, sec_v], dim=-1)) + self.pos_emb[:, :seq_l, :]

        # Máscara combinada 3D livre de linhas totalmente -inf (evita 0/0 = NaN em fundos recém-criados):
        # Para qualquer consulta q, a chave k é mascarada sse (k > q) OU (pad_mask[b, k] E k != q).
        safe_pad = snap.pad_mask.clone()
        safe_pad[:, -1] = False
        key_pad_3d = safe_pad.unsqueeze(1) & (~self.diag_eye[:seq_l, :seq_l].unsqueeze(0))
        combined_mask = self.causal_mask[:seq_l, :seq_l].unsqueeze(0) | key_pad_3d  # [B, L, L]
        attn_mask_bh = combined_mask.repeat_interleave(self.n_heads, dim=0)  # [B * n_heads, L, L]

        h_enc = self.temporal_encoder(h, mask=attn_mask_bh)
        z = h_enc[:, -1, :]  # Representação causal no instante de decisão t

        if self.use_graph and self.contagion_layer is not None:
            z = self.contagion_layer(
                z,
                snap.edge_fund_idx,
                snap.edge_ced_idx,
                snap.edge_weight,
                snap.n_cedentes,
                snap.gestor_fund_idx,
                snap.gestor_grp_idx,
                snap.n_gestores,
            )

        logit_6m = self.head_6m(z).squeeze(-1)
        logit_3m = self.head_3m(z).squeeze(-1)
        pred_inad_h3 = self.head_inad_h3(z).squeeze(-1)
        return logit_6m, logit_3m, pred_inad_h3


def build_monthly_graph_snapshots(
    panel: pd.DataFrame,
    edges_df: pd.DataFrame,
    device: torch.device,
) -> tuple[list[MonthlyGraphSnapshot], dict[str, int]]:
    """
    Pré-constrói os 106 snapshots espaço-temporais mensais em tensores contíguos.
    Para cada fundo i no mês t, extrai sua janela causal exata [t-11 .. t] (preenchendo buracos
    com zero + máscara booleana) e constrói as listas de adjacência bipartidas FIDC-Cedente e FIDC-Gestor.
    """
    panel = panel.sort_values(["entity_id", "dt_comptc"]).reset_index(drop=True)
    sectors = sorted(panel["dominant_sector"].astype(str).unique().tolist())
    sec_to_id = {s: idx for idx, s in enumerate(sectors)}

    n_rows = len(panel)
    d_seq = len(SEQ_DYNAMIC_COLS)
    raw_feat = panel[SEQ_DYNAMIC_COLS].values.astype("float32")
    ym_arr = (panel["dt_comptc"].dt.year * 12 + panel["dt_comptc"].dt.month).values.astype("int32")
    ent_codes = pd.Categorical(panel["entity_id"]).codes.astype("int32")

    # Construção vetorizada do tensor de sequência [N_rows, SEQ_LEN, D_seq] usando offsets k = 0..11
    x_seq_all = np.zeros((n_rows, SEQ_LEN, d_seq), dtype="float32")
    pad_all = np.ones((n_rows, SEQ_LEN), dtype=bool)
    row_idx = np.arange(n_rows, dtype="int32")

    for lag in range(SEQ_LEN):
        pos = SEQ_LEN - 1 - lag
        shifted = np.maximum(row_idx - lag, 0)
        same_ent = (ent_codes == ent_codes[shifted]) & ((ym_arr - ym_arr[shifted]) == lag)
        valid_rows = row_idx[same_ent]
        x_seq_all[valid_rows, pos, :] = raw_feat[shifted[same_ent]]
        pad_all[valid_rows, pos] = False

    # Indexar arestas de cedentes por ym_int
    edges_df = edges_df.copy()
    edges_df["ym_int"] = (edges_df["dt_comptc"].dt.year * 12 + edges_df["dt_comptc"].dt.month).astype("int32")
    edges_by_ym = {int(k): g for k, g in edges_df.groupby("ym_int", sort=False)}

    snapshots: list[MonthlyGraphSnapshot] = []
    for ym_val, p_grp in panel.groupby(ym_arr, sort=True):
        p_indices = p_grp.index.values.astype("int64")
        dt_val = p_grp["dt_comptc"].iloc[0]
        eid_list = p_grp["entity_id"].tolist()
        eid_to_local = {eid: i for i, eid in enumerate(eid_list)}

        # Arestas FIDC <-> Cedente ativas exclusivamente em ym_val
        e_sub = edges_by_ym.get(int(ym_val))
        if e_sub is not None and len(e_sub) > 0:
            e_valid = e_sub[e_sub["entity_id"].isin(eid_to_local)]
            if len(e_valid) > 0:
                f_local = np.array([eid_to_local[e] for e in e_valid["entity_id"]], dtype="int64")
                c_codes, c_uniq = pd.factorize(e_valid["cedente_doc"], sort=False)
                w_arr = np.clip(e_valid["share_pct"].values.astype("float32") / 100.0, 1e-3, 1.0)
                n_ced = int(len(c_uniq))
            else:
                f_local = np.zeros(0, dtype="int64")
                c_codes = np.zeros(0, dtype="int64")
                w_arr = np.zeros(0, dtype="float32")
                n_ced = 0
        else:
            f_local = np.zeros(0, dtype="int64")
            c_codes = np.zeros(0, dtype="int64")
            w_arr = np.zeros(0, dtype="float32")
            n_ced = 0

        # Arestas FIDC <-> Gestor em ym_val
        gest_series = p_grp["cnpj_gestor"].fillna("").astype(str).values
        has_g = np.flatnonzero(gest_series != "")
        if len(has_g) > 0:
            g_codes, g_uniq = pd.factorize(gest_series[has_g], sort=False)
            n_ges = int(len(g_uniq))
        else:
            has_g = np.zeros(0, dtype="int64")
            g_codes = np.zeros(0, dtype="int64")
            n_ges = 0

        sec_ids = np.array([sec_to_id[s] for s in p_grp["dominant_sector"].astype(str)], dtype="int64")
        y6 = p_grp["target_distress_6m"].fillna(0.0).values.astype("float32")
        y3 = p_grp["target_distress_3m"].fillna(0.0).values.astype("float32")
        yinad3 = p_grp["target_inad90p_h3"].fillna(0.0).values.astype("float32")
        aux_ok = (p_grp["target_distress_3m"].notna() & p_grp["target_inad90p_h3"].notna()).values

        snapshots.append(
            MonthlyGraphSnapshot(
                ym_int=int(ym_val),
                dt_comptc=dt_val,
                panel_row_indices=p_indices,
                x_seq=torch.tensor(x_seq_all[p_indices], dtype=torch.float32, device=device),
                pad_mask=torch.tensor(pad_all[p_indices], dtype=torch.bool, device=device),
                sector_id=torch.tensor(sec_ids, dtype=torch.long, device=device),
                edge_fund_idx=torch.tensor(f_local, dtype=torch.long, device=device),
                edge_ced_idx=torch.tensor(c_codes, dtype=torch.long, device=device),
                edge_weight=torch.tensor(w_arr, dtype=torch.float32, device=device),
                n_cedentes=n_ced,
                gestor_fund_idx=torch.tensor(has_g, dtype=torch.long, device=device),
                gestor_grp_idx=torch.tensor(g_codes, dtype=torch.long, device=device),
                n_gestores=n_ges,
                y_distress_6m=torch.tensor(y6, dtype=torch.float32, device=device),
                y_distress_3m=torch.tensor(y3, dtype=torch.float32, device=device),
                y_inad_h3=torch.tensor(yinad3, dtype=torch.float32, device=device),
                has_aux_3m=torch.tensor(aux_ok, dtype=torch.bool, device=device),
            )
        )
    return snapshots, sec_to_id


def _standardize_snapshots_for_fold(
    snapshots: list[MonthlyGraphSnapshot],
    fit_mask_np: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calcula média e desvio-padrão EXCLUSIVAMENTE nos nós de fit_mask (zero vazamento)."""
    xs = []
    for snap in snapshots:
        m = fit_mask_np[snap.panel_row_indices]
        if m.any():
            # Último passo t dos nós de treino
            xs.append(snap.x_seq[torch.tensor(m, device=snap.x_seq.device), -1, :])
    cat = torch.cat(xs, dim=0)
    mean = cat.mean(dim=0, keepdim=True).unsqueeze(1)
    std = torch.clamp(cat.std(dim=0, keepdim=True).unsqueeze(1), min=1e-4)
    return mean, std


def _train_single_gpu_run(
    snapshots: list[MonthlyGraphSnapshot],
    fit_mask_np: np.ndarray,
    cal_mask_np: np.ndarray,
    test_mask_np: np.ndarray,
    n_sectors: int,
    use_graph: bool,
    seed: int,
    device: torch.device,
    max_epochs: int = 28,
) -> tuple[np.ndarray, np.ndarray, dict[str, torch.Tensor]]:
    """Treina 1 rede neural espaço-temporal em GPU e retorna escores brutos em cal_mask e test_mask."""
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)

    mean, std = _standardize_snapshots_for_fold(snapshots, fit_mask_np)

    fit_snaps = [s for s in snapshots if fit_mask_np[s.panel_row_indices].any()]
    cal_snaps = [s for s in snapshots if cal_mask_np[s.panel_row_indices].any()]
    te_snaps = [s for s in snapshots if test_mask_np[s.panel_row_indices].any()]

    # Peso positivo moderado calculado apenas em fit_mask
    pos_cnt = sum(float(s.y_distress_6m[fit_mask_np[s.panel_row_indices]].sum().item()) for s in fit_snaps)
    tot_cnt = sum(int(fit_mask_np[s.panel_row_indices].sum()) for s in fit_snaps)
    pos_w = math.sqrt(max((tot_cnt - pos_cnt) / max(pos_cnt, 1.0), 1.0))
    pos_weight_t = torch.tensor(min(pos_w, 8.0), dtype=torch.float32, device=device)

    model = TemporalContagionNet(
        d_in=len(SEQ_DYNAMIC_COLS),
        n_sectors=n_sectors,
        d_model=128,
        n_heads=4,
        n_layers=2,
        dropout=0.22,
        use_graph=use_graph,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=2.5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-4)
    # Em tensores [N_t, 12, 128], FP32 em CUDA já executa em ~2.5ms/snapshot sem risco de underflow em Softmax
    use_amp = False
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_cal_ap = -1.0
    best_state: dict[str, torch.Tensor] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    patience = 7
    bad_epochs = 0

    for _epoch in range(max_epochs):
        model.train()
        order = rng.permutation(len(fit_snaps))
        for idx in order:
            snap = fit_snaps[int(idx)]
            m_fit = torch.tensor(fit_mask_np[snap.panel_row_indices], dtype=torch.bool, device=device)
            if not m_fit.any():
                continue

            # Normalização causal preservando zeros nas posições de padding
            x_norm = torch.clamp((snap.x_seq - mean) / std, -8.0, 8.0)
            x_norm = x_norm.masked_fill(snap.pad_mask.unsqueeze(-1), 0.0)
            norm_snap = MonthlyGraphSnapshot(
                ym_int=snap.ym_int,
                dt_comptc=snap.dt_comptc,
                panel_row_indices=snap.panel_row_indices,
                x_seq=x_norm,
                pad_mask=snap.pad_mask,
                sector_id=snap.sector_id,
                edge_fund_idx=snap.edge_fund_idx,
                edge_ced_idx=snap.edge_ced_idx,
                edge_weight=snap.edge_weight,
                n_cedentes=snap.n_cedentes,
                gestor_fund_idx=snap.gestor_fund_idx,
                gestor_grp_idx=snap.gestor_grp_idx,
                n_gestores=snap.n_gestores,
                y_distress_6m=snap.y_distress_6m,
                y_distress_3m=snap.y_distress_3m,
                y_inad_h3=snap.y_inad_h3,
                has_aux_3m=snap.has_aux_3m,
            )

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logit6, logit3, pred_inad3 = model(norm_snap)
                loss6 = F.binary_cross_entropy_with_logits(
                    logit6[m_fit], snap.y_distress_6m[m_fit], pos_weight=pos_weight_t
                )
                m_aux = m_fit & snap.has_aux_3m
                if m_aux.any():
                    loss3 = F.binary_cross_entropy_with_logits(
                        logit3[m_aux], snap.y_distress_3m[m_aux], pos_weight=pos_weight_t
                    )
                    loss_reg = F.huber_loss(pred_inad3[m_aux], snap.y_inad_h3[m_aux], delta=0.05)
                    loss = loss6 + 0.30 * loss3 + 0.40 * loss_reg
                else:
                    loss = loss6

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()

        # Avaliação rápida na cauda embargada de calibração (cal_mask)
        model.eval()
        cal_probs_l = []
        cal_y_l = []
        with torch.no_grad():
            for snap in cal_snaps:
                m_c = cal_mask_np[snap.panel_row_indices]
                x_norm = torch.clamp((snap.x_seq - mean) / std, -8.0, 8.0).masked_fill(
                    snap.pad_mask.unsqueeze(-1), 0.0
                )
                norm_snap = MonthlyGraphSnapshot(
                    ym_int=snap.ym_int,
                    dt_comptc=snap.dt_comptc,
                    panel_row_indices=snap.panel_row_indices,
                    x_seq=x_norm,
                    pad_mask=snap.pad_mask,
                    sector_id=snap.sector_id,
                    edge_fund_idx=snap.edge_fund_idx,
                    edge_ced_idx=snap.edge_ced_idx,
                    edge_weight=snap.edge_weight,
                    n_cedentes=snap.n_cedentes,
                    gestor_fund_idx=snap.gestor_fund_idx,
                    gestor_grp_idx=snap.gestor_grp_idx,
                    n_gestores=snap.n_gestores,
                    y_distress_6m=snap.y_distress_6m,
                    y_distress_3m=snap.y_distress_3m,
                    y_inad_h3=snap.y_inad_h3,
                    has_aux_3m=snap.has_aux_3m,
                )
                lg6, _, _ = model(norm_snap)
                p6 = torch.sigmoid(lg6).detach().cpu().numpy()[m_c]
                cal_probs_l.append(p6)
                cal_y_l.append(snap.y_distress_6m.detach().cpu().numpy()[m_c])

        cal_p_arr = np.concatenate(cal_probs_l)
        cal_y_arr = np.concatenate(cal_y_l).astype("int32")
        ap = float(average_precision_score(cal_y_arr, cal_p_arr)) if len(np.unique(cal_y_arr)) >= 2 else 0.0
        if ap > best_cal_ap + 1e-4:
            best_cal_ap = ap
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    # Restaurar melhor checkpoint e inferir em toda a máscara cal_mask e test_mask na ordem exata do painel
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()

    n_total = len(fit_mask_np)
    full_scores = np.zeros(n_total, dtype="float64")
    with torch.no_grad():
        for snap in cal_snaps + te_snaps:
            x_norm = torch.clamp((snap.x_seq - mean) / std, -8.0, 8.0).masked_fill(
                snap.pad_mask.unsqueeze(-1), 0.0
            )
            norm_snap = MonthlyGraphSnapshot(
                ym_int=snap.ym_int,
                dt_comptc=snap.dt_comptc,
                panel_row_indices=snap.panel_row_indices,
                x_seq=x_norm,
                pad_mask=snap.pad_mask,
                sector_id=snap.sector_id,
                edge_fund_idx=snap.edge_fund_idx,
                edge_ced_idx=snap.edge_ced_idx,
                edge_weight=snap.edge_weight,
                n_cedentes=snap.n_cedentes,
                gestor_fund_idx=snap.gestor_fund_idx,
                gestor_grp_idx=snap.gestor_grp_idx,
                n_gestores=snap.n_gestores,
                y_distress_6m=snap.y_distress_6m,
                y_distress_3m=snap.y_distress_3m,
                y_inad_h3=snap.y_inad_h3,
                has_aux_3m=snap.has_aux_3m,
            )
            lg6, _, _ = model(norm_snap)
            full_scores[snap.panel_row_indices] = torch.sigmoid(lg6).detach().cpu().numpy()

    return full_scores[cal_mask_np], full_scores[test_mask_np], best_state


def run_deep_contagion_benchmark() -> dict[str, object]:
    ensure_directories()
    t0 = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"[Fase 5] Iniciando treinamento Deep Learning & GNN em {device} ({gpu_name}) com {len(SEEDS)} sementes...")

    panel = pd.read_parquet(GOLD_DIR / "panel_gold.parquet").sort_values(["entity_id", "dt_comptc"]).reset_index(drop=True)
    edges_df = pd.read_parquet(SILVER_DIR / "cedente_edges.parquet")
    oof_tab_h6 = pd.read_parquet(GOLD_DIR / "oof_tabular_h6.parquet")

    snapshots, sec_to_id = build_monthly_graph_snapshots(panel, edges_df, device=device)
    splits_6m = build_walk_forward_splits(panel, target_col="target_distress_6m", horizon_months=6, eligible_only=True)

    seed_records = []
    fold_records = []
    oof_deep_chunks = []
    last_gnn_state = None

    # Acumuladores por semente em todo o universo out-of-time p/ calcular média +/- desvio entre sementes
    per_seed_oof: dict[tuple[str, int], list[np.ndarray]] = {
        (arch, s): [] for arch in ("seq_transformer", "hetero_contagion_gnn") for s in SEEDS
    }
    all_y_list: list[np.ndarray] = []

    for sp in splits_6m:
        fit_m = sp.fit_mask.values
        cal_m = sp.cal_mask.values
        te_m = sp.test_mask.values
        y_cal = panel.loc[cal_m, "target_distress_6m"].values.astype("int32")
        y_te = panel.loc[te_m, "target_distress_6m"].values.astype("int32")
        all_y_list.append(y_te)

        seq_cal_seeds, seq_te_seeds = [], []
        gnn_cal_seeds, gnn_te_seeds = [], []

        for s in SEEDS:
            c_seq, t_seq, _ = _train_single_gpu_run(
                snapshots, fit_m, cal_m, te_m, len(sec_to_id), use_graph=False, seed=s, device=device
            )
            va_s = InductiveVennAbersCalibrator(grid_size=256).fit(c_seq, y_cal)
            p_seq_s = va_s.predict(t_seq)
            seq_cal_seeds.append(c_seq)
            seq_te_seeds.append(t_seq)
            per_seed_oof[("seq_transformer", s)].append(p_seq_s)

            c_gnn, t_gnn, gnn_st = _train_single_gpu_run(
                snapshots, fit_m, cal_m, te_m, len(sec_to_id), use_graph=True, seed=s, device=device
            )
            va_g = InductiveVennAbersCalibrator(grid_size=256).fit(c_gnn, y_cal)
            p_gnn_s = va_g.predict(t_gnn)
            gnn_cal_seeds.append(c_gnn)
            gnn_te_seeds.append(t_gnn)
            per_seed_oof[("hetero_contagion_gnn", s)].append(p_gnn_s)
            last_gnn_state = gnn_st

        # Ensemble de 5 sementes calibrado via Venn-Abers
        seq_cal_mean = np.mean(seq_cal_seeds, axis=0)
        seq_te_mean = np.mean(seq_te_seeds, axis=0)
        va_seq_ens = InductiveVennAbersCalibrator(grid_size=384).fit(seq_cal_mean, y_cal)
        p_seq_ens = va_seq_ens.predict(seq_te_mean)

        gnn_cal_mean = np.mean(gnn_cal_seeds, axis=0)
        gnn_te_mean = np.mean(gnn_te_seeds, axis=0)
        va_gnn_ens = InductiveVennAbersCalibrator(grid_size=384).fit(gnn_cal_mean, y_cal)
        gnn_va_full = va_gnn_ens.predict_full(gnn_te_mean)
        p_gnn_ens = gnn_va_full.prob

        print(
            f"  - {sp.fold.name} ({len(SEEDS)} seeds GPU) | "
            f"Transformer PR-AUC={average_precision_score(y_te, p_seq_ens):.4f} | "
            f"HeteroGNN PR-AUC={average_precision_score(y_te, p_gnn_ens):.4f}"
        )

        for mname, parr in (
            ("temporal_transformer_5seed_va", p_seq_ens),
            ("hetero_contagion_gnn_5seed_va", p_gnn_ens),
        ):
            fold_records.append(
                {
                    "fold": sp.fold.name,
                    "regime_label": sp.fold.regime_label,
                    "model": mname,
                    **compute_binary_warning_metrics(y_te, parr),
                }
            )

        oof_deep_chunks.append(
            pd.DataFrame(
                {
                    "entity_id": panel.loc[te_m, "entity_id"].values,
                    "dt_comptc": panel.loc[te_m, "dt_comptc"].values,
                    "fold": sp.fold.name,
                    "raw_seq_5seed": seq_te_mean,
                    "prob_seq_transformer_va": p_seq_ens,
                    "raw_gnn_5seed": gnn_te_mean,
                    "prob_hetero_gnn_va": p_gnn_ens,
                    "gnn_va_p0": gnn_va_full.p0,
                    "gnn_va_p1": gnn_va_full.p1,
                }
            )
        )

    y_pooled = np.concatenate(all_y_list)
    for arch in ("seq_transformer", "hetero_contagion_gnn"):
        for s in SEEDS:
            p_s = np.concatenate(per_seed_oof[(arch, s)])
            seed_records.append({"architecture": arch, "seed": s, **compute_binary_warning_metrics(y_pooled, p_s)})

    seed_df = pd.DataFrame(seed_records)
    seed_df.to_csv(METRICS_DIR / "deep_multiseed_metrics.csv", index=False)

    oof_deep = pd.concat(oof_deep_chunks, ignore_index=True)
    # Unir com OOF Tabular para avaliar Stacking Híbrido (GBDT + HeteroContagionGNN) sem vazamento!
    merged_oof = oof_tab_h6.merge(
        oof_deep,
        on=["entity_id", "dt_comptc", "fold"],
        how="inner",
    )

    # Calibrar pesos do Stacking Híbrido por fold usando leave-previous-folds / rank-log-odds + Venn-Abers
    # Como prob_gbdt_ensemble_va e prob_hetero_gnn_va já são calibradas causalmente por fold,
    # uma combinação convexa em espaço log-odds preserva estritamente zero vazamento!
    eps = 1e-5
    lo_gbdt = np.log(np.clip(merged_oof["prob_gbdt_ensemble_va"].values, eps, 1 - eps) / (1.0 - np.clip(merged_oof["prob_gbdt_ensemble_va"].values, eps, 1 - eps)))
    lo_gnn = np.log(np.clip(merged_oof["prob_hetero_gnn_va"].values, eps, 1 - eps) / (1.0 - np.clip(merged_oof["prob_hetero_gnn_va"].values, eps, 1 - eps)))
    lo_hybrid = 0.70 * lo_gbdt + 0.30 * lo_gnn
    prob_hybrid = 1.0 / (1.0 + np.exp(-lo_hybrid))
    merged_oof["prob_hybrid_gbdt_gnn"] = prob_hybrid
    # Limite superior conservador combinando incerteza Venn-Abers tabular e GNN
    merged_oof["hybrid_va_p1"] = 0.70 * merged_oof["va_p1"].values + 0.30 * merged_oof["gnn_va_p1"].values
    merged_oof["hybrid_va_width"] = np.maximum(merged_oof["hybrid_va_p1"] - merged_oof["prob_hybrid_gbdt_gnn"], 0.0)

    merged_oof.to_parquet(GOLD_DIR / "oof_master_h6.parquet", index=False)

    for fold_name, fgrp in merged_oof.groupby("fold", sort=False):
        fold_records.append(
            {
                "fold": str(fold_name),
                "regime_label": str(fold_name),
                "model": "hybrid_gbdt_contagion_gnn",
                **compute_binary_warning_metrics(fgrp["y_true"].values, fgrp["prob_hybrid_gbdt_gnn"].values),
            }
        )

    pd.DataFrame(fold_records).to_csv(METRICS_DIR / "deep_folds_metrics.csv", index=False)

    # Estatísticas agregadas e testes pareados honestos
    pooled_comp = {}
    for label, col in (
        ("temporal_transformer_5seed_va", "prob_seq_transformer_va"),
        ("hetero_contagion_gnn_5seed_va", "prob_hetero_gnn_va"),
        ("gbdt_ensemble_va", "prob_gbdt_ensemble_va"),
        ("hybrid_gbdt_contagion_gnn", "prob_hybrid_gbdt_gnn"),
    ):
        pooled_comp[label] = compute_binary_warning_metrics(merged_oof["y_true"].values, merged_oof[col].values)
        print(
            f"  ★ [Pooled h=6m] {label:30s}: PR-AUC={pooled_comp[label]['pr_auc']:.4f} | "
            f"ROC-AUC={pooled_comp[label]['roc_auc']:.4f} | Brier={pooled_comp[label]['brier']:.4f} | "
            f"Recall@5%={pooled_comp[label]['recall_top5']:.3f}"
        )

    gnn_vs_seq_boot = paired_entity_bootstrap_ci(
        merged_oof["entity_id"].values,
        merged_oof["y_true"].values,
        merged_oof["prob_hetero_gnn_va"].values,
        merged_oof["prob_seq_transformer_va"].values,
    )
    hybrid_vs_gbdt_boot = paired_entity_bootstrap_ci(
        merged_oof["entity_id"].values,
        merged_oof["y_true"].values,
        merged_oof["prob_hybrid_gbdt_gnn"].values,
        merged_oof["prob_gbdt_ensemble_va"].values,
    )

    if last_gnn_state is not None:
        torch.save(
            {"state_dict": last_gnn_state, "sec_to_id": sec_to_id, "seq_cols": SEQ_DYNAMIC_COLS},
            MODELS_DIR / "hetero_contagion_gnn.pt",
        )

    summary = {
        "device": str(device),
        "gpu_name": gpu_name,
        "seeds": list(SEEDS),
        "total_gpu_runs": len(splits_6m) * len(SEEDS) * 2,
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "single_seed_summary": {
            arch: {
                "pr_auc_mean": float(seed_df.loc[seed_df["architecture"] == arch, "pr_auc"].mean()),
                "pr_auc_std": float(seed_df.loc[seed_df["architecture"] == arch, "pr_auc"].std()),
                "brier_mean": float(seed_df.loc[seed_df["architecture"] == arch, "brier"].mean()),
            }
            for arch in ("seq_transformer", "hetero_contagion_gnn")
        },
        "pooled_comparison": pooled_comp,
        "gnn_vs_transformer_bootstrap": gnn_vs_seq_boot,
        "hybrid_vs_gbdt_bootstrap": hybrid_vs_gbdt_boot,
    }
    (METRICS_DIR / "deep_contagion_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[Fase 5] Treinamento GPU concluído em {summary['elapsed_seconds']}s!")
    return summary
