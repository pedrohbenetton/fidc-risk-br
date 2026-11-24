"""Testes da Arquitetura PyTorch: Causalidade Temporal e Passagem de Mensagens LOO na GNN."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from fidc_risk.models.deep_contagion import (
    SEQ_DYNAMIC_COLS,
    SEQ_LEN,
    LeaveOneOutHeteroContagionLayer,
    MonthlyGraphSnapshot,
    TemporalContagionNet,
)


def _dummy_snapshot(n_funds: int = 4, device_str: str = "cpu") -> MonthlyGraphSnapshot:
    dev = torch.device(device_str)
    torch.manual_seed(7)
    x_seq = torch.randn(n_funds, SEQ_LEN, len(SEQ_DYNAMIC_COLS), device=dev)
    pad = torch.zeros(n_funds, SEQ_LEN, dtype=torch.bool, device=dev)
    pad[0, :4] = True  # Fundo jovem com apenas 8 meses de histórico
    pad[1, :-1] = True  # Fundo recém-constituído com APENAS o mês atual t válido (11 passos de padding!)

    # Cedente 0 compartilhado entre fundos 0 e 1; Cedente 1 exclusivo do fundo 2 (grau 1!)
    edge_fund = torch.tensor([0, 1, 2], dtype=torch.long, device=dev)
    edge_ced = torch.tensor([0, 0, 1], dtype=torch.long, device=dev)
    edge_w = torch.tensor([0.40, 0.25, 0.80], dtype=torch.float32, device=dev)

    # Gestor 0 compartilhado entre fundos 1 e 3; Gestor 1 exclusivo do fundo 2
    g_fund = torch.tensor([1, 3, 2], dtype=torch.long, device=dev)
    g_grp = torch.tensor([0, 0, 1], dtype=torch.long, device=dev)

    return MonthlyGraphSnapshot(
        ym_int=2025 * 12 + 6,
        dt_comptc=pd.Timestamp("2025-06-30"),
        panel_row_indices=np.arange(n_funds, dtype="int64"),
        x_seq=x_seq,
        pad_mask=pad,
        sector_id=torch.zeros(n_funds, dtype=torch.long, device=dev),
        edge_fund_idx=edge_fund,
        edge_ced_idx=edge_ced,
        edge_weight=edge_w,
        n_cedentes=2,
        gestor_fund_idx=g_fund,
        gestor_grp_idx=g_grp,
        n_gestores=2,
        y_distress_6m=torch.tensor([0.0, 1.0, 0.0, 0.0], device=dev),
        y_distress_3m=torch.tensor([0.0, 1.0, 0.0, 0.0], device=dev),
        y_inad_h3=torch.tensor([0.02, 0.18, 0.01, 0.03], device=dev),
        has_aux_3m=torch.ones(n_funds, dtype=torch.bool, device=dev),
    )


def test_loo_contagion_layer_prevents_degree1_echo_and_propagates_shared_peers() -> None:
    torch.manual_seed(42)
    layer = LeaveOneOutHeteroContagionLayer(d_model=32, dropout=0.0).eval()
    z_base = torch.randn(4, 32)
    snap = _dummy_snapshot(4, "cpu")

    out_base = layer(
        z_base,
        snap.edge_fund_idx,
        snap.edge_ced_idx,
        snap.edge_weight,
        snap.n_cedentes,
        snap.gestor_fund_idx,
        snap.gestor_grp_idx,
        snap.n_gestores,
    )

    # Perturbar APENAS o fundo 1 (que compartilha Cedente 0 com fundo 0 e Gestor 0 com fundo 3,
    # mas NÃO tem nenhuma aresta com o fundo 2!)
    z_pert = z_base.clone()
    z_pert[1] += 10.0

    out_pert = layer(
        z_pert,
        snap.edge_fund_idx,
        snap.edge_ced_idx,
        snap.edge_weight,
        snap.n_cedentes,
        snap.gestor_fund_idx,
        snap.gestor_grp_idx,
        snap.n_gestores,
    )

    # Fundo 2 (isolado em cedente de grau 1 e gestor de grau 1) deve permanecer 100% idêntico!
    assert torch.allclose(out_base[2], out_pert[2], atol=1e-6)
    # Fundo 0 (par via Cedente 0) e Fundo 3 (par via Gestor 0) DEVEM responder ao choque do fundo 1!
    assert not torch.allclose(out_base[0], out_pert[0], atol=1e-4)
    assert not torch.allclose(out_base[3], out_pert[3], atol=1e-4)


def test_temporal_contagion_net_forward_backward_cpu_and_cuda() -> None:
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for dev_name in devices:
        snap = _dummy_snapshot(4, dev_name)
        for use_g in (False, True):
            net = TemporalContagionNet(
                d_in=len(SEQ_DYNAMIC_COLS),
                n_sectors=4,
                d_model=64,
                n_heads=4,
                n_layers=2,
                dropout=0.1,
                use_graph=use_g,
            ).to(dev_name)
            lg6, lg3, inad3 = net(snap)
            assert lg6.shape == (4,)
            assert lg3.shape == (4,)
            assert inad3.shape == (4,)
            assert torch.all((inad3 >= 0.0) & (inad3 <= 1.0))

            loss = lg6.mean() + lg3.mean() + inad3.mean()
            loss.backward()
            grads_ok = [p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters() if p.requires_grad]
            assert all(grads_ok)
