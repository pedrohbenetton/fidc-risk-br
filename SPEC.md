# Especificação Técnica e Científica — `fidc-risk-br`

## 1. Motivação e Problema Real

Os **Fundos de Investimento em Direitos Creditórios (FIDCs)** constituem a espinha dorsal da securitização de crédito no Brasil (mais de R\$ 500 bilhões sob gestão, financiando desde cartões e crédito pessoal de fintechs até duplicatas industriais, agronegócio e precatórios). Apesar de comercializados no mercado de crédito privado como instrumentos com proteção estrutural via **subordinação de cotas** (cotas subordinadas absorvem as primeiras perdas antes de atingir as cotas mezanino e sênior), choques recorrentes entre 2020 e 2026 mostraram três falhas clássicas na análise tradicional de crédito:

1. **Maquiagem de inadimplência via recompras e substituições (*window-dressing*)**: cedentes ou coobrigados recompram recebíveis vencidos entre 1 e 60 dias antes que cruzem a faixa de >90 dias, mantendo o índice de inadimplência aparente baixo até que a liquidez do cedente se esgote e ocorra um salto abrupto de provisão (PDD) e corrosão da cota subordinada/sênior.
2. **Assimetria entre risco SCR e provisão efetiva**: a Tabela X do informe mensal reporta a distribuição da carteira nas faixas regulatórias AA–H da Resolução CMN/BCB nº 2.682, permitindo confrontar a provisão mínima implícita pelo critério do Banco Central com a provisão efetivamente contabilizada (`VL_REDUCAO_RECUP`).
3. **Contágio em rede via cedentes compartilhados e prestadores**: a Tabela I divulga mensalmente o CPF/CNPJ e a participação percentual dos até 18 maiores cedentes (`TAB_I2A12` com aquisição de risco e `TAB_I2B12` sem aquisição de risco). Quando um grande cedente corporativo entra em estresse, múltiplos FIDCs expostos a esse mesmo CNPJ sofrem contágio em defasagens distintas conforme o prazo de vencimento de suas carteiras.

Este projeto constrói, de ponta a ponta com dados 100% públicos e auditáveis da CVM e do Banco Central (2018–2026), um **Lakehouse Medallion**, um **motor forense de sinais contábeis e de rede**, uma **bateria de modelos tabulares calibrados, matrizes de roll-rate de Markov e redes neurais espaço-temporais / grafos heterogêneos treinadas em GPU**, e um **simulador de decisão e alocação de portfólio respeitando prazos reais de resgate**.

---

## 2. Fontes de Dados Oficiais (Zero Dados Sintéticos)

1. **CVM — Informe Mensal de FIDCs (`INF_MENSAL_FIDC`)**:
   - Histórico anual (`2018` a `2024`) em `dados.cvm.gov.br/dados/FIDC/DOC/INF_MENSAL/DADOS/HIST/` e arquivos mensais (`202501` a `202610`) em `DADOS/`.
   - 17 tabelas relacionais por competência mensal:
     - `tab_I`: Ativo, disponibilidades, carteira com/sem aquisição de risco, créditos em recuperação judicial, ação judicial, PDD (`VL_REDUCAO_RECUP`), cotas de outros FIDCs/FIDC-NP e **CPF/CNPJ + participação dos principais cedentes**.
     - `tab_II`: Segmentação setorial da carteira (industrial, imobiliário, comercial/varejo, serviços, agronegócio, financeiro/pessoal/consignado/veículos, factoring, precatórios/judicial).
     - `tab_III` e `tab_IV`: Passivo, derivativos de hedge e Patrimônio Líquido (`VL_PL`, `VL_PL_MEDIO`).
     - `tab_V` e `tab_VI`: Estrutura a termo dos vencimentos e **10 faixas de aging de inadimplência** (`até 30d`, `31–60d`, `61–90d`, `91–120d`, `121–150d`, `151–180d`, `181–360d`, `361–720d`, `721–1080d`, `>1080d`) para direitos com e sem aquisição de risco.
     - `tab_VII`: Fluxos mensais de aquisições, liquidações, **substituições (`VL_SUBST`)** e **recompras (`VL_RECOMPRA`)**.
     - `tab_IX`: Taxas de desconto praticadas na compra de direitos creditórios (mínima, média, máxima).
     - `tab_X`, `tab_X_1..7`: Distribuição da carteira nas faixas de risco **SCR AA–H**, perfil de cotistas (PF, bancos, EFPC, RPPS, FIIs), **quantidade e valor de cotas por classe/série (Sênior, Mezanino, Subordinada)**, **rentabilidade mensal realizada**, **captações/amortizações/resgates**, escada de liquidez (`0, 30, 60, 90, 180, 360, >360d`), **desempenho esperado vs. realizado** e garantias.
2. **CVM — Cadastro Estrutural RCVM 175 (`registro_fundo_classe.zip`)**:
   - Vínculo entre Fundo, Classe e Subclasse, data de adaptação à Resolução CVM 175, situação cadastral, público-alvo, condomínio, **Gestor (`CPF_CNPJ_Gestor`)**, **Administrador** e **Custodiante**.
3. **Banco Central do Brasil — API SGS (`api.bcb.gov.br`)**:
   - CDI mensal acumulado (`4391`), Meta Selic (`432`), IPCA mensal (`433`), IBC-Br (`24363`), Inadimplência SFN Total/PF/PJ (`21082`, `21083`, `21084`), Spread médio de crédito (`20783`), Endividamento das famílias (`29037`) e Comprometimento de renda (`29034`).

---

## 3. Governança Temporal, Anti-Vazamento e Reconciliação RCVM 175

### 3.1 Continuidade de Entidade (ICVM 356 $\to$ RCVM 175)
Com a entrada em vigor da Resolução CVM 175 (2023–2024), fundos passaram a reportar por `Classe` (`CNPJ_FUNDO_CLASSE` / `CNPJ_CLASSE`). A camada **Silver** constrói uma chave longitudinal determinística (`entity_id = regexp_replace(CNPJ_FUNDO_CLASSE, '[^0-9]', '')`), preservando a série histórica contínua de fundos monoclasse e tratando classes independentes sem dupla contagem.

### 3.2 Regra Estrita de Disponibilidade (*Point-in-Time / Publication Lag*)
- O informe mensal referente à competência $t$ (último dia do mês $t$) tem prazo regulatório de entrega até 15 dias após o encerramento do mês.
- Na data de decisão após o fechamento do informe de $t$:
  - Todas as variáveis de estado, lags ($t, t-1, \dots, t-11$), matrizes de roll-rate e arestas do grafo de contágio (`FIDC–Cedente–Gestor`) utilizam **exclusivamente** competências $\le t$.
  - Os alvos preditivos medem eventos estritamente futuros em $(t+1, \dots, t+h]$, **jamais** sobrepondo $t$.
  - Nas janelas de validação *expanding-window* com horizonte $h$, exige-se um **embargo temporal de $h$ meses** entre o último ponto de treino (`origin` $\le T_{\text{train}} - h$) e a origem de teste $T_{\text{train}}$, garantindo que nenhum rótulo de treino utilize realizações posteriores à data de corte!

---

## 4. Formulação das Três Tarefas Quantitativas

### Tarefa 1 — Alerta Precoce de Estresse Severo de Crédito (*Distress Onset*, $h \in \{3, 6\}$ meses)
Restringimos a coorte elegível em $t$ aos fundos **ativos, não-exclusivos de FIDC-NP estrutural e saudáveis na largada** ($\text{PL}_t \ge \text{R\$ 5 milhões}$, carteira de crédito ativa $\ge \text{R\$ 2 milhões}$, sem evento de estresse ativo em $t$). Definimos o evento binário $Y_{i, t}^{(h)} \in \{0, 1\}$ de **Entrada em Estresse Severo de Crédito** nos próximos $h$ meses se ocorrer pelo menos um dos gatilhos econômicos auditáveis em $\tau \in \{t+1, \dots, t+h\}$:
1. **Choque na Cota Sênior (Impairment / Drawdown Realizado)**: retorno acumulado da cota sênior na janela futura de até $-2,0\%$, ou algum mês com retorno de até $-1,5\%$ (em um instrumento atrelado ao CDI positivo, retorno nominal negativo representa perda severa por marcação de PDD/default), **ou**
2. **Quebra de Cobertura Estrutural / Explosão de Inadimplência**: o estoque ruim $\max(\text{Inadimplência}_{>90\text{d}}, \text{PDD/Carteira})$ sobe pelo menos $+5\text{ p.p.}$, atinge $12\%$ ou mais e deixa a folga da subordinação + mezanino sobre ele em $2\text{ p.p.}$ ou menos.

Os limiares vivem como constantes nomeadas em `src/fidc_risk/features/gold.py` (`TARGET_*`, `ELIGIBLE_*`, `ACUTE_*`), e a página de Metodologia os cita a partir delas.

Avaliamos adicionalmente a **sensibilidade a definições alternativas do alvo** (apenas drawdown da cota sênior vs. gatilho estrutural composto) para garantir que as conclusões não dependam de um limiar arbitrário.

**Métricas Primárias**:
- **PR-AUC** (Average Precision — métrica principal sob desbalanceamento),
- **ROC-AUC**,
- **Brier Score** e **ECE** (*Expected Calibration Error* em 10 quantis),
- **Recall@Top-5% Watchlist** e **Lift@Top-5%** (capacidade operacional de capturar crises reais inspecionando apenas 5% da indústria).

### Tarefa 2 — Previsão Probabilística da Trajetória de Inadimplência $>90\text{d}$ ($h \in \{1, 3, 6\}$ meses)
Previsão quantílica ($\alpha \in \{0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95\}$) da razão $\text{Inad}_{>90\text{d}, t+h} = \frac{\text{Vencidos}_{>90\text{d}, t+h}}{\text{Carteira}_{t+h}}$ comparando:
1. Persistência simples e persistência estratificada por patamar de atraso;
2. **Matriz de Transição de Roll-Rate de Markov Regularizada** ($S_t \to S_{t+1}$ nas faixas `Adimplente`, `1–30d`, `31–60d`, `61–90d`, `>90d`, ajustada por vazamento de recompras);
3. **LightGBM Quantílico Global**;
4. **Conformalized Quantile Regression (CQR)** estratificada por regime de atraso inicial;
5. **Cabeça Multi-Horizonte Neural em GPU**.

**Métricas Primárias**: **WIS** (*Weighted Interval Score*), **MAE** da mediana, e **Cobertura Empírica de 80% e 90%**.

### Tarefa 3 — Otimização de Decisão e Alocação de Portfólio de Cotas Sênior vs. CDI
Tradução das probabilidades calibradas (Platt, Isotônica e **Venn-Abers**) e intervalos conformais em políticas de investimento e monitoramento sobre o universo investível de **FIDCs abertos com cota sênior ativa**, respeitando estritamente o prazo contratual de conversão + pagamento de resgate (`PRAZO_CONVERSAO_COTA + PRAZO_PAGTO_RESGATE`):
- Comparação contra **100% CDI (SGS 4391)**, carteira *Equal-Weight*, *Yield-Chasing*, filtro tradicional de comitê de crédito, *Watchlist Cost-Sensitive* e **Alocação Penalizada por Risco Caudal / Incerteza Venn-Abers (CVaR-Aware)**.

---

## 5. Arquitetura de Software

```text
fidc-risk-br/
├── pyproject.toml              # Configuração uv, ruff, pytest, mypy e PyTorch CUDA
├── SPEC.md                     # Especificação matemática, invariantes e anti-leakage
├── README.md                   # Síntese executiva, resultados honestos e guia de reprodução
├── src/fidc_risk/
│   ├── config.py               # Caminhos, constantes regulatórias Bacen 2682 e janelas
│   ├── ingestion/              # Download assíncrono resiliente CVM (HIST+DADOS), CAD e BCB
│   ├── lakehouse/              # Bronze -> Silver em DuckDB/Parquet + invariantes contábeis
│   ├── features/               # Painel Gold point-in-time, sinais forenses e grafo LOO
│   ├── models/                 # Baselines, Logística, LightGBM/CatBoost, Roll-Rate, CQR,
│   │                           # Venn-Abers e GNN/Transformer temporal em PyTorch CUDA
│   ├── allocation/             # Estudo forense de recompras + backtest de portfólio c/ lag
│   ├── monitoring/             # SHAP, PSI / Jensen-Shannon drift e diagnóstico por regime
│   ├── export/                 # Geração de figuras de publicação e pacotes JSON p/ web
│   └── api/                    # Serviço FastAPI sobre DuckDB + modelos calibrados
├── web/                        # Observatório interativo (Radar, Grafo de Contágio, Simulador)
├── reports/                    # Relatórios científicos por fase + CSVs de métricas + figuras
└── tests/                      # Suíte completa (145 testes: invariantes, anti-leakage, API)
```
