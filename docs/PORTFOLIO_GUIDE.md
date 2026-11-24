# Guia de Portfólio, Publicação e Entrevistas Técnicas — `fidc-risk-br`

Este guia foi preparado para você publicar o projeto em 2 minutos ao acordar, atualizar seu perfil do GitHub / currículo e dominar qualquer entrevista técnica de estágio em **Ciência de Dados, Engenharia de ML, Engenharia de Dados ou Quant/Crédito** (Nubank, Itaú, BTG Pactual, Creditas, Stone, Mercado Livre/Mercado Pago, iFood, PicPay, CloudWalk, B3, XP).

---

## 1. Como Publicar no GitHub + Colocar o Observatório no Ar (demo anonimizado)

1. Crie um repositório público vazio no GitHub chamado **`fidc-risk-br`** (sem README inicial).
2. No terminal dentro desta pasta (`e:\Faculdade\Projetos\Megazord`), rode:
   ```powershell
   git remote add origin https://github.com/pedrohbenetton/fidc-risk-br.git
   git push -u origin main
   ```
3. O que vai a público é só o **build demo**, com fundos anonimizados (códigos estáveis, PL em faixas):
   ```powershell
   .\.venv\Scripts\fidc-risk.exe export-demo
   npx vercel deploy dist/demo --prod
   ```
   Não publique a pasta `web/` inteira de uma máquina que rodou o pipeline, nem ative o GitHub Pages
   apontando para ela. Detalhes na seção "Demo anonimizado (Vercel)" do README.
   - Para usar localmente com os fundos identificados e a **API FastAPI + DuckDB + Simulação Contrafactual Ao Vivo**:
     ```powershell
     .\.venv\Scripts\fidc-risk.exe serve
     # Acesse http://127.0.0.1:8000 (Observatório Web) e http://127.0.0.1:8000/docs (OpenAPI)
     ```

---

## 2. Trecho Pronto para Adicionar ao `pedrohbenetton/pedrohbenetton/README.md`

```markdown
### [fidc-risk-br](https://github.com/pedrohbenetton/fidc-risk-br)
Alerta precoce de estresse de crédito, contágio em grafos e alocação sob incerteza para os **6.146 FIDCs brasileiros** (2018–2026, 215 mil observações mensais da CVM + séries do Banco Central). Lakehouse Medallion em DuckDB/Parquet reconciliando a transição regulatória da RCVM 175, sinais forenses de maquiagem via recompras e hiato de provisão frente à Resolução BCB 2.682, calibração multiprobabilística de Venn-Abers, regressão quantílica conformal (CQR) e redes espaço-temporais em GPU (5 sementes). Inclui o que não funcionou: variáveis macroeconômicas em nível degradam árvores fora da amostra por atuarem como proxy espúrio de data calendário.
```

---

## 3. Bullets Prontos para o Currículo (PT-BR e Inglês)

### Versão em Português (`Currículo Estágio Dados / ML`)
- **`fidc-risk-br` — Risco de Crédito Estruturado, Contágio em Grafos e Alocação (Python, DuckDB, LightGBM, CatBoost, PyTorch CUDA, FastAPI)**: Construiu Lakehouse Medallion sobre 106 competências da CVM e Banco Central (215 mil fundos-mês, 135 mil arestas corporativas `FIDC–Cedente`), reconciliando schemas ICVM 356 → RCVM 175 e elevando a cobertura de gestores de 23,8% para 99,9% ao corrigir corrupção de zeros à esquerda em CNPJs. Em validação *walk-forward* com embargo estrito de 6 meses (98 mil previsões), elevou o PR-AUC de alerta precoce de 0,276 (regra de comitê) para 0,377+ com calibração Venn-Abers (`ECE < 0,8%`), provou empiricamente o risco de maquiagem via recompras e simulou alocação de cotas sênior contra o CDI respeitando prazos contratuais de resgate.

### English Version (`ML / Data Science / Data Engineering Intern Resume`)
- **`fidc-risk-br` — Structured Credit Risk Lakehouse, Contagion GNN & Portfolio Allocation (Python, DuckDB, LightGBM, CatBoost, PyTorch CUDA, FastAPI)**: Engineered a survivorship-bias-free Medallion Lakehouse across 9 years of Brazilian SEC (CVM) securitization filings and Central Bank series (215k fund-months, 6,146 funds, 135k corporate originator edges), enforcing automated accounting invariants (99.99% pass rate) and adversarial anti-leakage tests. Improved 6-month distress early-warning PR-AUC from 0.276 to 0.377+ (`ECE = 0.0079` via Inductive Venn-Abers) across 4 macro regimes, benchmarked 5-seed GPU Temporal Transformers & Leave-One-Out Contagion GNNs, and built a redemption-lag-aware CVaR portfolio simulator + interactive web observatory.

---

## 4. Roteiro de 3 Minutos para Entrevistas Técnicas (Perguntas & Respostas)

1. **"Por que você escolheu FIDCs da CVM em vez de um dataset pronto do Kaggle?"**
   - *Resposta*: Datasets prontos como LendingClub ou `creditcard.csv` já vêm limpos em uma tabela estática e não refletem os problemas reais de times de crédito no Brasil. Na CVM, cada mês tem 18 tabelas relacionais, houve mudança regulatória no meio da série (RCVM 175 em 2023–2024), existem fundos liquidados (viés de sobrevivência se você só olhar o cadastro ativo) e há fenômenos contábeis reais do mercado brasileiro — como cedentes recomprando títulos vencidos de 30–60 dias antes que cruzem 90 dias, ou divergência entre o risco SCR AA–H do Banco Central e a PDD provisionada.
2. **"Como você garantiu que não houve data leakage?"**
   - *Resposta*: Em três níveis testados no `pytest`: (i) perturbação adversarial do futuro — se eu corromper todos os dados após $t_0$, nenhuma das 103 features em $t \le t_0$ muda nem em $10^{-12}$; (ii) agregações de grafo *Leave-One-Out* estritamente dentro de cada mês $t$, subtraindo algebricamente o próprio fundo do soma do cedente/gestor; e (iii) embargo temporal igual ao horizonte $h$ entre treino e teste (para testar em jan/2023 com alvo de 6 meses, o treino para em jun/2022, pois o rótulo de jun/2022 usa realizações até dez/2022).
3. **"Qual foi a descoberta mais contraintuitiva do projeto?"**
   - *Resposta*: Na ablação das 6 famílias de features, todas as variáveis microeconômicas e de rede melhoraram o PR-AUC fora da amostra (`0,2858 → 0,3575`), mas adicionar séries macroeconômicas em nível (`Selic`, `IBC-Br`, `Inadimplência SFN`) **piorou** o modelo em `-1,60 p.p.`. Como a Selic é constante para todos os fundos num mesmo mês, árvores de decisão usam `Selic > X%` como identificador disfarçado de data calendário, memorizando taxas base passadas que falham quando o ciclo monetário vira.
