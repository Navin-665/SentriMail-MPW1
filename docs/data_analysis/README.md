# Raw-data analysis (EDA)

Exploratory audits of the four raw datasets, made on 2026-09-01 before SentriCorpus was built. They contain aggregate
counts and distributions only (no complaint or tweet text).

| Figure | Shows |
|---|---|
| `01_cfpb_product_and_narrative_audit.png` | CFPB product volumes, narrative availability, de-duplication |
| `02_cfpb_token_length_distribution.png` | Complaint length in subword tokens vs 256/512 context cut-offs |
| `03_twcs_response_quality_audit.png` | TWCS brand volumes, customer/brand ratio, reply-quality filter |
| `04_customer_support_audit.png` | E-commerce dataset categories, CSAT distribution, remark availability |
| `04/06_dataset_composition_comparison.png` | Raw vs cleaned vs planned corpus volumes |
| `05_csat_vs_remark_length.png` | Dissatisfied customers write longer remarks |
| `05_dataset_audit_dashboard.png`, `08_executive_dashboard.png` | Summary dashboards |
| `07_sentriformer_pipeline_architecture.png` | Planned data and model pipeline |

Note: these are planning-stage figures. Targets shown here (for example "~2.95M examples", "12–20M parameters") differ
from the final corpus (3,538,711 records) and model (11,654,698 parameters). See `research_publications/` for the
verified final numbers.
