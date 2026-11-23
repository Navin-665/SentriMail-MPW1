# SentriMail / SentriFormer — Data Sources Manifest

> **Purpose**: Document every raw dataset used in the SentriFormer pipeline with provenance, license, size, and intended role.
> **Last Updated**: 2026-09-01

---

## 1. CFPB Consumer Financial Complaints

| Field | Value |
|:------|:------|
| **Source** | Consumer Financial Protection Bureau (US Government) |
| **URL** | https://www.consumerfinance.gov/data-research/consumer-complaints/ |
| **License** | Public Domain (US Government Work) |
| **Local Path** | `rawdata/complaints.csv` |
| **File Size** | 9.18 GB |
| **Total Rows** | 17,281,869 |
| **Usable Narratives** | 3,841,422 (22.2% have text) |
| **Unique Narratives (≥50 chars)** | 2,579,403 (32.6% duplicate rate) |
| **Key Columns** | Product, Sub-product, Issue, Sub-issue, Consumer complaint narrative |
| **Taxonomy** | 21 Products → 86 Sub-products → 178 Issues |
| **Text Length** | Mean 188 words / Median 128 words |
| **Token Estimate** | Mean ~253 subword tokens |
| **Role in SentriFormer** | ★★★★★ Core pretraining corpus + hierarchical financial classification labels |

---

## 2. TWCS — Twitter Customer Support Conversations

| Field | Value |
|:------|:------|
| **Source** | Thought.Works Customer Support Dataset (Kaggle) |
| **URL** | https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter |
| **License** | CC BY-NC-SA 4.0 |
| **Local Path** | `rawdata/twcs.csv` |
| **File Size** | 516 MB |
| **Total Tweets** | 2,811,774 |
| **Inbound (Customer)** | 1,537,843 (54.7%) |
| **Outbound (Brand)** | 1,273,931 (45.3%) |
| **Unique Brands** | 108 handles (Top: @AmazonHelp, @AppleSupport, @Uber_Support) |
| **Conversations Initiated** | 787,346 top-of-thread complaints |
| **Brand Reply Rate** | 90.7% (714,245 matched responses) |
| **Boilerplate/DM Deflection** | 53.3% of replies are deflections |
| **Usable Clean Pairs** | 325,132 (45.5% of all paired interactions) |
| **Text Length** | Customer: ~20 words avg / Brand: ~20 words avg |
| **Role in SentriFormer** | ★★★★☆ Conversational customer-support language + dialogue pairs |

---

## 3. Bitext Customer Support LLM Training Dataset

| Field | Value |
|:------|:------|
| **Source** | Bitext (HuggingFace Hub) |
| **URL** | https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset |
| **License** | Apache 2.0 |
| **Local Path** | Downloaded via `datasets` library (HuggingFace) |
| **Total Examples** | 26,872 |
| **Columns** | flags, instruction, category, intent, response |
| **Categories** | 11 (ACCOUNT, ORDER, REFUND, INVOICE, PAYMENT, CONTACT, FEEDBACK, DELIVERY, SHIPPING, SUBSCRIPTION, CANCEL) |
| **Instruction Length** | Mean 46.9 chars (~8 words) |
| **Response Length** | Mean 634.1 chars (~100 words) |
| **Role in SentriFormer** | ★★★☆☆ Intent/response linguistic variation + chatbot response training |

---

## 4. Customer Support Data (Kaggle — Shopzilla/E-commerce)

| Field | Value |
|:------|:------|
| **Source** | Akash Bommidi (Kaggle) |
| **URL** | https://www.kaggle.com/datasets/akashbommidi/customer-support-data |
| **License** | MIT |
| **Local Path** | `rawdata/archive (1)/Customer_support_data.csv` |
| **File Size** | 20.2 MB |
| **Total Rows** | 85,907 |
| **Columns** | 20 (Unique id, channel_name, category, Sub-category, Customer Remarks, Order_id, order_date_time, Issue_reported at, issue_responded, Survey_response_Date, Customer_City, Product_category, Item_price, connected_handling_time, Agent_name, Supervisor, Manager, Tenure Bucket, Agent Shift, CSAT Score) |
| **Categories** | 12 (Returns, Order Related, Refund Related, Product Queries, Shopzilla Related, Payments related, Feedback, Cancellation, Offers & Cashback, Others, App/website, Onboarding related) |
| **Sub-categories** | 57 unique |
| **Customer Remarks (non-null)** | 28,742 (33.5%) |
| **Unique Remarks** | 18,231 (36.6% duplicate rate) |
| **Remark Length** | Mean 43.4 chars / Median 19 chars (~7.5 words avg) |
| **CSAT Score** | 1–5 scale (Mean 4.24, Median 5.0) |
| **Channels** | Inbound (68k), Outcall (15k), Email (3k) |
| **Product Categories** | 9 (Electronics, Lifestyle, Books, Mobile, Home, Furniture, etc.) |
| **Role in SentriFormer** | ★★★★★ Short complaint language + issue hierarchy + CSAT satisfaction signal |

### Usage Strategy for Dataset #4

```
Available AT complaint time          Generated AFTER handling
─────────────────────────────       ──────────────────────────────
Customer Remarks                    connected_handling_time
channel_name                        issue_responded
category                            Survey_response_Date
Sub-category                        Agent_name
Product_category                    Supervisor / Manager
Item_price                          CSAT Score (TARGET)
Order info
Issue_reported at
```

> **IMPORTANT**: CSAT Score is the **target variable**, not a pretraining feature.
> Do NOT give the model post-hoc information (response time, agent identity, etc.)
> when predicting satisfaction — that would constitute **target leakage**.

---

## Combined SentriCorpus Budget

| Source | Pretraining Texts | Supervised Labels | Response Pairs | CSAT Signal |
|:-------|------------------:|------------------:|---------------:|:-----------:|
| CFPB | 2,579,403 | ✅ (Product/Issue hierarchy) | — | — |
| TWCS | 325,132 | — | ✅ (dialogue pairs) | — |
| Bitext | 26,872 | ✅ (intent/category) | ✅ (instruction→response) | — |
| Customer Support | 18,231 | ✅ (category/sub-category) | — | ✅ (CSAT 1–5) |
| **Total** | **~2,949,638** | | | |

**Estimated total subword tokens**: ~600 Million

---

## Licensing Summary

| Dataset | License | Commercial Use | Attribution Required |
|:--------|:--------|:--------------:|:--------------------:|
| CFPB | Public Domain | ✅ | No |
| TWCS | CC BY-NC-SA 4.0 | ❌ (non-commercial) | Yes |
| Bitext | Apache 2.0 | ✅ | Yes |
| Customer Support | MIT | ✅ | Yes |

> **Note**: TWCS is CC BY-NC-SA which restricts commercial use. For academic/research purposes this is fine. If SentriMail were to be commercialized, TWCS data would need to be replaced or re-licensed.

---

*This manifest should be updated whenever a new dataset is added to the pipeline.*
