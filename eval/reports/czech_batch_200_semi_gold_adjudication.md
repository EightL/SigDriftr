# CZE Batch 200 Semi-Gold Adjudication

Generated from `eval/annotations/czech_batch_200_human.md` plus manual rereading of the lowest-agreement/high-impact cases and the article excerpts in `eval/annotations/czenec_batch_200.csv`.

## Baseline Rule

The semi-gold baseline defaults to the human annotation. Labels were changed only when rereading the article made a narrower decision clearly better than the original human label or the four-model consensus.

## Output

- Baseline CSV: `eval/annotations/czech_batch_200_semi_gold.csv`
- Rows: 200
- Reviewed articles: 26
- Articles with changed labels: 6
- Changed fields: 12

## Changed Labels

| article_id | title | field | human | semi_gold | reason | model labels |
|---|---|---|---|---|---|---|
| `047c1d315d283f32` | Stávka energetiků zažehnána | `avoidance_relevance` | `1` | `0` | Strike risk was avoided institutionally, but no consumer/user avoidance behavior is described. | chatgpt=0, deepseek=0, gemma=0, claude=0 (counts: {'0': 4}) |
| `047c1d315d283f32` | Stávka energetiků zažehnána | `purchase_relevance` | `2` | `0` | The defused energy strike affects wages/contracts, not a purchase or bill decision. | chatgpt=1, deepseek=0, gemma=0, claude=0 (counts: {'1': 1, '0': 3}) |
| `047c1d315d283f32` | Stávka energetiků zažehnána | `seg_family_relevance` | `1` | `0` | No household/family-specific angle appears; the article is mainly energy-sector labor/B2B. | chatgpt=1, deepseek=1, gemma=0, claude=0 (counts: {'1': 2, '0': 2}) |
| `047c1d315d283f32` | Stávka energetiků zažehnána | `seg_young_urban_relevance` | `1` | `0` | No youth/urban-specific angle appears. | chatgpt=0, deepseek=0, gemma=0, claude=0 (counts: {'0': 4}) |
| `06f7e52f6478a183` | Půjčky jsou zatím díky inflaci výhodné | `topic_relevance` | `1` | `2` | Inflation is not just incidental; the advice hinges on taking loans while inflation erodes debt value. | chatgpt=2, deepseek=2, gemma=2, claude=2 (counts: {'2': 4}) |
| `0c33263d15025ddb` | Růst mezd nevyvolá inflaci | `concern_bucket` | `medium` | `low` | The article says wage growth does not create inflation pressure; risk is downplayed. | chatgpt=low, deepseek=low, gemma=low, claude=low (counts: {'low': 4}) |
| `0c33263d15025ddb` | Růst mezd nevyvolá inflaci | `purchase_relevance` | `1` | `0` | No concrete buying, borrowing, spending, or bill-payment decision is present. | chatgpt=0, deepseek=0, gemma=0, claude=0 (counts: {'0': 4}) |
| `0c33263d15025ddb` | Růst mezd nevyvolá inflaci | `topic_relevance` | `1` | `2` | Wage growth is explicitly judged through inflation pressure, so inflation is central. | chatgpt=2, deepseek=2, gemma=2, claude=2 (counts: {'2': 4}) |
| `80e932822fdeda63` | Ode dneška je jízda vlakem dražší - další zdražování zřejmě v červnu | `seg_senior_relevance` | `2` | `1` | Senior passes are mentioned as mostly unaffected; relevant but not a strong senior-impact story. | chatgpt=1, deepseek=1, gemma=1, claude=1 (counts: {'1': 4}) |
| `cdc3b2bb7bbc1665` | Osud neplatičů za vodné a energii? Upomínky, odpojení a soudní spor | `topic_relevance` | `1` | `2` | Utility nonpayment, disconnection, and court action make energy/bills the core article topic. | chatgpt=2, deepseek=2, gemma=2, claude=2 (counts: {'2': 4}) |
| `ed62176ea13f9924` | Islámští rebelové obsadili nemocnici ve filipínském Lamitanu | `avoidance_relevance` | `2` | `0` | The article describes danger, not a reader/person choosing to avoid, delay, stop, or switch behavior. | chatgpt=2, deepseek=0, gemma=0, claude=0 (counts: {'2': 1, '0': 3}) |
| `ed62176ea13f9924` | Islámští rebelové obsadili nemocnici ve filipínském Lamitanu | `topic_relevance` | `2` | `1` | A hospital is occupied, but the story is primarily armed conflict rather than healthcare-system behavior. | chatgpt=1, deepseek=1, gemma=1, claude=1 (counts: {'1': 4}) |

## Reviewed But Kept As Human

| article_id | title | decision |
|---|---|---|
| `13c1b2cf3d7cc4da` | Vláda schválila akční plán zaměstnanosti | Kept human: consumer-credit protection and employment support are explicit policy actions. |
| `160cee6a3e7f1eae` | Fišer dá Motolu do konce roku pár miliónů | Kept human: funding fixes poor conditions in a children's transplant unit; impact is direct. |
| `169d18ceb8e68d85` | Ceny benzinu se na Vysočině ustálily | Kept human: fuel prices are central, but inflation itself is only indirect. |
| `1896ae5afbbbbf8b` | Vláda schválila koncepci důchodové reformy | Kept human: pension reform is directly about public benefits/contributions. |
| `2fa9a1f5af8c9a92` | A. Grebeníček bude zřejmě souzen v nemocnici | Kept human: poor health is used to delay/change trial logistics, so avoidance is legitimate. |
| `31849718b4718734` | Míry nezaměstnanosti a inflace rostou | Kept human: unemployment plus the highest inflation in years justifies the risk framing. |
| `367881a222a612f1` | Elektřina bude výrazně dražší | Kept human: electricity price rise directly affects household bills; models split on action/segments. |
| `3ae4df437ea80459` | Při pitvě lékař údajně našel dvě střely | Kept human: the doctor/autopsy mention is a crime-investigation detail, not a healthcare topic. |
| `403db6f109c4ce5f` | Václav Havel leží v Ústřední vojenské nemocnici v Praze | Kept human: hospitalization and possible pneumonia carry a health-risk frame. |
| `4ac0dfd9c8684d59` | Od ledna znovu porostou náklady na energie | Kept human: household electricity/gas increases are direct bill pressure. |
| `5ad2d2f2c8c89356` | Vláda zavedla víza pro Kanaďany | Kept human: the article mixes visas, school milk, and mortgage support; human labels best preserve that mix. |
| `5b9b572c0519912b` | Maďarsko by mohla trápit inflace | Kept human: inflation is one concern in a broader Hungary/EU report; models over-focused the keyword. |
| `661c1ff03b39f5ec` | Izraelská vláda národní jednoty zatím není | Kept human: violence and failed government talks justify high concern. |
| `74699a19cfa01f70` | Levná ropa dělá Rusům starosti | Kept human: low oil prices threaten Russia's budget, so high concern is justified. |
| `77adf576deb5f275` | Zdražení nemusí být podle vzoru Český Telecom | Kept human: telecom price hikes are present, but inflation is not visible in the provided text. |
| `d183b3876169cfbc` | Povinné ručení příští rok podraží | Kept human: mandatory insurance price increases are a direct household/driver cost. |
| `d9dfb20c681c5288` | Ceny rostly v listopadu jen minimálně | Kept human: minimal price growth and cheaper fuel are framed as reassuring/opportunity. |
| `da5ae3b988c2012f` | Afghánci si stěžují na nedostatek elektřiny | Kept human: sharply restricted electricity supply is the dominant concern despite price cuts. |
| `f5011fc19ba7110c` | Premiéři ČR a Rakouska se možná sejdou v prosinci | Kept human: anti-nuclear border blockades are direct avoidance/obstruction behavior. |
| `f5dfc440db876e96` | Kosmonaut: Sex na palubě je tabu! | Kept human: space-medicine uncertainty explains the ban; the article is niche but labelable. |

## Adjudication Notes

- Treat 4/4 model agreement as useful evidence, not truth. Several 4/4 disagreements were model over-inference from keywords.
- For `topic_relevance`, require the canonical topic to shape the article, not merely appear as a word.
- For `purchase_relevance`, utility bills, fares, loans, insurance, mortgages, and direct price/payment changes count as concrete economic action.
- For `avoidance_relevance`, danger alone is not enough; there must be avoid, delay, stop, block, restrict, switch, or protection behavior.
- Segment labels should be explicit or strongly implied by affected actors, not inferred from generic public relevance.
