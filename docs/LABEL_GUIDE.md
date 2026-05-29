# Human Label Guide

Use this guide when annotating the CZE-NEC evaluation articles. Label from the
article content only: title, summary, and body excerpt.

## General Rules

- Prefer the article's main point over one isolated sentence.
- Use `notes` when a label feels ambiguous or the excerpt is too short.
- If two labels seem possible, choose the more conservative one.
- Segment labels are independent. More than one segment can be strongly relevant.
- Do not infer public opinion. Label what the article says or frames.

## `topic_relevance`

How directly the article is about the target topic.

- `0 = irrelevant`: the topic is absent or only a coincidental word match.
- `1 = related`: the topic is present but not the main subject.
- `2 = core`: the topic is central to the article.

Examples:

- `2`: an article about inflation rate changes, energy prices, elections, or hospital funding when that is the target topic.
- `1`: an article about household budgets that briefly mentions inflation.
- `0`: an article where `energie` appears only as a sports metaphor or company name unrelated to energy.

## `dominant_frame`

The main framing lens of the article.

- `fear`: risk, harm, danger, loss, uncertainty, threat, anxiety.
- `opportunity`: benefit, improvement, savings, growth, solution, positive change.
- `conflict`: dispute, blame, protest, political fight, institutional clash.
- `neutral`: factual update, explainer, routine report, balanced coverage.
- `other`: clear dominant frame exists but does not fit the above.

Examples:

- `fear`: "Prices may rise sharply and households will struggle."
- `opportunity`: "New subsidies may reduce household energy costs."
- `conflict`: "Government and opposition clash over healthcare reform."
- `neutral`: "Inflation fell to 3.2 percent, statistics office reports."

## `concern_bucket`

How much concern or worry the article content would reasonably signal.

- `low`: little or no worry; routine, optimistic, or mild impact.
- `medium`: concrete concern exists, but not severe or urgent.
- `high`: strong risk, urgency, crisis, major harm, or widespread anxiety.

Examples:

- `low`: inflation is stable or falling; officials expect no major impact.
- `medium`: prices rose and may affect household budgets.
- `high`: severe shortages, major price shocks, hospital collapse, large protests.

## `purchase_relevance`

Whether the article is relevant to buying, spending, investment, subscriptions,
borrowing, switching providers, or other market action.

- `0 = none`: no meaningful purchase or spending implication.
- `1 = weak`: indirect or minor purchase implication.
- `2 = strong`: buying/spending/investment behavior is central.

Examples:

- `2`: "How to protect savings from inflation"; mortgage or energy-tariff decisions.
- `1`: inflation affects prices but the article is mainly macroeconomic.
- `0`: parliamentary procedure with no consumer or business action.

## `avoidance_relevance`

Whether the article suggests avoiding, delaying, reducing, rejecting, boycotting,
or steering clear of something.

- `0 = none`: no avoidance behavior implied.
- `1 = weak`: indirect or minor avoidance implication.
- `2 = strong`: avoidance behavior is central.

Examples:

- `2`: people avoid travel, products, hospitals, loans, or public places because of risk or cost.
- `1`: article mentions consumers may reduce spending.
- `0`: neutral statistics report with no avoidance implication.

## Segment Relevance Fields

Use the same scale for each segment:

- `0 = not relevant`: no clear special relevance to this segment.
- `1 = somewhat relevant`: plausible relevance, but not focused on the segment.
- `2 = strongly relevant`: the segment is directly addressed or highly affected.

### `seg_young_urban_relevance`

Young adults, students, renters, city residents, commuters, digital-first
consumers, culture/nightlife users.

Examples:

- `2`: student housing, city rents, public transport, urban jobs.
- `1`: general household inflation with some relevance to young workers.
- `0`: pension indexation with no youth or urban angle.

### `seg_family_relevance`

Parents, households with children, school-related costs, family budgets,
housing, childcare, groceries, energy bills.

Examples:

- `2`: food prices, school costs, family benefits, childcare.
- `1`: general consumer prices affecting households.
- `0`: business regulation with no household angle.

### `seg_senior_relevance`

Pensioners, retirement income, healthcare access, medicine, elder care,
fixed-income vulnerability.

Examples:

- `2`: pension changes, medicine costs, hospital access for elderly people.
- `1`: inflation affects fixed incomes but seniors are not central.
- `0`: startup investment story with no senior relevance.

### `seg_b2b_relevance`

Companies, employers, industry, supply chains, regulation, business costs,
investment, professional decision-makers.

Examples:

- `2`: energy costs for manufacturers, employer taxes, business regulation.
- `1`: macroeconomic policy that may affect companies.
- `0`: family grocery spending with no business angle.

## `notes`

Use notes for:

- unclear excerpt,
- mixed frame,
- label definition problem,
- article too short,
- disagreement-worthy edge case.

Keep notes short and concrete.
