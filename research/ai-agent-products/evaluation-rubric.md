# AI Agent Products Evaluation Rubric

## Overview

This rubric provides a systematic framework for evaluating AI agent product ideas based on market viability, technical complexity, time investment, and competitive landscape.

---

## Evaluation Dimensions

### 1. Market Opportunity (1-5)
How large and accessible is the target market?

| Score | Description |
|-------|-------------|
| 1 | Niche market, <$10M TAM |
| 2 | Small market, $10-50M TAM |
| 3 | Medium market, $50-500M TAM |
| 4 | Large market, $500M-5B TAM |
| 5 | Massive market, >$5B TAM |

### 2. Technical Difficulty (1-5)
How hard is it to build a working MVP?

| Score | Description |
|-------|-------------|
| 1 | Simple - API wrappers, basic CRUD, standard integrations |
| 2 | Easy - Some AI integration, single-platform focus |
| 3 | Moderate - Multi-platform, complex AI workflows, real-time requirements |
| 4 | Hard - Deep integrations, complex state management, high reliability needs |
| 5 | Very Hard - Novel AI capabilities, infrastructure-heavy, regulatory compliance |

### 3. Time to MVP (Estimated)
How long to build a functional product?

| Category | Timeline |
|----------|----------|
| Quick | 1-2 weeks |
| Fast | 2-4 weeks |
| Moderate | 1-2 months |
| Long | 2-4 months |
| Extended | 4+ months |

### 4. Barrier to Entry (1-5)
How defensible is the product against competition?

| Score | Description |
|-------|-------------|
| 1 | Very Low - Easy to clone, commoditized space |
| 2 | Low - Some differentiation possible, but many competitors |
| 3 | Moderate - Requires specific expertise or integrations |
| 4 | High - Network effects, proprietary data, or complex tech moat |
| 5 | Very High - Strong moats, regulatory barriers, or first-mover lock-in |

### 5. Revenue Potential (1-5)
What's the realistic revenue opportunity for a solo founder?

| Score | Description |
|-------|-------------|
| 1 | Low - <$5K MRR ceiling |
| 2 | Modest - $5-20K MRR ceiling |
| 3 | Good - $20-100K MRR ceiling |
| 4 | Strong - $100K-500K MRR ceiling |
| 5 | Excellent - $500K+ MRR ceiling |

### 6. Customer Acquisition Ease (1-5)
How easy is it to find and convert customers?

| Score | Description |
|-------|-------------|
| 1 | Very Hard - Long sales cycles, enterprise-only, requires trust |
| 2 | Hard - Significant education needed, competitive paid acquisition |
| 3 | Moderate - Clear value prop, multiple channels available |
| 4 | Easy - Viral potential, SEO-friendly, word-of-mouth |
| 5 | Very Easy - Built-in distribution, immediate value demonstration |

---

## Composite Scoring

### Attractiveness Score Formula
```
Attractiveness = (Market × 0.20) + (5 - Difficulty) × 0.15 + TimeScore × 0.15 +
                 (5 - BarrierEntry) × 0.10 + Revenue × 0.25 + Acquisition × 0.15
```

*Note: Lower barrier to entry is favorable for quick entry but unfavorable for defensibility. Adjust weighting based on strategy.*

### Time Score Conversion
| Timeline | Score |
|----------|-------|
| Quick (1-2 weeks) | 5 |
| Fast (2-4 weeks) | 4 |
| Moderate (1-2 months) | 3 |
| Long (2-4 months) | 2 |
| Extended (4+ months) | 1 |

---

## Research Checklist

For each product, investigate:

- [ ] **Existing competitors** - Who's already solving this?
- [ ] **Pricing models** - What do competitors charge?
- [ ] **Customer pain points** - Reddit, Twitter, forums feedback
- [ ] **Technical stack** - What APIs/services are needed?
- [ ] **Distribution channels** - How will customers find this?
- [ ] **Regulatory concerns** - Any compliance requirements?
- [ ] **Integration complexity** - How many third-party services?
- [ ] **AI requirements** - Simple prompts vs fine-tuned models?

---

## Quick Reference Matrix

| Dimension | Weight | Ideal Target |
|-----------|--------|--------------|
| Market Opportunity | 20% | 3-5 (avoid tiny niches) |
| Technical Difficulty | 15% | 1-3 (manageable for solo) |
| Time to MVP | 15% | Quick-Moderate |
| Barrier to Entry | 10% | 2-4 (balance) |
| Revenue Potential | 25% | 3-5 (worth the effort) |
| Acquisition Ease | 15% | 3-5 (sustainable growth) |

---

## Category Tags

Products can be tagged for quick filtering:

- **Infrastructure** - APIs, developer tools, platform services
- **Marketing** - SEO, social, content, PR tools
- **Operations** - Monitoring, analytics, internal tools
- **Communication** - Email, chat, support
- **Productivity** - Project management, automation
- **Finance** - Accounting, legal, payments
- **Content** - Generation, curation, directories

---

## Risk Factors

Additional considerations that may affect evaluation:

| Risk Factor | Impact |
|-------------|--------|
| API dependency | High - subject to pricing changes, rate limits |
| Platform risk | Medium - dependent on third-party platforms |
| AI accuracy requirements | Variable - some domains tolerate errors better |
| Data sensitivity | High - healthcare, finance, legal have compliance needs |
| Enterprise vs SMB | Different sales cycles and support needs |
