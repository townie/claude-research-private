# Metrics & Analytics Deep Dive

## The Data-Driven Operator

Success in the Eyeballs Method isn't about guessing—it's about measuring, testing, and optimizing. This document covers every metric you need to track and how to use data to make decisions.

## The Metrics Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                    NORTH STAR METRICS                       │
│        Net Profit / Day   |   Return on Time Invested       │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────┼─────────────────────────────┐
│           REVENUE METRICS   │    TRAFFIC METRICS          │
│  • Daily Revenue            │  • Daily Views              │
│  • AOV                      │  • Daily Clicks             │
│  • LTV                      │  • CTR                      │
│  • Refund Rate              │  • Views/Post Average       │
└─────────────────────────────┼─────────────────────────────┘
                              │
┌─────────────────────────────┼─────────────────────────────┐
│         CONTENT METRICS     │    FUNNEL METRICS           │
│  • Posts/Day                │  • Landing Page CVR         │
│  • Engagement Rate          │  • Bump Rate                │
│  • Watch Time               │  • OTO Take Rates           │
│  • Viral Rate               │  • Email Opt-in Rate        │
└─────────────────────────────┴─────────────────────────────┘
```

## Traffic Metrics (Top of Funnel)

### Views

**Definition:** Total video views across all accounts

**Tracking:**
- Daily total views
- Views per account
- Views per niche
- Views per content type

**Benchmarks:**
| Stage | Daily Views Target |
|-------|-------------------|
| Testing | 50k-100k |
| Validation | 100k-500k |
| Growth | 500k-2M |
| Scale | 2M-10M+ |

**Analysis Questions:**
- Which accounts drive most views?
- Which content types get most views?
- What times produce best views?
- Which hooks are working?

### Click-Through Rate (CTR)

**Definition:** (Bio Clicks ÷ Views) × 100

**Formula:**
```
CTR = (Link Clicks / Total Views) × 100
```

**Benchmarks:**
| CTR | Grade | Action |
|-----|-------|--------|
| <0.5% | F | Major CTA/offer problem |
| 0.5-1% | D | Improve hooks and CTAs |
| 1-2% | C | Acceptable, optimize |
| 2-3% | B | Good performance |
| 3%+ | A | Scale aggressively |

**Optimization Levers:**
1. Stronger CTAs in content
2. More curiosity-driven hooks
3. Better bio optimization
4. Landing page promise alignment

### Engagement Rate

**Definition:** (Likes + Comments + Shares) ÷ Views × 100

**Benchmarks:**
| Engagement | Quality |
|------------|---------|
| <2% | Low |
| 2-5% | Average |
| 5-10% | Good |
| 10%+ | Excellent |

**Why It Matters:**
- Higher engagement = more algorithm push
- Comments indicate content resonance
- Shares = organic viral distribution

### Watch Time / Retention

**Definition:** Average percentage of video watched

**Benchmarks:**
| Watch Time | Grade |
|------------|-------|
| <30% | Poor - hook failing |
| 30-50% | Average |
| 50-70% | Good |
| 70%+ | Excellent |

**Optimization:**
- Front-load value
- Pattern interrupts every 3-5 seconds
- Loop endings back to beginning
- Keep videos under 60 seconds initially

### Viral Coefficient

**Definition:** Videos hitting 100k+ views ÷ Total videos posted

**Target:** 5-10% of videos should hit 100k+

**Formula:**
```
Viral Rate = (Videos with 100k+ views / Total videos) × 100
```

## Funnel Metrics (Middle of Funnel)

### Landing Page Conversion Rate

**Definition:** (Opt-ins or Purchases) ÷ Landing Page Visitors × 100

**Benchmarks (Opt-in Pages):**
| CVR | Grade |
|-----|-------|
| <20% | Poor |
| 20-30% | Average |
| 30-40% | Good |
| 40%+ | Excellent |

**Benchmarks (Sales Pages):**
| CVR | Grade |
|-----|-------|
| <1% | Poor |
| 1-2% | Average |
| 2-4% | Good |
| 4%+ | Excellent |

### Order Bump Rate

**Definition:** Customers who add bump ÷ Total customers × 100

**Target:** 30-50%

**Optimization:**
- Bump relevance to main offer
- Price point (30-50% of front-end)
- Copy clarity (one-sentence benefit)
- Visual prominence

### OTO Take Rates

**Definition:** Customers who purchase OTO ÷ Customers who see OTO × 100

**Benchmarks:**
| OTO | Target Take Rate |
|-----|------------------|
| OTO1 | 10-20% |
| OTO2 | 8-15% |
| OTO3+ | 5-10% |

**Optimization:**
- Video sales letter quality
- Offer irresistibility
- Urgency/scarcity elements
- Price anchoring

### Email Metrics

**Open Rate:**
| Rate | Grade |
|------|-------|
| <15% | Poor |
| 15-25% | Average |
| 25-35% | Good |
| 35%+ | Excellent |

**Click Rate:**
| Rate | Grade |
|------|-------|
| <1% | Poor |
| 1-3% | Average |
| 3-5% | Good |
| 5%+ | Excellent |

**Unsubscribe Rate:**
- Target: <0.5% per email
- If higher: Review frequency and value

## Revenue Metrics (Bottom of Funnel)

### Average Order Value (AOV)

**Definition:** Total Revenue ÷ Number of Orders

**Formula:**
```
AOV = (Front-End Revenue + Bump Revenue + OTO Revenue) / Total Orders
```

**Example Calculation:**
```
100 customers
× $27 front-end = $2,700
× 40% bump ($9) = $360
× 15% OTO1 ($97) = $1,455
× 10% OTO2 ($67) = $670
─────────────────────
Total: $5,185
AOV: $51.85
```

**AOV Optimization:**
1. Add order bump (+20-40% to AOV)
2. Add/improve OTO1 (+30-50% to AOV)
3. Raise prices (test +20%)
4. Add OTO2/3 (+15-25% each)
5. Payment plans for OTOs (+30% take rate)

### Customer Lifetime Value (LTV)

**Definition:** Total revenue from a customer over their entire relationship

**Simple Formula:**
```
LTV = AOV × Average Purchase Frequency × Average Customer Lifespan
```

**For single-purchase funnels:**
```
LTV ≈ AOV × (1 + Email Purchase Rate over 90 days)
```

**Target:** LTV should be 2-3x front-end price

### Refund Rate

**Definition:** Refunds ÷ Total Orders × 100

**Benchmarks:**
| Rate | Status |
|------|--------|
| <5% | Excellent |
| 5-10% | Acceptable |
| 10-15% | Concerning |
| 15%+ | Critical - fix immediately |

**High Refund Causes:**
- Overselling in content/sales page
- Product quality issues
- Buyer's remorse (expected some)
- Technical delivery problems

### Revenue Per View (RPV)

**Definition:** Total Revenue ÷ Total Views

**Formula:**
```
RPV = Total Daily Revenue / Total Daily Views
```

**Example:**
```
$5,000 revenue / 1,000,000 views = $0.005 per view
= $5 per 1,000 views
```

**Benchmark:** $1-10 per 1,000 views depending on niche

This metric helps compare different niches and content strategies.

## Operational Metrics

### Content Production Rate

**Track:**
- Videos produced per day
- Videos posted per day
- Posting consistency (days without gaps)
- Content backlog size

**Targets:**
| Stage | Posts/Day |
|-------|-----------|
| Phase 1 Start | 30-50 |
| Phase 1 End | 100-150 |
| Phase 2 | 500-5000+ |

### Account Health Metrics

**Per Account Track:**
- Follower count
- Follower growth rate
- Average views per post
- Engagement trend (up/down/flat)
- Warning/restriction status

**Account Health Score:**
```
Score = (Avg Views × Engagement Rate × Growth Rate) / 1000
```

### Poster Performance (Phase 2)

**Per Poster Track:**
- Posts per week
- Views generated
- Clicks driven
- Sales attributed
- Commission paid
- Response time
- Quality score

**Poster Leaderboard:**
| Rank | Poster | Posts | Views | Sales | Commission |
|------|--------|-------|-------|-------|------------|
| 1 | Name | ### | ###M | ### | $### |
| 2 | Name | ### | ###M | ### | $### |
| ... | ... | ... | ... | ... | ... |

## Analytics Dashboards

### Daily Dashboard (Check Every Morning)

```
┌─────────────────────────────────────────────────────────────┐
│                  DAILY PERFORMANCE                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  YESTERDAY'S NUMBERS                                        │
│  ├── Total Views: _____________ (vs 7-day avg: ___%)       │
│  ├── Bio Clicks: ______________ (CTR: ___%)                │
│  ├── Landing Visits: __________                            │
│  ├── Opt-ins: _________________ (Rate: ___%)               │
│  ├── Sales: ___________________ (CVR: ___%)                │
│  ├── Revenue: $________________                            │
│  ├── AOV: $____________________                            │
│  └── Refunds: _________________ (Rate: ___%)               │
│                                                             │
│  CONTENT POSTED: _____ videos                              │
│  TOP PERFORMER: [Video ID] - _____ views                   │
│  WORST PERFORMER: [Video ID] - _____ views                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Weekly Dashboard (Strategic Review)

```
┌─────────────────────────────────────────────────────────────┐
│                   WEEKLY SUMMARY                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  TRAFFIC                    │  REVENUE                      │
│  Total Views: ___________   │  Total: $______________       │
│  CTR: ___________%          │  AOV: $________________       │
│  Week/Week: ___________%    │  Week/Week: ___________% │
│                             │                               │
│  CONTENT                    │  OPERATIONS                   │
│  Posts Published: ________  │  Active Accounts: ______      │
│  Viral Hits (100k+): _____  │  Active Posters: _______      │
│  Viral Rate: ___________% │  Avg Posts/Poster: _____      │
│                             │                               │
│  TOP 3 WINNERS THIS WEEK:                                   │
│  1. [Video] - _________ views - ________ clicks            │
│  2. [Video] - _________ views - ________ clicks            │
│  3. [Video] - _________ views - ________ clicks            │
│                                                             │
│  INSIGHTS & ACTIONS:                                        │
│  • ___________________________________________________     │
│  • ___________________________________________________     │
│  • ___________________________________________________     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Monthly Dashboard (Business Health)

```
┌─────────────────────────────────────────────────────────────┐
│                  MONTHLY BUSINESS REVIEW                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  REVENUE                                                    │
│  Gross Revenue: $________________                           │
│  Refunds: $______________________                           │
│  Net Revenue: $__________________                           │
│  Month/Month Growth: ___________%                           │
│                                                             │
│  EXPENSES                                                   │
│  Tools/Software: $_______________                           │
│  Proxies/Infra: $________________                           │
│  Commissions Paid: $______________                          │
│  Other: $________________________                           │
│  Total Expenses: $_______________                           │
│                                                             │
│  PROFIT                                                     │
│  Net Profit: $___________________                           │
│  Profit Margin: ________________%                           │
│                                                             │
│  KEY METRICS TREND (3-month)                               │
│  ┌─────┬─────────┬─────────┬─────────┐                     │
│  │     │ Month-2 │ Month-1 │ Current │                     │
│  ├─────┼─────────┼─────────┼─────────┤                     │
│  │ AOV │ $____   │ $____   │ $____   │                     │
│  │ CTR │ ____%   │ ____%   │ ____%   │                     │
│  │ CVR │ ____%   │ ____%   │ ____%   │                     │
│  └─────┴─────────┴─────────┴─────────┘                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Setting Up Tracking

### UTM Parameter Structure

```
URL Structure:
https://yourdomain.com/offer?utm_source=X&utm_medium=Y&utm_campaign=Z&utm_content=W

Parameters:
- utm_source: Platform (tiktok, ig, youtube)
- utm_medium: Account ID or name
- utm_campaign: Content type or campaign name
- utm_content: Specific video ID
```

**Example:**
```
https://offer.com/start?utm_source=tiktok&utm_medium=acc_001&utm_campaign=sidehustle&utm_content=vid_2024_001
```

### Tracking Links Per Account

Each account gets unique tracking link:
```
Account 1: yourdomain.com/go/a1
Account 2: yourdomain.com/go/a2
Account 3: yourdomain.com/go/a3
...
```

Or use URL parameters for single link:
```
linkin.bio/yourbrand?ref=a1
linkin.bio/yourbrand?ref=a2
```

### Attribution Models

**First Touch:**
- Credits the first TikTok video someone watched
- Best for: Understanding discovery

**Last Touch:**
- Credits the final video before purchase
- Best for: Understanding conversion triggers

**Linear:**
- Splits credit across all touchpoints
- Best for: Understanding full journey

**Recommended:** Start with last touch, move to linear as you scale

## A/B Testing Framework

### What to Test (Priority Order)

1. **Hooks** (30-50% impact)
2. **Landing page headline** (20-30% impact)
3. **Price points** (10-30% impact)
4. **CTAs** (10-20% impact)
5. **Video format** (10-20% impact)
6. **Posting times** (5-15% impact)

### Statistical Significance

**Minimum sample sizes for confidence:**
| Test Type | Minimum Actions |
|-----------|-----------------|
| Hook test | 10k+ views per variant |
| CTR test | 1k+ clicks total |
| CVR test | 100+ purchases total |
| AOV test | 200+ purchases total |

**Formula (simplified):**
```
If Variant B beats Variant A by 20%+ with 100+ conversions each:
→ Likely significant, implement winner
```

### Testing Calendar

**Week 1:** Hook variations
**Week 2:** CTA variations
**Week 3:** Format variations
**Week 4:** Price testing
**Week 5:** Landing page elements
**Week 6:** Analyze and implement winners
*Repeat*

## Red Flags & Alerts

### Set Up Alerts For:

| Metric | Alert Threshold | Action |
|--------|-----------------|--------|
| CTR drops | <50% of average | Investigate content/offer |
| CVR drops | <50% of average | Check funnel, pricing |
| Refund spike | >15% | Review product/expectations |
| Views crash | <30% of average | Check account health |
| Revenue drop | <70% of average | Full diagnostic |

### Diagnostic Decision Tree

```
Revenue Down?
├── Views down?
│   ├── Yes → Account issues, content quality, or algorithm
│   └── No → CTR or CVR problem
│
├── CTR down?
│   ├── Yes → Content/hook problem or offer mismatch
│   └── No → Funnel conversion problem
│
├── CVR down?
│   ├── Yes → Landing page, pricing, or offer issue
│   └── No → AOV problem (check OTOs)
│
└── AOV down?
    └── Check bump rates, OTO take rates, refunds
```

## Tools for Analytics

### Free/Low-Cost Stack
- Google Analytics 4 (web)
- TikTok native analytics
- Google Sheets (manual tracking)
- Notion databases

### Professional Stack
- Hyros (attribution)
- TripleWhale (e-commerce)
- Mixpanel (product analytics)
- Databox (dashboards)

### Enterprise Stack
- Custom data warehouse
- Looker/Tableau visualization
- Segment (data collection)
- Amplitude (product analytics)

---

**Next:** [BUILD-PLAN.md](../BUILD-PLAN.md) - Complete Implementation Roadmap
