# Technology Stack Guide

## Overview

This document provides a comprehensive guide to all tools, platforms, and technologies needed to build and operate the Eyeballs Method system at scale.

## Tech Stack Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      CONTENT CREATION                           │
│  AI Writing │ TTS/Voice │ Video Editing │ Asset Libraries       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      DISTRIBUTION                               │
│  Schedulers │ Automation │ Multi-Account │ Proxy Management     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      MONETIZATION                               │
│  Funnel Builder │ Payment │ Email │ CRM                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      ANALYTICS & OPS                            │
│  Tracking │ Attribution │ Team Management │ Communication       │
└─────────────────────────────────────────────────────────────────┘
```

## Content Creation Stack

### AI Script Writing

| Tool | Purpose | Pricing | Rating |
|------|---------|---------|--------|
| ChatGPT Plus | Script generation, ideation | $20/mo | ⭐⭐⭐⭐⭐ |
| Claude | Long-form, nuanced scripts | $20/mo | ⭐⭐⭐⭐⭐ |
| Copy.ai | Marketing copy, ads | $49/mo | ⭐⭐⭐⭐ |
| Jasper | Marketing-focused AI | $49/mo | ⭐⭐⭐⭐ |
| Writesonic | Bulk content | $19/mo | ⭐⭐⭐ |

**Recommended:** ChatGPT Plus + Claude for maximum versatility

### Text-to-Speech (TTS)

| Tool | Quality | Voices | Pricing | Best For |
|------|---------|--------|---------|----------|
| ElevenLabs | A+ | 100+ | $5-22/mo | Premium content |
| Play.ht | A | 900+ | $14/mo | Variety needed |
| Murf.ai | A | 120+ | $19/mo | Professional |
| Speechify | A- | 30+ | $11/mo | Simple needs |
| Amazon Polly | B+ | 60+ | Pay-per-use | High volume |
| TikTok Native | B | 10+ | Free | Quick tests |

**Recommended:** ElevenLabs for quality, Amazon Polly for scale

### Video Editing

| Tool | Platform | Skill Level | Pricing | Best For |
|------|----------|-------------|---------|----------|
| CapCut | Desktop/Mobile | Beginner | Free | Most creators |
| InShot | Mobile | Beginner | Free/$4/mo | Quick edits |
| Canva Video | Web | Beginner | Free/$13/mo | Templates |
| DaVinci Resolve | Desktop | Intermediate | Free | Full control |
| Adobe Premiere | Desktop | Advanced | $23/mo | Professional |
| Final Cut Pro | Mac | Intermediate | $300 one-time | Mac users |

**Recommended:** CapCut for most, DaVinci Resolve for advanced

### AI Video Generation

| Tool | Output Type | Pricing | Quality |
|------|-------------|---------|---------|
| Synthesia | AI avatars | $30/mo | A |
| HeyGen | AI avatars | $24/mo | A |
| Pictory | Text-to-video | $19/mo | B+ |
| InVideo | Templates + AI | $25/mo | B+ |
| Runway | AI editing | $15/mo | A |
| Lumen5 | Blog-to-video | $19/mo | B |

**Recommended:** HeyGen for avatars, Pictory for bulk faceless

### Stock Assets

| Type | Platform | Pricing | Library Size |
|------|----------|---------|--------------|
| Video | Pexels | Free | 30k+ |
| Video | Pixabay | Free | 20k+ |
| Video | Storyblocks | $15/mo | 1M+ |
| Video | Artgrid | $25/mo | Premium |
| Images | Unsplash | Free | 3M+ |
| Images | Freepik | Free/$10/mo | 20M+ |
| Music | Epidemic Sound | $15/mo | 40k+ |
| Music | Artlist | $17/mo | 30k+ |
| SFX | Envato Elements | $17/mo | 100k+ |

**Recommended:** Pexels + Pixabay (free), Storyblocks + Epidemic Sound (paid)

### Image Generation

| Tool | Quality | Style Control | Pricing |
|------|---------|---------------|---------|
| Midjourney | A+ | High | $10-30/mo |
| DALL-E 3 | A | Medium | Via ChatGPT |
| Stable Diffusion | A | Very High | Free/Self-host |
| Leonardo.ai | A | High | Free/$12/mo |
| Adobe Firefly | A- | Medium | $5/mo |

**Recommended:** Midjourney for quality, Leonardo for free tier

## Distribution Stack

### Social Media Schedulers

| Tool | Platforms | Features | Pricing |
|------|-----------|----------|---------|
| Metricool | TikTok, IG, YT | Analytics, scheduling | Free/$22/mo |
| Later | IG, TikTok, Pinterest | Visual planning | $18/mo |
| Publer | Multi-platform | Bulk scheduling | $12/mo |
| SocialBee | Multi-platform | Category rotation | $29/mo |
| Buffer | Multi-platform | Team features | $6/mo/channel |
| Hootsuite | Enterprise | Full suite | $99/mo |

**Recommended:** Metricool for TikTok focus, Publer for budget

### Automation Tools

| Tool | Purpose | Technical Level | Pricing |
|------|---------|-----------------|---------|
| Puppeteer | Browser automation | Advanced | Free |
| Playwright | Cross-browser | Advanced | Free |
| Selenium | Classic automation | Intermediate | Free |
| Make (Integromat) | No-code automation | Beginner | Free/$9/mo |
| Zapier | App connections | Beginner | Free/$20/mo |
| n8n | Self-hosted workflows | Intermediate | Free/Self-host |

**Recommended:** Make for non-technical, Puppeteer for custom

### Multi-Account Management

| Tool | Purpose | Pricing |
|------|---------|---------|
| GoLogin | Browser profiles | $24/mo |
| Multilogin | Anti-detection | $99/mo |
| Incogniton | Profile management | $30/mo |
| Kameleo | Mobile + desktop | $59/mo |
| AdsPower | Budget option | $9/mo |

**Recommended:** GoLogin for most, Multilogin for serious scale

### Proxy Services

| Type | Provider | Use Case | Pricing |
|------|----------|----------|---------|
| Residential | Bright Data | Premium | $10/GB |
| Residential | Smartproxy | Good value | $7/GB |
| Residential | IPRoyal | Budget | $3/GB |
| Mobile | Soax | Highest trust | $15/GB |
| Datacenter | Webshare | Testing only | $0.05/IP |

**Recommended:** Smartproxy for balance, Mobile proxies for critical accounts

### VPN Services

| Provider | Speed | Locations | Pricing |
|----------|-------|-----------|---------|
| NordVPN | Fast | 60+ countries | $4/mo |
| ExpressVPN | Very Fast | 90+ countries | $8/mo |
| Surfshark | Good | 100+ countries | $2/mo |

**Recommended:** Surfshark for value, NordVPN for speed

## Monetization Stack

### Funnel Builders

| Platform | Ease of Use | Features | Pricing |
|----------|-------------|----------|---------|
| ClickFunnels 2.0 | Easy | Full suite | $147-297/mo |
| Systeme.io | Easy | All-in-one | Free-97/mo |
| Kartra | Medium | Comprehensive | $99-499/mo |
| Kajabi | Easy | Course focus | $149-399/mo |
| WordPress + Elementor | Medium | Full control | ~$50/mo |
| Webflow | Medium | Design freedom | $29/mo |

**Recommended:** Systeme.io for start, ClickFunnels for scale

### Checkout/Payment

| Platform | Features | Fees | Best For |
|----------|----------|------|----------|
| ThriveCart | OTOs, bumps, affiliates | $495-690 lifetime | One-time payment |
| SamCart | Conversion optimized | $59-199/mo | Checkout focus |
| Stripe | Developer-friendly | 2.9% + $0.30 | Custom builds |
| PayPal | Universal | 3.5% + $0.49 | Accessibility |
| Gumroad | Simple | 10% | Quick setup |
| Lemon Squeezy | Modern | 5% + $0.50 | Digital products |

**Recommended:** ThriveCart for serious operators, Gumroad for testing

### Email Marketing

| Platform | List Size | Automation | Pricing |
|----------|-----------|------------|---------|
| ConvertKit | Unlimited | Good | $9/mo (300) |
| Mailchimp | Scalable | Good | Free (500) |
| ActiveCampaign | Unlimited | Advanced | $29/mo |
| Klaviyo | Scalable | E-commerce focus | Free (250) |
| Beehiiv | Unlimited | Newsletter focus | Free (2.5k) |
| GetResponse | Scalable | Webinars included | $15/mo |

**Recommended:** ConvertKit for creators, ActiveCampaign for advanced

### Landing Page Builders

| Tool | Ease | Speed | Pricing |
|------|------|-------|---------|
| Carrd | Very Easy | Very Fast | $9/year |
| Leadpages | Easy | Fast | $49/mo |
| Unbounce | Medium | Fast | $99/mo |
| Instapage | Medium | Very Fast | $199/mo |

**Recommended:** Carrd for simple, Leadpages for full features

### Link-in-Bio Tools

| Tool | Features | Pricing |
|------|----------|---------|
| Linktree | Basic links | Free/$5/mo |
| Stan Store | Native checkout | 5% fee |
| Beacons | Feature-rich | Free/$10/mo |
| Koji | Interactive | Free |
| Bio.fm | Analytics | Free |

**Recommended:** Stan Store for monetization, Beacons for features

## Analytics & Operations Stack

### Analytics Platforms

| Tool | Focus | Pricing |
|------|-------|---------|
| Google Analytics 4 | Web traffic | Free |
| TikTok Analytics | Native metrics | Free |
| TripleWhale | E-commerce attribution | $100/mo |
| Hyros | Advanced attribution | $99/mo |
| Voluum | Affiliate tracking | $199/mo |

**Recommended:** GA4 + TikTok native, Hyros for scale

### Link Tracking

| Tool | Features | Pricing |
|------|----------|---------|
| Bit.ly | Basic shortening | Free/$8/mo |
| Rebrandly | Custom domains | Free/$13/mo |
| ClickMagick | Full tracking | $47/mo |
| Voluum | Advanced | $199/mo |

**Recommended:** Rebrandly for brand, ClickMagick for conversion tracking

### Project Management

| Tool | Best For | Pricing |
|------|----------|---------|
| Notion | All-in-one workspace | Free/$8/mo |
| Trello | Visual boards | Free/$5/mo |
| Asana | Task management | Free/$11/mo |
| Monday.com | Team workflows | $8/mo |
| ClickUp | Feature-rich | Free/$7/mo |

**Recommended:** Notion for solo/small, ClickUp for teams

### Team Communication

| Tool | Features | Pricing |
|------|----------|---------|
| Discord | Communities, voice | Free |
| Slack | Professional | Free/$7/mo |
| Telegram | Groups, bots | Free |
| Loom | Video messages | Free/$12/mo |
| Zoom | Video calls | Free/$15/mo |

**Recommended:** Discord for poster community, Slack for internal team

### File Storage

| Service | Storage | Pricing |
|---------|---------|---------|
| Google Drive | 15GB free | $2/mo (100GB) |
| Dropbox | 2GB free | $12/mo (2TB) |
| OneDrive | 5GB free | $2/mo (100GB) |
| pCloud | 10GB free | $4/mo (500GB) |

**Recommended:** Google Drive for collaboration

### Password Management

| Tool | Features | Pricing |
|------|----------|---------|
| BitWarden | Open source | Free/$10/year |
| 1Password | Teams | $3/mo |
| LastPass | Popular | $3/mo |
| Dashlane | VPN included | $5/mo |

**Recommended:** BitWarden for budget, 1Password for teams

## Budget Recommendations

### Minimum Viable Stack ($0-100/month)

```
Content:
- ChatGPT Free or Claude Free
- CapCut (Free)
- Pexels/Pixabay (Free)
- TikTok Native TTS (Free)

Distribution:
- Manual posting initially
- Metricool free tier

Monetization:
- Systeme.io free tier
- Stripe (pay per transaction)
- Beehiiv free tier

Operations:
- Notion (Free)
- Google Drive (Free)
- Discord (Free)

Total: ~$50/month (transaction fees only)
```

### Growth Stack ($100-500/month)

```
Content:
- ChatGPT Plus ($20)
- ElevenLabs ($22)
- CapCut Pro ($10)
- Storyblocks ($15)

Distribution:
- Metricool Pro ($22)
- GoLogin ($24)
- Smartproxy ($50)

Monetization:
- Systeme.io Pro ($47)
- ThriveCart (amortized $50)
- ConvertKit ($29)

Operations:
- Notion Team ($16)
- Loom ($12)

Total: ~$317/month
```

### Scale Stack ($500-2000/month)

```
Content:
- ChatGPT Plus ($20)
- Claude Pro ($20)
- ElevenLabs Pro ($99)
- Storyblocks ($30)
- Epidemic Sound ($15)
- Midjourney ($30)

Distribution:
- Custom automation (dev cost)
- Multilogin ($99)
- Premium proxies ($200)
- Multiple schedulers ($100)

Monetization:
- ClickFunnels ($197)
- ThriveCart Pro
- ActiveCampaign ($99)
- Hyros ($99)

Operations:
- ClickUp ($30)
- Slack ($30)
- Zoom ($15)
- Loom ($12)

Total: ~$1,200/month
```

## Integration Map

```
TikTok Account
      │
      ├── Scheduler (Metricool)
      │       │
      │       └── Content Library (Google Drive)
      │               │
      │               └── AI Tools (ChatGPT + ElevenLabs)
      │
      └── Bio Link (Stan Store)
              │
              ├── Landing Page (Carrd/ClickFunnels)
              │       │
              │       └── Email Capture (ConvertKit)
              │
              └── Checkout (ThriveCart)
                      │
                      ├── Payment (Stripe)
                      │
                      └── CRM (Systeme.io)
                              │
                              └── Analytics (GA4 + Hyros)
```

## Tool Selection Framework

When choosing tools, evaluate:

1. **Current needs** - Don't overbuy features
2. **Scale path** - Can it grow with you?
3. **Integration** - Does it connect to your stack?
4. **Support** - Is help available when needed?
5. **Pricing model** - Monthly vs usage-based vs lifetime
6. **Exit difficulty** - Can you migrate data out?

---

**Next:** [08-metrics-analytics.md](./08-metrics-analytics.md) - Metrics & Analytics Deep Dive
