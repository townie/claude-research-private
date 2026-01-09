# Location Page Template

## URL Structure
`/[service]-in-[city]-[state]/` or `/[city]/[service]/`

---

## Page Template

```html
<title>[Service] in [City], [State] | [Business Name]</title>
<meta name="description" content="Professional [service] in [City], [State]. [Unique value proposition]. Call [phone] for [service type] today.">
```

### H1: [Service] in [City], [State]

**Opening Paragraph (150-200 words)**
Introduce your service in the specific location. Mention:
- How long you've served this area
- Specific neighborhoods or areas within the city
- What makes your service unique in this location

### H2: Our [Service] Services in [City]

List services offered with brief descriptions:
- Service 1: Description with local context
- Service 2: Description with local context
- Service 3: Description with local context

### H2: Why Choose [Business Name] for [Service] in [City]?

- Local expertise and knowledge
- Response time specific to this area
- Local team members
- Community involvement

### H2: Service Areas in [City]

List specific neighborhoods, districts, or zip codes served:
- Neighborhood 1
- Neighborhood 2
- Neighborhood 3
- Surrounding areas

### H2: [Service] FAQs for [City] Residents

**Q: How quickly can you provide [service] in [City]?**
A: [Specific answer with local context]

**Q: What areas of [City] do you serve?**
A: [List neighborhoods and surrounding areas]

**Q: What are common [service-related issues] in [City]?**
A: [Local-specific answer, e.g., climate, infrastructure, common problems]

### H2: Recent [Service] Projects in [City]

Brief case studies or project highlights:
- Project 1: Location, problem, solution, result
- Project 2: Location, problem, solution, result

### H2: Contact Us for [Service] in [City]

- Phone number
- Local address (if applicable)
- Service hours
- Contact form

---

## Content Guidelines

### DO:
- Include genuine local knowledge
- Mention specific neighborhoods and landmarks
- Add unique content for each location (minimum 500 words unique)
- Include local reviews/testimonials
- Add location-specific images
- Embed Google Maps for the service area

### DON'T:
- Copy/paste the same content changing only city names
- Create pages for areas you don't actually serve
- Use AI-generated content without human review
- Stuff keywords unnaturally
- Neglect mobile optimization

---

## Schema Markup

```json
{
  "@context": "https://schema.org",
  "@type": "LocalBusiness",
  "name": "[Business Name]",
  "description": "[Service] provider in [City], [State]",
  "url": "https://example.com/[service]-in-[city]/",
  "telephone": "[phone]",
  "areaServed": {
    "@type": "City",
    "name": "[City]",
    "containedInPlace": {
      "@type": "State",
      "name": "[State]"
    }
  },
  "serviceArea": {
    "@type": "GeoCircle",
    "geoMidpoint": {
      "@type": "GeoCoordinates",
      "latitude": "[lat]",
      "longitude": "[long]"
    },
    "geoRadius": "[radius in miles]"
  }
}
```
