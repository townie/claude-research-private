#!/usr/bin/env python3
"""
Location-Specific Page Generator for Local SEO

This script generates location-specific landing pages based on
a list of services and locations. Each page is customized to
avoid duplicate content issues.
"""

import os
import json
from dataclasses import dataclass
from typing import Optional
from string import Template


@dataclass
class Location:
    """Represents a service location."""
    city: str
    state: str
    state_abbr: str
    neighborhoods: list[str]
    zip_codes: list[str]
    local_context: str  # Unique info about this location
    latitude: Optional[float] = None
    longitude: Optional[float] = None


@dataclass
class Service:
    """Represents a service offering."""
    name: str
    slug: str
    description: str
    features: list[str]
    common_issues: list[str]


@dataclass
class Business:
    """Represents the business information."""
    name: str
    phone: str
    email: str
    website: str
    years_in_business: int


# Page template
PAGE_TEMPLATE = Template("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>${service_name} in ${city}, ${state_abbr} | ${business_name}</title>
    <meta name="description" content="Professional ${service_name_lower} services in ${city}, ${state}. Serving ${neighborhoods_text}. Call ${phone} today!">
    <script type="application/ld+json">
    ${schema_markup}
    </script>
</head>
<body>
    <main>
        <h1>${service_name} in ${city}, ${state}</h1>

        <section>
            <p>
                Looking for reliable ${service_name_lower} in ${city}? ${business_name} has been
                proudly serving ${city} and surrounding areas for over ${years} years.
                ${local_context}
            </p>
        </section>

        <section>
            <h2>Our ${service_name} Services in ${city}</h2>
            <ul>
${features_list}
            </ul>
        </section>

        <section>
            <h2>Areas We Serve in ${city}</h2>
            <p>We provide ${service_name_lower} throughout ${city}, including:</p>
            <ul>
${neighborhoods_list}
            </ul>
            <p>Service available in zip codes: ${zip_codes_text}</p>
        </section>

        <section>
            <h2>Common ${service_name} Issues in ${city}</h2>
            <ul>
${issues_list}
            </ul>
        </section>

        <section>
            <h2>Contact Us for ${service_name} in ${city}</h2>
            <p>Phone: <a href="tel:${phone_clean}">${phone}</a></p>
            <p>Email: <a href="mailto:${email}">${email}</a></p>
        </section>
    </main>
</body>
</html>
""")


def generate_schema_markup(
    business: Business,
    service: Service,
    location: Location
) -> str:
    """Generate JSON-LD schema markup for the page."""
    schema = {
        "@context": "https://schema.org",
        "@type": "LocalBusiness",
        "name": business.name,
        "description": f"{service.name} provider in {location.city}, {location.state}",
        "url": f"{business.website}/{service.slug}-in-{location.city.lower().replace(' ', '-')}/",
        "telephone": business.phone,
        "email": business.email,
        "areaServed": {
            "@type": "City",
            "name": location.city,
            "containedInPlace": {
                "@type": "State",
                "name": location.state
            }
        }
    }

    if location.latitude and location.longitude:
        schema["geo"] = {
            "@type": "GeoCoordinates",
            "latitude": location.latitude,
            "longitude": location.longitude
        }

    return json.dumps(schema, indent=4)


def generate_page(
    business: Business,
    service: Service,
    location: Location
) -> str:
    """Generate a location-specific landing page."""

    # Prepare template variables
    features_list = "\n".join(
        f"                <li>{feature}</li>"
        for feature in service.features
    )

    neighborhoods_list = "\n".join(
        f"                <li>{n}</li>"
        for n in location.neighborhoods
    )

    issues_list = "\n".join(
        f"                <li>{issue}</li>"
        for issue in service.common_issues
    )

    schema_markup = generate_schema_markup(business, service, location)

    return PAGE_TEMPLATE.substitute(
        service_name=service.name,
        service_name_lower=service.name.lower(),
        city=location.city,
        state=location.state,
        state_abbr=location.state_abbr,
        business_name=business.name,
        phone=business.phone,
        phone_clean=business.phone.replace("-", "").replace(" ", "").replace("(", "").replace(")", ""),
        email=business.email,
        years=business.years_in_business,
        local_context=location.local_context,
        neighborhoods_text=", ".join(location.neighborhoods[:3]) + " and more",
        neighborhoods_list=neighborhoods_list,
        zip_codes_text=", ".join(location.zip_codes),
        features_list=features_list,
        issues_list=issues_list,
        schema_markup=schema_markup
    )


def generate_all_pages(
    business: Business,
    services: list[Service],
    locations: list[Location],
    output_dir: str = "generated-pages"
) -> list[str]:
    """Generate all service/location page combinations."""

    os.makedirs(output_dir, exist_ok=True)
    generated_files = []

    for service in services:
        for location in locations:
            # Generate page content
            content = generate_page(business, service, location)

            # Create filename
            city_slug = location.city.lower().replace(" ", "-")
            filename = f"{service.slug}-in-{city_slug}.html"
            filepath = os.path.join(output_dir, filename)

            # Write file
            with open(filepath, "w") as f:
                f.write(content)

            generated_files.append(filepath)
            print(f"Generated: {filepath}")

    return generated_files


# Example usage
if __name__ == "__main__":
    # Example business
    example_business = Business(
        name="ABC Plumbing Co",
        phone="(555) 123-4567",
        email="info@abcplumbing.com",
        website="https://abcplumbing.com",
        years_in_business=15
    )

    # Example services
    example_services = [
        Service(
            name="Emergency Plumbing",
            slug="emergency-plumbing",
            description="24/7 emergency plumbing services",
            features=[
                "24/7 availability",
                "Fast response times",
                "Licensed and insured plumbers",
                "Upfront pricing"
            ],
            common_issues=[
                "Burst pipes",
                "Severe leaks",
                "Backed up sewers",
                "No hot water"
            ]
        ),
        Service(
            name="Drain Cleaning",
            slug="drain-cleaning",
            description="Professional drain cleaning services",
            features=[
                "Video camera inspection",
                "Hydro jetting",
                "Rooter service",
                "Preventive maintenance"
            ],
            common_issues=[
                "Slow drains",
                "Recurring clogs",
                "Tree root intrusion",
                "Grease buildup"
            ]
        )
    ]

    # Example locations
    example_locations = [
        Location(
            city="Austin",
            state="Texas",
            state_abbr="TX",
            neighborhoods=["Downtown", "South Congress", "East Austin", "Mueller", "Hyde Park"],
            zip_codes=["78701", "78702", "78703", "78704", "78705"],
            local_context="With Austin's rapid growth and aging infrastructure in many neighborhoods, plumbing issues are increasingly common.",
            latitude=30.2672,
            longitude=-97.7431
        ),
        Location(
            city="Round Rock",
            state="Texas",
            state_abbr="TX",
            neighborhoods=["Downtown", "Old Settlers Park", "Brushy Creek", "Forest Creek"],
            zip_codes=["78664", "78665", "78681"],
            local_context="Round Rock's mix of historic homes and new construction presents unique plumbing challenges.",
            latitude=30.5083,
            longitude=-97.6789
        )
    ]

    # Generate pages
    generated = generate_all_pages(
        example_business,
        example_services,
        example_locations,
        output_dir="generated-pages"
    )

    print(f"\nGenerated {len(generated)} pages")
