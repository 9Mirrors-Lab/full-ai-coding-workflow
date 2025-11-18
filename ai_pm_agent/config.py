"""
Configuration data for work item templates, launch readiness, and GTM phases.
"""

from typing import Dict, List, Any


def get_work_item_templates() -> Dict[str, str]:
    """
    Get ADO work item templates.

    Returns:
        Dictionary of work item type to template string
    """
    return {
        "Epic": """Title: [Clear, concise epic name]

Vision Statement:
[What is this epic trying to achieve?]

Business Value:
[Why is this important? What value does it deliver?]

Success Metrics:
- [Measurable metric 1]
- [Measurable metric 2]

Tags: [Relevant tags]""",

        "Feature": """Title: [User-facing capability name]

Description:
[What capability does this provide to users?]

Acceptance Criteria:
- [ ] Criterion 1
- [ ] Criterion 2
- [ ] Criterion 3

Technical Notes:
[Implementation considerations, dependencies, etc.]

Tags: [Relevant tags]""",

        "User Story": """Title: [Brief user story summary]

User Story:
As a [user type]
I want [capability]
So that [benefit]

Acceptance Criteria (Gherkin format):

Scenario 1: [Scenario name]
Given [initial context]
When [action taken]
Then [expected outcome]

Scenario 2: [Another scenario name]
Given [initial context]
When [action taken]
Then [expected outcome]

Tags: [Relevant tags]"""
    }


def get_launch_readiness_categories() -> List[Dict[str, str]]:
    """
    Get launch readiness categories with descriptions.

    Returns:
        List of category dictionaries
    """
    return [
        {
            "name": "Product Readiness",
            "description": "Core product features, UX completeness, feature parity",
            "examples": "Feature development, UX improvements, Product specs"
        },
        {
            "name": "Technical Readiness",
            "description": "Infrastructure, performance, scalability, technical debt",
            "examples": "API development, Database optimization, Bug fixes"
        },
        {
            "name": "Operational Readiness",
            "description": "Support processes, documentation, monitoring",
            "examples": "Support docs, Monitoring setup, Operational procedures"
        },
        {
            "name": "Commercial Readiness",
            "description": "Pricing, billing, contracts, sales enablement",
            "examples": "Pricing strategy, Billing integration, Sales materials"
        },
        {
            "name": "Customer Readiness",
            "description": "Onboarding, training, customer success",
            "examples": "Onboarding flows, Training materials, Customer success"
        },
        {
            "name": "Security & Compliance",
            "description": "Security audits, compliance, data protection",
            "examples": "Security review, GDPR compliance, Data encryption"
        }
    ]


def get_gtm_phase_definitions() -> Dict[str, Dict[str, str]]:
    """
    Get GTM phase definitions.

    Returns:
        Dictionary of phase name to phase details
    """
    return {
        "Foundation": {
            "description": "Core development, scaffolding, essential features",
            "typical_work": "Architecture setup, core APIs, database schema, authentication"
        },
        "Validation": {
            "description": "Testing, onboarding, user feedback",
            "typical_work": "Alpha/beta testing, user interviews, bug fixes, UX refinement"
        },
        "Launch Prep": {
            "description": "Pricing, support setup, documentation",
            "typical_work": "Pricing strategy, support docs, customer success processes"
        },
        "Go-to-Market": {
            "description": "Marketing, training, initial customer acquisition",
            "typical_work": "Marketing campaigns, sales training, customer onboarding"
        },
        "Growth": {
            "description": "Scaling, optimization, performance improvements",
            "typical_work": "Performance optimization, scaling infrastructure, analytics"
        },
        "Market Leadership": {
            "description": "Advanced features, automation, competitive differentiation",
            "typical_work": "Advanced AI features, workflow automation, integrations"
        }
    }


def get_current_sprint_info() -> Dict[str, Any]:
    """
    Get current sprint information.

    Note: In production, this would fetch from Supabase or ADO API.
    For now, returns a default configuration.

    Returns:
        Dictionary with sprint number and capacity info
    """
    return {
        "sprint": 42,
        "capacity": 100,
        "start_date": "2025-01-13",
        "end_date": "2025-01-26"
    }
