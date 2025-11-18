# Azure DevOps Work Item Standards

> Complete standards for creating all ADO work item types in the NorthStar project

---

## Quick Reference

### Workflow and Templates
- **Complete Workflow:** [/Users/ryan_riley/Develop/CoDeveloper/projects/work-items/tracker/workflow.md](cci:7://file:///Users/ryan_riley/Develop/CoDeveloper/projects/work-items/tracker/workflow.md:0:0-0:0)
- **Feature Template:** [/Users/ryan_riley/Develop/CoDeveloper/projects/work-items/tracker/feature-template.md](cci:7://file:///Users/ryan_riley/Develop/CoDeveloper/projects/work-items/tracker/feature-template.md:0:0-0:0)
- **User Story Template:** [/Users/ryan_riley/Develop/CoDeveloper/projects/work-items/tracker/user-story-template.md](cci:7://file:///Users/ryan_riley/Develop/CoDeveloper/projects/work-items/tracker/user-story-template.md:0:0-0:0)
- **Bug Template:** `docs/BUG_TEMPLATE.md`

### Post-Creation
**REQUIRED:** After creating ANY work item, update the tracker:
- **Tracker File:** `work-items/tracker/work-items-tracker.md`
- **Instructions:** See `work-item-tracking.md`

---

## Universal Rules (All Work Item Types)

### Project
- **MUST** use the **NorthStar** project

### Title Format
- **MUST** capitalize only the first word
- ✅ Good: "Admin user receives 403 error when accessing dashboard"
- ❌ Bad: "Admin User Receives 403 Error When Accessing Dashboard"

### Content Format
- **MUST** use **Markdown** format for ALL work item types
- Do NOT use HTML

### User Perspective
- **MUST** focus on user-facing issues and frontend perspective
- Do NOT include technical implementation details or code snippets
- Developers will determine technical solutions

### Tags
- Do NOT add tags (tagging strategy under discussion in work item #1584)

---

## Voice and Tone

### Writing Style
You are writing as a **product manager** who communicates clearly and naturally. The goal is to sound like a real person wrote it, not a robot or corporate memo.

**MUST follow:**
- Write like you're explaining to a colleague over coffee
- Use simple, everyday language
- Keep it conversational but clear
- Write as if you (the PM) discovered and are reporting the issue or need

**MUST avoid:**
- Technical jargon or developer terminology
- Business buzzwords ("leverage," "synergy," "optimize," "actionable insights")
- Corporate speak ("per our discussion," "as per," "moving forward")
- Overly formal language ("hereby," "aforementioned," "utilize" instead of "use")
- Marketing language ("cutting-edge," "best-in-class," "world-class")
- Hedging phrases ("it appears that," "it seems like," "potentially")

### Examples

**❌ Too Technical:**
"The API endpoint returns a 403 status code when the admin role attempts to access the dashboard route due to authorization middleware failure."

**✅ Natural PM Voice:**
"Admin users get a 403 error when trying to open the dashboard. They can log in fine but hit the error right after."

**❌ Business Jargon:**
"We need to leverage the system's capabilities to optimize the user experience and deliver actionable insights that drive engagement."

**✅ Natural PM Voice:**
"Users need to see their data faster so they can make decisions without waiting."

**❌ Overly Formal:**
"As per our discussion, it appears that the aforementioned functionality requires enhancement to facilitate improved user interaction paradigms."

**✅ Natural PM Voice:**
"Users are having trouble with the search feature. It needs to work faster and show better results."

### Tone Guidelines

**For Bugs:**
- Describe what's broken from user perspective
- Keep it factual and clear
- No need to apologize or add emotion

**For Features:**
- Explain why users need this
- Focus on the problem being solved
- Keep it simple and direct

**For User Stories:**
- Write from the user's point of view
- Focus on what they're trying to do
- Explain the value in plain terms

---

## Bugs

### Required Fields
- **Title:** Brief description from user perspective
- **Repro Steps:** Field name `Microsoft.VSTS.TCM.ReproSteps` (Markdown format)
- **System Info:** Field name `Microsoft.VSTS.TCM.SystemInfo` (Markdown format, if needed)
- **Priority:** 1-4 (1 = highest)
- **Severity:** 1 = Critical, 2 = High, 3 = Medium, 4 = Low

### Content Structure (Markdown)

Follow this exact order in the Repro Steps field:

```markdown
**Tenant:** [Tenant name]
**User Role:** [Role of affected user]
**Reported By:** [Name/team of reporter]
**Environment:** [Dev | Prod]

## Issue summary
[What is broken from a user perspective]

## Steps to reproduce
1. [First step]
2. [Second step]
3. [Third step]

## Expected behavior
[What should happen]

## Actual behavior
[What actually happens]
```

### Do NOT Include
- Additional context as separate section
- Code snippets or technical implementation details
- Suggested fixes or solutions

---

## Features

### Required Information
- Follow the complete workflow and template:
  - [Workflow](cci:7://file:///Users/ryan_riley/Develop/CoDeveloper/projects/work-items/tracker/workflow.md:0:0-0:0)
  - [Feature Template](cci:7://file:///Users/ryan_riley/Develop/CoDeveloper/projects/work-items/tracker/feature-template.md:0:0-0:0)

### Format
- **MUST** use **Markdown** format
- Include clear description of feature value
- Link to parent Epic if applicable

---

## User Stories

### Required Information
- Follow the complete workflow and template:
  - [Workflow](cci:7://file:///Users/ryan_riley/Develop/CoDeveloper/projects/work-items/tracker/workflow.md:0:0-0:0)
  - [User Story Template](cci:7://file:///Users/ryan_riley/Develop/CoDeveloper/projects/work-items/tracker/user-story-template.md:0:0-0:0)

### Format
- **MUST** use **Markdown** format
- **MUST** follow Gherkin format for acceptance criteria

### User Story Structure

**Required Sections:**
1. User Story (As a/I want/So that)
2. Description
3. Acceptance Criteria (Gherkin format)

**User Story Format:**
```markdown
**As a** [type of user]
**I want** [capability or action]
**So that** [benefit or value]
```

**Description:**
- 1-2 sentences providing brief context about the user need
- Focus on the problem, not the solution

**Acceptance Criteria Format:**
```markdown
## Acceptance Criteria

**Scenario 1: [Scenario name]**
- **Given** [initial context/state]
- **When** [action taken]
- **Then** [expected outcome]
- **And** [additional outcomes as needed]

**Scenario 2: [Scenario name]**
- **Given** [initial context/state]
- **When** [action taken]
- **Then** [expected outcome]
```

### Writing Principles

**MUST follow:**
- Write from user's perspective (food scientists, R&D professionals)
- Focus on observable user behaviors and outcomes
- Use natural PM voice (see Voice and Tone section above)
- Use simple, conversational language appropriate for domain
- Express clear business value in plain terms (trust, efficiency, ease of use)
- Include performance criteria when relevant (e.g., "within 5 seconds")
- Handle edge cases with separate scenarios

**MUST avoid:**
- Implementation details, internal mechanisms, or technical specifics
- Stakeholder or system perspective
- Business jargon or overly formal language
- Technical terminology unless it's standard domain language
- Vague or ambiguous language

### Domain Context

**Sous (The AI Assistant):**
- AI-powered food science expert within CoDeveloper
- Describe as confident, knowledgeable, and expert-level
- Helps with formulation, ingredients, and regulatory guidance

**User Roles:**
- Food scientist (primary user)
- Formulation scientist
- R&D professional
- Product developer

### Quality Checklist

Before creating a user story, verify:

1. ✅ Written from user's perspective
2. ✅ Focuses on observable behaviors and outcomes
3. ✅ Uses natural PM voice (see Voice and Tone section)
4. ✅ Avoids technical jargon and business buzzwords
5. ✅ Avoids implementation details
6. ✅ Expresses clear business value in simple terms
7. ✅ Acceptance criteria describe user-facing results
8. ✅ Language is conversational and clear
9. ✅ Multiple scenarios cover different cases
10. ✅ Uses proper Gherkin format

### Example User Story

**Title:** Sous provides expert-level food science explanations

**User Story:**
```markdown
**As a** food scientist
**I want** Sous to provide confident, scientifically accurate explanations when I ask formulation questions
**So that** I can rely on its input in product development
```

**Description:**
Food scientists need to trust AI recommendations before incorporating them into product development. Confidence in the explanation increases adoption and reduces need for external verification.

**Acceptance Criteria:**
```markdown
## Acceptance Criteria

**Scenario 1: Basic nutrient interaction question**
- **Given** I am developing a formula
- **When** I ask Sous to explain a nutrient interaction
- **Then** Sous provides an accurate and concise explanation using correct food science terminology
- **And** the tone reflects confidence and expertise

**Scenario 2: Complex formulation question**
- **Given** I am working on a dairy-free formula
- **When** I ask about protein stability in plant-based systems
- **Then** Sous explains the mechanisms clearly
- **And** provides actionable guidance within 5 seconds
- **And** cites relevant food science principles

**Scenario 3: Regulatory question**
- **Given** I need to verify ingredient compliance
- **When** I ask about FDA regulations for a specific ingredient
- **Then** Sous provides current regulatory status
- **And** includes relevant CFR citations if applicable
```

---

## Post-Creation Requirements

### Update Tracker
**REQUIRED:** After creating any Epic, Feature, or User Story:
1. Update `work-items/tracker/work-items-tracker.md`
2. Follow instructions in `work-item-tracking.md`

This tracking is MANDATORY and must happen immediately after work item creation.

