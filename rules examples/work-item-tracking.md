# Work Item Tracking

> How to maintain the work items tracker when creating ADO work items

---

## Rule

**REQUIRED:** After creating any Epic, Feature, or User Story in Azure DevOps, you MUST update the tracker file immediately.

**Tracker Location:** `work-items/tracker/work-items-tracker.md`

---

## When to Update

Update the tracker immediately after:
- Creating a new Epic
- Creating a new Feature
- Creating a new User Story
- Modifying state or key fields of existing work items

---

## How to Update

Follow these four steps in order:

### Step 1: Add to Work Item Table

Add a new row to the corresponding table based on work item type:

**For Epics:**
```markdown
| {ID} | {Title} | {State} | {Created Date} | [Link]({URL}) |
```

**For Features:**
```markdown
| {ID} | {Title} | {State} | {Parent Epic ID or "-"} | {Created Date} | [Link]({URL}) |
```

**For User Stories:**
```markdown
| {ID} | {Title} | {State} | {Parent Feature ID or "-"} | {Created Date} | [Link]({URL}) |
```

**URL Format:**
```
https://dev.azure.com/{org}/{project}/_workitems/edit/{id}
```

**Date Format:** ISO format `YYYY-MM-DD`

**State Values:** `New`, `Active`, `Resolved`, `Closed`, `Removed`

### Step 2: Update Summary Counts

Increment the appropriate counter in the Summary table at the top of the tracker file.

### Step 3: Add to Recent Activity

Add an entry under the current date in the Recent Activity section:

```markdown
### YYYY-MM-DD
- Created Epic #{ID}: {Title}
- Created Feature #{ID}: {Title} (under Epic #{ParentID})
- Created User Story #{ID}: {Title} (under Feature #{ParentID})
```

### Step 4: Update Status Counts

Update the "By Status" section with current counts based on all work items.

---

## Example

### Before Adding First Epic

```markdown
## 🎯 Epics

| ID | Title | State | Created | Link |
|----|-------|-------|---------|------|
| - | No epics tracked yet | - | - | - |
```

### After Adding First Epic

```markdown
## 🎯 Epics

| ID | Title | State | Created | Link |
|----|-------|-------|---------|------|
| 1001 | Launch Readiness | New | 2025-10-27 | [View](https://dev.azure.com/IFT1/NorthStar/_workitems/edit/1001) |
```

**Note:** Remove placeholder rows (like "No epics tracked yet") when adding the first item.

---

## Maintenance Schedule

### Daily
- Review Recent Activity section
- Ensure all new work items are tracked

### Weekly
- Verify counts in Summary table
- Update status counts
- Clean up old activity entries (keep last 30 days)

### Monthly
- Archive old entries if needed
- Review and update notes section

---

## Integration with Automation

The tracking file complements automated scripts:

- **`scripts/ado/retrieve_work_items.py`**: Fetches ALL work items and saves to `data/work_items/`
- **`work-items/tracker/work-items-tracker.md`**: Human-readable quick reference for tracking

**Purpose:**
- Use the tracker for quick visibility of recent work
- Use JSON files from scripts for detailed analysis

---

## Best Practices

1. **Update immediately** - Don't let the tracker get stale
2. **Remove placeholders** - Delete "No items tracked yet" rows when adding first item
3. **Use consistent formatting** - Follow the examples exactly
4. **Keep it concise** - Only essential info in the tracker
5. **Link to details** - Use hyperlinks to ADO for full details
6. **Verify counts** - Double-check summary and status counts after updates

---

## Quick Checklist

After creating a work item, verify you:
- ✅ Added row to appropriate table (Epic/Feature/User Story)
- ✅ Updated Summary count
- ✅ Added Recent Activity entry
- ✅ Updated Status counts
- ✅ Used correct URL format
- ✅ Used ISO date format (YYYY-MM-DD)
- ✅ Removed placeholder rows if applicable




