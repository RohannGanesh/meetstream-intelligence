"""
Synthetic meeting title generator for Build 1 — Meeting Title Classifier.

Produces 1000+ labeled titles across 8 categories with realistic variation:
abbreviations, names, dates, project codes, and mixed conventions.
"""

import random
import csv
import os
from datetime import date, timedelta

SEED = 42
random.seed(SEED)

FIRST_NAMES = [
    "Alice", "Bob", "Carlos", "Diana", "Ethan", "Fatima", "George", "Hannah",
    "Ivan", "Julia", "Kevin", "Laura", "Marcus", "Nina", "Omar", "Priya",
    "Quinn", "Rachel", "Sam", "Tara", "Uma", "Victor", "Wendy", "Xiang",
    "Yusuf", "Zoe", "Alex", "Jordan", "Morgan", "Taylor", "Casey", "Riley",
]

LAST_NAMES = [
    "Smith", "Johnson", "Lee", "Patel", "Kim", "Chen", "Garcia", "Brown",
    "Davis", "Wilson", "Martinez", "Anderson", "Taylor", "Thomas", "Moore",
]

TEAMS = [
    "Eng", "Engineering", "Product", "Design", "Marketing", "Sales",
    "Data", "Infra", "Backend", "Frontend", "ML", "Platform", "Growth",
    "DevOps", "QA", "Security", "Finance", "Legal", "HR", "Ops",
]

PROJECTS = [
    "Phoenix", "Atlas", "Orion", "Vega", "Helix", "Nova", "Apex",
    "Horizon", "Pulse", "Nexus", "Titan", "Zephyr", "Echo", "Forge",
    "Q1", "Q2", "Q3", "Q4", "2024", "2025", "v2", "v3", "Beta", "Alpha",
    "MVP", "R&D", "POC", "OKR",
]

COMPANIES = [
    "Acme", "GlobalTech", "RetailCo", "HealthFirst", "FinServ", "EduBase",
    "CloudNine", "DataBridge", "NetCore", "SkyLine", "OpenPath", "BrightEdge",
]

def _name() -> str:
    return random.choice(FIRST_NAMES)

def _full_name() -> str:
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"

def _team() -> str:
    return random.choice(TEAMS)

def _project() -> str:
    return random.choice(PROJECTS)

def _company() -> str:
    return random.choice(COMPANIES)

def _date_str() -> str:
    base = date(2024, 1, 1)
    d = base + timedelta(days=random.randint(0, 365))
    fmt = random.choice(["%m/%d", "%b %d", "%Y-%m-%d", "W%W", "Q%q"])
    if fmt == "Q%q":
        return f"Q{(d.month - 1) // 3 + 1}"
    return d.strftime(fmt)

def _maybe(val: str, prob: float = 0.4) -> str:
    """Return val or empty string with given probability."""
    return val if random.random() < prob else ""

def _join(*parts: str, sep: str = " ") -> str:
    return sep.join(p for p in parts if p).strip()


def _standup_titles() -> list[str]:
    templates = [
        lambda: _join(_team(), random.choice(["Standup", "Stand-up", "Stand Up", "Sync", "Daily"])),
        lambda: _join(random.choice(["Daily", "Morning", "EOD", "Weekly"]), _team(), random.choice(["Standup", "Sync", "Check-in"])),
        lambda: random.choice(["Standup", "Stand-up", "Daily Standup", "Daily Scrum", "Morning Sync"]),
        lambda: _join(_team(), "Daily", _maybe(_date_str())),
        lambda: _join(random.choice(["Eng", "Dev", "Team"]), "Standup", _maybe(f"- {_date_str()}")),
        lambda: _join(_project(), "Daily Standup"),
        lambda: random.choice(["DSU", "Daily Scrum", "DS", "AM Sync", "EOD Sync"]),
        lambda: _join(_team(), "DSU", _maybe(_date_str())),
        lambda: f"{_team()} standup ({_date_str()})",
        lambda: _join("Scrum", _maybe(f"- {_team()}")),
        lambda: _join(_project(), "Scrum", _maybe(_date_str())),
        lambda: f"[{_team()}] Daily",
        lambda: _join(_team(), "/", _team(), "Standup"),
        lambda: random.choice(["10am Standup", "9am Sync", "Morning Standup", "EOD Check-in"]),
        lambda: _join(_project(), random.choice(["Sprint Sync", "Sprint Daily"])),
    ]
    return [random.choice(templates)() for _ in range(160)]


def _planning_titles() -> list[str]:
    templates = [
        lambda: _join(_team(), random.choice(["Planning", "Sprint Planning", "Roadmap Review", "Backlog Grooming"])),
        lambda: _join(_project(), random.choice(["Planning", "Kickoff", "Roadmap", "Strategy Session"])),
        lambda: _join(random.choice(["Q1", "Q2", "Q3", "Q4", "2025", "H1", "H2"]), random.choice(["Planning", "Roadmap", "OKR Review", "Goal Setting"])),
        lambda: random.choice(["Sprint Planning", "Backlog Refinement", "Backlog Grooming", "Roadmap Review"]),
        lambda: _join(_team(), "Sprint", random.choice(["Planning", "Retro", "Review", "Kickoff"])),
        lambda: _join(_project(), "Milestone", random.choice(["Planning", "Review", "Check"])),
        lambda: f"{_team()} {random.choice(['Planning', 'Strategy'])} - {_date_str()}",
        lambda: _join(random.choice(["Annual", "Monthly", "Weekly"]), _team(), "Planning"),
        lambda: _join("PI Planning", _maybe(f"- {_project()}")),
        lambda: _join(_project(), "v2 Planning"),
        lambda: random.choice(["Release Planning", "Launch Planning", "Go-to-Market Planning"]),
        lambda: _join(_team(), "OKR", random.choice(["Planning", "Review", "Check-in"])),
        lambda: f"[{_project()}] Sprint Planning",
        lambda: _join(_team(), "Roadmap", _maybe(_date_str())),
        lambda: random.choice(["Capacity Planning", "Resource Planning", "Headcount Planning"]),
    ]
    return [random.choice(templates)() for _ in range(160)]


def _one_on_one_titles() -> list[str]:
    templates = [
        lambda: _join(_name(), "<>", _name()),
        lambda: _join(_name(), "/", _name()),
        lambda: _join(_name(), "&", _name()),
        lambda: _join(_name(), "x", _name()),
        lambda: f"1:1 {_name()} / {_name()}",
        lambda: f"1-1: {_name()} & {_name()}",
        lambda: f"1on1 - {_name()}",
        lambda: f"Weekly 1:1 {_name()}",
        lambda: _join(_full_name(), "<>", _full_name()),
        lambda: f"{_name()} 1:1",
        lambda: f"Catch up: {_name()} & {_name()}",
        lambda: f"Sync: {_name()} / {_name()}",
        lambda: random.choice(["Manager 1:1", "Skip-level 1:1", "Weekly Catch-up", "Monthly 1:1"]),
        lambda: f"[1:1] {_name()} - {_name()}",
        lambda: f"{_name()} Check-in",
        lambda: f"Career Chat: {_name()}",
        lambda: f"{_name()} <> {_name()} ({_date_str()})",
        lambda: f"Weekly: {_name()} / {_name()}",
    ]
    return [random.choice(templates)() for _ in range(150)]


def _client_titles() -> list[str]:
    templates = [
        lambda: _join(_company(), random.choice(["Call", "Sync", "Check-in", "Meeting", "Review"])),
        lambda: _join(_company(), random.choice(["QBR", "Quarterly Review", "Monthly Review", "Weekly Sync"])),
        lambda: _join(random.choice(["Client", "Customer", "Account"]), _company(), random.choice(["Call", "Sync"])),
        lambda: _join(_company(), "Onboarding", _maybe(_date_str())),
        lambda: _join(_company(), "Demo"),
        lambda: _join(_company(), random.choice(["Discovery Call", "Sales Call", "Intro Call", "Kickoff"])),
        lambda: f"{_company()} — {random.choice(['Technical Review', 'Product Review', 'Executive Review'])}",
        lambda: _join(_company(), _project(), "Update"),
        lambda: _join("External:", _company(), random.choice(["Sync", "Call"])),
        lambda: f"[Client] {_company()} {random.choice(['Sync', 'Call', 'Review'])}",
        lambda: _join(_company(), "Partnership", random.choice(["Call", "Discussion", "Review"])),
        lambda: f"{_company()} x {_team()} Sync",
        lambda: _join(_full_name(), f"({_company()})", random.choice(["Call", "Intro"])),
        lambda: random.choice(["Customer Discovery", "Client Onboarding", "Account Review", "QBR"]),
        lambda: _join(_company(), "POC", random.choice(["Review", "Kickoff", "Update"])),
        lambda: f"{_company()} Renewal Discussion",
    ]
    return [random.choice(templates)() for _ in range(150)]


def _all_hands_titles() -> list[str]:
    templates = [
        lambda: _join(_team(), random.choice(["All Hands", "All-Hands", "Town Hall", "Company Meeting"])),
        lambda: random.choice(["All Hands", "All-Hands Meeting", "Company All Hands", "Org All Hands"]),
        lambda: _join(random.choice(["Monthly", "Quarterly", "Annual", "Weekly"]), random.choice(["All Hands", "Town Hall", "Company Meeting"])),
        lambda: _join(_team(), "Town Hall", _maybe(_date_str())),
        lambda: f"{random.choice(['Q1', 'Q2', 'Q3', 'Q4'])} All Hands",
        lambda: f"{_team()} All-Hands ({_date_str()})",
        lambda: random.choice(["Leadership Town Hall", "Executive All Hands", "Board All Hands"]),
        lambda: _join(random.choice(["Eng", "Product", "Sales", "Company"]), "All Hands", _maybe(f"- {_date_str()}")),
        lambda: f"[All Hands] {_date_str()}",
        lambda: random.choice(["Company Update", "Org Update", "Company-wide Meeting", "Full Team Sync"]),
        lambda: _join(_project(), "Launch All Hands"),
        lambda: f"{_team()} Town Hall - {random.choice(['Q1', 'Q2', 'Q3', 'Q4'])} {random.choice(['2024', '2025'])}",
        lambda: random.choice(["Fireside Chat", "AMA Session", "Leadership AMA"]),
        lambda: _join(random.choice(["Weekly", "Monthly"]), "Company", "Sync"),
    ]
    return [random.choice(templates)() for _ in range(120)]


def _interview_titles() -> list[str]:
    templates = [
        lambda: _join(_full_name(), random.choice(["Interview", "- Interview", "| Interview"])),
        lambda: _join(random.choice(["SWE", "PM", "DS", "ML", "Design", "Sales", "Marketing"]), "Interview:", _full_name()),
        lambda: _join("Interview:", _name(), random.choice(["- Round 1", "- Round 2", "- Final", "- Technical", "- Onsite"])),
        lambda: _join(_name(), random.choice(["Technical Screen", "Phone Screen", "Coding Interview", "System Design"])),
        lambda: f"Hiring: {_full_name()} ({random.choice(['SWE', 'PM', 'DS', 'Lead', 'Senior', 'Staff'])})",
        lambda: _join("Recruiter Screen:", _name()),
        lambda: f"{_name()} - {random.choice(['Onsite', 'Virtual Onsite', 'Take-home Review', 'Bar Raiser'])}",
        lambda: _join(random.choice(["HM", "Hiring Manager"]), "Interview:", _name()),
        lambda: f"[Interview] {_full_name()} - {random.choice(['Eng', 'Product', 'Data'])}",
        lambda: _join(_name(), "Behavioral Interview"),
        lambda: f"Debrief: {_name()} {random.choice(['SWE', 'PM'])} Interview",
        lambda: random.choice(["Candidate Debrief", "Hiring Committee Review", "Offer Review"]),
        lambda: f"{_full_name()} | {random.choice(['Round 1', 'Round 2', 'Final Round', 'Technical Round'])}",
        lambda: _join(_team(), "Interview:", _name()),
    ]
    return [random.choice(templates)() for _ in range(130)]


def _workshop_titles() -> list[str]:
    templates = [
        lambda: _join(_team(), random.choice(["Workshop", "Offsite", "Working Session", "Deep Dive"])),
        lambda: _join(_project(), random.choice(["Workshop", "Design Sprint", "Brainstorm", "Hackathon"])),
        lambda: _join(random.choice(["Design", "Strategy", "Architecture", "Process"]), "Workshop", _maybe(f"- {_team()}")),
        lambda: random.choice(["Design Sprint", "Hackathon", "Innovation Workshop", "Team Offsite"]),
        lambda: _join(_team(), "Offsite", _maybe(_date_str())),
        lambda: f"{_project()} Architecture Review",
        lambda: _join(_team(), "Brainstorm:", _project()),
        lambda: random.choice(["Lunch & Learn", "L&L:", "Knowledge Share", "Tech Talk"]),
        lambda: f"L&L: {_project()}",
        lambda: _join("Training:", random.choice(["Python", "AWS", "Leadership", "Agile", "Security", "ML"])),
        lambda: f"{_team()} Deep Dive - {_project()}",
        lambda: random.choice(["Postmortem", "Incident Review", "RCA Workshop", "Retro Workshop"]),
        lambda: _join(_project(), "Design Review"),
        lambda: f"[Workshop] {_team()} x {_team()}",
        lambda: _join(random.choice(["Quarterly", "Annual"]), _team(), "Offsite"),
    ]
    return [random.choice(templates)() for _ in range(130)]


def _social_titles() -> list[str]:
    templates = [
        lambda: random.choice(["Team Lunch", "Team Dinner", "Team Breakfast", "Team Coffee"]),
        lambda: _join(_team(), random.choice(["Lunch", "Happy Hour", "Team Lunch", "Coffee Chat", "Tea Time"])),
        lambda: random.choice(["Virtual Happy Hour", "Virtual Coffee", "Virtual Lunch", "Remote Social"]),
        lambda: _join(_team(), "Social", _maybe(_date_str())),
        lambda: random.choice(["Game Night", "Team Games", "Trivia Night", "Escape Room"]),
        lambda: _join(_team(), random.choice(["Offsite Dinner", "Celebration", "Team Outing"])),
        lambda: random.choice(["Welcome Lunch", "Onboarding Lunch", "New Hire Lunch"]),
        lambda: f"Coffee: {_name()} & {_name()}",
        lambda: random.choice(["Holiday Party", "End of Year Party", "Summer Party", "Quarterly Social"]),
        lambda: _join(_team(), random.choice(["Bowling", "Cooking Class", "Paint Night", "Volunteering"])),
        lambda: random.choice(["Birthday Celebration", "Work Anniversary", "Farewell Lunch", "Welcome Drinks"]),
        lambda: f"[Social] {_team()} {random.choice(['Hangout', 'Meetup', 'Gathering'])}",
        lambda: random.choice(["Donut Chat", "Random Coffee", "Peer Connect", "Water Cooler"]),
        lambda: _join(_name(), "Farewell", random.choice(["Lunch", "Drinks", "Party"])),
    ]
    return [random.choice(templates)() for _ in range(120)]


CATEGORY_GENERATORS: dict[str, callable] = {
    "standup":    _standup_titles,
    "planning":   _planning_titles,
    "one_on_one": _one_on_one_titles,
    "client":     _client_titles,
    "all_hands":  _all_hands_titles,
    "interview":  _interview_titles,
    "workshop":   _workshop_titles,
    "social":     _social_titles,
}


def generate_dataset(oversample_factor: int = 3) -> list[dict[str, str]]:
    """Generate labeled meeting titles across all categories.

    Oversamples then deduplicates per category to maximise unique titles.
    """
    rows: list[dict[str, str]] = []
    for label, gen_fn in CATEGORY_GENERATORS.items():
        seen: set[str] = set()
        for _ in range(oversample_factor):
            for title in gen_fn():
                t = title.strip()
                if t not in seen:
                    seen.add(t)
                    rows.append({"title": t, "label": label})
    random.shuffle(rows)
    return rows


def save_dataset(output_path: str) -> int:
    """Save dataset to CSV and return row count."""
    rows = generate_dataset()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["title", "label"])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "meeting_titles.csv")
    count = save_dataset(out)
    print(f"Generated {count} labeled titles → {out}")

    from collections import Counter
    rows = generate_dataset()
    dist = Counter(r["label"] for r in rows)
    print("\nLabel distribution:")
    for label, n in sorted(dist.items()):
        print(f"  {label:<12} {n}")
