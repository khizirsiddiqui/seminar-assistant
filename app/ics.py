from datetime import datetime, time
from zoneinfo import ZoneInfo

from icalendar import Calendar, Event

TZ = ZoneInfo("Europe/Berlin")


def build_calendar(rows: list[dict], venue: str, start_time: str, end_time: str) -> str:
    """rows: [{date: YYYY-MM-DD, student, paper}]. Returns full ICS text."""
    cal = Calendar()
    cal.add("PRODID", "-//seminar-guide//EN")
    cal.add("VERSION", "2.0")
    cal.add("X-WR-CALNAME", "Seminar")
    sh, sm = map(int, start_time.split(":"))
    eh, em = map(int, end_time.split(":"))
    for r in rows:
        d = datetime.fromisoformat(r["date"]).date()
        ev = Event()
        ev.add("uid", f"{r['date']}-{r['student']}@seminar-guide")
        ev.add("summary", f"{r['paper']} — {r['student']}")
        ev.add("dtstart", datetime.combine(d, time(sh, sm), tzinfo=TZ))
        ev.add("dtend", datetime.combine(d, time(eh, em), tzinfo=TZ))
        ev.add("location", venue)
        ev.add("description", f"Paper: {r['paper']}\nPresenter: {r['student']}")
        cal.add_component(ev)
    return cal.to_ical().decode("utf-8")
