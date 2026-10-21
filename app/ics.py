from datetime import datetime, time
from zoneinfo import ZoneInfo

from icalendar import Calendar, Event

TZ = ZoneInfo("Europe/Berlin")


def build_calendar(rows: list[dict], venue: str, start_time: str, end_time: str,
                   course_title: str = "") -> str:
    """rows: [{date: YYYY-MM-DD, student, paper}]. Returns full ICS text.

    Rows sharing a date are grouped into one VEVENT; the description lists every
    student/paper pair for that date.
    """
    cal = Calendar()
    cal.add("PRODID", "-//seminar-assistant//EN")
    cal.add("VERSION", "2.0")
    cal.add("X-WR-CALNAME", course_title or "Seminar")
    sh, sm = map(int, start_time.split(":"))
    eh, em = map(int, end_time.split(":"))

    grouped: dict[str, list[dict]] = {}
    for r in rows:
        grouped.setdefault(r["date"], []).append(r)

    for date_key, group in grouped.items():
        d = datetime.fromisoformat(date_key).date()
        ev = Event()
        ev.add("uid", f"{date_key}@seminar-assistant")
        ev.add("summary", course_title or "Seminar")
        ev.add("dtstart", datetime.combine(d, time(sh, sm), tzinfo=TZ))
        ev.add("dtend", datetime.combine(d, time(eh, em), tzinfo=TZ))
        ev.add("location", venue)
        ev.add("description", "\n".join(
            f"{r['paper']} — {r['student']}" for r in group))
        cal.add_component(ev)
    return cal.to_ical().decode("utf-8")
