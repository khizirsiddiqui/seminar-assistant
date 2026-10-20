from icalendar import Calendar

from app.ics import build_calendar

ROWS = [
    {"date": "2026-10-27", "student": "Alice", "paper": "Attention Is All You Need"},
    {"date": "2026-11-03", "student": "Bob", "paper": "BERT"},
]


def test_build_calendar_contains_events():
    ics = build_calendar(ROWS, venue="Room 1.06, Sand 14", start_time="14:15", end_time="15:45")
    assert ics.startswith("BEGIN:VCALENDAR")
    assert ics.count("BEGIN:VEVENT") == 2
    assert "Attention Is All You Need" in ics
    assert "Alice" in ics
    assert "20261027T141500" in ics  # DTSTART local time rendered by icalendar
    assert "SUMMARY:" in ics
    # roundtrip-parse to verify values (commas are escaped in raw ICS per RFC 5545)
    cal = Calendar.from_ical(ics.encode("utf-8"))
    events = [e for e in cal.walk("VEVENT")]
    assert len(events) == 2
    assert str(events[0]["LOCATION"]) == "Room 1.06, Sand 14"
    assert str(events[0]["SUMMARY"]) == "Attention Is All You Need — Alice"
    assert events[0]["DTSTART"].dt.hour == 14 and events[0]["DTSTART"].dt.minute == 15
    assert str(events[0]["DTSTART"].params["TZID"]) == "Europe/Berlin"
