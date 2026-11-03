from pathlib import Path

p = Path("d:/projects/Test_ScoRe/score/agents.py")
raw = p.read_bytes()

OLD = (
    b"        \"content\": P.PLANNER_PROMPT.format(\r\n"
    b"            guide=P.APPWORLD_GUIDE,\r\n"
    b"            k_default=k_default, k_min=k_min, k_max=k_max,\r\n"
    b"            task_line=gstate.task_line,\r\n"
    b"        ) + \"\\n\\n\" + P.PLANNER_FIRST_TURN_NOTE,"
)
NEW = (
    b"        \"content\": P.PLANNER_PROMPT.format(\r\n"
    b"            guide=P.APPWORLD_GUIDE,\r\n"
    b"            k=k_default,\r\n"
    b"            task_line=gstate.task_line,\r\n"
    b"        ) + \"\\n\\n\" + P.PLANNER_FIRST_TURN_NOTE,"
)

assert OLD in raw, "Target format call in agents.py not found!"
raw = raw.replace(OLD, NEW)
p.write_bytes(raw)
print("Updated agents.py to pass k=k_default.")
