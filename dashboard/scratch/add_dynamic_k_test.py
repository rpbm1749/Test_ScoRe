from pathlib import Path

p = Path("d:/projects/Test_ScoRe/tests/test_mock_loop.py")
raw = p.read_bytes()

TEST_DYNAMIC_K = (
    b"\r\n\r\ndef test_dynamic_k_declared_by_planner():\r\n"
    b"    # Planner sets EXECUTOR_STEPS: 4 explicitly, overriding the default k=2\r\n"
    b"    env = FakeEnv()\r\n"
    b"    dyn_instr = \"SUBGOAL: work\\nEXECUTOR_STEPS: 4\\nSTEPS:\\n1. step\\nKNOWN:\\nSUCCESS CHECK:\\nHAND BACK IF:\"\r\n"
    b"    planner = ScriptedLLM(\"planner\", [\r\n"
    b"        plan(\"p1 = 1\", dyn_instr),\r\n"
    b"        code(\"apis.supervisor.complete_task()\"),\r\n"
    b"    ])\r\n"
    b"    executor = ScriptedLLM(\"executor\", [\r\n"
    b"        code(\"e1 = 1\"),\r\n"
    b"        code(\"e2 = 1\"),\r\n"
    b"        code(\"e3 = 1\"),\r\n"
    b"        code(\"e4 = 1\"),\r\n"
    b"    ])\r\n"
    b"    reason, traj, extra = run_score(env, planner, executor, CFG, verbose=False)\r\n"
    b"    assert reason == \"completed\"\r\n"
    b"    # Window 1 should have 4 executor steps because planner declared 4\r\n"
    b"    assert extra[\"windows\"][0][\"executor_steps\"] == [2, 3, 4, 5]\r\n"
    b"    assert \"budget_exhausted (k=4)\" in extra[\"windows\"][0][\"handback_reason\"]\r\n"
)

raw = raw + TEST_DYNAMIC_K
p.write_bytes(raw)
print("Added test_dynamic_k_declared_by_planner to test_mock_loop.py")
