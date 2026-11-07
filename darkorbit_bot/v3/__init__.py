"""
V3 - one object-token policy trained on raw recordings, for any game that
has a profile in games/<name>/profile.yaml (DarkOrbit is the first).

    agent.py    record your play / run the bot (F1 pause, F2 takeover, F3 stop)
    train.py    train the policy from recorded sessions
    features.py the ONLY place inputs are built (shared by agent and dataset)
    profile.py  game profiles: detector, class -> role rules, HUD bars, keys

V2 (darkorbit_bot/v2) is untouched; V3 reuses its perception and input code.
"""
