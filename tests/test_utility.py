from mp_deep_research.tools.utility import think

def test_think_tool():
    result = think.invoke({"thought": "I should check the stability."})
    assert result == "Thought recorded: I should check the stability."