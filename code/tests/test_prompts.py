from dora_repro.prompts import TrainingSample, extract_prediction, format_training_prompt


def test_format_training_prompt_contains_sections() -> None:
    prompt = format_training_prompt(
        TrainingSample(instruction="Solve it", input="2+2", output="4"),
    )
    assert "### Instruction:" in prompt
    assert "### Input:" in prompt
    assert "### Response:" in prompt


def test_extract_prediction_for_supported_tasks() -> None:
    assert extract_prediction("boolq", "The answer is true.") == "true"
    assert extract_prediction("piqa", "solution2 looks better") == "solution2"
    assert extract_prediction("social_i_qa", "I pick answer3.") == "answer3"
    assert extract_prediction("hellaswag", "ending4") == "ending4"
    assert extract_prediction("winogrande", "option1") == "option1"
