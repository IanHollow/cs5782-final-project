from dora_repro.prompts import (
    EvalSample,
    TrainingSample,
    extract_prediction,
    format_eval_prompt,
    format_training_prompt,
    format_user_prompt,
)


def test_format_training_prompt_contains_sections() -> None:
    prompt = format_training_prompt(
        TrainingSample(instruction="Solve it", input="2+2", output="4"),
    )
    assert "### Instruction:" in prompt
    assert "### Input:" in prompt
    assert "### Response:" in prompt


def test_format_user_and_eval_prompts_use_shared_wrapper() -> None:
    user_prompt = format_user_prompt(
        TrainingSample(instruction="Solve it", input="", output="4"),
    )
    eval_prompt = format_eval_prompt(
        EvalSample(
            id="1", task="boolq", instruction="Answer true or false.", choices=(), label="true"
        ),
    )
    assert user_prompt.endswith("### Response:\n")
    assert "### Instruction:\nSolve it" in user_prompt
    assert eval_prompt.endswith("### Response:\n")
    assert "### Instruction:\nAnswer true or false." in eval_prompt


def test_extract_prediction_for_supported_tasks() -> None:
    assert extract_prediction("boolq", "The answer is true.") == "true"
    assert extract_prediction("piqa", "solution2 looks better") == "solution2"
    assert extract_prediction("social_i_qa", "I pick answer3.") == "answer3"
    assert extract_prediction("hellaswag", "ending4") == "ending4"
    assert extract_prediction("winogrande", "option1") == "option1"
