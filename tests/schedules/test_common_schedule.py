import pytest
from schedules import common_schedule


@pytest.mark.parametrize("hyper", [64, 128, 256])
@pytest.mark.parametrize("warmup_steps", [2000, 4000, 8000])
def test_common_schedule(hyper, warmup_steps):
    schedule = common_schedule(hyper, warmup_steps)

    before_warmup = []
    for step in range(0, warmup_steps, 250):
        before_warmup.append(schedule(step))

    after_warmup = []
    for step in range(warmup_steps + 250, warmup_steps * 2, 250):
        after_warmup.append(schedule(step))

    at_warmup = schedule(warmup_steps)

    for value in before_warmup + after_warmup:
        assert at_warmup >= value
