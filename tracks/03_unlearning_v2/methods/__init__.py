"""Strategy registry for Track 3 unlearning benchmark."""

from methods.abliteration import run_abliteration
from methods.bad_teacher import run_bad_teacher
from methods.lora_unlearn import run_lora_npo
from methods.salun_npo import run_salun_npo
from methods.ssd import run_ssd

STRATEGY_RUNNERS = {
    "salun_npo": run_salun_npo,
    "bad_teacher": run_bad_teacher,
    "ssd": run_ssd,
    "lora_npo": run_lora_npo,
    "abliteration": run_abliteration,
}
