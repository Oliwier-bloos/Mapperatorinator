import torch
import torch.nn.functional as F
from transformers import LogitsProcessor

from osuT5.osuT5.dataset.data_utils import TIMED_EVENTS
from osuT5.osuT5.event import EventType, Event
from osuT5.osuT5.tokenizer import Tokenizer

MILISECONDS_PER_SECOND = 1000
MILISECONDS_PER_STEP = 10


def get_beat_type_tokens(tokenizer: Tokenizer) -> tuple[int, ...]:
    beat_range = [
        tokenizer.event_start[EventType.BEAT],
        tokenizer.event_start[EventType.MEASURE],
    ]
    if EventType.TIMING_POINT in tokenizer.event_start:
        beat_range.append(tokenizer.event_start[EventType.TIMING_POINT])
    return tuple(beat_range)


def get_mania_type_tokens(tokenizer: Tokenizer) -> tuple[int, ...]:
    return tuple([
        tokenizer.event_start[EventType.CIRCLE],
        tokenizer.event_start[EventType.HOLD_NOTE],
        tokenizer.event_start[EventType.HOLD_NOTE_END],
    ] if EventType.HOLD_NOTE_END in tokenizer.event_start else [])


def get_scroll_speed_tokens(tokenizer: Tokenizer) -> tuple[int, ...]:
    return tuple(range(tokenizer.event_start[EventType.SCROLL_SPEED], tokenizer.event_end[EventType.SCROLL_SPEED])
                 if EventType.SCROLL_SPEED in tokenizer.event_start else [])


def get_snapping_tokens(tokenizer: Tokenizer) -> tuple[int, ...]:
    return tuple(range(tokenizer.event_start[EventType.SNAPPING], tokenizer.event_end[EventType.SNAPPING])
                 if EventType.SNAPPING in tokenizer.event_start else [])


def get_mania_column_tokens(tokenizer: Tokenizer) -> tuple[int, ...]:
    return tuple(range(tokenizer.event_start[EventType.MANIA_COLUMN], tokenizer.event_end[EventType.MANIA_COLUMN])
                 if EventType.MANIA_COLUMN in tokenizer.event_start else [])


class TimeshiftBias(LogitsProcessor):
    def __init__(self, timeshift_bias: float, time_start: int, time_end: int):
        self.timeshift_bias = timeshift_bias
        self.time_range = slice(time_start, time_end)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.Tensor:
        scores_processed = scores.clone()
        scores_processed[:, self.time_range] += self.timeshift_bias
        return scores_processed


class ConditionalTemperatureLogitsWarper(LogitsProcessor):
    """Temperature warper with separate temperatures for timing, mania columns and taiko hits.

    Which tokens get which temperature depends on the token order the model was trained with.

    types_first=True (e.g. V28, V29, V31): every event group starts with its type token, so
    the time shift right after a beat, measure or timing point type is timing, and the token
    three after a mania note type is its column.

    types_first=False (e.g. V30, V32): every event group ends with its type token, so those
    look-back rules cannot work. Instead:
      - the TIMING context pass uses the timing temperature for every token, because that
        pass only contains timing events;
      - mania column tokens are resampled at the mania column temperature within their own
        category, which keeps the total probability of picking a column unchanged.

    In both orders a taiko hit (whose don/kat type is stored in the hitsound token) follows
    a snapping token and a scroll speed token, and no other object has that pattern, so the
    taiko hit temperature uses the same rule for both.
    """

    def __init__(
            self,
            temperature: float,
            timing_temperature: float,
            mania_column_temperature: float,
            taiko_hit_temperature: float,
            types_first: bool,
            beat_type_tokens: tuple[int, ...],
            mania_type_tokens: tuple[int, ...],
            scroll_speed_tokens: tuple[int, ...],
            snapping_tokens: tuple[int, ...] = (),
            mania_column_tokens: tuple[int, ...] = (),
            timing_context: bool = False,
    ):
        for name, value in [
            ("temperature", temperature),
            ("timing_temperature", timing_temperature),
            ("mania_column_temperature", mania_column_temperature),
            ("taiko_hit_temperature", taiko_hit_temperature),
        ]:
            if not value > 0:
                raise ValueError(f"`{name}` has to be a strictly positive float, but is {value}")

        # Each rule is (temperature, [(offset, tokens), ...]). A rule matches a sequence when,
        # for every entry, the token `offset` positions from the end is one of `tokens`.
        # Rules are checked in order and the first match wins.
        self.conditionals: list[tuple[float, list[tuple[int, torch.Tensor]]]] = []
        # Each entry is (temperature, start, end): a token range sampled at its own temperature.
        self.category_temperatures: list[tuple[float, int, int]] = []

        if not types_first and timing_context:
            # The timing pass only contains timing events, so every token is timing
            self.temperature = timing_temperature
            return
        self.temperature = temperature

        def tensor(tokens):
            return torch.tensor(tokens, dtype=torch.long)

        if types_first:
            if timing_temperature != temperature and len(beat_type_tokens) > 0:
                self.conditionals.append((timing_temperature, [(1, tensor(beat_type_tokens))]))
            if mania_column_temperature != temperature and len(mania_type_tokens) > 0:
                self.conditionals.append((mania_column_temperature, [(3, tensor(mania_type_tokens))]))
        elif mania_column_temperature != temperature and len(mania_column_tokens) > 0:
            start, end = mania_column_tokens[0], mania_column_tokens[-1] + 1
            if end - start != len(mania_column_tokens):
                raise ValueError("Mania column tokens must be a contiguous token range")
            self.category_temperatures.append((mania_column_temperature, start, end))

        if taiko_hit_temperature != temperature and len(scroll_speed_tokens) > 0 and len(snapping_tokens) > 0:
            self.conditionals.append((taiko_hit_temperature, [(2, tensor(snapping_tokens)), (1, tensor(scroll_speed_tokens))]))

    def _row_temperatures(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.Tensor:
        temperatures = torch.full((scores.shape[0], 1), self.temperature, dtype=scores.dtype, device=scores.device)
        # Apply the rules in reverse so that the first matching rule wins
        for i in reversed(range(len(self.conditionals))):
            temperature, pattern = self.conditionals[i]
            if input_ids.shape[1] < max(offset for offset, _ in pattern):
                continue
            matches = torch.ones(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
            for j, (offset, tokens) in enumerate(pattern):
                if tokens.device != input_ids.device:
                    tokens = tokens.to(input_ids.device)
                    pattern[j] = (offset, tokens)
                matches &= torch.isin(input_ids[:, -offset], tokens)
            temperatures = torch.where(matches.to(scores.device).unsqueeze(1), temperature, temperatures)
        return temperatures

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.Tensor:
        if len(self.conditionals) == 0 and len(self.category_temperatures) == 0:
            return scores / self.temperature

        temperatures = self._row_temperatures(input_ids, scores) if len(self.conditionals) > 0 else self.temperature
        scores_processed = scores / temperatures

        for temperature, start, end in self.category_temperatures:
            category = scores[:, start:end]
            warped = category / temperature
            # Keep the category's total probability mass, only reshape the distribution inside it
            mass = torch.logsumexp(scores_processed[:, start:end], dim=-1, keepdim=True)
            warped_mass = torch.logsumexp(warped, dim=-1, keepdim=True)
            shift = torch.where(torch.isfinite(mass) & torch.isfinite(warped_mass), mass - warped_mass, 0)
            scores_processed[:, start:end] = warped + shift

        return scores_processed


class LookbackBiasLogitsWarper(LogitsProcessor):
    """Suppress repeated lookback events without inflating unrelated token categories.

    For time-first streams, redistribute past-timestamp mass among allowed timestamps.
    Type-first streams correct the preceding event/EOS decision when its time is sampled.
    """
    def __init__(self, lookback_max_time: float, tokenizer: Tokenizer, types_first: bool, device):
        self.types_first = types_first
        self.lookback_start = tokenizer.event_start[EventType.TIME_SHIFT]
        self.lookback_end = tokenizer.encode(Event(EventType.TIME_SHIFT, int(lookback_max_time / MILISECONDS_PER_STEP)))
        self.time_shift_end = tokenizer.event_end[EventType.TIME_SHIFT]
        self.eos_id = tokenizer.eos_id
        self.lookback_range = torch.full((tokenizer.vocab_size_out,), False, dtype=torch.bool, device=device)
        self.lookback_range[self.lookback_start:self.lookback_end] = True
        self.other_range = ~self.lookback_range
        self.eos_ids = torch.tensor([tokenizer.eos_id] + [tokenizer.context_eos[context] for context in tokenizer.context_eos], dtype=torch.long, device=device)

        self.last_scores = None
        self.timed_tokens = []
        for event_type in TIMED_EVENTS:
            if event_type in tokenizer.event_start:
                self.timed_tokens.extend(list(range(tokenizer.event_start[event_type], tokenizer.event_end[event_type])))
        self.timed_tokens = torch.tensor(self.timed_tokens, dtype=torch.long, device=device)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.Tensor:
        if not self.types_first:
            scores_processed = scores.clone()
            time_mass = torch.logsumexp(scores[:, self.lookback_start:self.time_shift_end], dim=-1)
            allowed_scores = scores[:, self.lookback_end:self.time_shift_end]
            allowed_mass = torch.logsumexp(allowed_scores, dim=-1)
            has_allowed_time = torch.isfinite(allowed_mass)
            # Preserve the total timestamp mass and the relative scores of the
            # allowed timestamps. Other categories retain their original logits.
            shift = torch.where(has_allowed_time, time_mass - allowed_mass, 0)
            scores_processed[:, self.lookback_end:self.time_shift_end] = allowed_scores + shift.unsqueeze(-1)
            scores_processed[:, self.lookback_range] = -torch.inf
            # If all timestamps are masked, terminate rather than boosting
            # positions/hitsounds or producing NaNs from an empty category.
            scores_processed[:, self.eos_id] = torch.where(
                has_allowed_time, scores[:, self.eos_id],
                torch.logaddexp(scores[:, self.eos_id], time_mass),
            )
            return scores_processed

        scores_processed = scores

        if input_ids.shape[1] != 0 and self.last_scores is not None:
            last_token = input_ids[:, -1]
            last_timed = torch.isin(last_token, self.timed_tokens)
            if last_timed.any():
                # The scores are for a timeshift event
                last_probs = F.softmax(self.last_scores, dim=-1)
                probs = F.softmax(scores, dim=-1)
                prob_eos = last_probs[:, self.eos_ids].sum(dim=-1)
                prob_event = 1 - prob_eos
                s = 1 / (probs[:, self.other_range].sum(dim=-1) * prob_event + prob_eos)
                probs[:, self.lookback_range] = 0
                probs[:, self.other_range] *= s.unsqueeze(1)
                # Probability of eos now which should have been at the previous token
                prob_eos_extra = torch.clip((s - 1) * prob_eos / prob_event, 0, 1)  # Clip to avoid numerical instability
                probs[:, self.lookback_start] = prob_eos_extra  # This will be treated as eos if trim lookback is true

                scores_processed = torch.where(last_timed.unsqueeze(1), torch.log(probs), scores)

        self.last_scores = scores
        return scores_processed


class MonotonicTimeShiftLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.time_shift_start = tokenizer.event_start[EventType.TIME_SHIFT]
        self.time_shift_end = tokenizer.event_end[EventType.TIME_SHIFT]
        self.sos_ids = torch.tensor([tokenizer.sos_id] + list(getattr(tokenizer, "context_sos", {}).values()))

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        self.sos_ids = self.sos_ids.to(device)

        # Create masks for time_shift and sos tokens
        is_time_shift = (input_ids >= self.time_shift_start) & (input_ids < self.time_shift_end)
        is_sos = torch.isin(input_ids, self.sos_ids)

        # Create a sequence of indices [0, 1, ..., seq_len-1]
        indices = torch.arange(seq_len, device=device).expand(batch_size, -1)

        # Find the index of the last occurrence of each event type
        # If not found, it will be -1
        last_time_shift_idx = torch.max(torch.where(is_time_shift, indices, -1), dim=1).values
        last_sos_idx = torch.max(torch.where(is_sos, indices, -1), dim=1).values

        # Get the value of the last time_shift token for each sequence
        # If no time_shift token, value is 0
        last_time_shift_values = torch.where(
            last_time_shift_idx != -1,
            input_ids[torch.arange(batch_size), last_time_shift_idx] - self.time_shift_start,
            0
        )

        # Determine which sequences in the batch need masking
        # Mask if a time_shift token was found and it did not appear after the last SOS token
        apply_mask = (last_time_shift_idx != -1) & (last_time_shift_idx > last_sos_idx)

        # Create the mask for invalid time_shift tokens
        time_shift_vocab = torch.arange(self.time_shift_start, self.time_shift_end, device=device)
        # Shape: (batch_size, num_time_shift_tokens)
        invalid_mask = time_shift_vocab.unsqueeze(0) < (self.time_shift_start + last_time_shift_values).unsqueeze(1)

        # Apply the mask to the scores for the relevant sequences
        # Shape: (batch_size, vocab_size)
        batch_mask = torch.full_like(scores, False, dtype=torch.bool)
        batch_mask[:, self.time_shift_start:self.time_shift_end] = invalid_mask
        scores[apply_mask] = scores[apply_mask].masked_fill(batch_mask[apply_mask], -torch.inf)

        return scores
