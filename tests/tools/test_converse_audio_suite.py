"""Server-side proof that the converse voice pipeline hears speech — including QUIET speech.

Everything here drives the REAL VAD, endpointer and WAV-capture path (only STT is mocked),
using audio whose block-RMS envelope mimics real speech across a range of volumes. This is
the regression wall for "I'm talking and the server just reports idle": it fails if the
detector stops hearing a soft voice, and it fails if it starts hearing silence/room noise.
"""
import numpy as np
import pytest

from tests.support.converse_audio import (
    RMS_AT_ROOM_FLOOR, RMS_LOUD_SPEECH, RMS_NORMAL_SPEECH, RMS_QUIET_SPEECH,
    heard, room_noise, run_utterances, silence, speech_like,
)


class TestConverseHearsSpeechAcrossVolume:
    """Volume matrix through the real ConverseSession (VAD + endpoint + capture; STT mocked)."""

    @pytest.mark.parametrize("rms", [RMS_QUIET_SPEECH, 700, 1000, RMS_NORMAL_SPEECH, RMS_LOUD_SPEECH])
    def test_speech_is_heard(self, rms):
        # A soft voice (500 RMS) through a loud one must all produce a transcript. This is the
        # core "quiet speech works" guarantee — the 0.6 sustained-window + adaptive floor.
        assert heard(speech_like(rms, seed=rms)), f"speech at {rms} RMS was not heard"

    def test_quiet_speech_across_several_takes(self):
        # Robustness to the random envelope: several independent soft-voice takes are all heard.
        for seed in range(5):
            assert heard(speech_like(RMS_QUIET_SPEECH, seed=100 + seed)), f"soft take {seed} missed"


class TestConverseRejectsNonSpeech:
    """The other half of a real VAD: it must NOT fire on silence or steady room noise."""

    def test_silence_is_not_heard(self):
        assert not heard(silence(), timeout=2.5)

    @pytest.mark.parametrize("rms", [120, 200, 250])
    def test_room_noise_is_not_heard(self, rms):
        assert not heard(room_noise(rms, seed=rms), timeout=2.5), f"room noise {rms} false-tripped"

    def test_signal_at_the_room_floor_is_not_heard(self):
        # ~300 RMS "speech" sits in the room tone — indistinguishable, must not trip (avoids
        # spurious turns). A real client normalizes a soft voice well above this.
        assert not heard(speech_like(RMS_AT_ROOM_FLOOR, seed=3), timeout=2.5)


class TestConverseMultiTurn:
    """The socket stays open the whole time: several utterances on one session each produce a
    turn, with quiet gaps between them left un-triggered."""

    def test_three_utterances_yield_three_transcripts(self):
        segs = [speech_like(RMS_NORMAL_SPEECH, seed=s) for s in (11, 22, 33)]
        got = run_utterances(segs, transcript="ok", timeout=6.0)
        assert got == ["ok", "ok", "ok"], f"expected 3 turns, got {got}"

    def test_gap_between_utterances_is_not_its_own_turn(self):
        # A soft voice, a long quiet gap, then a soft voice → exactly two turns (the gap alone
        # never trips).
        segs = [speech_like(RMS_QUIET_SPEECH, seed=44), speech_like(RMS_QUIET_SPEECH, seed=55)]
        got = run_utterances(segs, timeout=6.0)
        assert len(got) == 2, f"expected 2 turns, got {len(got)}: {got}"


class TestBoundaryIsDocumented:
    """Pin the sensitivity boundary so a future change that regresses it is caught: a soft
    voice at 400 RMS is heard, room tone at 250 is not."""

    def test_400_rms_heard_250_noise_silent(self):
        assert heard(speech_like(400, seed=400)), "400 RMS speech should be heard"
        assert not heard(room_noise(250, seed=8), timeout=2.5), "250 RMS noise should be silent"
